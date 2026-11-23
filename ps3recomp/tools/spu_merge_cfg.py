#!/usr/bin/env python3
"""spu_merge_cfg.py -- Merge SPU_XFER basic-block functions into multi-block C functions.

spu_lifter.py emits branch targets as separate C functions chained via SPU_XFER.
Inside hot loops and branches, every SPU_XFER hop spills dirty GPRs to ctx->gpr[],
executes a tail-call boundary, and reloads live GPRs in the next function.

This build-time source-to-source pass operates on register-cached lifted SPU TUs
(spu_recomp_rc.c or spu_recomp_rc_only.c):
  1. Parses all <P>_rc_spu_func_XXXXXXXX functions and their static SPU_XFER graph.
  2. Groups connected functions (WCCs by default, or SCCs with --mode scc; capped
     at --max-comp blocks per merged function).
  3. Un-caches each block back to architectural ctx->gpr[] references while
     preserving v200 spu_f*_k, SPU_IBF_SITE/RETSELF/CALL, and lodx hooks.
  4. Replaces internal SPU_XFER edges with fast-path local `goto blk_XXXXXXXX;`
     (preserving exact ctx->pc, ctx->steps++, interrupt, lockstep, slow-hook,
     and stack-reset semantics via a cold spill + trampoline fallback).
  5. Runs unified forward may-dirty + backward liveness register caching over the
     entire merged CFG, loading only live-in GPRs per entry `case 0xXXXXu:`.
  6. Emits thin musttail entry shims for each original <P>_rc_spu_func_XXXXXXXX
     so dispatch tables (s_rc_only_table, s_ibf_fn) and direct callers work
     unchanged with zero host stack growth.

Usage:
  spu_merge_cfg.py <in.c> <out.c> [--mode wcc|scc] [--max-comp 160]
                   [--prefix-rename NEW_P] [--stats]
"""
import collections
import os
import re
import sys

GPR_RE = re.compile(r'ctx->gpr\[(\d+)\]')
ASSIGN_RE = re.compile(r'ctx->gpr\[(\d+)\]\s*=(?!=)')
REG_LOCAL_RE = re.compile(r'\br(\d+)\b')
SPILL_ONLY_RE = re.compile(r'^\s*(?:ctx->gpr\[\d+\]\s*=\s*r\d+;\s*)+$')
RELOAD_ONLY_RE = re.compile(r'^\s*(?:r\d+\s*=\s*ctx->gpr\[\d+\];\s*)*spu_rc_ls\s*=\s*ctx->ls;\s*$')
DECL_U128_RE = re.compile(r'^\s*u128\s+(?:r\d+\s*=\s*ctx->gpr\[\d+\](?:,\s*)?)+;\s*$')
DECL_LS_RE = re.compile(r'^\s*uint8_t\*\s*(?:__restrict\s+)?spu_rc_ls\s*=\s*ctx->ls;.*$')
NORET_RE = re.compile(r'\breturn\b|\bSPU_RET\b|\bSPU_RET_REG\b|\bSPU_FAST_XFER\b|\bSPU_TAILCALL\b')
LABEL_RE = re.compile(r'^\s*([A-Za-z_]\w*):\s*;?\s*$')
GOTO_RE = re.compile(r'^\s*(?:if\s*\((.*)\)\s*)?goto\s+([A-Za-z_]\w*);\s*$')

CFG_HELPER_MACROS = r'''
/* spu_merge_cfg.py: fast-path condition matching SPU_XFER + spu_tlc (pc != 0) */
#ifndef SPU_CFG_FAST_COND
#  if defined(SPU_HOP_RECENT)
#    define SPU_CFG_RECENT_NOTE(pcv) (g_spu_recent[g_spu_recent_n++ & 31] = (uint32_t)(pcv) & SPU_LS_MASK)
#  else
#    define SPU_CFG_RECENT_NOTE(pcv) ((void)0)
#  endif
#  define SPU_CFG_FAST_COND(ctx) \
       (__builtin_expect(g_spu_dtail > 0, 1) && \
        __builtin_expect((g_spu_tlc_slow | (int)g_spu_stack_reset_n | (int)g_yz_lockstep_on) == 0, 1) && \
        __builtin_expect(!((ctx)->int_enable && ((ctx)->event_status & (ctx)->event_mask)), 1))
#  if defined(__clang__)
#    define SPU_CFG_MUSTTAIL(call) __attribute__((musttail)) return call
#  else
#    define SPU_CFG_MUSTTAIL(call) do { call; return; } while (0)
#  endif
#endif
'''


def is_pure(s):
    t = GPR_RE.sub('', s).replace('spu_ls_read128(ctx,', '').replace('spu_ls_write128(ctx,', '')
    return not re.search(r'\bctx\b|\breturn\b|\bextern\b|g_spu_|\bSPU_|^\s*#', t)


def split_if(s):
    """'if (C) { B }' on one line -> (indent, C, B) or None."""
    st = s.lstrip()
    if not st.startswith('if ('):
        return None
    ind = s[:len(s) - len(st)]
    dep = 0
    end_j = -1
    for j in range(3, len(st)):
        if st[j] == '(':
            dep += 1
        elif st[j] == ')':
            dep -= 1
            if dep == 0:
                end_j = j
                break
    if end_j < 0:
        return None
    cond = st[4:end_j]
    rest = st[end_j + 1:].rstrip()
    if not (rest.startswith(' {') and rest.endswith('}')):
        return None
    body = rest[2:-1]
    d = 0
    for ch in body:
        if ch == '{':
            d += 1
        elif ch == '}':
            d -= 1
            if d < 0:
                return None
    if d != 0:
        return None
    return ind, cond, body


def loc_regs(s):
    return GPR_RE.sub(lambda m: 'r' + m.group(1), s)


def uncache_block(pc, body_lines, prefix):
    """Convert a register-cached function body back to architectural ctx->gpr[] lines,
    namespacing local labels (loc_*, spu_ibf_top) to this block PC."""
    ibf_lbl = 'blk_%s_ibf' % pc
    out = []
    has_ibf_top = False

    # First check which local labels exist in this block
    local_labels = set()
    for line in body_lines:
        m = re.match(r'^\s*(loc_[0-9A-Fa-f]+):\s*$', line)
        if m:
            local_labels.add(m.group(1))
        if line.strip().startswith('spu_ibf_top:'):
            has_ibf_top = True

    def rename_labels(s):
        for lb in local_labels:
            s = re.sub(r'\b' + re.escape(lb) + r'\b', 'loc_%s_%s' % (pc, lb[4:]), s)
        if has_ibf_top:
            s = s.replace('goto spu_ibf_top;', 'goto %s;' % ibf_lbl)
            # Rewrite SPU_IBF_SITE(...) and SPU_IBF_RETSELF(...) to *_L(..., ibf_lbl)
            s = re.sub(
                r'SPU_IBF_SITE\(([^)]+)\)',
                lambda mm: 'SPU_IBF_SITE_L(%s, %s)' % (mm.group(1), ibf_lbl),
                s
            )
            s = re.sub(
                r'SPU_IBF_RETSELF\(([^)]+)\)',
                lambda mm: 'SPU_IBF_RETSELF_L(%s, %s)' % (mm.group(1), ibf_lbl),
                s
            )
        return s

    for raw in body_lines:
        s = raw.strip()
        if not s:
            continue
        if DECL_LS_RE.match(s) or DECL_U128_RE.match(s):
            continue
        if SPILL_ONLY_RE.match(s) or RELOAD_ONLY_RE.match(s):
            continue
        if s.startswith('spu_ibf_top:'):
            out.append('%s: ;' % ibf_lbl)
            continue

        # Handle v200 loop-back line: convert rN -> ctx->gpr[N], rename spu_ibf_top
        if '/* v200 loop-back without write-back */' in raw:
            line2 = REG_LOCAL_RE.sub(lambda m: 'ctx->gpr[%s]' % m.group(1), raw)
            line2 = rename_labels(line2)
            out.append(line2)
            continue

        # Handle `ctx->gpr[N] = spu_link(0xADDR); { ctx->pc = 0xADDR; SPU_XFER(...); }`
        m_link_xfer = re.match(
            r'^(\s*)(ctx->gpr\[\d+\]\s*=\s*spu_link\([^)]+\);)\s*(\{\s*ctx->pc\s*=\s*0x[0-9A-Fa-f]+;\s*SPU_XFER\(ctx,\s*\w+\);\s*\})\s*$',
            raw
        )
        if m_link_xfer:
            ind, link_stmt, xfer_stmt = m_link_xfer.groups()
            out.append(ind + link_stmt)
            out.append(ind + xfer_stmt)
            continue

        # Handle `if (cond) { <spills> <body...> <reloads> }`
        sp = split_if(raw)
        if sp:
            ind, cond, ibody = sp
            cond2 = REG_LOCAL_RE.sub(lambda m: 'ctx->gpr[%s]' % m.group(1), cond)
            ib2 = ibody.strip()
            # Strip leading spills `ctx->gpr[N] = rN;`
            ib2 = re.sub(r'^(?:ctx->gpr\[\d+\]\s*=\s*r\d+;\s*)+', '', ib2)
            # Strip trailing reloads `rN = ctx->gpr[N]; ... spu_rc_ls = ctx->ls;`
            ib2 = re.sub(r'(?:\s*r\d+\s*=\s*ctx->gpr\[\d+\];)*\s*spu_rc_ls\s*=\s*ctx->ls;\s*$', '', ib2)
            ib2 = rename_labels(ib2.strip())
            out.append('%sif (%s) { %s }' % (ind, cond2, ib2))
            continue

        # Pure line or impure line:
        # In spu_regcache.py, impure lines never have rN locals (only ctx->gpr[N]).
        # Pure lines have rN locals (never ctx->gpr[N]).
        line2 = rename_labels(raw)
        if 'ctx->gpr[' not in line2:
            # Verify it is not an impure line that happens to contain no gpr (like spu_stop/lodx/etc.)
            # None of those use identifier r\d+, except pure lines. Check safety:
            if not re.search(r'\b(spu_wrch|spu_rdch|spu_rchcnt|spu_stop|spu_halt|spu_drain_call|SPU_IBF_|SPU_RET|SPU_XFER|lodx|spu_unsupported)\b', line2):
                line2 = REG_LOCAL_RE.sub(lambda m: 'ctx->gpr[%s]' % m.group(1), line2)
        out.append(line2)

    return out


def tarjan_sccs(nodes, edges):
    index = {}
    lowlink = {}
    stack = []
    on_stack = set()
    sccs = []
    cur = [0]

    def dfs(v):
        index[v] = lowlink[v] = cur[0]
        cur[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in edges.get(v, ()):
            if w not in nodes:
                continue
            if w not in index:
                dfs(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in sorted(nodes):
        if v not in index:
            dfs(v)
    return sccs


def compute_components(nodes, edges, mode='wcc', max_comp=160):
    """Partition function PCs into merge groups (each group has >= 2 blocks or a self-loop)."""
    if mode == 'scc':
        sccs = tarjan_sccs(nodes, edges)
        comps = [sorted(s) for s in sccs if len(s) > 1 or s[0] in edges.get(s[0], ())]
        return comps

    # WCC mode
    adj = collections.defaultdict(set)
    for u in nodes:
        for v in edges.get(u, ()):
            if v in nodes and v != u:
                adj[u].add(v)
                adj[v].add(u)

    seen = set()
    raw_wccs = []
    for u in sorted(nodes):
        if u in seen:
            continue
        q = collections.deque([u])
        seen.add(u)
        comp = []
        while q:
            x = q.popleft()
            comp.append(x)
            for y in sorted(adj[x]):
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        if len(comp) > 1 or u in edges.get(u, ()):
            raw_wccs.append(sorted(comp))

    # Enforce max_comp by splitting oversized WCCs along SCC boundaries
    comps = []
    for wcc in raw_wccs:
        if len(wcc) <= max_comp:
            comps.append(wcc)
            continue
        sub_nodes = set(wcc)
        sccs = tarjan_sccs(sub_nodes, edges)
        # Pack SCCs in topological order (Tarjan emits reverse topological) into chunks <= max_comp
        sccs.reverse()
        cur_chunk = []
        for s in sccs:
            if len(s) > max_comp:
                if len(cur_chunk) > 1 or (len(cur_chunk) == 1 and cur_chunk[0] in edges.get(cur_chunk[0], ())):
                    comps.append(sorted(cur_chunk))
                cur_chunk = []
                comps.append(sorted(s))
                continue
            if len(cur_chunk) + len(s) > max_comp:
                if len(cur_chunk) > 1 or (len(cur_chunk) == 1 and cur_chunk[0] in edges.get(cur_chunk[0], ())):
                    comps.append(sorted(cur_chunk))
                cur_chunk = list(s)
            else:
                cur_chunk.extend(s)
        if len(cur_chunk) > 1 or (len(cur_chunk) == 1 and cur_chunk[0] in edges.get(cur_chunk[0], ())):
            comps.append(sorted(cur_chunk))

    return comps


BRSL_CALL_RE = re.compile(
    r'^(\s*)ctx->gpr\[0\]\s*=\s*spu_link\((0x[0-9A-Fa-f]+)\);\s*\{\s*'
    r'int32_t\s+_si\s*=\s*\(int32_t\)ctx->image_id;\s*ctx->host_depth\+\+;\s*'
    r'(\w+_rc_spu_func_([0-9A-Fa-f]{8}))\(ctx\);\s*'
    r'spu_drain_call\(ctx,\s*(0x[0-9A-Fa-f]+)\);\s*ctx->host_depth--;\s*spu_img_restore\(ctx,\s*_si\);\s*\}\s*$'
)


def is_pure_leaf_block(uncached_lines):
    """Return list of pure body lines (excluding final SPU_RET) if this single-block
    function is a pure leaf that never modifies r0, never branches/transfers/calls,
    and exits exclusively via a single final SPU_RET(ctx);."""
    non_empty = [ln for ln in uncached_lines if ln.strip()]
    if not non_empty or len(non_empty) > 180:
        return None
    if non_empty[-1].strip() != 'SPU_RET(ctx);':
        return None
    body = non_empty[:-1]
    for ln in body:
        st = ln.strip()
        if LABEL_RE.match(ln) or GOTO_RE.match(ln):
            return None
        if not is_pure(ln):
            return None
        # Must never overwrite link register r0
        for m in ASSIGN_RE.finditer(ln):
            if int(m.group(1)) == 0:
                return None
    return body


def build_merged_function(comp_id, comp_pcs, uncached_map, prefix, fn_prefix, entry_pcs=None, regcache=False, pure_leaves=None):
    """Build one merged static function across all blocks in comp_pcs.

    When regcache=False (default, recommended): blocks access ctx->gpr[N] directly
    with a single invariant `uint8_t* __restrict const spu_rc_ls = ctx->ls;`.
    LLVM's SSA/LICM/DSE/RegAlloc promotes only the hottest ~12-14 GPRs per loop
    into XMM registers with zero stack-frame bloat, zero entry loads, zero cold
    spill blocks, and zero brsl spill/reload storms.

    entry_pcs restricts the top `switch (ctx->pc)` to externally-reachable entries
    (or omits the switch entirely when len(entry_pcs) == 1), keeping internal
    natural loops reducible for LLVM loop optimizations.
    """
    comp_set = set(comp_pcs)
    if entry_pcs is None:
        entry_list = list(comp_pcs)
    else:
        entry_list = [pc for pc in comp_pcs if pc in entry_pcs]
        if not entry_list:
            entry_list = [comp_pcs[0]]
    # Place single entry first so fall-through without switch works cleanly
    if len(entry_list) == 1 and comp_pcs[0] != entry_list[0]:
        e0 = entry_list[0]
        comp_pcs = [e0] + [pc for pc in comp_pcs if pc != e0]
    xfer_fn_re = re.compile(
        r'ctx->pc\s*=\s*(0x[0-9A-Fa-f]+);\s*SPU_XFER\(ctx,\s*(' + re.escape(fn_prefix) + r'([0-9A-Fa-f]+))\);'
    )

    # Assemble flat instruction stream (kinds) across all blocks in comp_pcs.
    # Each item is (kind, payload):
    #   'lbl' : label string (e.g. 'blk_000069DC', 'blk_000044C8_ibf', 'loc_000069DC_...')
    #   'p'   : pure line (ctx->gpr[N] -> rN)
    #   'g'   : pure goto line (raw_line, target_lbl, is_cond)
    #   'lb'  : v200 loop-back without write-back (raw_line, target_lbl)
    #   'ix'  : unconditional internal XFER -> (ind, target_pc_hex, target_pc_str, target_fn)
    #   'cix' : conditional internal XFER -> (ind, cond_str, target_pc_hex, target_pc_str, target_fn)
    #   'if'  : impure conditional line -> (ind, cond_str, body_str, ibf_goto_lbl_or_None)
    #   'x'   : impure statement -> (raw_line, is_noret, ibf_goto_lbl_or_None)
    kinds = []
    block_start_idx = {}

    for pc in comp_pcs:
        blk_lbl = 'blk_' + pc
        block_start_idx[pc] = len(kinds)
        kinds.append(('lbl', blk_lbl))
        lines = uncached_map[pc]
        for s in lines:
            st = s.strip()
            if not st:
                continue

            # Label line
            ml = LABEL_RE.match(s)
            if ml:
                kinds.append(('lbl', ml.group(1)))
                continue

            # v200 loop-back line
            if '/* v200 loop-back without write-back */' in s:
                mg = re.search(r'goto\s+([A-Za-z_]\w*);', s)
                if not mg:
                    raise ValueError('bad v200 loop-back: ' + s)
                kinds.append(('lb', (s, mg.group(1))))
                continue

            # Unconditional internal SPU_XFER: `{ ctx->pc = 0xADDR; SPU_XFER(ctx, fn); }`
            if st.startswith('{ ctx->pc =') and 'SPU_XFER(' in st:
                mx = xfer_fn_re.search(st)
                if mx and mx.group(3) in comp_set:
                    ind = s[:len(s) - len(st)]
                    kinds.append(('ix', (ind, mx.group(1), mx.group(3), mx.group(2))))
                    continue

            # Conditional line `if (C) { B }`
            sp = split_if(s)
            if sp and is_pure('if (' + sp[1] + ')'):
                ind, cond, ibody = sp
                if 'SPU_XFER(' in ibody:
                    mx = xfer_fn_re.search(ibody)
                    if mx and mx.group(3) in comp_set:
                        kinds.append(('cix', (ind, cond, mx.group(1), mx.group(3), mx.group(2))))
                        continue
                ibf_tgt = None
                mi = re.search(r'SPU_IBF_(?:SITE|RETSELF)_L\([^)]*,\s*([A-Za-z_]\w*)\)', ibody)
                if mi:
                    ibf_tgt = mi.group(1)
                kinds.append(('if', (ind, cond, ibody, ibf_tgt)))
                continue

            # Pure line / pure goto
            if is_pure(s):
                mg = GOTO_RE.match(s)
                if mg:
                    kinds.append(('g', (s, mg.group(2), mg.group(1) is not None)))
                else:
                    kinds.append(('p', s))
                continue

            if re.search(r'\bgoto\b', s):
                raise ValueError('unexpected impure goto in %s: %s' % (pc, s))

            # Impure line
            is_noret = bool((not re.search(r'\bif\b', s)) and NORET_RE.search(s))
            ibf_tgt = None
            mi = re.search(r'SPU_IBF_(?:SITE|RETSELF)_L\([^)]*,\s*([A-Za-z_]\w*)\)', s)
            if mi:
                ibf_tgt = mi.group(1)
            kinds.append(('x', (s, is_noret, ibf_tgt)))

        # Guard against block fall-through into the next merged block in C source order:
        # if the last statement of this block can fall through, emit an explicit return.
        if kinds:
            last_k, last_v = kinds[-1]
            can_ft = True
            if last_k == 'ix':
                can_ft = False
            elif last_k == 'g' and not last_v[2]:
                can_ft = False
            elif last_k == 'x' and last_v[1]:
                can_ft = False
            if can_ft:
                kinds.append(('x', ('        return;', True, None)))

    # Fast direct-GPR emission (default): zero manual u128 locals, zero stack-frame explosion,
    # zero cold spill blocks, invariant __restrict const spu_rc_ls = ctx->ls.
    if not regcache:
        merged_name = '%s_cfg_%d' % (prefix, comp_id)
        out = ['static void %s(spu_context* __restrict ctx) {' % merged_name]
        out.append('\tuint8_t* __restrict const spu_rc_ls = ctx->ls; (void)spu_rc_ls;')
        if len(entry_list) > 1:
            out.append('\tswitch (ctx->pc & SPU_LS_MASK) {')
            for pc in entry_list:
                out.append('\tcase 0x%Xu: goto blk_%s;' % (int(pc, 16), pc))
            out.append('\tdefault: __builtin_unreachable();')
            out.append('\t}')
        for k, v in kinds:
            if k == 'lbl':
                out.append('%s: ;' % v)
            elif k == 'p':
                out.append(v)
            elif k == 'g':
                out.append(v[0])
            elif k == 'lb':
                out.append(v[0])
            elif k == 'ix':
                ind, pc_hex, tgt_pc, tgt_fn = v
                out.append(
                    '%sif (SPU_CFG_FAST_COND(ctx)) { ctx->pc = %s; ctx->steps++; SPU_CFG_RECENT_NOTE(%s); SPU_PCH_NOTE(ctx); goto blk_%s; }'
                    % (ind, pc_hex, pc_hex, tgt_pc)
                )
                out.append('%s{ ctx->pc = %s; g_spu_trampoline_fn = %s; return; }' % (ind, pc_hex, tgt_fn))
            elif k == 'cix':
                ind, cond, pc_hex, tgt_pc, tgt_fn = v
                out.append('%sif (%s) {' % (ind, cond))
                out.append(
                    '%s    if (SPU_CFG_FAST_COND(ctx)) { ctx->pc = %s; ctx->steps++; SPU_CFG_RECENT_NOTE(%s); SPU_PCH_NOTE(ctx); goto blk_%s; }'
                    % (ind, pc_hex, pc_hex, tgt_pc)
                )
                out.append('%s    ctx->pc = %s; g_spu_trampoline_fn = %s; return;' % (ind, pc_hex, tgt_fn))
                out.append('%s}' % ind)
            elif k == 'if':
                ind, cond, ibody, _ibf_tgt = v
                out.append('%sif (%s) { %s }' % (ind, cond, ibody.strip()))
            elif k == 'x':
                raw_x = v[0]
                mb = BRSL_CALL_RE.match(raw_x) if pure_leaves else None
                if mb and mb.group(4) in pure_leaves and mb.group(2).lower() == mb.group(5).lower():
                    ind = mb.group(1)
                    link_hex = mb.group(2)
                    leaf_pc = mb.group(4)
                    leaf_fn = '%s_leaf_%s' % (prefix, leaf_pc)
                    out.append(
                        '%sif (__builtin_expect(g_spu_drain_fastret_pub > 0 && !g_spu_trampoline_fn && !ctx->irq_frame && '
                        'ctx->host_depth + 65u < g_spu_depth_lim && g_irq_live_n == 0, 1)) '
                        '{ ctx->gpr[0] = spu_link(%s); %s(ctx, spu_rc_ls); ctx->pc = %su; ctx->drain_ret_pc = 0; } '
                        'else { %s }' % (ind, link_hex, leaf_fn, link_hex, raw_x.strip())
                    )
                else:
                    out.append(raw_x)
        out.append('}')
        return merged_name, '\n'.join(out)

    # Collect used / written GPRs across all pure/conditional/loop-back statements
    used = set()
    wr = set()
    for k, v in kinds:
        if k == 'p':
            used.update(int(x) for x in GPR_RE.findall(v))
            wr.update(int(x) for x in ASSIGN_RE.findall(v))
        elif k == 'g':
            used.update(int(x) for x in GPR_RE.findall(v[0]))
        elif k == 'lb':
            used.update(int(x) for x in GPR_RE.findall(v[0]))
        elif k == 'cix':
            used.update(int(x) for x in GPR_RE.findall(v[1]))
        elif k == 'if':
            used.update(int(x) for x in GPR_RE.findall(v[1]))

    U = sorted(used)
    n = len(kinds)

    # Map every label to its index in kinds
    labels = {}
    for i, (k, v) in enumerate(kinds):
        if k == 'lbl':
            labels[v] = i

    # Build CFG successor edges
    succ = []
    for i, (k, v) in enumerate(kinds):
        nx = [i + 1] if i + 1 < n else []
        if k == 'lbl' or k == 'p':
            succ.append(nx)
        elif k == 'g':
            tgt = labels[v[1]]
            succ.append((nx if v[2] else []) + [tgt])
        elif k == 'lb':
            tgt = labels[v[1]]
            succ.append(nx + [tgt])
        elif k == 'ix':
            # Fast path jumps to blk_<target_pc>; slow path returns (no fall-through)
            succ.append([labels['blk_' + v[2]]])
        elif k == 'cix':
            # Taken fast path jumps to blk_<target_pc>; not-taken falls through to nx
            succ.append(nx + [labels['blk_' + v[3]]])
        elif k == 'if':
            # Not-taken falls through; if taken has SPU_IBF_*_L, it can jump to ibf_tgt
            s_list = list(nx)
            if v[3] is not None:
                s_list.append(labels[v[3]])
            succ.append(s_list)
        elif k == 'x':
            s_list = [] if v[1] else list(nx)
            if v[2] is not None:
                s_list.append(labels[v[2]])
            succ.append(s_list)

    # ---- Forward may-dirty analysis ----
    # Every block entry is reachable from the top entry switch with empty dirty set!
    din = [None] * n
    work = []
    for pc in comp_pcs:
        bi = block_start_idx[pc]
        din[bi] = frozenset()
        work.append(bi)

    while work:
        i = work.pop()
        k, v = kinds[i]
        d = din[i]
        if k == 'p':
            out_d = d | frozenset(int(x) for x in ASSIGN_RE.findall(v))
            for j in succ[i]:
                if din[j] is None:
                    din[j] = out_d; work.append(j)
                elif not out_d <= din[j]:
                    din[j] = din[j] | out_d; work.append(j)
        elif k == 'x':
            # Impure line spills all dirty regs before executing -> empty dirty set on all outgoing edges
            out_d = frozenset()
            for j in succ[i]:
                if din[j] is None:
                    din[j] = out_d; work.append(j)
                elif not out_d <= din[j]:
                    din[j] = din[j] | out_d; work.append(j)
        elif k == 'if':
            # Not-taken path keeps d; taken ibf_tgt path spilled before SPU_IBF_*_L so empty dirty set
            if i + 1 < n:
                j = i + 1
                if din[j] is None:
                    din[j] = d; work.append(j)
                elif not d <= din[j]:
                    din[j] = din[j] | d; work.append(j)
            if v[3] is not None:
                j = labels[v[3]]
                if din[j] is None:
                    din[j] = frozenset(); work.append(j)
        else:
            # 'lbl', 'g', 'lb', 'ix', 'cix': fast paths keep d unchanged
            for j in succ[i]:
                if din[j] is None:
                    din[j] = d; work.append(j)
                elif not d <= din[j]:
                    din[j] = din[j] | d; work.append(j)

    # ---- Backward liveness of cached locals ----
    def uses_defs(t):
        u = set(int(x) for x in GPR_RE.findall(t))
        d = set(int(x) for x in ASSIGN_RE.findall(t))
        if len(d) == 1 and t.count(';') == 1:
            m = ASSIGN_RE.search(t)
            rhs = t[m.end():]
            u = set(int(x) for x in GPR_RE.findall(t[:m.start()])) | set(int(x) for x in GPR_RE.findall(rhs))
            return u, d
        return u, set()

    def dset(i):
        return set(din[i]) if din[i] is not None else set()

    ud = []
    for k, v in kinds:
        if k == 'p':
            ud.append(uses_defs(v))
        elif k in ('g', 'lb'):
            ud.append((set(int(x) for x in GPR_RE.findall(v[0])), set()))
        elif k == 'cix':
            ud.append((set(int(x) for x in GPR_RE.findall(v[1])), set()))
        elif k == 'if':
            ud.append((set(int(x) for x in GPR_RE.findall(v[1])), set()))
        else:
            ud.append(None)

    lin = [set() for _ in range(n)]
    changed = True
    while changed:
        changed = False
        for i in range(n - 1, -1, -1):
            k, v = kinds[i]
            lo = set()
            for j in succ[i]:
                lo |= lin[j]
            if k == 'p':
                u, d = ud[i]
                new = u | (lo - d)
            elif k == 'lbl':
                new = lo
            elif k in ('g', 'lb'):
                new = ud[i][0] | lo
            elif k == 'ix':
                # Fast path needs lo; cold slow-path spill reads dset(i)
                new = dset(i) | lo
            elif k == 'cix':
                # Condition reads ud[i][0]; fast path/fall-through need lo; cold slow-path spill reads dset(i)
                new = ud[i][0] | dset(i) | lo
            elif k == 'if':
                # Condition reads ud[i][0]; taken path spills dset(i) (and if ibf_tgt, jumps with locals live);
                # not-taken path needs lin[i+1] (which is already in lo)
                new = ud[i][0] | dset(i) | lo
            elif k == 'x':
                # Spills dset(i). If it has ibf_tgt, SPU_IBF_*_L jumps to ibf_tgt BEFORE any reload,
                # so lin[ibf_tgt] must ALSO be live in locals BEFORE this line!
                new = set(dset(i))
                if v[2] is not None:
                    new |= lin[labels[v[2]]]
            if new != lin[i]:
                lin[i] = new
                changed = True

    def spill_str(i):
        regs = sorted(din[i] or ())
        return ' '.join('ctx->gpr[%d] = r%d;' % (r, r) for r in regs)

    def reload_str(i):
        # Only reload registers needed on fall-through (i+1), not ibf_tgt (which was already live)
        if i + 1 >= n:
            return ''
        need = sorted(lin[i + 1] & used)
        return ' '.join('r%d = ctx->gpr[%d];' % (r, r) for r in need)

    # ---- Emit merged C function ----
    merged_name = '%s_cfg_%d' % (prefix, comp_id)
    out = ['static void %s(spu_context* __restrict ctx) {' % merged_name]
    out.append('\tuint8_t* __restrict spu_rc_ls = ctx->ls;')

    # Determine which used registers are live-in at ANY block entry of this component.
    # Initializing them once before the entry switch avoids repeating N*M loads across
    # every case arm while keeping Clang's jump table minimal (1 instruction per case).
    entry_live = set()
    for pc in comp_pcs:
        entry_live |= (lin[block_start_idx[pc]] & used)

    if U:
        decls = []
        for r in U:
            if r in entry_live:
                decls.append('r%d = ctx->gpr[%d]' % (r, r))
            else:
                decls.append('r%d' % r)
        out.append('\tu128 ' + ', '.join(decls) + ';')

    # Entry switch: 1-line jump table into each block
    out.append('\tswitch (ctx->pc & SPU_LS_MASK) {')
    for pc in comp_pcs:
        pc_val = int(pc, 16)
        out.append('\tcase 0x%Xu: goto blk_%s;' % (pc_val, pc))
    out.append('\tdefault: __builtin_unreachable();')
    out.append('\t}')

    # Deduplicate and chain cold trampoline fallback spill blocks by subset inclusion:
    # if dirty set D_b is a strict superset of D_a, emit only (D_b - D_a) + goto cfg_cold_a.
    cold_spill_map = {}   # frozenset(regs) -> label_name
    cold_spill_sets = []  # [(label_name, frozenset(regs))]

    def get_cold_label(i):
        regs = frozenset(din[i] or ())
        if not regs:
            return None
        lbl = cold_spill_map.get(regs)
        if lbl is None:
            lbl = 'cfg_cold_%d' % len(cold_spill_sets)
            cold_spill_map[regs] = lbl
            cold_spill_sets.append((lbl, regs))
        return lbl

    for i, (k, v) in enumerate(kinds):
        if k == 'lbl':
            out.append('%s: ;' % v)
        elif k == 'p':
            out.append(loc_regs(v))
        elif k == 'g':
            out.append(loc_regs(v[0]))
        elif k == 'lb':
            out.append(loc_regs(v[0]))
        elif k == 'ix':
            ind, pc_hex, tgt_pc, tgt_fn = v
            clbl = get_cold_label(i)
            out.append(
                '%sif (SPU_CFG_FAST_COND(ctx)) { ctx->pc = %s; ctx->steps++; SPU_CFG_RECENT_NOTE(%s); SPU_PCH_NOTE(ctx); goto blk_%s; }'
                % (ind, pc_hex, pc_hex, tgt_pc)
            )
            if clbl:
                out.append('%s{ ctx->pc = %s; g_spu_trampoline_fn = %s; goto %s; }' % (ind, pc_hex, tgt_fn, clbl))
            else:
                out.append('%s{ ctx->pc = %s; g_spu_trampoline_fn = %s; return; }' % (ind, pc_hex, tgt_fn))
        elif k == 'cix':
            ind, cond, pc_hex, tgt_pc, tgt_fn = v
            clbl = get_cold_label(i)
            out.append('%sif (%s) {' % (ind, loc_regs(cond)))
            out.append(
                '%s    if (SPU_CFG_FAST_COND(ctx)) { ctx->pc = %s; ctx->steps++; SPU_CFG_RECENT_NOTE(%s); SPU_PCH_NOTE(ctx); goto blk_%s; }'
                % (ind, pc_hex, pc_hex, tgt_pc)
            )
            if clbl:
                out.append('%s    ctx->pc = %s; g_spu_trampoline_fn = %s; goto %s;' % (ind, pc_hex, tgt_fn, clbl))
            else:
                out.append('%s    ctx->pc = %s; g_spu_trampoline_fn = %s; return;' % (ind, pc_hex, tgt_fn))
            out.append('%s}' % ind)
        elif k == 'if':
            ind, cond, ibody, _ibf_tgt = v
            sp = spill_str(i)
            if NORET_RE.search(ibody):
                tail = ''
            else:
                rl = reload_str(i)
                tail = (' ' + rl if rl else '') + ' spu_rc_ls = ctx->ls;'
            out.append('%sif (%s) { %s%s%s }' % (ind, loc_regs(cond), (sp + ' ') if sp else '', ibody.strip(), tail))
        elif k == 'x':
            raw_line, is_noret, _ibf_tgt = v
            ind = raw_line[:len(raw_line) - len(raw_line.lstrip())]
            sp = spill_str(i)
            if sp:
                out.append(ind + sp)
            out.append(raw_line)
            if not is_noret:
                rl = reload_str(i)
                out.append(ind + ((rl + ' ') if rl else '') + 'spu_rc_ls = ctx->ls;')

    # Emit cold spill blocks in ascending size order so every strict-subset target is well-founded (DAG)
    cold_by_size = sorted(cold_spill_sets, key=lambda x: len(x[1]))
    emitted_cold = []  # [(lbl, frozenset)]
    for clbl, regs in cold_by_size:
        best_parent = None
        best_len = 0
        for plbl, pregs in emitted_cold:
            if len(pregs) > best_len and pregs < regs:
                best_parent = (plbl, pregs)
                best_len = len(pregs)
        if best_parent is not None:
            plbl, pregs = best_parent
            delta = sorted(regs - pregs)
            sp = ' '.join('ctx->gpr[%d] = r%d;' % (r, r) for r in delta)
            out.append('%s: %s goto %s;' % (clbl, sp, plbl))
        else:
            sp = ' '.join('ctx->gpr[%d] = r%d;' % (r, r) for r in sorted(regs))
            out.append('%s: %s return;' % (clbl, sp))
        emitted_cold.append((clbl, regs))

    out.append('}')
    return merged_name, '\n'.join(out)


def transform_source(src, mode='wcc', max_comp=64, prefix_rename=None, ab_twin=False, regcache=False):
    # Detect prefix P from <P>_rc_spu_func_XXXXXXXX
    m_pref = re.search(r'\b(\w+)_rc_spu_func_[0-9A-Fa-f]{8}\b', src)
    if not m_pref:
        raise ValueError('could not detect <P>_rc_spu_func_ prefix')
    P = m_pref.group(1)
    fn_prefix = P + '_rc_spu_func_'

    # Parse all function definitions (preserve exact spans so non-merged code stays intact)
    fn_head_re = re.compile(
        r'^(static\s+)?void\s+(' + re.escape(fn_prefix) + r'([0-9A-Fa-f]{8}))\(spu_context\*\s*(?:__restrict\s+)?ctx\)\s*\{\s*$'
    )
    lines = src.split('\n')
    funcs = {}      # pc -> dict(start, end, is_static, name, body_lines)
    order = []
    i = 0
    while i < len(lines):
        m = fn_head_re.match(lines[i])
        if m:
            is_static = bool(m.group(1))
            name = m.group(2)
            pc = m.group(3)
            j = i + 1
            while j < len(lines) and lines[j] != '}':
                j += 1
            funcs[pc] = {
                'start': i,
                'end': j,
                'is_static': is_static,
                'name': name,
                'body': lines[i + 1:j],
            }
            order.append(pc)
            i = j + 1
        else:
            i += 1

    # Build SPU_XFER directed graph + identify direct brsl call targets and ila constants
    xfer_re = re.compile(r'SPU_XFER\(ctx,\s*' + re.escape(fn_prefix) + r'([0-9A-Fa-f]{8})\)')
    call_re = re.compile(r'ctx->host_depth\+\+;\s*' + re.escape(fn_prefix) + r'([0-9A-Fa-f]{8})\(ctx\)')
    ila_re = re.compile(r'\bspu_ila\((0x[0-9A-Fa-f]+)\)')
    edges = collections.defaultdict(list)
    brsl_targets = set()
    brsl_counts = collections.Counter()
    ila_pcs = set()
    for pc in order:
        body_txt = '\n'.join(funcs[pc]['body'])
        for tgt in xfer_re.findall(body_txt):
            edges[pc].append(tgt)
        for tgt in call_re.findall(body_txt):
            brsl_targets.add(tgt)
            brsl_counts[tgt] += 1
        for hx in ila_re.findall(body_txt):
            ila_pcs.add('%08X' % (int(hx, 16) & 0x3FFFF))
    # Also include module entry PC from prefix `_eXXXX`
    m_ent = re.search(r'_e([0-9A-Fa-f]+)$', P)
    if m_ent:
        brsl_targets.add('%08X' % (int(m_ent.group(1), 16) & 0x3FFFF))

    comps = compute_components(set(funcs.keys()), edges, mode=mode, max_comp=max_comp)
    pc_to_comp = {}
    for cid, comp_pcs in enumerate(comps):
        for pc in comp_pcs:
            pc_to_comp[pc] = cid

    # Identify pure-leaf brsl callees (single-block, pure ops, no r0 write, single final SPU_RET)
    pure_leaves = {}   # pc -> list of pure body lines
    leaf_defs = []
    if not regcache:
        for pc in order:
            if pc in pc_to_comp:
                continue
            if pc not in brsl_targets:
                continue
            ul = uncache_block(pc, funcs[pc]['body'], P)
            pbody = is_pure_leaf_block(ul)
            if pbody is not None:
                pure_leaves[pc] = pbody
                lname = '%s_leaf_%s' % (P, pc)
                force_inl = (len(pbody) <= 35) or (len(pbody) <= 50 and brsl_counts[pc] <= 10) or (len(pbody) <= 80 and brsl_counts[pc] <= 4)
                inl_attr = 'static inline __attribute__((always_inline))' if force_inl else 'static inline'
                leaf_lines = [
                    '%s void %s(spu_context* __restrict ctx, uint8_t* __restrict const spu_rc_ls) {' % (inl_attr, lname),
                    '\t(void)spu_rc_ls;',
                ] + pbody + ['}']
                leaf_defs.append('\n'.join(leaf_lines))

    # Un-cache and merge each component
    merged_defs = []
    merged_decls = []
    if leaf_defs:
        merged_decls.extend(leaf_defs)
    unm_decls = []
    unm_defs = []
    shim_map = {}   # pc -> replacement lines for original function definition
    unm_prefix = P + '_unm_spu_func_'

    if ab_twin:
        merged_decls.extend([
            'extern volatile int g_ab_phase;',
            'static inline int %s_cfg_use_merged(void) {' % P,
            '    static int ph = -2;',
            '    if (__builtin_expect(ph == -2, 0)) {',
            '        const char* e = getenv("SPU_CFG_MERGE_PHASE");',
            '        if (e && *e) ph = atoi(e);',
            '        else { const char* on = getenv("SPU_CFG_MERGE"); ph = (on && *on == \'0\') ? -3 : -1; }',
            '    }',
            '    if (ph == -1) return 1;',
            '    if (ph == -3) return 0;',
            '    return (g_ab_phase & 1) == ph;',
            '}',
        ])

    for cid, comp_pcs in enumerate(comps):
        comp_set = set(comp_pcs)
        # Compute externally reachable entries of this component so internal loop blocks
        # are excluded from the entry switch (keeping natural loops reducible for LLVM).
        in_from_in = set()
        in_from_out = set()
        for src_pc, tgts in edges.items():
            for tgt in tgts:
                if tgt in comp_set:
                    if src_pc in comp_set:
                        in_from_in.add(tgt)
                    else:
                        in_from_out.add(tgt)
        entry_pcs = set()
        for pc in comp_pcs:
            if (pc not in in_from_in) or (pc in in_from_out) or (pc in brsl_targets) or (pc in ila_pcs):
                entry_pcs.add(pc)
        if not entry_pcs:
            # Pure cycle with no obvious outside caller: pick lowest PC as entry
            entry_pcs.add(comp_pcs[0])

        uncached_map = {pc: uncache_block(pc, funcs[pc]['body'], P) for pc in comp_pcs}
        mname, mcode = build_merged_function(cid, comp_pcs, uncached_map, P, fn_prefix,
                                             entry_pcs=entry_pcs, regcache=regcache, pure_leaves=pure_leaves)
        merged_decls.append('static void %s(spu_context* __restrict ctx);' % mname)
        merged_defs.append(mcode)
        for pc in comp_pcs:
            info = funcs[pc]
            qual = 'static ' if info['is_static'] else ''
            pc_val = int(pc, 16)
            is_entry = (pc in entry_pcs)
            if ab_twin:
                unm_name = unm_prefix + pc
                unm_decls.append('static void %s(spu_context* __restrict ctx);' % unm_name)
                # Preserve exact unmerged body, redirecting internal SPU_XFERs to unmerged twins
                ubody = '\n'.join(info['body'])
                ubody = re.sub(
                    r'\b' + re.escape(fn_prefix) + r'([0-9A-Fa-f]{8})\b',
                    lambda mm: (unm_prefix + mm.group(1)) if mm.group(1) in comp_set else mm.group(0),
                    ubody
                )
                unm_defs.append('static void %s(spu_context* __restrict ctx) {\n%s\n}' % (unm_name, ubody))
                if is_entry:
                    shim_map[pc] = [
                        '%svoid %s(spu_context* __restrict ctx) {' % (qual, info['name']),
                        '    if (__builtin_expect(%s_cfg_use_merged(), 1)) { ctx->pc = 0x%Xu; SPU_CFG_MUSTTAIL(%s(ctx)); }'
                        % (P, pc_val, mname),
                        '    SPU_CFG_MUSTTAIL(%s(ctx));' % unm_name,
                        '}',
                    ]
                else:
                    # Non-entry block: rarely entered directly (interrupt resume / indirect jump).
                    # Keep original 1-block body so merged function stays reducible; on unmerged A/B phase
                    # dispatch to unm_name, on merged phase execute 1 block whose SPU_XFER re-enters mname.
                    shim_map[pc] = [
                        '%svoid %s(spu_context* __restrict ctx) {' % (qual, info['name']),
                        '    if (!__builtin_expect(%s_cfg_use_merged(), 1)) { SPU_CFG_MUSTTAIL(%s(ctx)); }' % (P, unm_name),
                    ] + info['body'] + ['}']
            else:
                if is_entry:
                    shim_map[pc] = [
                        '%svoid %s(spu_context* __restrict ctx) { ctx->pc = 0x%Xu; SPU_CFG_MUSTTAIL(%s(ctx)); }'
                        % (qual, info['name'], pc_val, mname)
                    ]
                # Non-entry block in non-AB mode: leave original 1-block function untouched
                # so interrupt resume / indirect jumps work bit-identically and re-enter mname at next entry.

    if unm_decls:
        merged_decls.extend(unm_decls)
    if unm_defs:
        merged_defs.extend(unm_defs)

    # Reconstruct output source
    out_lines = []
    inserted_macros = False
    inserted_decls = False
    first_fn_start = funcs[order[0]]['start'] if order else 0

    skip_until = -1
    for idx, line in enumerate(lines):
        if idx <= skip_until:
            continue

        if not inserted_macros and line.startswith('#include "spu_helpers_vec.h"'):
            out_lines.append(line)
            out_lines.append(CFG_HELPER_MACROS)
            inserted_macros = True
            continue

        if not inserted_decls and idx == first_fn_start:
            if not inserted_macros:
                out_lines.append(CFG_HELPER_MACROS)
                inserted_macros = True
            if merged_decls:
                out_lines.append('/* spu_merge_cfg.py: %d merged CFG components (%d blocks) */'
                                 % (len(comps), len(pc_to_comp)))
                out_lines.extend(merged_decls)
                out_lines.append('')
            inserted_decls = True

        m = fn_head_re.match(line)
        if m and m.group(3) in shim_map:
            pc = m.group(3)
            out_lines.extend(shim_map[pc])
            skip_until = funcs[pc]['end']
            continue

        out_lines.append(line)

    # Insert merged function definitions right before the final registration / ibf table section
    # (or at the very end if neither exists, since all forward declarations are already emitted).
    if merged_defs:
        insert_pos = len(out_lines)
        for idx, line in enumerate(out_lines):
            if line.startswith('/* PORT_SPU_RC_ONLY:') or line.startswith('/* v177 spu_ibfast:') or line.startswith('/* v95: (original, register-cached) pairs'):
                insert_pos = idx
                break
        out_lines[insert_pos:insert_pos] = ['', '/* spu_merge_cfg.py: merged CFG functions */'] + merged_defs + ['']

    result = '\n'.join(out_lines)

    if prefix_rename:
        # Preserve #include paths while renaming all symbol prefixes
        out_ren = []
        for ln in result.split('\n'):
            if ln.lstrip().startswith('#include'):
                out_ren.append(ln)
            else:
                out_ren.append(ln.replace(P, prefix_rename))
        result = '\n'.join(out_ren)

    stats = {
        'prefix': P,
        'funcs_total': len(funcs),
        'comps': len(comps),
        'funcs_merged': len(pc_to_comp),
        'xfers_before': len(re.findall(r'SPU_XFER\(', src)),
        'xfers_after': len(re.findall(r'SPU_XFER\(', result)),
        'loads_before': len(re.findall(r'=\s*ctx->gpr\[\d+\]', src)),
        'loads_after': len(re.findall(r'=\s*ctx->gpr\[\d+\]', result)),
        'stores_before': len(re.findall(r'ctx->gpr\[\d+\]\s*=', src)),
        'stores_after': len(re.findall(r'ctx->gpr\[\d+\]\s*=', result)),
    }
    return result, stats


def main():
    args = sys.argv[1:]
    mode = 'wcc'
    max_comp = 64
    prefix_rename = None
    show_stats = False
    ab_twin = False
    regcache = False

    pos = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--mode':
            mode = args[i + 1]; i += 2
        elif a == '--max-comp':
            max_comp = int(args[i + 1]); i += 2
        elif a == '--prefix-rename':
            prefix_rename = args[i + 1]; i += 2
        elif a == '--ab-twin':
            ab_twin = True; i += 1
        elif a == '--regcache':
            regcache = True; i += 1
        elif a == '--stats':
            show_stats = True; i += 1
        else:
            pos.append(a); i += 1

    if len(pos) != 2:
        sys.exit('usage: spu_merge_cfg.py <in.c> <out.c> [--mode wcc|scc] [--max-comp 64] [--prefix-rename NEW_P] [--ab-twin] [--regcache] [--stats]')

    in_path, out_path = pos
    with open(in_path, 'r', encoding='utf-8', errors='surrogateescape') as f:
        src = f.read()

    out_src, st = transform_source(src, mode=mode, max_comp=max_comp, prefix_rename=prefix_rename, ab_twin=ab_twin, regcache=regcache)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.isfile(out_path):
        with open(out_path, 'r', encoding='utf-8', errors='surrogateescape') as f:
            if f.read() == out_src:
                if show_stats:
                    print('%(prefix)s: comps=%(comps)d merged=%(funcs_merged)d/%(funcs_total)d '
                          'XFER=%(xfers_before)d->%(xfers_after)d '
                          'gpr_ld=%(loads_before)d->%(loads_after)d '
                          'gpr_st=%(stores_before)d->%(stores_after)d (unchanged)' % st)
                return
    with open(out_path, 'w', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
        f.write(out_src)

    if show_stats:
        print('%(prefix)s: comps=%(comps)d merged=%(funcs_merged)d/%(funcs_total)d '
              'XFER=%(xfers_before)d->%(xfers_after)d '
              'gpr_ld=%(loads_before)d->%(loads_after)d '
              'gpr_st=%(stores_before)d->%(stores_after)d' % st)


if __name__ == '__main__':
    main()
