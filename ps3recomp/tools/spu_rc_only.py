#!/usr/bin/env python3
"""spu_rc_only.py -- Build-time transform for PORT_SPU_RC_ONLY=ON.

Emits a standalone RC-only lifted SPU translation unit from (spu_recomp.c,
spu_recomp_rc.c) so spu_recomp.c does not need to be compiled or linked:
  - strips the s_spu_rc_pairs[] table and its constructor (which referenced
    every non-RC symbol in spu_recomp.c)
  - marks all <P>_rc_spu_func_* internal functions static
  - emits <P>_spu_recomp_register() registering the <P>_rc_spu_func_* symbols
    directly from spu_recomp.c's spu_function_table
  - emits thin musttail shims for any <P>_spu_func_* entry symbols referenced
    by spu_workloads.c (passed via --workloads)

Usage: spu_rc_only.py <orig spu_recomp.c> <rc spu_recomp_rc.c> <out.c> [--workloads <spu_workloads.c>]
"""
import os
import re
import sys

def main():
    args = sys.argv[1:]
    workloads_path = None
    if '--workloads' in args:
        idx = args.index('--workloads')
        workloads_path = args[idx + 1]
        del args[idx:idx + 2]
    if len(args) != 3:
        sys.exit('usage: spu_rc_only.py <orig.c> <rc.c> <out.c> [--workloads spu_workloads.c]')
    orig_path, rc_path, out_path = args

    with open(orig_path, 'r', encoding='utf-8') as f:
        orig = f.read()
    with open(rc_path, 'r', encoding='utf-8') as f:
        rc = f.read()

    # Extract prefix P and registration entries from orig spu_recomp.c
    m_reg = re.search(r'void\s+(\w+)_spu_recomp_register\s*\(\s*void\s*\)', orig)
    if not m_reg:
        sys.exit('spu_rc_only: could not find *_spu_recomp_register in ' + orig_path)
    P = m_reg.group(1)

    # Extract { 0xADDRu, <P>_spu_func_XXXX, "..." } entries from spu_function_table
    tab_entries = re.findall(
        r'\{\s*(0x[0-9A-Fa-f]+u?)\s*,\s*(' + re.escape(P) + r'_spu_func_[0-9A-Fa-f_]+)\s*,\s*"([^"]+)"\s*\}',
        orig
    )
    if not tab_entries:
        sys.exit('spu_rc_only: empty spu_function_table in ' + orig_path)

    # Find external entry symbols referenced by spu_workloads.c for this prefix
    ext_entries = set()
    if workloads_path and os.path.isfile(workloads_path):
        with open(workloads_path, 'r', encoding='utf-8') as f:
            wsrc = f.read()
        for m in re.finditer(r'\b(' + re.escape(P) + r'_spu_func_[0-9A-Fa-f_]+)\b', wsrc):
            ext_entries.add(m.group(1))

    # Rewrite any #include "local.h" that lives beside rc_path (spu_recomp.h, lodx_v231.h, etc.)
    src_dir = os.path.dirname(os.path.abspath(rc_path)).replace('\\', '/')
    def _fix_inc(m):
        hdr = m.group(1)
        if os.path.isfile(os.path.join(src_dir, hdr)):
            return '#include "%s/%s"' % (src_dir, hdr)
        return m.group(0)
    rc = re.sub(r'#include\s+"([^"/\\]+)"', _fix_inc, rc)
    if 'LODX233_OWNER' in orig and 'LODX233_OWNER' not in rc:
        rc = '#define LODX233_OWNER 1\n' + rc

    # Remove the s_spu_rc_pairs[] table and its constructor (references orig symbols)
    marker = '/* v95: (original, register-cached) pairs'
    m_start = rc.find(marker)
    if m_start >= 0:
        ctor_hdr = '__attribute__((constructor)) static void ' + P + '_rc_pairs_ctor(void)'
        c_pos = rc.find(ctor_hdr, m_start)
        if c_pos >= 0:
            c_end = rc.find('}', c_pos)
            if c_end >= 0:
                rc = rc[:m_start] + rc[c_end + 1:]

    # Make all <P>_rc_spu_func_* declarations and definitions static so Clang can
    # optimize across internal callers and omit unused symbol exports.
    fn_pat = re.compile(
        r'^void\s+(' + re.escape(P) + r'_rc_spu_func_[0-9A-Fa-f_]+\(spu_context\*\s*(?:__restrict\s+)?ctx\)\s*[;{])',
        re.M
    )
    rc = fn_pat.sub(r'static void \1', rc)

    # Also drop unused <P>_rc_spu_recomp_register if present in rc
    rc_lines = [rc, '', '/* PORT_SPU_RC_ONLY: direct registration of RC functions */']
    for sym in sorted(ext_entries):
        rc_sym = sym.replace(P + '_spu_func_', P + '_rc_spu_func_', 1)
        rc_lines.append('void %s(spu_context* ctx) { __attribute__((musttail)) return %s(ctx); }' % (sym, rc_sym))

    rc_lines.append('static const spu_func_entry s_rc_only_table[] = {')
    for addr, orig_sym, name in tab_entries:
        rc_sym = orig_sym.replace(P + '_spu_func_', P + '_rc_spu_func_', 1)
        rc_lines.append('    { %s, %s, "%s" },' % (addr, rc_sym, name))
    rc_lines.append('    { 0, NULL, NULL }')
    rc_lines.append('};')
    rc_lines.append('void %s_spu_recomp_register(void) {' % P)
    rc_lines.append('    for (const spu_func_entry* e = s_rc_only_table; e->func; ++e)')
    rc_lines.append('        spu_register_function(e->addr, e->func);')
    rc_lines.append('}')
    rc_lines.append('')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    new_content = '\n'.join(rc_lines)
    if os.path.isfile(out_path):
        with open(out_path, 'r', encoding='utf-8') as f:
            if f.read() == new_content:
                return
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

if __name__ == '__main__':
    main()
