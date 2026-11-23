/*
 * ps3recomp - platform-neutral RSX draw engine (see rsx_draw_engine.h)
 *
 * The orchestration carried across from libs/video/rsx_live_draw.c, the
 * NV4097 -> D3D12 engine a title has shipped on. Line numbers in the comments
 * below are that file's, so the two can be read side by side; the file itself
 * is vendored and stays exactly as it is.
 *
 * What is deliberately left out, because it is not renderer behaviour a title
 * needs: the shader disk cache, movie mode and its compositor, the a010
 * probe, and every YZ_PERF_PROFILE block.
 */
#include "rsx_draw_engine.h"
#include <time.h>

/* XXH3 is the content hash for every identity in this file: it runs at
 * memory speed where the byte-at-a-time FNV it replaces ran at about
 * 1.2 GB/s, and every one of these hashes (69 MB of texture content, the two
 * microcodes per draw) was measurable in the frame. Hash quality only has to
 * separate live entries inside one run, and the cached keys are always
 * compared field by field, never trusted on the hash alone. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#define XXH_INLINE_ALL
#include "xxhash.h"
#pragma clang diagnostic pop

#include "rsx_fp_decompiler.h"
#include "rsx_primitives.h"
#include "rsx_restart_cuts.h"
#include "rsx_upscale_common.h"
#include "rsx_vertex_formats.h"
#include "rsx_vp_decompiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* Guest memory and the RSX offset resolvers, declared the way every backend
 * that reads guest data declares them (rsx_vertex_fetch.h). */
extern u8* vm_base;
u32 cellGcmResolveLocated(int local, u32 offset);
/* Non-zero only when the offset's page is in the IO table, so a caller can
 * tell IO-mapped main memory from "assume VRAM". */
u32 cellGcmResolveIO(u32 offset);
/* The guest VM's OOB guard: the size a host backed, or 0 for "the whole 32-bit
 * space is backed and no check is needed" (runtime/memory/vm.h). */
extern u32 ppu_vm_size;

/* ------------------------------------------------------------------------- */

#define ENG_MAX_SURFACES   64
#define ENG_MAX_ZDEPTHS    64
#define ENG_MAX_TEXTURES   RSX_DRAW_ENGINE_TEXTURE_CACHE
#define ENG_MAX_VTEXTURES  64
#define ENG_MAX_PIPELINES  8192
#define ENG_MAX_BATCHES    256
#define ENG_VERT_STRIDE    (RSX_DSP_NUM_VERTEX_ATTR * 16u)
#define ENG_INVALID        0xFFFFFFFFu
/* Guest textures larger than this are a misdecoded register, not an image;
 * rsx_live_draw.c rejects the same 4096 bound (2033). */
#define ENG_MAX_TEX_DIM    4096u
/* A surface declaration beyond this is a guest pointer read as clip
 * dimensions. rsx_live_draw.c's surface-guard (2625) preserves the existing
 * target rather than destroying a live one. */
#define ENG_MAX_SURFACE_DIM 8192u

#define ENG_VP_CB_BYTES    ((RSX_DSP_NUM_CONSTANTS + 2) * 16u)
#define ENG_FP_MAX_BYTES   0x10000u

#define M_SURFACE_COLOR_TARGET 0x0220
#define M_ALPHA_TEST_ENABLE   0x0304
#define M_ALPHA_FUNC          0x0308
#define M_ALPHA_REF           0x030C
#define M_BLEND_ENABLE        0x0310
#define M_BLEND_SFACTOR       0x0314
#define M_BLEND_DFACTOR       0x0318
#define M_BLEND_COLOR         0x031C
#define M_BLEND_EQUATION      0x0320
#define M_BLEND_COLOR2        0x037C
#define M_COLOR_MASK          0x0324
#define M_STENCIL_TEST_ENABLE 0x0328
#define M_STENCIL_MASK        0x032C
#define M_STENCIL_FUNC        0x0330
#define M_STENCIL_FUNC_REF    0x0334
#define M_STENCIL_FUNC_MASK   0x0338
#define M_STENCIL_OP_FAIL     0x033C
#define M_STENCIL_OP_ZFAIL    0x0340
#define M_STENCIL_OP_ZPASS    0x0344
#define M_TWO_SIDED_STENCIL   0x0348
#define M_BACK_STENCIL_FUNC   0x0350
#define M_BACK_STENCIL_OP_FAIL  0x035C
#define M_BACK_STENCIL_OP_ZFAIL 0x0360
#define M_BACK_STENCIL_OP_ZPASS 0x0364
#define M_SCISSOR_HORIZONTAL  0x08C0
#define M_SCISSOR_VERTICAL    0x08C4
#define M_POLY_OFFSET_POINT   0x0A60
#define M_POLY_OFFSET_LINE    0x0A64
#define M_POLY_OFFSET_FILL    0x0A68
#define M_DEPTH_FUNC          0x0A6C
#define M_DEPTH_WRITE         0x0A70
#define M_DEPTH_TEST_ENABLE   0x0A74
#define M_POLY_OFFSET_FACTOR  0x0A78
#define M_POLY_OFFSET_UNITS   0x0A7C
#define M_ZSTENCIL_CLEAR      0x1D8C

/* ---- module state -------------------------------------------------------- */

typedef struct {
    u32 location, offset;
    u32 w, h;
    rsx_be_format fmt;
    u32 handle;
    u32 feedback;          /* the copy a draw sampling this target binds   */
    u64 feedback_pass;     /* g.pass_serial that copy may still be shared in (#814), or 0 */
    int cwritten;          /* at least one draw wrote colour into this RT  */
    u32 feedback_last_draw;
    u32 feedback_runlen;
    u32 feedback_frame;
    u8  feedback_valid;    /* feedback copy matches current target content */
} eng_surface;

typedef struct {
    u32 location, offset;
    u32 w, h;
    u32 handle;
    u32 snapshot;          /* sampleable R32F depth copy, or 0             */
    u32 color_snapshot;    /* sampleable A8R8G8B8 D24S8 copy, or 0         */
    int cleared;
    int had_write;
    int snapshot_valid;
    int color_snapshot_valid;
} eng_zdepth;

typedef struct {
    u32 location, offset, format, width, height, pitch, remap, cubemap, levels;
    u32 handle;
    u64 content_hash;      /* full hash of the span the uploaded copy came from */
    u32 sample_hash;       /* sparse hash taken with that same full hash    */
    u32 span;              /* guest bytes the content hash covers           */
    u32 last_hash_frame;   /* the frame it was last validated in            */
    u32 full_hash_frame;   /* the frame the FULL hash was last taken        */
    u32 swd_ver_frame;     /* s_swd_frame at last content verification      */
    u32 create_frame;      /* frame when this cache entry was created       */
    u16 swd_clo, swd_chi;  /* coarse SWD bucket range when swd_ok == 1      */
    u8  swd_ok;            /* safe for cross-frame coarse SWD fast hit      */
    u8  always_hash;       /* non-SPU writer detected: always hash          */
    u64 last_use_serial;
} eng_texture;

typedef struct { u64 key; u32 handle; u8 fixed; } eng_pipeline;

/* Everything derived from one pair of microcode programs that a pipeline key
 * needs, memoized on the identity the REGISTERS name (transform-program write
 * counter + start slot, SET_SHADER_PROGRAM, SHADER_CONTROL, the cube and
 * vertex-texture masks) and validated against both content hashes, so a
 * fragment program rewritten in place under the same address is caught. On a
 * hit the per-draw cost is the two content hashes; the four passes over the
 * microcode (two end-marker scans, the transform-program input analysis, the
 * fragment structural hash) ran once per program instead of once per draw. */
/* Inline constants a program may have and still be cached whole. The block
 * limit (RSX_FP_MAX_INLINE_CONSTANTS) is 2048; a title's shaders carry a
 * handful, and a bigger one just recomputes per draw. */
#define ENG_PROG_CONST_MAX 16u
#define ENG_VP_SPARSE_MAX  64u

typedef struct {
    u64 id;                  /* register-side identity; 0 = free slot         */
    u64 vp_hash, fp_hash;    /* content hashes this entry was verified against */
    u64 fp_struct_hash;
    u32 vp_instrs, fp_size;
    u32 vp_gen, vp_start, fp_reg;
    u32 fp_ctrl, cube_mask, vtex_mask;
    rsx_vertex_layout_plan layout;
    u8  fixed;
    u8  vp_const_packed;
    u8  vp_const_sparse;
    u8  vp_const_blk_mask;
    u16 vp_const_slots;
    u16 vp_const_count;
    u16 vp_const_list[ENG_VP_SPARSE_MAX];
    u8  vp_cuse_valid;
    u32 vp_cuse[16];
    u32 fp_ea;
    u32 fp_ver_frame;
    /* The program's inline constants, hoisted into the buffered block. Count
     * and values are what the constant buffer is filled from, so they are
     * kept here: collecting them per draw meant zeroing and rescanning a
     * 32 KB block 10k times a frame. A program with more constants than this
     * takes the uncached path. */
    u8  const_cached;
    u32 const_count;
    u64 const_hash;
    u32 const_values[ENG_PROG_CONST_MAX][4];
} eng_program;

/* Open-addressed indexes over the entry arrays: a draw's pipeline key and a
 * texture unit's identity are looked up 10k times a frame each, and both used
 * to be linear scans (1.35 M and 47 M entries walked per frame on the
 * preload). The key itself is always compared in full, never trusted from the
 * hash alone. */
#define ENG_PROGRAM_SLOTS  1024u
#define ENG_PIPELINE_BUCKETS (ENG_MAX_PIPELINES * 2u)
#define ENG_TEXTURE_BUCKETS  (ENG_MAX_TEXTURES * 2u)
#define ENG_TEXTURE_TOMBSTONE 0xFFFFFFFFu
#define ENG_SURFACE_BUCKETS 128u

/* ---- converted-vertex cache (ENG_VCACHE) types -------------------------- */
/*
 * The policy and the reasoning live with the code that drives it, above
 * dc_fetch; these are the sizes, which the module state below has to know.
 */
#define ENG_VC_CLASSES   12u              /* 1 KB .. 2 MB blocks */
#define ENG_VC_ENTS      (1u << 17)
#define ENG_VC_BUCKETS   (1u << 19)
#define ENG_VC_NONE      0xFFFFFFFFu
/* The most guest bytes one keyed group may hash. A sparse index list would
 * otherwise hash a whole mesh's arrays for a handful of vertices. */
#define ENG_VC_HASH_MAX  (4u << 20)

typedef struct {
    u64 key;
    u64 serial;      /* monotonic immutable ID for GPU-side VB caching */
    u32 off;         /* byte offset of the converted block in the arena */
    u32 bytes;
    u32 last_frame;  /* the last frame a draw used it                     */
    u32 placed;      /* the frame it took (or last re-armed) its block    */
    int prev, next;  /* the size class's LRU list                         */
    u8 cls;
} eng_vcent;

typedef struct { u64 key; u32 ent; } eng_vcslot;

typedef struct { u32 first, count; } eng_batch;

/* ---- fast draw-batch mesh cache (ENG_MESH_CACHE) + SPU page-dirty map --- */
#define ENG_MC_MAX_SPANS    32u
#define ENG_MC_INLINE_SPANS 4u
#define ENG_MC_SETS         8192u
#define ENG_MC_WAYS         4u
#define ENG_MC_ENTS         (ENG_MC_SETS * ENG_MC_WAYS)

typedef struct {
    u32 ea;
    u32 len;
} eng_mc_span;

typedef struct {
    u64 lo;
    u64 hi;
} eng_mctag;

typedef struct __attribute__((aligned(64))) {
    /* Hot MC-hit + VK-gcache fields packed into the FIRST 64-byte cache line. */
    rsx_eng_gcache_memo gcache;
    u32 n_verts;
    u32 n_draw;
    u32 verified_frame;
    u32 nonspu_checked_gen;
    u32 vc_evict_gen;
    u32 next_slot;
    u16 desc_bytes;
    u16 coarse_lo;
    u16 coarse_hi;
    u16 last_use_16;
    u8  coarse_single;
    u8  vc_promoted;
    u8  indexed;
    u8  always_hash;
    u8  valid;
    /* Cold fields (cache line 2+): touched only on gcache miss / verify / eviction. */
    u32* idx_buf;
    u64 vc_serial;
    u64 idx_serial;
    u32 vc_ent;
    u32 verts_off;
    u32 n_source_refs;
    u32 last_use_frame;
    u64 mem_hash;
    u64 vc_key;
    eng_mc_span* extra_spans;  /* non-NULL only when n_spans > ENG_MC_INLINE_SPANS */
    u32 verts_bytes;
    u32 idx_cap;
    u16 n_spans;
    eng_mc_span spans[ENG_MC_INLINE_SPANS];
} eng_mcent;
_Static_assert(offsetof(eng_mcent, idx_buf) == 64, "hot MC fields must fit in first 64B cache line");

static inline const eng_mc_span* eng_mc_spans_ptr(const eng_mcent* e)
{
    return e->extra_spans ? e->extra_spans : e->spans;
}

/* What each texture unit of the traced draw resolved to, so a black frame can
 * be read as "this unit sampled guest memory" rather than guessed at:
 * 0 none, 1 live surface, 2 feedback copy, 3 zeta snapshot, 4 guest texture. */
static u8 s_tunit[RSX_BE_MAX_TEXTURES];


typedef struct {
    u32 location, offset, pitch, width, height;
    int valid;
} eng_display_buffer;

static struct {
    const rsx_draw_backend* be;
    int ready;
    int default_on;
    u32 width, height;

    rsx_dispatch rsx;

    eng_surface surfaces[ENG_MAX_SURFACES];
    u32 n_surfaces;
    u32 surface_gen;
    eng_zdepth zdepths[ENG_MAX_ZDEPTHS];
    u32 n_zdepths;
    eng_texture textures[ENG_MAX_TEXTURES];
    u32 n_textures;
    u32 tex_gen;
    eng_pipeline pipelines[ENG_MAX_PIPELINES];
    u32 n_pipelines;
    u32 pipeline_bucket[ENG_PIPELINE_BUCKETS];   /* entry index + 1, 0 = empty */
    eng_program programs[ENG_PROGRAM_SLOTS];
    u32 tex_bucket[ENG_TEXTURE_BUCKETS];         /* entry index + 1, 0 = empty */
    u32 tex_tombstones;
    /* Render targets and zeta by (location, offset): a draw's texture unit asks
     * whether its address is a registered target on every bind, and the answer
     * used to come from a scan of every surface and then every zeta. */
    u32 surface_bucket[ENG_SURFACE_BUCKETS];
    u32 zdepth_bucket[ENG_SURFACE_BUCKETS];
    eng_display_buffer display_buffers[8];

    u32 frames;
    u32 draw_seq;
    u32 vc_evict_gen;
    u64 texture_use_serial;
    u32 guest_draws;
    u32 last_guest_draws;
    u32 last_present_surface;
    u32 last_flip_buffer;
    /* sp00nznet/ps3recomp #814: current render pass identity (colour targets + zeta).
     * Bumped on target/zeta change or present; an own-target feedback copy is
     * shared across the pass when PS3RECOMP_ENG_FEEDBACK_PER_PASS=1. */
    u64 pass_serial;
    u32 pass_n, pass_targets[RSX_BE_MAX_COLOR_TARGETS];
    u32 pass_zloc, pass_zoff;

    /* staging: decoded texture levels, the constant blocks, the index list */
    u8* tex_staging;
    u32 tex_staging_cap;
    u8* vp_cb;
    u32 last_vp_const_gen;
    u32 last_vp_block_gen[8];
    u32 last_vp_bytes;
    int last_vp_packed;
    int last_vp_sparse;
    const eng_program* last_vp_prog;
    u8  last_vp_sparse_data[(ENG_VP_SPARSE_MAX + 2u) * 16u];
    float last_vp_xf[8];
    int vp_cb_valid;
    u8* fp_cb;
    u32 fp_cb_cap;
    u64 last_fp_hash;
    u32 last_fp_bytes;
    int fp_cb_valid;
    u32 last_tex_mask;
    u32 last_textures[RSX_BE_MAX_TEXTURES];
    u32 cur_textures[RSX_BE_MAX_TEXTURES];
    u8  cur_sh_nibble[RSX_BE_MAX_TEXTURES];
    u32 cur_sh_mask;
    u32 cur_tex_resolve_dirty;
    u32 cur_tex_valid_mask;
    rsx_be_sampler_desc last_samplers[RSX_BE_MAX_TEXTURES];
    u32 last_samp_gen[RSX_BE_MAX_TEXTURES];
    int tex_bind_valid;
    u32 last_tex_state_gen;
    u32 last_tex_frame;
    u32 last_tex_surface_gen;
    u32 last_tex_tex_gen;
    u32 last_tex_surf_gen;
    u32 last_tex_clear_gen;
    u64 last_tex_sh_key;
    u8  last_tex_had_dynamic;
    u32 clear_gen;
    int surf_cache_valid;
    u32 surf_cache_gen;
    u32 surf_cache_surface_gen;
    u32 surf_cache_nzdepths;
    rsx_dsp_surface surf_cache_sf;
    u32 surf_cache_targets[RSX_BE_MAX_COLOR_TARGETS];
    u32 surf_cache_ntargets;
    u32 surf_cache_zslot;
    int vp_cache_valid;
    u32 vp_cache_gen;
    u32 vp_cache_w, vp_cache_h;
    float vp_cache_xf[8];
    u32 last_dyn_gen;
    u32 last_dyn_surf_gen;
    u32 last_vtex_mask;
    const eng_program* last_fp_prog;
    u64 last_fp_prog_const_hash;
    u32 last_fp_prog_const_cnt;
    u32 last_fp_alpha_raw;
    u32 last_fp_alpha_fmt;
    float last_fp_wscale_x, last_fp_wscale_y;
    u8  bind_dedup_valid;
    u32 last_rt_count;
    u32 last_rt_handles[RSX_BE_MAX_COLOR_TARGETS];
    u32 last_depth_handle;
    u32 last_pipeline;
    float last_vp_w, last_vp_h;
    u32 last_sc[4];
    u32 last_sref;
    u32 last_db_lo, last_db_hi;
    u8  last_db_valid;
    u32 last_bc_c, last_bc_c2;
    u8  last_bc_fp16, last_bc_valid;
    u32 last_bias_en, last_bias_sf, last_bias_u, last_bias_bits;
    u8  last_bias_valid;
    u32* indices;
    u32 index_cap;

    rsx_fp_constant_block fp_constants;

    /* PS3RECOMP_ENG_STATS=1: what each presented frame cost the walker, so a
     * frame-rate drop can be attributed without a profiler. Filled in by
     * sink_end and eng_texture_slot, printed and reset in eng_present. */
    u64 st_draws, st_hashes, st_hash_bytes, st_uploads, st_verts;
    /* Sparse validation: how many of this frame's revalidations had to take
     * the full hash (a new entry, samples that disagreed, or the every-8th
     * verification) rather than the ~4 KB sample. */
    u64 st_verifies;
    /* Texture units served from a copy of a target the same draw writes. */
    u64 st_feedback;
    double st_ms_draw, st_ms_present;
    /* [ENGTIME] phase split, same knob: summed over the presented frame's
     * groups and divided by the frame count of the print window. st_ms_group
     * is the walker's whole sink_end time, which is what a frame's wall time
     * is made of besides the backend's fence waits. */
    u64 st_groups;
    double st_ms_group, st_ms_fetch, st_ms_pipeline, st_ms_textures;
    double st_ms_bind, st_ms_backend_draw;
    /* Where the texture and pipeline phases go: slot/pipeline cache lookups
     * and how many entries their scans touch, plus the share of the texture
     * phase spent hashing content. This is what a cache change has to move. */
    u64 st_lookups, st_probes, st_plookups, st_pprobes;
    /* Robustness counters a soak run reads: texture entries whose first upload
     * failed and are being retried, and pipeline-table misses that hit the
     * cap. Either one silently degrades the picture, so both are printed. */
    u64 st_handle0, st_pipe_full;
    double st_ms_hash, st_ms_prog, st_ms_consts;

    /* ENG_VCACHE: converted vertices in one arena, keyed by what they were
     * converted from (see the cache section above dc_fetch). */
    u8* vc_arena;
    u64 vc_arena_bytes, vc_bump;
    eng_vcent* vc_ents;
    u32 vc_nents;
    eng_vcslot* vc_ht;
    u64* vc_seen;          /* 2-way ghost filter (2-hit admission for ENGVC) */
    u32 vc_tombs;
    int vc_head[ENG_VC_CLASSES], vc_tail[ENG_VC_CLASSES];
    /* Per presented frame, like the [eng stats] counters: groups served from
     * the cache and groups converted. */
    u64 vc_hits, vc_misses, vc_hit_verts, vc_miss_verts;
    u64 vc_hash_bytes, vc_evict, vc_unkeyed;
    u32 vc_thrash_frames, vc_last_reset;
    /* Where the fetch phase goes once the cache is on: building the reference
     * list from the index array, preparing the plan and hashing the key, and
     * converting the misses. */
    double st_ms_vcrefs, st_ms_vckey, st_ms_vcconvert;

    /* ENG_MESH_CACHE: fast descriptor + SPU page-dirty mesh cache skipping
     * index parsing, vertex remapping, 40 MB/frame hashing and topology index
     * rebuild on unchanged draws. */
    eng_mcent* mc_ents;
    eng_mctag* mc_tags;
    u64* mc_seen;          /* 2-way descriptor ghost filter for cold MC misses */
    u64 vc_serial, mc_idx_serial;
    u64 mc_hits, mc_hits_swd, mc_hits_ver, mc_hits_ahash;
    u64 mc_misses, mc_miss_dirty, mc_miss_ver, mc_miss_cold, mc_miss_unelig;
    u64 mc_nonspu_pages;

    /* Lighting defect-1 kill switches (read from env in init, default 1). */
    u8 tex_xf_on;          /* TEX_XF=0 disables bx2/sRGB texture transforms */
    u8 sun_ndl_clamp_on;   /* SUN_NDL_CLAMP=0 disables max(N.L, 0) clamp    */
    u8 ct_none_on;         /* CT_NONE=0 disables COLOR_TARGET=0 cmask=0     */
    u8 zs_cwritten_on;     /* ZS_CWRITTEN=0 disables cwritten zeta priority */
    u8 zeta_exact_size_on; /* ZETA_EXACT_SIZE=0 restores padded zeta keying */
    u8 zconv_color_on;     /* ZCONV_COLOR=0 disables 0x85 zeta->RGBA8 copy  */
    u8 shadows_on;         /* SHADOWS=0 disables depth-compare PCF shadows  */
    u8 shadow_lit_on;      /* SHADOW_LIT=0 disables 1.0 lit fallback        */
    u8 shadow_flip;        /* SHADOW_FLIP=1 flips GL compare op order       */
    u8 shadow_rp_on;       /* SHADOW_RP=0 uses slope bias instead of RP PCF */
    u8 samp_border_on;     /* SAMP_BORDER=0 ignores guest border colour     */
    double shadow_bias, shadow_rp_max, shadow_slope;
    u32 shadow_lit_tex;    /* lazy 1x1 white (1.0) texture for SHADOW_LIT   */

    /* Optional RES_SCALE + temporal upscaler hooks (zero cost when off). */
    u32 render_w, render_h;        /* physical size for guest-size targets  */
    u8  res_scaled;                /* 1 when render != guest size           */
    u8  upscale_hooks;             /* cached rsx_upscale_hooks_on()         */
    u8  upscale_depth_captured;    /* scene depth already copied this frame */
    u32 upscale_scene_zslot;       /* active main-scene depth target slot   */
    u32 upscale_rec_idx;           /* per-frame draw counter for tracking   */
} g;

static u32 s_req_render_w, s_req_render_h;

static inline void eng_scale_dims(u32 gw, u32 gh, u32* pw, u32* ph)
{
    if (__builtin_expect(g.res_scaled, 0) && gw == g.width && gh == g.height) {
        *pw = g.render_w;
        *ph = g.render_h;
    } else {
        *pw = gw;
        *ph = gh;
    }
}

static int eng_is_display_surface(u32 location, u32 offset)
{
    for (u32 b = 0; b < 8; b++) {
        if (g.display_buffers[b].valid &&
            g.display_buffers[b].location == location &&
            g.display_buffers[b].offset == offset)
            return 1;
    }
    return 0;
}

/* The decompilers' HLSL. The VP decompiler builds bodies up to 192 KB. */
static char s_vs_hlsl[256 * 1024];
static char s_ps_hlsl[256 * 1024];

/* The draw accumulator: one BEGIN_END group's batches, references and cuts. */
static struct {
    eng_batch arr[ENG_MAX_BATCHES];
    u32 n_arr;
    eng_batch idx[ENG_MAX_BATCHES];
    u32 n_idx;
    u32 n_packets;

    rsx_vertex_ref* refs;
    u32 n_refs, cap_refs, n_source_refs;
    int refs_remapped;
    rsx_vertex_remap ref_remap;

    u32* cuts;
    u32 n_cuts, cap_cuts;

    u8* verts;
    u64 verts_cap;
    u32 n_verts;
    /* The converted block this draw uses: the cache's when the group's key
     * was already there, the scratch buffer above otherwise. */
    const u8* verts_cache;
    const u32* indices_cache;
    u64 vc_key;
    u32 vc_ent;
    u64 cur_vid;
    u64 cur_iid;
    rsx_eng_gcache_memo* cur_gcache;
    eng_mcent* mc_hit_ent;
    int mc_hit;
    int mc_elig;
    int mc_was_cold;
    int mc_seen_hit;
    int mc_indexed;
    u32 mc_n_draw;
    u32 mc_snap_frame;
    u64 mc_desc_lo;
    u64 mc_desc_hi;
    u16 mc_desc_bytes;

    rsx_vertex_layout_plan layout;
    rsx_vertex_fetch_plan fetch_plan;
    int fetch_ok;

    /* INLINE_ARRAY stream for this group, when the vertices came down the
     * FIFO rather than from a vertex array. */
    const u8* inl;
    u32 inl_bytes;

    /* SET_VERTEX_DATA* immediate mode: completing attribute 0 emits a vertex
     * containing the current value of every attribute (the Bink/UI quads). */
    float (*imm)[RSX_DSP_NUM_VERTEX_ATTR][4];
    u32 n_imm, cap_imm;
} dc;

/* ---- guest memory -------------------------------------------------------- */

/* rsx_dsp_* locations are RSX_LOCATION_LOCAL = 0 / _MAIN = 1;
 * cellGcmResolveLocated's argument is the opposite sense (1 = local), and
 * getting that backwards resolves every VRAM object through the IO table. */
static rsx_vertex_guest_ptr_fn s_guest_reader;
static void* s_guest_user;

void rsx_draw_engine_set_guest_memory(rsx_vertex_guest_ptr_fn reader, void* user)
{
    s_guest_reader = reader;
    s_guest_user = user;
}

static const u8* eng_guest_ptr(void* user, u32 location, u32 offset,
                               u32 min_bytes)
{
    (void)user;
    if (!min_bytes || min_bytes > (256u << 20)) return NULL;
    if (s_guest_reader) return s_guest_reader(s_guest_user, location, offset, min_bytes);
    if (!vm_base) return NULL;
    const u32 ea = cellGcmResolveLocated(location == RSX_LOCATION_LOCAL, offset);
    if (!ea) {
        static int trace = -1, logs;
        if (trace < 0) trace = getenv("PS3RECOMP_ENG_TRACE") ? 1 : 0;
        if (trace && logs++ < 24)
            fprintf(stderr, "[rsx engine] unresolved guest pointer loc=%u offset=0x%08X bytes=%u\n",
                    location, offset, min_bytes);
        return NULL;
    }
    if (ppu_vm_size && (u64)ea + min_bytes > ppu_vm_size) return NULL;
    return vm_base + ea;
}

static double eng_now_ms(void);
static int eng_stats_on(void);

/* The A/B phase the runtime's driver advances (ppu_loader.cpp), for the knobs
 * that measure in-run: even phases are the new path. Weak, because a tool or a
 * unit test links this file without the runtime: it then reports phase 0, the
 * new path, which is what it would have run anyway. */
#if defined(__GNUC__) || defined(__clang__)
extern volatile int g_ab_phase __attribute__((weak));
#define ENG_AB_PHASE() (&g_ab_phase ? g_ab_phase : 0)
#else
#define ENG_AB_PHASE() 0
#endif

/* Content hash: microcode and texels. Exact, not sampled (sampling the
 * texels is a separate change), but at XXH3's speed instead of one byte per
 * dependent multiply. */
static u64 eng_hash_bytes(const void* data, u32 n, u64 seed)
{
    return XXH3_64bits_withSeed(data, n, seed);
}

/* Key-field combiner: folds one 32-bit state field into a pipeline or cache
 * key. Kept separate from eng_hash_bytes so a key made of many small fields
 * does not pay a call per field. */
static u64 eng_fnv1a(const void* data, u32 n, u64 hash)
{
    const u8* p = (const u8*)data;
    for (u32 i = 0; i < n; i++) { hash ^= p[i]; hash *= 1099511628211ull; }
    return hash;
}

/* ---- render state -------------------------------------------------------- */

void rsx_draw_engine_decode_render_state(const rsx_dispatch* rsx,
                                         rsx_be_render_state* rs)
{
    memset(rs, 0, sizeof(*rs));
    rs->alpha_test_enable = rsx_dsp_reg(rsx, M_ALPHA_TEST_ENABLE) & 1;
    rs->alpha_func   = rsx_dsp_reg(rsx, M_ALPHA_FUNC);
    rs->alpha_ref_raw = rsx_dsp_reg(rsx, M_ALPHA_REF);
    rsx_dsp_surface sf;
    rsx_dsp_get_surface(rsx, &sf);
    rs->alpha_ref_format = sf.color_format;
    rs->rt_fp16 = sf.color_format == RSX_SURFACE_FMT_F_W16Z16Y16X16;
    rs->blend_enable = rsx_dsp_reg(rsx, M_BLEND_ENABLE) & 1;
    const u32 sfac = rsx_dsp_reg(rsx, M_BLEND_SFACTOR);
    const u32 dfac = rsx_dsp_reg(rsx, M_BLEND_DFACTOR);
    const u32 eq   = rsx_dsp_reg(rsx, M_BLEND_EQUATION);
    rs->sf_rgb = sfac & 0xFFFF; rs->sf_a = sfac >> 16;
    rs->df_rgb = dfac & 0xFFFF; rs->df_a = dfac >> 16;
    rs->eq_rgb = eq & 0xFFFF;   rs->eq_a = eq >> 16;
    rs->depth_test  = rsx_dsp_reg(rsx, M_DEPTH_TEST_ENABLE) & 1;
    rs->depth_write = rsx_dsp_reg(rsx, M_DEPTH_WRITE) & 1;
    rs->depth_func  = rsx_dsp_reg(rsx, M_DEPTH_FUNC);
    rs->cull_enable = rsx_dsp_reg(rsx, 0x183C) & 1;
    rs->cull_face   = rsx_dsp_reg(rsx, 0x1830);
    rs->front_face  = rsx_dsp_reg(rsx, 0x1834);
    /* The RAW register: a game-written 0 is a real "write no colour channel"
     * (a depth-prime pass), and rsx_dispatch_init seeds the nv40 reset value
     * so never-written reads as all-on. */
    rs->color_mask  = rsx_dsp_reg(rsx, M_COLOR_MASK);
    /* MRT targets 1..3: COLOR_MASK_MRT (0x0370) packs one A,R,G,B nibble per
     * target at [4i..4i+3]; BLEND_ENABLE_MRT (0x036C) has one bit per target.
     * Target 0 keeps color_mask/blend_enable above. A backend that ignores
     * these still renders the target-0 path; one that honours them gets the
     * extra attachments the title declared. */
    rs->color_mask_mrt = rsx_dsp_reg(rsx, 0x0370);
    rs->blend_mrt      = rsx_dsp_reg(rsx, 0x036C);
    rs->depth_bounds_test = rsx_dsp_reg(rsx, 0x0380) & 1;
    rs->depth_clamp = 0;   /* 0x1D78 is not modelled by this dispatcher yet */
    rs->stencil_enable    = rsx_dsp_reg(rsx, M_STENCIL_TEST_ENABLE) & 1;
    rs->stencil_two_sided = rsx_dsp_reg(rsx, M_TWO_SIDED_STENCIL) & 1;
    rs->s_func       = rsx_dsp_reg(rsx, M_STENCIL_FUNC);
    rs->s_func_mask  = rsx_dsp_reg(rsx, M_STENCIL_FUNC_MASK) & 0xFF;
    rs->s_write_mask = rsx_dsp_reg(rsx, M_STENCIL_MASK) & 0xFF;
    rs->s_fail       = rsx_dsp_reg(rsx, M_STENCIL_OP_FAIL);
    rs->s_zfail      = rsx_dsp_reg(rsx, M_STENCIL_OP_ZFAIL);
    rs->s_zpass      = rsx_dsp_reg(rsx, M_STENCIL_OP_ZPASS);
    rs->bs_func      = rsx_dsp_reg(rsx, M_BACK_STENCIL_FUNC);
    rs->bs_fail      = rsx_dsp_reg(rsx, M_BACK_STENCIL_OP_FAIL);
    rs->bs_zfail     = rsx_dsp_reg(rsx, M_BACK_STENCIL_OP_ZFAIL);
    rs->bs_zpass     = rsx_dsp_reg(rsx, M_BACK_STENCIL_OP_ZPASS);
}

u64 rsx_draw_engine_hash_render_state(const rsx_be_render_state* rs, u64 hash)
{
#define ENG_HASH_FIELD(name) hash = eng_fnv1a(&rs->name, sizeof(rs->name), hash)
    /* The alpha REFERENCE is deliberately absent: it lives in the fragment
     * constant block, so a title animating a fade must not compile a new
     * pipeline per frame. Enable and compare mode still select a variant.
     * Every field is listed by name so struct padding can never be identity
     * (rsx_live_draw.c:3151-3185). */
    ENG_HASH_FIELD(alpha_test_enable);
    ENG_HASH_FIELD(alpha_func);
    ENG_HASH_FIELD(blend_enable);
    ENG_HASH_FIELD(sf_rgb);
    ENG_HASH_FIELD(df_rgb);
    ENG_HASH_FIELD(sf_a);
    ENG_HASH_FIELD(df_a);
    ENG_HASH_FIELD(eq_rgb);
    ENG_HASH_FIELD(eq_a);
    ENG_HASH_FIELD(depth_test);
    ENG_HASH_FIELD(depth_write);
    ENG_HASH_FIELD(depth_func);
    ENG_HASH_FIELD(cull_enable);
    ENG_HASH_FIELD(cull_face);
    ENG_HASH_FIELD(front_face);
    ENG_HASH_FIELD(color_mask);
    ENG_HASH_FIELD(color_mask_mrt);
    ENG_HASH_FIELD(blend_mrt);
    ENG_HASH_FIELD(depth_bounds_test);
    ENG_HASH_FIELD(depth_clamp);
    ENG_HASH_FIELD(rt_fp16);
    ENG_HASH_FIELD(stencil_enable);
    ENG_HASH_FIELD(stencil_two_sided);
    ENG_HASH_FIELD(s_func);
    ENG_HASH_FIELD(s_func_mask);
    ENG_HASH_FIELD(s_write_mask);
    ENG_HASH_FIELD(s_fail);
    ENG_HASH_FIELD(s_zfail);
    ENG_HASH_FIELD(s_zpass);
    ENG_HASH_FIELD(bs_func);
    ENG_HASH_FIELD(bs_fail);
    ENG_HASH_FIELD(bs_zfail);
    ENG_HASH_FIELD(bs_zpass);
#undef ENG_HASH_FIELD
    return hash;
}

/* ---- topology expansion -------------------------------------------------- */

/* Every expansion is bounded by the restart cuts, so a strip that the guest
 * broke with the sentinel index gets no connecting triangle across the break.
 * Missing that decode is what upstream's aa7fb63 records as "the exploded
 * spiky mesh that occluded the scene". */
u32 rsx_draw_engine_topology_index_count(u32 primitive, u32 source_refs,
                                         const u32* cuts, u32 cut_count)
{
    const u32 segments = cut_count + 1;
    switch (primitive) {
    case RSX_PRIMITIVE_TRIANGLES:
        return source_refs - source_refs % 3u;
    case RSX_PRIMITIVE_TRIANGLE_STRIP:
    case RSX_PRIMITIVE_TRIANGLE_FAN:
    case RSX_PRIMITIVE_POLYGON: {
        u32 total = 0;
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            (void)begin;
            if (count >= 3) total += (count - 2) * 3;
        }
        return total;
    }
    case RSX_PRIMITIVE_QUAD_STRIP: {
        /* Two vertices per quad after the first pair, so a segment of n
         * carries (n - 2) / 2 quads and an odd trailing vertex is dropped. */
        u32 total = 0;
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            (void)begin;
            if (count >= 4) total += ((count - 2u) / 2u) * 6u;
        }
        return total;
    }
    case RSX_PRIMITIVE_QUADS:
        return (source_refs / 4u) * 6u;
    default:
        return 0;
    }
}

static u32 eng_topology_vertex(const u32* occurrence_to_unique, u32 occurrence)
{
    return occurrence_to_unique ? occurrence_to_unique[occurrence] : occurrence;
}

void rsx_draw_engine_write_topology_indices(u32 primitive, u32 source_refs,
                                            const u32* cuts, u32 cut_count,
                                            const u32* occurrence_to_unique,
                                            u32* indices)
{
    u32 write = 0;
    const u32 segments = cut_count + 1;
    switch (primitive) {
    case RSX_PRIMITIVE_TRIANGLES: {
        const u32 count = source_refs - source_refs % 3u;
        for (u32 o = 0; o < count; o++)
            indices[write++] = eng_topology_vertex(occurrence_to_unique, o);
        break;
    }
    case RSX_PRIMITIVE_TRIANGLE_STRIP:
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            if (count < 3) continue;
            /* Odd triangles swap their first two vertices, so the whole strip
             * keeps one winding. */
            for (u32 i = 0; i + 2 < count; i++) {
                indices[write++] = eng_topology_vertex(
                    occurrence_to_unique, begin + i + (i & 1u));
                indices[write++] = eng_topology_vertex(
                    occurrence_to_unique, begin + i + 1u - (i & 1u));
                indices[write++] = eng_topology_vertex(
                    occurrence_to_unique, begin + i + 2u);
            }
        }
        break;
    /* A polygon is a convex fan around its own first vertex, which is what
     * both of this tree's other renderers make of one. */
    case RSX_PRIMITIVE_TRIANGLE_FAN:
    case RSX_PRIMITIVE_POLYGON:
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            if (count < 3) continue;
            for (u32 i = 1; i + 1 < count; i++) {
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + i);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + i + 1u);
            }
        }
        break;
    case RSX_PRIMITIVE_QUAD_STRIP:
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            if (count < 4) continue;
            /* Every quad is split along the diagonal between the two pairs,
             * so each one keeps the strip's winding without alternating the
             * way a triangle strip has to; a cut simply starts the pairing
             * again from the segment's own first vertex. This is the
             * expansion the vtable path's emit_vertices performs. */
            for (u32 q = 0; q + 3 < count; q += 2) {
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 1u);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 2u);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 1u);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 3u);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 2u);
            }
        }
        break;
    case RSX_PRIMITIVE_QUADS:
        for (u32 q = 0; q < source_refs / 4u; q++) {
            const u32 base = q * 4u;
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base + 1u);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base + 2u);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base + 2u);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base + 3u);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base);
        }
        break;
    default:
        break;
    }
}

/* Does this primitive become an indexed triangle list, and is the index buffer
 * needed at all? rsx_vertex_topology_plan answers that for the reference
 * engine, whose own expansion stops at quads. Quad strips and polygons are
 * this engine's, so their two cases are decided here rather than in the
 * shared helper, which the vendored rsx_live_draw.c also calls. */
static int eng_topology_rebuild(u32 primitive, int refs_remapped, int* indexed)
{
    if (primitive == RSX_PRIMITIVE_QUAD_STRIP ||
        primitive == RSX_PRIMITIVE_POLYGON) {
        (void)refs_remapped;
        *indexed = 1;
        return 1;
    }
    return rsx_vertex_topology_plan(primitive, refs_remapped, indexed);
}

/* ---- surfaces ------------------------------------------------------------ */

static rsx_be_format eng_surface_format(u32 color_format)
{
    switch (color_format & 0x1Fu) {
    case RSX_SURFACE_FMT_F_W16Z16Y16X16: return RSX_BE_FMT_R16G16B16A16F;
    case 0x0C: return RSX_BE_FMT_R32G32B32A32F;
    case 0x0D: return RSX_BE_FMT_R32F;
    default:   return RSX_BE_FMT_R8G8B8A8;
    }
}

/* Decode the guest's own bytes behind a colour surface into host R,G,B,A rows
 * so a freshly created target starts with what the title CPU-initialised it
 * to. Returns the staging pointer, or NULL when the bytes are unreadable or
 * the format is not one the decoder handles. */
static const void* eng_surface_seed(u32 location, u32 offset, u32 w, u32 h,
                                    rsx_be_format fmt, u32* out_row_bytes)
{
    *out_row_bytes = 0;
    if (fmt != RSX_BE_FMT_R8G8B8A8) return NULL;
    /* Only a surface in IO-mapped MAIN memory is seeded. RSX local memory is
     * reserved and not backed by this tree's guest VM -- runtime/memory/vm.h
     * maps the whole space PROT_NONE and commits main memory and the stacks,
     * leaving a runner to commit what its own title needs -- so reading a VRAM
     * surface's backing would fault long before it could help. Both halves are
     * checked: the DMA context says which space the offset is in, and the IO
     * table says the page is really mapped. */
    if (location != RSX_LOCATION_MAIN ||
        (!s_guest_reader && !cellGcmResolveIO(offset))) return NULL;
    rsx_tex_layout tl;
    /* A8R8G8B8 with the LN bit: a surface is linear, never swizzled. */
    rsx_texture_layout(0x85u | 0x20u, w, h, &tl);
    if (!tl.face_bytes || !tl.dst_row_bytes) return NULL;
    const u8* src = eng_guest_ptr(NULL, location, offset, tl.face_bytes);
    if (!src) return NULL;
    const u32 bytes = tl.dst_row_bytes * tl.rows;
    if (g.tex_staging_cap < bytes) {
        u8* n = (u8*)realloc(g.tex_staging, bytes);
        if (!n) return NULL;
        g.tex_staging = n;
        g.tex_staging_cap = bytes;
    }
    rsx_texture_decode(g.tex_staging, tl.dst_row_bytes, src, w, h, &tl,
                       rsx_texture_argb_is_rgba());
    *out_row_bytes = tl.dst_row_bytes;
    return g.tex_staging;
}

/* The colour target for this (location, offset), created or reallocated when
 * the size or format moved. Returns a slot index, or ENG_INVALID.
 * rsx_live_draw.c's surface_get (2605-2690). */
/* A target's slot by (location, offset), which is what an address names. Two
 * of these lookups happen per draw's target set and up to sixteen more per
 * draw for the texture units that ask whether their address is a live target,
 * so both ran as scans of the whole surface or zeta table. Targets are only
 * ever appended or reused with the same address, so the index needs no
 * deletion. */
static u32 eng_addr_key(u32 location, u32 offset)
{
    u64 key = ((u64)offset << 1) ^ (u64)location;
    key ^= key >> 17;
    key *= 0x9E3779B97F4A7C15ull;
    return (u32)((key >> 32) % ENG_SURFACE_BUCKETS);
}

static u32 eng_surface_find(u32 location, u32 offset)
{
    u32 at = eng_addr_key(location, offset);
    for (u32 walked = 0; walked < ENG_SURFACE_BUCKETS; walked++) {
        const u32 slot = g.surface_bucket[at];
        if (!slot) break;
        const eng_surface* e = &g.surfaces[slot - 1u];
        if (e->location == location && e->offset == offset) return slot - 1u;
        at = (at + 1u) % ENG_SURFACE_BUCKETS;
    }
    return ENG_MAX_SURFACES;
}

/* The target a texture unit means by an address. A guest can declare the same
 * address at more than one size (inFamous points 0x0F340000 at a 1024x1024
 * shadow atlas and at a 512x288 half-res depth copy), so the unit's own
 * declared size decides between them; with no size, or no match on one, the
 * first live declaration answers, which is what an address-only lookup did. */
static u32 eng_surface_find_sized(u32 location, u32 offset, u32 w, u32 h)
{
    u32 at = eng_addr_key(location, offset), first = ENG_MAX_SURFACES;
    for (u32 walked = 0; walked < ENG_SURFACE_BUCKETS; walked++) {
        const u32 slot = g.surface_bucket[at];
        if (!slot) break;
        const eng_surface* e = &g.surfaces[slot - 1u];
        if (e->location == location && e->offset == offset) {
            if ((w && e->w == w && h && e->h == h) || !w) return slot - 1u;
            if (first == ENG_MAX_SURFACES ||
                (!g.surfaces[first].cwritten && e->cwritten))
                first = slot - 1u;
        }
        at = (at + 1u) % ENG_SURFACE_BUCKETS;
    }
    return first;
}

static void eng_surface_index(u32 slot)
{
    u32 at = eng_addr_key(g.surfaces[slot].location, g.surfaces[slot].offset);
    while (g.surface_bucket[at]) at = (at + 1u) % ENG_SURFACE_BUCKETS;
    g.surface_bucket[at] = slot + 1u;
    g.surface_gen++;
}

static u32 eng_zdepth_find(u32 location, u32 offset)
{
    u32 at = eng_addr_key(location, offset);
    for (u32 walked = 0; walked < ENG_SURFACE_BUCKETS; walked++) {
        const u32 slot = g.zdepth_bucket[at];
        if (!slot) break;
        const eng_zdepth* e = &g.zdepths[slot - 1u];
        if (e->location == location && e->offset == offset) return slot - 1u;
        at = (at + 1u) % ENG_SURFACE_BUCKETS;
    }
    return ENG_MAX_ZDEPTHS;
}

static u32 eng_zdepth_find_sized(u32 location, u32 offset, u32 w, u32 h)
{
    u32 at = eng_addr_key(location, offset), best = ENG_MAX_ZDEPTHS;
    for (u32 walked = 0; walked < ENG_SURFACE_BUCKETS; walked++) {
        const u32 slot = g.zdepth_bucket[at];
        if (!slot) break;
        const eng_zdepth* e = &g.zdepths[slot - 1u];
        if (e->location == location && e->offset == offset) {
            if ((w && h && e->w == w && e->h == h) || !w || !h)
                return slot - 1u;
            if (best == ENG_MAX_ZDEPTHS ||
                (!g.zdepths[best].had_write && e->had_write))
                best = slot - 1u;
        }
        at = (at + 1u) % ENG_SURFACE_BUCKETS;
    }
    return best;
}

static void eng_zdepth_index(u32 slot)
{
    u32 at = eng_addr_key(g.zdepths[slot].location, g.zdepths[slot].offset);
    while (g.zdepth_bucket[at]) at = (at + 1u) % ENG_SURFACE_BUCKETS;
    g.zdepth_bucket[at] = slot + 1u;
    g.surface_gen++;
}

/* The colour target for this (location, offset) at this size and format,
 * created when it does not exist yet. A guest may declare one address at two
 * sizes and alternate between them every frame; keying on the address alone
 * made each declaration recreate the other's target and drop its contents, so
 * a pass that rendered at one size and was read back at the other read a
 * freshly cleared image (inFamous renders at two sizes for this reason, and
 * rsx_live_draw.c's surface_get (2823-2900) records the same defect and keys
 * on the size as the fix). Two declarations of one address do not alias, so a
 * pass that renders at one size and samples at the other sees nothing; that
 * is strictly better than both seeing nothing. Returns a slot index, or
 * ENG_INVALID. */
static u32 eng_surface_get(u32 location, u32 offset, u32 want_w, u32 want_h,
                          rsx_be_format want_fmt)
{
    if (!want_w) want_w = g.width;
    if (!want_h) want_h = g.height;

    /* Exact match first: the common case, and the only one that may reuse. */
    for (u32 at = eng_addr_key(location, offset), walked = 0;
         walked < ENG_SURFACE_BUCKETS; walked++, at = (at + 1u) % ENG_SURFACE_BUCKETS) {
        const u32 slot = g.surface_bucket[at];
        if (!slot) break;
        const eng_surface* e = &g.surfaces[slot - 1u];
        if (e->location == location && e->offset == offset &&
            e->w == want_w && e->h == want_h && e->fmt == want_fmt)
            return slot - 1u;
    }

    /* Never destroy a usable render target because one malformed command
     * decoded a guest pointer as clip dimensions. */
    if (want_w > ENG_MAX_SURFACE_DIM || want_h > ENG_MAX_SURFACE_DIM) {
        const u32 slot = eng_surface_find(location, offset);
        static u32 logs = 0;
        if ((logs < 8 && logs++ < 8))
            fprintf(stderr, "[rsx engine] rejected implausible surface 0x%X %ux%u;"
                            " keeping the %s target\n", offset, want_w, want_h,
                    slot < ENG_MAX_SURFACES ? "existing" : "absent");
        return slot < ENG_MAX_SURFACES ? slot : ENG_INVALID;
    }
    u32 slot = g.n_surfaces;
    if (slot >= ENG_MAX_SURFACES) {
        const u32 any = eng_surface_find(location, offset);
        static u32 logs = 0;
        if (logs < 8 && logs++ < 8)
            fprintf(stderr, "[rsx engine] surface table full (%u); keeping 0x%08X %ux%u\n",
                    (unsigned)ENG_MAX_SURFACES, offset, want_w, want_h);
        return any < ENG_MAX_SURFACES ? any : ENG_INVALID;
    }

    u32 phys_w, phys_h;
    eng_scale_dims(want_w, want_h, &phys_w, &phys_h);
    u32 seed_row = 0;
    const void* seed = (phys_w == want_w && phys_h == want_h)
        ? eng_surface_seed(location, offset, want_w, want_h, want_fmt, &seed_row)
        : NULL;
    const u32 handle = g.be->color_target_create(g.be->user, want_fmt,
                                                 phys_w, phys_h, seed, seed_row);
    if (!handle) return ENG_INVALID;

    eng_surface* s = &g.surfaces[slot];
    memset(s, 0, sizeof *s);
    s->location = location; s->offset = offset;
    s->w = want_w; s->h = want_h; s->fmt = want_fmt;
    s->handle = handle;
    g.n_surfaces++;
    eng_surface_index(slot);
    g.surface_gen++;
    { static u32 logs = 0; if ((logs < 24 && logs++ < 24))
        fprintf(stderr, "[rsx engine] surface %u:0x%08X %ux%u fmt %d slot %u%s\n",
                location, offset, want_w, want_h, (int)want_fmt, slot,
                seed ? " (seeded from guest memory)" : ""); }
    return slot;
}

/* The colour targets SET_SURFACE_COLOR_TARGET names, as slots into
 * g.surfaces, with A first. One for an ordinary draw; an MRT set adds B, C
 * and D, each created at target A's size and format, which is the rule the
 * vtable path's target_for binds by. Resolution stops at the first member
 * that cannot be created, as that path stops at the first gap. Returns how
 * many were resolved, or 0. */
static u32 eng_current_target_set(u32 slots[RSX_BE_MAX_COLOR_TARGETS])
{
    rsx_dsp_surface sf;
    rsx_dsp_get_surface(&g.rsx, &sf);
    /* SET_SURFACE_COLOR_TARGET 2 selects B alone; every other value starts at
     * A, and the MRT selectors add B, C and D on top of it. */
    const u32 sel = (sf.color_target == CELL_GCM_SURFACE_TARGET_1) ? 1u : 0u;
    const rsx_be_format fmt = eng_surface_format(sf.color_format);
    const u32 first = eng_surface_get(sf.color_location[sel], sf.color_offset[sel],
                                      sf.clip_w, sf.clip_h, fmt);
    if (first == ENG_INVALID) return 0;
    slots[0] = first;
    u32 n = 1;

    static const u32 mrt_from[3] = {
        CELL_GCM_SURFACE_TARGET_MRT1, CELL_GCM_SURFACE_TARGET_MRT2,
        CELL_GCM_SURFACE_TARGET_MRT3
    };
    for (u32 i = 0; i < 3; i++) {
        if (sf.color_target < mrt_from[i]) break;
        const u32 slot = eng_surface_get(sf.color_location[i + 1],
                                         sf.color_offset[i + 1],
                                         sf.clip_w, sf.clip_h, fmt);
        if (slot == ENG_INVALID) break;
        /* A set naming one buffer twice would attach the same target twice,
         * which no host API allows. */
        int seen = 0;
        for (u32 k = 0; k < n; k++) if (slots[k] == slot) seen = 1;
        if (seen) break;
        slots[n++] = slot;
    }
    return n;
}

static u32 eng_current_surface(void)
{
    u32 slots[RSX_BE_MAX_COLOR_TARGETS];
    return eng_current_target_set(slots) ? slots[0] : ENG_INVALID;
}

/* ---- per-zeta depth ------------------------------------------------------ */

/* One depth target per guest zeta address. Sharing a single resource across a
 * title's shadow, scene and post passes cross-contaminates later depth tests;
 * upstream's ef3271b records that as "the black player-character mass". */
static u32 eng_zdepth_get(u32 location, u32 offset, u32 rt_w, u32 rt_h)
{
    if (g.zeta_exact_size_on) {
        /* Key on (location, offset, w, h) without padding to canvas size:
         * shadow maps (e.g. 1024x1024) and half-res depth (512x288) can
         * share one guest offset and are sampled in normalized UV space,
         * so canvas-width padding stretches coordinates and aliasing two
         * sizes on one slot drops the other pass's contents. */
        const u32 want_w = rt_w ? rt_w : g.width;
        const u32 want_h = rt_h ? rt_h : g.height;
        if (want_w > ENG_MAX_SURFACE_DIM || want_h > ENG_MAX_SURFACE_DIM)
            return ENG_INVALID;
        for (u32 at = eng_addr_key(location, offset), walked = 0;
             walked < ENG_SURFACE_BUCKETS;
             walked++, at = (at + 1u) % ENG_SURFACE_BUCKETS) {
            const u32 slot = g.zdepth_bucket[at];
            if (!slot) break;
            const eng_zdepth* e = &g.zdepths[slot - 1u];
            if (e->location == location && e->offset == offset &&
                e->w == want_w && e->h == want_h)
                return slot - 1u;
        }
        if (g.n_zdepths >= ENG_MAX_ZDEPTHS) {
            const u32 any = eng_zdepth_find(location, offset);
            return any < ENG_MAX_ZDEPTHS ? any : ENG_INVALID;
        }
        const u32 slot = g.n_zdepths;
        u32 phys_w, phys_h;
        eng_scale_dims(want_w, want_h, &phys_w, &phys_h);
        const u32 handle = g.be->depth_target_create(g.be->user, phys_w, phys_h);
        if (!handle) return ENG_INVALID;
        eng_zdepth* z = &g.zdepths[slot];
        memset(z, 0, sizeof *z);
        z->location = location; z->offset = offset;
        z->w = want_w; z->h = want_h;
        z->handle = handle;
        g.n_zdepths++;
        eng_zdepth_index(slot);
        g.surface_gen++;
        return slot;
    }

    /* Legacy ZETA_EXACT_SIZE=0 path: one padded depth target per address. */
    u32 want_w = rt_w > g.width ? rt_w : g.width;
    u32 want_h = rt_h > g.height ? rt_h : g.height;
    if (want_w > ENG_MAX_SURFACE_DIM || want_h > ENG_MAX_SURFACE_DIM)
        return ENG_INVALID;

    u32 slot = eng_zdepth_find(location, offset);
    if (slot < ENG_MAX_ZDEPTHS) {
        eng_zdepth* z = &g.zdepths[slot];
        if (z->w >= want_w && z->h >= want_h) return slot;
        if (want_w < z->w) want_w = z->w;
        if (want_h < z->h) want_h = z->h;
    }
    if (slot == ENG_MAX_ZDEPTHS) {
        if (g.n_zdepths >= ENG_MAX_ZDEPTHS) return ENG_INVALID;
        slot = g.n_zdepths;
    }

    u32 phys_w, phys_h;
    eng_scale_dims(want_w, want_h, &phys_w, &phys_h);
    const u32 handle = g.be->depth_target_create(g.be->user, phys_w, phys_h);
    if (!handle)
        return (slot < g.n_zdepths && g.zdepths[slot].handle) ? slot : ENG_INVALID;

    eng_zdepth* z = &g.zdepths[slot];
    if (z->handle) g.be->depth_target_release(g.be->user, z->handle);
    z->location = location; z->offset = offset;
    z->w = want_w; z->h = want_h;
    z->handle = handle;
    z->snapshot = 0;
    z->color_snapshot = 0;
    z->cleared = 0;
    z->had_write = 0;
    z->snapshot_valid = 0;
    z->color_snapshot_valid = 0;
    if (slot == g.n_zdepths) { g.n_zdepths++; eng_zdepth_index(slot); }
    g.surface_gen++;
    return slot;
}

/* A depth target read back as a texture, but only once the pass has executed
 * a depth-WRITING draw: a write-enable bit alone does not prove the pass
 * produced a usable depth map, and a clear-only zeta falls through to guest
 * memory instead (rsx_live_draw.c:6016-6018, 6265-6269). */
static u32 eng_zdepth_snapshot(u32 slot)
{
    eng_zdepth* z = &g.zdepths[slot];
    if (!z->handle || !z->had_write) return 0;
    if (z->snapshot_valid && z->snapshot) return z->snapshot;
    if (!g.be->depth_snapshot) return 0;
    u32 pw, ph;
    eng_scale_dims(z->w, z->h, &pw, &ph);
    const u32 tex = g.be->depth_snapshot(g.be->user, z->handle, pw, ph);
    if (!tex) return 0;
    z->snapshot = tex;
    z->snapshot_valid = 1;
    return tex;
}

/* Convert a written depth+stencil target into an A8R8G8B8 colour surface
 * packing float4(mid, lo, stencil, hi) for passes that sample a zeta address
 * as format 0x85 (inFamous sun/moon light pass and half-res depth consumers). */
static u32 eng_zdepth_color_snapshot(u32 slot)
{
    eng_zdepth* z = &g.zdepths[slot];
    if (!z->handle || !z->had_write || !g.zconv_color_on) return 0;
    if (z->color_snapshot_valid && z->color_snapshot) return z->color_snapshot;
    if (!g.be->depth_color_snapshot) return 0;
    u32 pw, ph;
    eng_scale_dims(z->w, z->h, &pw, &ph);
    const u32 tex = g.be->depth_color_snapshot(g.be->user, z->handle, pw, ph);
    if (!tex) return 0;
    z->color_snapshot = tex;
    z->color_snapshot_valid = 1;
    return tex;
}

/* ---- textures ------------------------------------------------------------ */

static rsx_be_format eng_texfmt(rsx_texfmt f)
{
    switch (f) {
    case RSX_TEXFMT_R8:              return RSX_BE_FMT_R8;
    case RSX_TEXFMT_R8G8:            return RSX_BE_FMT_R8G8;
    case RSX_TEXFMT_R8G8B8A8:        return RSX_BE_FMT_R8G8B8A8;
    case RSX_TEXFMT_BC1:             return RSX_BE_FMT_BC1;
    case RSX_TEXFMT_BC2:             return RSX_BE_FMT_BC2;
    case RSX_TEXFMT_BC3:             return RSX_BE_FMT_BC3;
    case RSX_TEXFMT_R16:             return RSX_BE_FMT_R16;
    case RSX_TEXFMT_R16G16:          return RSX_BE_FMT_R16G16;
    case RSX_TEXFMT_R16G16F:         return RSX_BE_FMT_R16G16F;
    case RSX_TEXFMT_R16G16B16A16F:   return RSX_BE_FMT_R16G16B16A16F;
    case RSX_TEXFMT_R32F:            return RSX_BE_FMT_R32F;
    case RSX_TEXFMT_R32G32B32A32F:   return RSX_BE_FMT_R32G32B32A32F;
    default:                         return RSX_BE_FMT_R8;
    }
}

/* How many guest bytes a texture occupies: the whole mip chain, times six for
 * a cube map. That is what the content hash has to cover, or a title
 * animating one face or one lower level keeps its stale upload. */
static u32 eng_texture_span(u32 fmt, u32 w, u32 h, u32 levels, u32 pitch, int cube)
{
    if (cube) return rsx_texture_cube_face_stride(fmt, w, h, levels, pitch) * 6u;
    rsx_tex_level lv[RSX_MAX_TEXTURE_LEVELS];
    const u32 n = rsx_texture_mip_chain(fmt, w, h, levels, pitch, lv);
    if (!n) return 0;
    return lv[n - 1].offset + lv[n - 1].tl.face_bytes;
}

/* ---- sparse texture validation -------------------------------------------
 * A cached texture has to be re-read every frame it is bound, or an animated
 * UI, video or water surface freezes. Hashing every texel of every bound
 * texture cost 69 MB a frame on the preload (0.5 GB/s of the walker at FNV,
 * still 4.7 ms at XXH3). The D3D12 cache's policy is ported here instead
 * (rsx_d3d12_backend.c:5254-5338, :5653-5682):
 *
 *   - every frame, ~1024 words spread across the span (4 KB of a large
 *     texture), compared against the sparse hash stored with the last full
 *     verification;
 *   - a FULL hash when the samples differ, when the entry is new, and at
 *     least every ENG_TEX_VERIFY_FRAMES frames per texture regardless.
 *
 * The last rule is what keeps a change the samples cannot see -- one texel
 * between two sample points -- from freezing a surface forever: it is the
 * VC_SWD_VERIFY rule the D3D12 cache applies to its own unverified writers,
 * and it is mandatory, not an optimization. */
#define ENG_TEX_VERIFY_FRAMES 8u

/* ENG_TEX_SPARSE=0 restores the old policy (the full hash on every bind in
 * every frame); ENG_TEX_SPARSE_PHASE=<0|1> runs the sparse policy only on that
 * A/B parity, so its cost is the difference between two phases of ONE run. */
static int eng_texture_sparse_use(void)
{
    static int cfg = -1, phase = -1;
    if (cfg < 0) {
        const char* e = getenv("ENG_TEX_SPARSE");
        const char* p = getenv("ENG_TEX_SPARSE_PHASE");
        cfg = (e && *e == '0') ? 0 : 1;
        if (p && *p) phase = atoi(p) & 1;
    }
    if (!cfg) return 0;
    return phase < 0 || (ENG_AB_PHASE() & 1) == phase;
}

/* FNV-1a over ~1024 words spread across the span: tex_csum_step's sampling
 * policy, including its eight-independent-loads batching, which overlaps the
 * latency of samples that land in different cache lines instead of walking
 * one dependent load/hash chain. Same operation order, so the value is
 * stable; it is only ever compared against the same function's older output.
 * `bytes` receives the bytes actually hashed, for the [eng stats] total. */
static u32 eng_texture_sample_hash(const u8* src, u32 span, u32* bytes)
{
    const u32 step = span > 4096u ? span / 1024u : 4u;
    const u64 batch = (u64)step * 7u + 3u;
    u32 h = 2166136261u, i = 0, n = 0;
    for (; (u64)i + batch < span; i += step * 8u) {
        u32 w[8];
        for (u32 k = 0; k < 8; k++) memcpy(&w[k], src + i + (size_t)k * step, sizeof w[k]);
        for (u32 k = 0; k < 8; k++) { h ^= w[k]; h *= 16777619u; }
        n += 8;
    }
    for (; i + 3u < span; i += step) {
        u32 w; memcpy(&w, src + i, sizeof w);
        h ^= w; h *= 16777619u; n++;
    }
    if (bytes) *bytes = n * 4u;
    return h;
}

static int eng_staging_reserve(u32 bytes)
{
    if (g.tex_staging_cap >= bytes) return 1;
    u8* n = (u8*)realloc(g.tex_staging, bytes);
    if (!n) return 0;
    g.tex_staging = n;
    g.tex_staging_cap = bytes;
    return 1;
}

/* Decode a guest texture out of guest memory and hand every face and level to
 * the backend. Returns the backend handle, or 0. */
static u32 eng_texture_upload(u32 location, u32 offset, u32 fmt, u32 w, u32 h,
                              u32 levels, u32 pitch, int cube, u32 remap)
{
    rsx_tex_level lv[RSX_MAX_TEXTURE_LEVELS];
    const u32 nlv = rsx_texture_mip_chain(fmt, w, h, levels, pitch, lv);
    if (!nlv || !lv[0].tl.face_bytes) return 0;
    const u32 faces = cube ? 6u : 1u;
    const u32 face_stride = cube
        ? rsx_texture_cube_face_stride(fmt, w, h, levels, pitch) : 0u;
    const u32 span = eng_texture_span(fmt, w, h, levels, pitch, cube);
    const u8* src = eng_guest_ptr(NULL, location, offset, span);
    if (!src) return 0;

    g.st_uploads++;
    const u32 handle = g.be->texture_create(g.be->user, eng_texfmt(lv[0].tl.fmt),
                                            w, h, nlv, faces, remap, fmt);
    if (!handle) return 0;

    for (u32 f = 0; f < faces; f++) {
        const u8* face = src + (size_t)f * face_stride;
        for (u32 m = 0; m < nlv; m++) {
            const rsx_tex_layout* tl = &lv[m].tl;
            if (!eng_staging_reserve(tl->dst_row_bytes * tl->rows)) {
                g.be->texture_release(g.be->user, handle);
                return 0;
            }
            rsx_texture_decode(g.tex_staging, tl->dst_row_bytes,
                               face + lv[m].offset, lv[m].w, lv[m].h, tl,
                               rsx_texture_argb_is_rgba());
            g.be->texture_upload(g.be->user, handle, f, m, lv[m].w, lv[m].h,
                                 g.tex_staging, tl->dst_row_bytes, tl->rows);
        }
    }
    return handle;
}

/* The cache slot for a texture unit's bytes, keyed on where they are and what
 * the registers say they are. An already-cached entry is re-hashed at most
 * once per presented frame and re-decoded when the guest changed it, which is
 * what makes an animated UI or a video texture update at all. A miss with the
 * cache full evicts the least recently used entry rather than returning
 * nothing: returning white "made the recovered orphanage render as flat
 * green/black geometry" (rsx_live_draw.c:2326-2334). */
/* The cache slot's key, as one hash for the bucket index. The fields are also
 * compared one by one on every probe: the hash never decides on its own. */
static u64 eng_texture_key(u32 location, u32 offset, u32 fmt, u32 w, u32 h,
                           u32 levels, u32 pitch, u32 remap, int cube)
{
    /* Packed and mixed rather than folded byte by byte: this runs once per
     * bound texture unit per draw (68k times a frame on the preload) and the
     * fields' bit ranges do not overlap. */
    u64 k = (u64)offset
          ^ ((u64)(fmt & 0xFFFFu) << 40)
          ^ ((u64)(w & 0x1FFFu) << 27)
          ^ ((u64)(h & 0x1FFFu) << 14)
          ^ ((u64)(pitch & 0x7FFFu) << 1)
          ^ ((u64)location << 51)
          ^ ((u64)(cube != 0) << 50);
    /* The level count fixes the span an entry hashes and uploads, so a re-bind
     * with a different mip chain must not match the old entry. */
    k ^= (u64)(levels & 0xFFu) * 0xC2B2AE3D27D4EB4Full;
    k ^= (u64)((remap & 0xFFFFu) + 1u) * 0x9E3779B97F4A7C15ull;
    k ^= k >> 33;
    k *= 0xFF51AFD7ED558CCDull;
    k ^= k >> 29;
    return k;
}

static int eng_texture_key_matches(const eng_texture* e, u32 location, u32 offset,
                                   u32 fmt, u32 w, u32 h, u32 levels, u32 pitch,
                                   u32 remap, int cube)
{
    return e->location == location && e->offset == offset &&
           e->format == fmt && e->width == w && e->height == h &&
           e->levels == levels && e->pitch == pitch && e->remap == remap &&
           e->cubemap == (u32)(cube != 0);
}

static void eng_texture_table_remove(const eng_texture* e)
{
    if (!e->handle && !e->width) return;   /* never inserted */
    const u64 key = eng_texture_key(e->location, e->offset, e->format, e->width,
                                    e->height, e->levels, e->pitch, e->remap,
                                    (int)e->cubemap);
    u32 at = (u32)(key % ENG_TEXTURE_BUCKETS);
    for (u32 walked = 0; walked < ENG_TEXTURE_BUCKETS; walked++) {
        u32* slot = &g.tex_bucket[at];
        if (*slot == 0) return;
        if (*slot != ENG_TEXTURE_TOMBSTONE) {
            const eng_texture* o = &g.textures[*slot - 1u];
            if (o->location == e->location && o->offset == e->offset &&
                o->format == e->format && o->width == e->width &&
                o->height == e->height && o->pitch == e->pitch &&
                o->remap == e->remap && o->cubemap == e->cubemap) {
                *slot = ENG_TEXTURE_TOMBSTONE;
                g.tex_tombstones++;
                return;
            }
        }
        at = (at + 1u) % ENG_TEXTURE_BUCKETS;
    }
}

static void eng_texture_table_rebuild(void)
{
    memset(g.tex_bucket, 0, sizeof g.tex_bucket);
    g.tex_tombstones = 0;
    for (u32 i = 0; i < g.n_textures; i++) {
        const eng_texture* e = &g.textures[i];
        if (!e->width) continue;
        const u64 key = eng_texture_key(e->location, e->offset, e->format,
                                        e->width, e->height, e->levels, e->pitch,
                                        e->remap, (int)e->cubemap);
        u32 at = (u32)(key % ENG_TEXTURE_BUCKETS);
        while (g.tex_bucket[at] && g.tex_bucket[at] != ENG_TEXTURE_TOMBSTONE)
            at = (at + 1u) % ENG_TEXTURE_BUCKETS;
        g.tex_bucket[at] = i + 1u;
    }
}

static void eng_texture_table_insert(u32 index)
{
    if (g.tex_tombstones > ENG_TEXTURE_BUCKETS / 2u) eng_texture_table_rebuild();
    const eng_texture* e = &g.textures[index];
    const u64 key = eng_texture_key(e->location, e->offset, e->format, e->width,
                                    e->height, e->levels, e->pitch, e->remap,
                                    (int)e->cubemap);
    u32 at = (u32)(key % ENG_TEXTURE_BUCKETS);
    u32 tomb = ENG_TEXTURE_BUCKETS;
    for (;;) {
        u32* slot = &g.tex_bucket[at];
        if (*slot == 0) {
            u32* where = (tomb != ENG_TEXTURE_BUCKETS) ? &g.tex_bucket[tomb] : slot;
            *where = index + 1u;
            return;
        }
        if (*slot == ENG_TEXTURE_TOMBSTONE) { if (tomb == ENG_TEXTURE_BUCKETS) tomb = at; }
        at = (at + 1u) % ENG_TEXTURE_BUCKETS;
    }
}

/* The cache slot for a texture unit's bytes, keyed on where they are and what
 * the registers say they are. An already-cached entry is re-hashed at most
 * once per presented frame and re-decoded when the guest changed it, which is
 * what makes an animated UI or a video texture update at all. A miss with the
 * cache full evicts the least recently used entry rather than returning
 * nothing: returning white "made the recovered orphanage render as flat
 * green/black geometry" (rsx_live_draw.c:2326-2334).
 *
 * The lookup is a hash index (eng_texture_key) rather than a scan of every
 * entry in turn: the preload walked 47 M entries a frame doing this, for 801
 * distinct textures. The span the content hash covers is computed only when
 * the hash is actually taken, not on every bind. */
static u32* s_swd_gen = NULL;
static u32  s_swd_coarse[4096];  /* 1 MB superpage max dirty gen (16 KB, L1) */
static u8*  s_swd_nonspu = NULL;
static u32  s_swd_frame = 1u;
static int  s_mc_cfg = -1, s_mc_phase = -1, s_mc_ver = 64;
static int  s_mc_swd = 0, s_mc_swd_phase = -1;

/* ENG_M2D=0 disables M2d optimizations (SWD texture validation, whole-set
 * texture/program/pipeline fast memos, VC thrash compaction, fast MC desc).
 * ENG_M2D_PHASE=<0|1> runs M2d only on that A/B parity. */
static int s_vp_pack_cfg = -1;
static int eng_vp_pack_use(void)
{
    if (s_vp_pack_cfg < 0) {
        const char* e = getenv("ENG_VP_PACK");
        s_vp_pack_cfg = (e && *e == '0') ? 0 : 1;
    }
    return s_vp_pack_cfg;
}

static int s_vp_sparse_cfg = -1;
static int eng_vp_sparse_use(void)
{
    if (s_vp_sparse_cfg < 0) {
        const char* e = getenv("ENG_VP_SPARSE");
        s_vp_sparse_cfg = (e && *e == '0') ? 0 : 1;
    }
    return s_vp_sparse_cfg && eng_vp_pack_use();
}

static int s_m2d_cfg = -1, s_m2d_phase = -1;
static int eng_m2d_use(void)
{
    if (s_m2d_cfg < 0) {
        const char* e = getenv("ENG_M2D");
        const char* p = getenv("ENG_M2D_PHASE");
        s_m2d_cfg = (e && *e == '0') ? 0 : 1;
        if (p && *p) s_m2d_phase = atoi(p) & 1;
    }
    if (!s_m2d_cfg) return 0;
    return s_m2d_phase < 0 || (ENG_AB_PHASE() & 1) == s_m2d_phase;
}

static int s_m2e4_cfg = -1, s_m2e4_phase = -1;
static int eng_m2e4_use(void)
{
    if (s_m2e4_cfg < 0) {
        const char* e = getenv("M2E4");
        if (e && (e[0] == 'a' || e[0] == 'A')) { s_m2e4_cfg = 1; s_m2e4_phase = 0; }
        else { s_m2e4_cfg = (e && *e == '0') ? 0 : 1; }
        const char* p = getenv("M2E4_PHASE");
        if (p && *p) s_m2e4_phase = atoi(p) & 1;
    }
    if (!s_m2e4_cfg) return 0;
    return s_m2e4_phase < 0 || (ENG_AB_PHASE() & 1) == s_m2e4_phase;
}

static int s_m2e4b_cfg = -1, s_m2e4b_phase = -1;
static int eng_m2e4b_use(void)
{
    if (s_m2e4b_cfg < 0) {
        const char* e = getenv("M2E4B");
        if (e && (e[0] == 'a' || e[0] == 'A')) { s_m2e4b_cfg = 1; s_m2e4b_phase = 0; }
        else { s_m2e4b_cfg = (e && *e == '0') ? 0 : 1; }
        const char* p = getenv("M2E4B_PHASE");
        if (p && *p) s_m2e4b_phase = atoi(p) & 1;
    }
    if (!s_m2e4b_cfg) return 0;
    return s_m2e4b_phase < 0 || (ENG_AB_PHASE() & 1) == s_m2e4b_phase;
}

static int s_m2f_cfg = -1, s_m2f_phase = -1;
static int eng_m2f_use(void)
{
    if (s_m2f_cfg < 0) {
        const char* e = getenv("ENG_M2F");
        if (e && (e[0] == 'a' || e[0] == 'A')) { s_m2f_cfg = 1; s_m2f_phase = 0; }
        else { s_m2f_cfg = (e && *e == '0') ? 0 : 1; }
        const char* p = getenv("ENG_M2F_PHASE");
        if (p && *p) s_m2f_phase = atoi(p) & 1;
    }
    if (!s_m2f_cfg) return 0;
    return s_m2f_phase < 0 || (ENG_AB_PHASE() & 1) == s_m2f_phase;
}

/* Per-phase cached hot flags (eliminates ~150k per-draw helper calls/frame). */
static struct {
    int phase;
    u8  m2c, m2d, m2e4, m2e4b, m2f, stats, trace, mc, mc_swd;
} s_ef = { -999999, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
static int eng_m2c_use(void);
static int eng_stats_on(void);
static int eng_trace(void);
static int eng_mc_use(void);
static void eng_refresh_flags(int ph)
{
    s_ef.m2c    = (u8)(eng_m2c_use() != 0);
    s_ef.m2d    = (u8)(eng_m2d_use() != 0);
    s_ef.m2e4   = (u8)(eng_m2e4_use() != 0);
    s_ef.m2e4b  = (u8)(eng_m2e4b_use() != 0);
    s_ef.m2f    = (u8)(eng_m2f_use() != 0);
    s_ef.stats  = (u8)(eng_stats_on() != 0);
    s_ef.trace  = (u8)(eng_trace() != 0);
    s_ef.mc     = (u8)(eng_mc_use() != 0);
    s_ef.mc_swd = (u8)(s_mc_swd && (s_mc_swd_phase < 0 || (ph & 1) == s_mc_swd_phase));
    s_ef.phase  = ph;
}

rsx_eng_gcache_memo* g_rsx_eng_cur_gcache = NULL;
u32 g_rsx_vk_gcache_gen = 0u;
static u32 s_mc_prev_slot = 0xFFFFFFFFu;
static u32 s_mc_next_pred = 0xFFFFFFFFu;

#define ENG_TEX_FAST_SETS   2048u
#define ENG_TEX_FAST_WAYS   4u
#define ENG_TEX_FAST_SLOTS  (ENG_TEX_FAST_SETS * ENG_TEX_FAST_WAYS)

static inline u32 eng_tex_fast_idx(u32 offset, u32 format, u32 width, u32 height, u32 remap)
{
    return (((offset >> 4) ^ (format * 31u) ^
             (width << 5) ^ (height << 13) ^
             ((remap & 0xFFFFu) * 17u)) & (ENG_TEX_FAST_SETS - 1u)) * ENG_TEX_FAST_WAYS;
}

static void eng_tex_invalidate_handle(u32 old_handle, u32 tf_idx);
static u32 s_last_tex_swd_ver, s_last_tex_full_frame;
static u16 s_last_tex_clo, s_last_tex_chi;
static u8  s_last_tex_swd_ok;

static u32 eng_texture_slot(u32 location, u32 offset, u32 fmt, u32 w, u32 h,
                            u32 levels, u32 pitch, int cube, u32 remap)
{
    s_last_tex_swd_ok = 0;
    if (!w || !h || w > ENG_MAX_TEX_DIM || h > ENG_MAX_TEX_DIM) return 0;
    const int stats = eng_stats_on();
    if (stats) g.st_lookups++;

    const u64 key = eng_texture_key(location, offset, fmt, w, h, levels, pitch, remap, cube);
    u32 at = (u32)(key % ENG_TEXTURE_BUCKETS);
    for (u32 walked = 0; walked < ENG_TEXTURE_BUCKETS; walked++) {
        const u32 slot = g.tex_bucket[at];
        if (slot == 0) break;
        if (slot != ENG_TEXTURE_TOMBSTONE) {
            if (stats) g.st_probes++;
            eng_texture* e = &g.textures[slot - 1u];
            if (!eng_texture_key_matches(e, location, offset, fmt, w, h, levels,
                                         pitch, remap, cube)) {
                at = (at + 1u) % ENG_TEXTURE_BUCKETS;
                continue;
            }
            /* An entry whose first upload failed was inserted with handle 0
             * and then served nothing for the rest of the run: the guest span
             * can be unmapped when a streamed texture is bound before its DMA
             * lands. Retry the upload instead, and count the binds it still
             * cannot fill so a soak run shows them. */
            if (!e->handle) {
                e->handle = eng_texture_upload(location, offset, fmt, w, h, levels,
                                               pitch, cube, remap);
                if (!e->handle && stats) g.st_handle0++;
            }
            if (e->handle && e->last_hash_frame != g.frames) {
                const u8* src = e->span ? eng_guest_ptr(NULL, location, offset, e->span)
                                        : NULL;
                if (src) {
                    const int stats = eng_stats_on();
                    const int m2d = eng_m2d_use();
                    int swd_skip = 0, swd_eligible = 0, swd_dirty = 0, swd_nonspu = 0;
                    u32 swd_a = 0, swd_b = 0;
                    const u32 cur_swd = s_mc_swd ? __atomic_load_n(&s_swd_frame, __ATOMIC_ACQUIRE) : 0u;
                    const u32 min_swd_span = eng_m2f_use() ? 64u : 16384u;
                    if (m2d && s_mc_swd && !e->always_hash && e->span >= min_swd_span &&
                        (u32)(g.frames - e->create_frame) >= 4u && vm_base && src >= vm_base) {
                        const size_t ea = (size_t)(src - vm_base);
                        swd_a = (u32)(ea >> 12);
                        swd_b = (u32)((ea + e->span - 1u) >> 12);
                        if (swd_b < (1u << 20)) {
                            swd_eligible = 1;
                            const u32 vgen = e->swd_ver_frame;
                            const u32 ca = swd_a >> 8, cb = swd_b >> 8;
                            int coarse_dirty = 0;
                            for (u32 c = ca; c <= cb; c++) {
                                const u32 cg = __atomic_load_n(&s_swd_coarse[c], __ATOMIC_RELAXED);
                                if (cg && cg + 1u >= vgen) { coarse_dirty = 1; break; }
                            }
                            for (u32 pg = swd_a; pg <= swd_b; pg++) {
                                if (s_swd_nonspu[pg]) { swd_nonspu = 1; break; }
                                if (coarse_dirty) {
                                    const u32 sg = __atomic_load_n(&s_swd_gen[pg], __ATOMIC_RELAXED);
                                    if (sg && sg + 1u >= vgen) { swd_dirty = 1; break; }
                                }
                            }
                            if (!swd_nonspu && !swd_dirty) {
                                const u32 tver = s_ef.m2f ? 256u : 64u;
                                const int pver = (u32)(g.frames - e->full_hash_frame) >= tver ||
                                                 (g.frames & (tver - 1u)) == ((u32)key & (tver - 1u));
                                if (!pver) swd_skip = 1;
                            }
                        }
                    }
                    e->last_hash_frame = g.frames;
                    if (!swd_skip) {
                        u32 sbytes = 0;
                        g.st_hashes++;
                        /* The verification is spread across frames by the
                         * entry's own key: every cached texture is created in the
                         * same couple of frames, so a plain "every 8th frame"
                         * counter piles the whole 69 MB into one frame of eight.
                         * The age test stays as the rule's floor. */
                        int verify = (u32)(g.frames - e->full_hash_frame) >=
                                         ENG_TEX_VERIFY_FRAMES ||
                                     (g.frames & (ENG_TEX_VERIFY_FRAMES - 1u)) ==
                                         ((u32)key & (ENG_TEX_VERIFY_FRAMES - 1u)) ||
                                     !eng_texture_sparse_use();
                        if (!verify) {
                            const double t0 = stats ? eng_now_ms() : 0.0;
                            const u32 sample = eng_texture_sample_hash(src, e->span, &sbytes);
                            if (stats) { g.st_ms_hash += eng_now_ms() - t0;
                                         g.st_hash_bytes += sbytes; }
                            verify = sample != e->sample_hash;
                        }
                        if (verify) {
                            const double t0 = stats ? eng_now_ms() : 0.0;
                            const u64 hash = eng_hash_bytes(src, e->span,
                                                            1469598103934665603ull);
                            const u32 sample = eng_texture_sample_hash(src, e->span, &sbytes);
                            if (stats) { g.st_ms_hash += eng_now_ms() - t0;
                                         g.st_hash_bytes += e->span + sbytes;
                                         g.st_verifies++; }
                            e->full_hash_frame = g.frames;
                            e->sample_hash = sample;
                            if (hash != e->content_hash) {
                                if (swd_eligible && !swd_dirty && !swd_nonspu) {
                                    e->always_hash = 1;
                                    for (u32 pg = swd_a; pg <= swd_b; pg++) {
                                        if (!s_swd_nonspu[pg]) {
                                            s_swd_nonspu[pg] = 1;
                                            g.mc_nonspu_pages++;
                                        }
                                    }
                                }
                                const u32 fresh = eng_texture_upload(location, offset, fmt, w, h,
                                                                     levels, pitch, cube, remap);
                                if (fresh) {
                                    const u32 old_h = e->handle;
                                    g.be->texture_release(g.be->user, old_h);
                                    e->handle = fresh;
                                    e->content_hash = hash;
                                    if (eng_m2e4_use())
                                        eng_tex_invalidate_handle(old_h, eng_tex_fast_idx(offset, fmt, w, h, remap));
                                    else
                                        g.tex_gen++;
                                }
                            }
                            if (s_mc_swd) e->swd_ver_frame = cur_swd;
                        }
                    }
                    if (swd_eligible && !swd_nonspu && !e->always_hash && !swd_dirty) {
                        e->swd_clo = (u16)(swd_a >> 8);
                        e->swd_chi = (u16)(swd_b >> 8);
                        e->swd_ok = 1;
                    } else {
                        e->swd_ok = 0;
                    }
                }
            }
            e->last_use_serial = ++g.texture_use_serial;
            if (e->swd_ok && !e->always_hash) {
                s_last_tex_swd_ok = 1;
                s_last_tex_swd_ver = e->swd_ver_frame;
                s_last_tex_full_frame = e->full_hash_frame;
                s_last_tex_clo = e->swd_clo;
                s_last_tex_chi = e->swd_chi;
            }
            return e->handle;
        }
        at = (at + 1u) % ENG_TEXTURE_BUCKETS;
    }

    const u32 span = eng_texture_span(fmt, w, h, levels, pitch, cube);
    if (!span) return 0;

    u32 index;
    u32 evicted = 0;
    u32 ev_tf_idx = 0;
    if (g.n_textures < ENG_MAX_TEXTURES) {
        index = g.n_textures++;
    } else {
        index = 0;
        for (u32 i = 1; i < g.n_textures; i++)
            if (g.textures[i].last_use_serial < g.textures[index].last_use_serial)
                index = i;
        evicted = g.textures[index].handle;
        ev_tf_idx = eng_tex_fast_idx(g.textures[index].offset, g.textures[index].format,
                                     g.textures[index].width, g.textures[index].height,
                                     g.textures[index].remap);
        eng_texture_table_remove(&g.textures[index]);
    }

    eng_texture e;
    memset(&e, 0, sizeof(e));
    e.location = location; e.offset = offset; e.format = fmt;
    e.width = w; e.height = h; e.pitch = pitch; e.levels = levels;
    e.remap = remap; e.cubemap = (u32)(cube != 0);
    e.span = span;
    e.last_hash_frame = e.full_hash_frame = e.create_frame = g.frames;
    e.swd_ver_frame = s_mc_swd ? __atomic_load_n(&s_swd_frame, __ATOMIC_ACQUIRE) : 0u;
    e.always_hash = 0;
    e.last_use_serial = ++g.texture_use_serial;
    /* A new entry is uploaded below, so both hashes describe the bytes it was
     * decoded from; the sparse one is what later frames compare against. */
    { const u8* src = eng_guest_ptr(NULL, location, offset, span);
      u32 sbytes = 0;
      e.content_hash = src ? eng_hash_bytes(src, span, 1469598103934665603ull) : 0;
      e.sample_hash = src ? eng_texture_sample_hash(src, span, &sbytes) : 0; }
    e.handle = eng_texture_upload(location, offset, fmt, w, h, levels, pitch,
                                  cube, remap);
    if (e.handle && evicted) {
        g.be->texture_release(g.be->user, evicted);
        if (eng_m2e4_use()) eng_tex_invalidate_handle(evicted, ev_tf_idx);
        else g.tex_gen++;
    }
    if (e.handle || !evicted) {
        g.textures[index] = e;
        eng_texture_table_insert(index);
    }
    return e.handle;
}

/* SET_TEXTURE_FILTER's min field is [18:16] (1 NEAREST, 2 LINEAR, then 3..6,
 * the four combinations of a nearest/linear minification with a
 * nearest/linear mip filter) and mag [26:24]; SET_TEXTURE_CONTROL0 carries
 * max LOD at [18:7] and min LOD at [30:19], both 4.8 fixed point. RSX's LOD
 * bias, SET_TEXTURE_FILTER [12:0], is not applied: the reference engine
 * leaves it at zero too. */
static void eng_decode_sampler(u32 filter, u32 wrap, u32 control0,
                               u32 border_color, rsx_be_sampler_desc* out)
{
    const u32 minf = (filter >> 16) & 7u;
    const u32 magf = (filter >> 24) & 7u;
    out->min_linear  = (u8)(minf == 2 || minf == 4 || minf == 6);
    out->mag_linear  = (u8)(magf == 2);
    out->mip_present = (u8)(minf >= 3);
    out->mip_linear  = (u8)(minf == 5 || minf == 6);
    out->wrap_s = (u8)(wrap & 0xFu);
    out->wrap_t = (u8)((wrap >> 8) & 0xFu);
    out->wrap_r = (u8)((wrap >> 16) & 0xFu);
    out->min_lod = (float)((control0 >> 19) & 0xFFFu) / 256.0f;
    out->max_lod = out->mip_present ? (float)((control0 >> 7) & 0xFFFu) / 256.0f
                                    : 0.0f;
    if (out->max_lod < out->min_lod) out->max_lod = out->min_lod;
    out->border_color = g.samp_border_on ? border_color : 0u;
}

/* The decoded render state, kept until a write to one of the registers the
 * decode reads (the dispatcher's pipe_gen). A draw usually follows several
 * others with the same state, and the decode is 30 register reads plus a
 * memset of the whole struct. */
static struct {
    u32 gen;
    u64 hash;              /* of the structural state, folded in below */
    rsx_be_render_state rs;
} s_render_state = { 0xFFFFFFFFu, 0, { 0 } };

static const rsx_be_render_state* eng_render_state(void)
{
    if (s_render_state.gen != g.rsx.pipe_gen) {
        rsx_draw_engine_decode_render_state(&g.rsx, &s_render_state.rs);
        /* CT_NONE (default 1): SET_SURFACE_COLOR_TARGET = 0 means depth/stencil
         * only. The host pass still binds target A, so mask colour writes off
         * to avoid overwriting target A with the depth pass's FP output. */
        if (g.ct_none_on && rsx_dsp_reg(&g.rsx, M_SURFACE_COLOR_TARGET) == 0u) {
            s_render_state.rs.color_mask = 0u;
            s_render_state.rs.color_mask_mrt = 0u;
        }
        /* Hashed once per state change, not once per draw: the structural
         * state rarely changes between two draws of a frame, and folding its
         * thirty-three fields per draw was measurable. */
        s_render_state.hash = rsx_draw_engine_hash_render_state(
            &s_render_state.rs, 1469598103934665603ull);
        s_render_state.gen = g.rsx.pipe_gen;
    }
    return &s_render_state.rs;
}

/* ---- pipelines ----------------------------------------------------------- */

/* One decode per draw of the sixteen fragment and four vertex texture units.
 * The cube mask, the bind itself and the vertex-unit bind all read the same
 * registers, and rsx_dsp_get_texture fills a ~90-byte struct from fifteen of
 * them, so it ran about three times per draw for the same answer. */
static rsx_dsp_texture s_units[RSX_DSP_NUM_TEXTURES];
static rsx_dsp_vertex_texture s_vunits[RSX_DSP_NUM_VERTEX_TEXTURES];
static u32 s_enabled_tex_mask;
static u32 s_cube_mask;
static u32 s_vtex_mask;
static u32 s_tex_xf_key;
static u32 s_tex_state_gen;
#define ENG_PROG_FAST_SLOTS 1024u
#define ENG_PIPE_FAST_SLOTS 1024u
/* ENG_TEX_FAST_SLOTS defined above eng_tex_fast_idx */

typedef struct {
    u32 vp_gen, vp_start, fp_reg, fp_ctrl, cube_mask, vtex_mask, frame, epoch;
    eng_program* pg;
    const u8* fp_uc;
    u32 fp_size;
    u32 page_a, page_b;
    u32 full_ver_frame, swd_ver_frame, swd_checked_frame;
    u16 coarse_lo, coarse_hi;
    u8  valid;
} eng_prog_fast_ent;

typedef struct {
    const eng_program* pg;
    u64 rs_hash;
    u64 sh_key;
    u32 epoch, rt_count, tex_xf_key, handle;
    rsx_be_format rt_fmt;
    u8  fixed;
    u8  valid;
} eng_pipe_fast_ent;

typedef struct {
    u32 location, offset, format, width, height, levels, pitch, remap;
    u32 frame, surface_gen, tex_gen, handle;
    u32 swd_ver_frame, full_hash_frame;
    u16 coarse_lo, coarse_hi;
    u8  swd_ok;
    u8  cubemap;
    u8  valid;
} eng_tex_fast_ent;

static eng_prog_fast_ent s_prog_fast[ENG_PROG_FAST_SLOTS];
static eng_pipe_fast_ent s_pipe_fast[ENG_PIPE_FAST_SLOTS];
static eng_tex_fast_ent  s_tex_fast[ENG_TEX_FAST_SLOTS];
static eng_prog_fast_ent* s_last_prog;
static eng_pipe_fast_ent* s_last_pipe;
static u32 s_prog_epoch = 1u;
static rsx_be_sampler_desc s_unit_samp[RSX_DSP_NUM_TEXTURES];
static u32 s_unit_samp_gen[RSX_DSP_NUM_TEXTURES];
static u32 s_unit_samp_raw[RSX_DSP_NUM_TEXTURES][4];
static u8  s_unit_samp_raw_ok[RSX_DSP_NUM_TEXTURES];

/* ENG_M2C=0 disables dirty-unit texture caching and fast program checks;
 * ENG_M2C_PHASE=<0|1> runs M2c only on that A/B parity. */
static int s_m2c_cfg = -1, s_m2c_phase = -1;
static int eng_m2c_use(void)
{
    if (s_m2c_cfg < 0) {
        const char* e = getenv("ENG_M2C");
        const char* p = getenv("ENG_M2C_PHASE");
        s_m2c_cfg = (e && *e == '0') ? 0 : 1;
        if (p && *p) s_m2c_phase = atoi(p) & 1;
    }
    if (!s_m2c_cfg) return 0;
    return s_m2c_phase < 0 || (ENG_AB_PHASE() & 1) == s_m2c_phase;
}

typedef struct {
    u32 handle;
    rsx_be_sampler_desc samp;
    u32 frame;
    u32 surface_gen;
    u32 tex_gen;
    u16 slot;
    u8  valid;
    u8  kind;       /* 1=live surface, 3=depth snapshot, 4=guest texture */
    u8  sh_nibble;
} eng_unit_cache;

static eng_unit_cache s_ucache[RSX_DSP_NUM_TEXTURES];
static int s_units_seeded;

static void eng_tex_invalidate_handle(u32 old_handle, u32 tf_idx)
{
    if (!old_handle) return;
    for (u32 w = 0; w < ENG_TEX_FAST_WAYS; w++) {
        if (s_tex_fast[tf_idx + w].valid && s_tex_fast[tf_idx + w].handle == old_handle)
            s_tex_fast[tf_idx + w].valid = 0;
    }
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) {
        if (s_ucache[u].valid && s_ucache[u].handle == old_handle) {
            s_ucache[u].valid = 0;
            g.cur_tex_resolve_dirty |= 1u << u;
        }
    }
    if (g.tex_bind_valid) {
        u32 m = g.last_tex_mask;
        while (m) {
            const u32 u = (u32)__builtin_ctz(m);
            m &= m - 1u;
            if (g.last_textures[u] == old_handle) { g.tex_bind_valid = 0; break; }
        }
    }
}

#define ENG_RT_VIEW_CACHE 64u
typedef struct {
    u32 surface_gen, slot, format, remap, handle;
    u8  valid;
} eng_rt_view_ent;
static eng_rt_view_ent s_rt_view_cache[ENG_RT_VIEW_CACHE];

static void eng_units_decode(void)
{
    const int fast = (s_ef.phase >= 0) ? (int)s_ef.m2c : eng_m2c_use();
    const int m2e4 = (s_ef.phase >= 0) ? (int)s_ef.m2e4 : eng_m2e4_use();
    const int m2e4b = (s_ef.phase >= 0) ? (int)s_ef.m2e4b : eng_m2e4b_use();
    const int m2f = (s_ef.phase >= 0) ? (int)s_ef.m2f : 0;
    u32 td = (fast && s_units_seeded) ? g.rsx.tex_dirty_mask : 0xFFFFu;
    u32 id = (fast && s_units_seeded && m2e4b) ? g.rsx.tex_img_dirty_mask : td;
    u32 sd = (fast && s_units_seeded && m2e4) ? g.rsx.tex_samp_dirty_mask : 0xFFFFu;
    u32 vd = (fast && s_units_seeded) ? g.rsx.vtex_dirty_mask : 0xFu;
    if (td || !s_units_seeded) {
        const u32* regs = g.rsx.regs;
        u32 m = td;
        while (m) {
            const u32 u = (u32)__builtin_ctz(m);
            m &= m - 1u;
            const u32 bit = 1u << u;
            rsx_dsp_texture* t = &s_units[u];
            const int img_chg = !s_units_seeded || ((id & bit) != 0u);
            if (m2e4b) {
                const u32 bidx = (0x1A00u + u * 0x20u) >> 2;
                const u32 ctrl0 = regs[bidx + 3u];
                const u32 wrap  = regs[bidx + 2u];
                const u32 filt  = regs[bidx + 5u];
                const u32 bcol  = regs[bidx + 7u];
                t->control0     = ctrl0;
                t->enabled      = ctrl0 >> 31;
                t->wrap         = wrap;
                t->filter       = filt;
                t->border_color = bcol;
                if (img_chg) {
                    const u32 fmt  = regs[bidx + 1u];
                    const u32 rect = regs[bidx + 6u];
                    const u32 sz1  = regs[(0x1840u >> 2) + u];
                    t->offset    = regs[bidx + 0u];
                    t->location  = (fmt & 3u) == 2u ? RSX_LOCATION_MAIN : RSX_LOCATION_LOCAL;
                    t->cubemap   = (fmt >> 2) & 1u;
                    t->dimension = (fmt >> 4) & 0xFu;
                    t->format    = (fmt >> 8) & 0xFFu;
                    t->mipmaps   = fmt >> 16;
                    t->remap     = regs[bidx + 4u];
                    t->width     = rect >> 16;
                    t->height    = rect & 0xFFFFu;
                    t->pitch     = sz1 & 0xFFFFFu;
                    t->depth     = sz1 >> 20;
                    s_ucache[u].valid = 0;
                    g.cur_tex_resolve_dirty |= bit;
                    if (t->enabled) s_enabled_tex_mask |= bit;
                    else s_enabled_tex_mask &= ~bit;
                    if (t->enabled && t->cubemap && t->width && t->width == t->height)
                        s_cube_mask |= bit;
                    else
                        s_cube_mask &= ~bit;
                    u32 mode = 0u;
                    if (g.tex_xf_on && t->enabled) {
                        const u32 base_fmt = t->format & RSX_TEX_FMT_BASE_MASK & ~(u32)RSX_TEX_FMT_UNNORM;
                        if (!(base_fmt >= 0x90u && base_fmt <= 0x93u)) {
                            const u32 ur = (wrap >> 12) & 0xFu;
                            const u32 ga = (wrap >> 20) & 0xFu;
                            mode = (ur == 1u ? 1u : 0u) | (ga ? 2u : 0u);
                        }
                    }
                    s_tex_xf_key = (s_tex_xf_key & ~(3u << (2u * u))) | (mode << (2u * u));
                }
                if (t->enabled && (((sd & bit) != 0u) || !s_unit_samp_gen[u])) {
                    const u32 rf = filt & 0x07070000u;
                    const u32 rw = wrap & 0x000F0F0Fu;
                    const u32 rc = ctrl0 & 0x7FFFFF80u;
                    const u32 rb = g.samp_border_on ? bcol : 0u;
                    if (!m2f || !s_unit_samp_raw_ok[u] || !s_unit_samp_gen[u] ||
                        s_unit_samp_raw[u][0] != rf || s_unit_samp_raw[u][1] != rw ||
                        s_unit_samp_raw[u][2] != rc || s_unit_samp_raw[u][3] != rb) {
                        rsx_be_sampler_desc ns;
                        eng_decode_sampler(filt, wrap, ctrl0, bcol, &ns);
                        s_unit_samp_raw[u][0] = rf; s_unit_samp_raw[u][1] = rw;
                        s_unit_samp_raw[u][2] = rc; s_unit_samp_raw[u][3] = rb;
                        s_unit_samp_raw_ok[u] = 1;
                        if (!s_unit_samp_gen[u] || memcmp(&s_unit_samp[u], &ns, sizeof ns) != 0) {
                            s_unit_samp[u] = ns;
                            s_unit_samp_gen[u]++;
                            if (!s_unit_samp_gen[u]) s_unit_samp_gen[u] = 1u;
                        }
                    }
                }
            } else {
                rsx_dsp_get_texture(&g.rsx, u, t);
                s_ucache[u].valid = 0;
                g.cur_tex_resolve_dirty |= bit;
                if (t->enabled && (((sd >> u) & 1u) || !s_unit_samp_gen[u])) {
                    rsx_be_sampler_desc ns;
                    eng_decode_sampler(t->filter, t->wrap, t->control0, t->border_color, &ns);
                    if (!s_unit_samp_gen[u] || memcmp(&s_unit_samp[u], &ns, sizeof ns) != 0) {
                        s_unit_samp[u] = ns;
                        s_unit_samp_gen[u]++;
                        if (!s_unit_samp_gen[u]) s_unit_samp_gen[u] = 1u;
                    }
                }
                if (t->enabled) s_enabled_tex_mask |= bit;
                else s_enabled_tex_mask &= ~bit;
                if (t->enabled && t->cubemap && t->width && t->width == t->height)
                    s_cube_mask |= bit;
                else
                    s_cube_mask &= ~bit;
                u32 mode = 0u;
                if (g.tex_xf_on && t->enabled) {
                    const u32 base_fmt = t->format & RSX_TEX_FMT_BASE_MASK & ~(u32)RSX_TEX_FMT_UNNORM;
                    if (!(base_fmt >= 0x90u && base_fmt <= 0x93u)) {
                        const u32 ur = (t->wrap >> 12) & 0xFu;
                        const u32 ga = (t->wrap >> 20) & 0xFu;
                        mode = (ur == 1u ? 1u : 0u) | (ga ? 2u : 0u);
                    }
                }
                s_tex_xf_key = (s_tex_xf_key & ~(3u << (2u * u))) | (mode << (2u * u));
            }
        }
        g.rsx.tex_dirty_mask = 0;
        g.rsx.tex_img_dirty_mask = 0;
        g.rsx.tex_samp_dirty_mask = 0;
        s_tex_state_gen++;
    }
    if (vd || !s_units_seeded) {
        u32 m = vd;
        while (m) {
            const u32 u = (u32)__builtin_ctz(m);
            m &= m - 1u;
            rsx_dsp_get_vertex_texture(&g.rsx, u, &s_vunits[u]);
            const rsx_dsp_vertex_texture* vt = &s_vunits[u];
            if (vt->enabled && vt->width && vt->height) s_vtex_mask |= 1u << u;
            else s_vtex_mask &= ~(1u << u);
        }
        g.rsx.vtex_dirty_mask = 0;
    }
    s_units_seeded = 1;
}

static u32 eng_vtex_mask(void)
{
    if (eng_m2d_use()) return s_vtex_mask;
    u32 mask = 0;
    for (u32 u = 0; u < RSX_DSP_NUM_VERTEX_TEXTURES; u++) {
        const rsx_dsp_vertex_texture* vt = &s_vunits[u];
        if (vt->enabled && vt->width && vt->height) mask |= 1u << u;
    }
    return mask;
}

static u32 eng_cube_mask(void)
{
    if (eng_m2d_use()) return s_cube_mask;
    u32 mask = 0;
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) {
        const rsx_dsp_texture* t = &s_units[u];
        /* A cube face is square by construction, so a non-square image
         * cannot be one; the fragment program is compiled against this mask
         * and a texturecube slot filled with a 2D texture is a validation
         * failure, not a wrong pixel. */
        if (t->enabled && t->cubemap && t->width && t->width == t->height)
            mask |= 1u << u;
    }
    return mask;
}

/* Pointers and sizes of the program pair the registers name. `known` is a
 * memo hit, whose sizes let both end-marker scans be skipped; without it both
 * programs are scanned for their end marker. */
static int eng_program_fetch(const eng_program* known, const u8** out_vp,
                             u32* out_vp_instrs, const u8** out_fp, u32* out_fp_size)
{
    const u8* vp_uc = NULL;
    u32 vp_instrs = 0;
    const u32 start = rsx_dsp_vp_start(&g.rsx);
    if (start < RSX_DSP_VP_INSTR) {
        vp_uc = (const u8*)(g.rsx.vp + start * 4);
        vp_instrs = known ? known->vp_instrs
                          : rsx_vp_program_size_instrs(
                                vp_uc, (RSX_DSP_VP_INSTR - start) * 16u);
    }

    /* A zero SET_SHADER_PROGRAM is "none", not "the program at offset 0": the
     * register carries the location in its low two bits, so a real program can
     * never read back as zero. */
    const u8* fp_uc = NULL;
    u32 fp_size = 0;
    if (rsx_dsp_reg(&g.rsx, 0x08E4)) {
        u32 fp_loc = 0;
        const u32 fp_off = rsx_dsp_fragment_program(&g.rsx, &fp_loc);
        fp_uc = eng_guest_ptr(NULL, fp_loc, fp_off,
                              known ? known->fp_size : 16);
        fp_size = known ? known->fp_size
                        : (fp_uc ? rsx_fp_program_size(fp_uc, ENG_FP_MAX_BYTES) : 0u);
        if (fp_size) fp_uc = eng_guest_ptr(NULL, fp_loc, fp_off, fp_size);
        if (!fp_uc) fp_size = 0;
    }

    if (out_vp) *out_vp = vp_uc;
    if (out_vp_instrs) *out_vp_instrs = vp_instrs;
    if (out_fp) *out_fp = fp_uc;
    if (out_fp_size) *out_fp_size = fp_size;
    return vp_instrs && fp_size;
}

/* The memoized identity of the program pair this draw runs, or NULL when the
 * fragment program's structural hash cannot be taken. Identity comes from the
 * registers, which is what an upload changes; the two content hashes validate
 * the memo, since a guest can rewrite fragment microcode in place without ever
 * touching SET_SHADER_PROGRAM, and the engine must not keep an old pipeline
 * for that. On a hit the per-draw cost is those two hashes. */
static eng_program* eng_program_get(const u8** out_fp_uc, u32* out_fp_size)
{
    const u32 vp_start = rsx_dsp_vp_start(&g.rsx);
    const u32 fp_ctrl  = rsx_dsp_shader_control(&g.rsx);
    const u32 fp_reg   = rsx_dsp_reg(&g.rsx, 0x08E4);
    const u32 cube_mask = eng_cube_mask();
    const u32 vtex_mask = eng_vtex_mask();
    const u32 vp_gen = g.rsx.vp_gen;
    const int m2d = eng_m2d_use();
    const int m2e4b = eng_m2e4b_use();
    const u32 cur_swd = s_mc_swd ? __atomic_load_n(&s_swd_frame, __ATOMIC_RELAXED) : 0u;

    eng_prog_fast_ent* pfe = s_last_prog;
    u32 pf_idx = 0;
    int pf_match = (m2d && s_mc_swd && pfe && pfe->valid &&
                    pfe->epoch == s_prog_epoch &&
                    pfe->vp_gen == vp_gen && pfe->vp_start == vp_start &&
                    pfe->fp_reg == fp_reg && pfe->fp_ctrl == fp_ctrl &&
                    pfe->cube_mask == cube_mask && pfe->vtex_mask == vtex_mask &&
                    (m2e4b || pfe->frame == g.frames));
    if (pf_match) {
        pf_idx = (u32)(pfe - s_prog_fast);
    } else {
        pf_idx = ((vp_start * 2654435761u) ^ (fp_reg >> 2) ^ (fp_ctrl * 31u) ^
                  (cube_mask << 5) ^ (vtex_mask << 21)) & (ENG_PROG_FAST_SLOTS - 1u);
        pfe = &s_prog_fast[pf_idx];
        pf_match = (m2d && s_mc_swd && pfe->valid &&
                    pfe->epoch == s_prog_epoch &&
                    pfe->vp_gen == vp_gen && pfe->vp_start == vp_start &&
                    pfe->fp_reg == fp_reg && pfe->fp_ctrl == fp_ctrl &&
                    pfe->cube_mask == cube_mask && pfe->vtex_mask == vtex_mask &&
                    (m2e4b || pfe->frame == g.frames));
    }
    if (pf_match) {
        int ok = 1;
        if (m2e4b && pfe->frame == g.frames && pfe->swd_checked_frame == cur_swd) {
            /* Already SWD-checked in this frame/swd generation: zero atomic loads. */
        } else {
            if (pfe->frame != g.frames) {
                /* Periodic full re-verify every 64 (or 256 under ENG_M2F) frames, spread by slot index. */
                const u32 pver = s_ef.m2f ? 256u : 64u;
                if ((u32)(g.frames - pfe->full_ver_frame) >= pver ||
                    (g.frames & (pver - 1u)) == (pf_idx & (pver - 1u)))
                    ok = 0;
            }
            if (ok) {
                const u32 vgen = m2e4b ? pfe->swd_ver_frame : cur_swd;
                if (pfe->coarse_lo == pfe->coarse_hi) {
                    const u32 cg = __atomic_load_n(&s_swd_coarse[pfe->coarse_lo], __ATOMIC_RELAXED);
                    if (cg && cg + 1u >= vgen) {
                        for (u32 pg_i = pfe->page_a; pg_i <= pfe->page_b; pg_i++) {
                            const u32 sg = __atomic_load_n(&s_swd_gen[pg_i], __ATOMIC_RELAXED);
                            if (sg && sg + 1u >= vgen) { ok = 0; break; }
                        }
                    }
                } else {
                    for (u32 pg_i = pfe->page_a; pg_i <= pfe->page_b; pg_i++) {
                        const u32 sg = __atomic_load_n(&s_swd_gen[pg_i], __ATOMIC_RELAXED);
                        if (sg && sg + 1u >= vgen) { ok = 0; break; }
                    }
                }
            }
            if (ok) {
                pfe->frame = g.frames;
                pfe->swd_checked_frame = cur_swd;
            }
        }
        if (ok) {
            s_last_prog = pfe;
            if (out_fp_uc) *out_fp_uc = pfe->fp_uc;
            if (out_fp_size) *out_fp_size = pfe->fp_size;
            return pfe->pg;
        }
    }

    u64 id = ((u64)vp_gen << 32) | vp_start;
    id *= 0x9E3779B97F4A7C15ull;
    id ^= ((u64)fp_reg << 32) | fp_ctrl;
    id *= 0xBF58476D1CE4E5B9ull;
    id ^= ((u64)cube_mask << 16) | vtex_mask | ((u64)vp_start << 40);
    id ^= id >> 33;
    id *= 0xFF51AFD7ED558CCDull;
    id ^= id >> 33;
    if (!id) id = 1;

    eng_program* pg = &g.programs[id & (ENG_PROGRAM_SLOTS - 1u)];
    const int memo = pg->id == id &&
                     pg->vp_gen == vp_gen && pg->vp_start == vp_start &&
                     pg->fp_reg == fp_reg && pg->fp_ctrl == fp_ctrl &&
                     pg->cube_mask == cube_mask && pg->vtex_mask == vtex_mask;

    const u8* vp_uc = NULL;
    const u8* fp_uc = NULL;
    u32 vp_instrs = 0, fp_size = 0;
    const int fast_prog = eng_m2c_use();
    int resident = eng_program_fetch(memo ? pg : NULL, &vp_uc, &vp_instrs,
                                     &fp_uc, &fp_size);
    int fixed = !resident;
    /* vp_gen is part of the verified memo key, so on a non-fixed memo hit vp_uc is identical. */
    if (memo && fast_prog && !fixed && !pg->fixed) {
        int fp_ok = 0;
        u32 pa = 0, pb = 0;
        if (s_mc_swd && vm_base && fp_uc >= vm_base) {
            const size_t ea = (size_t)(fp_uc - vm_base);
            pa = (u32)(ea >> 12);
            pb = (u32)((ea + fp_size - 1u) >> 12);
            if (pg->fp_ver_frame == g.frames && pb < (1u << 20)) {
                fp_ok = 1;
                const u32 vgen = __atomic_load_n(&s_swd_frame, __ATOMIC_RELAXED);
                const u32 ca = pa >> 8, cb = pb >> 8;
                int cdirty = 0;
                for (u32 c = ca; c <= cb; c++) {
                    const u32 cg = __atomic_load_n(&s_swd_coarse[c], __ATOMIC_RELAXED);
                    if (cg && cg + 1u >= vgen) { cdirty = 1; break; }
                }
                if (cdirty) {
                    for (u32 pg_i = pa; pg_i <= pb; pg_i++) {
                        const u32 sg = __atomic_load_n(&s_swd_gen[pg_i], __ATOMIC_RELAXED);
                        if (sg && sg + 1u >= vgen) { fp_ok = 0; break; }
                    }
                }
            }
        }
        if (fp_ok) {
            pfe->valid = 1; pfe->epoch = s_prog_epoch; pfe->frame = g.frames;
            pfe->vp_gen = vp_gen; pfe->vp_start = vp_start;
            pfe->fp_reg = fp_reg; pfe->fp_ctrl = fp_ctrl;
            pfe->cube_mask = cube_mask; pfe->vtex_mask = vtex_mask;
            pfe->pg = pg; pfe->fp_uc = fp_uc; pfe->fp_size = fp_size;
            pfe->page_a = pa; pfe->page_b = pb;
            pfe->full_ver_frame = g.frames; pfe->swd_ver_frame = cur_swd; pfe->swd_checked_frame = cur_swd;
            pfe->coarse_lo = (u16)(pa >> 8); pfe->coarse_hi = (u16)(pb >> 8);
            s_last_prog = pfe;
            if (out_fp_uc) *out_fp_uc = fp_uc;
            if (out_fp_size) *out_fp_size = fp_size;
            return pg;
        }
        const u64 fp_hash = eng_hash_bytes(fp_uc, fp_size, 0x46505247ull);
        if (pg->fp_hash == fp_hash) {
            pg->fp_ver_frame = g.frames;
            if (s_mc_swd && vm_base && fp_uc >= vm_base && pb < (1u << 20)) {
                pfe->valid = 1; pfe->epoch = s_prog_epoch; pfe->frame = g.frames;
                pfe->vp_gen = vp_gen; pfe->vp_start = vp_start;
                pfe->fp_reg = fp_reg; pfe->fp_ctrl = fp_ctrl;
                pfe->cube_mask = cube_mask; pfe->vtex_mask = vtex_mask;
                pfe->pg = pg; pfe->fp_uc = fp_uc; pfe->fp_size = fp_size;
                pfe->page_a = pa; pfe->page_b = pb;
                pfe->full_ver_frame = g.frames; pfe->swd_ver_frame = cur_swd; pfe->swd_checked_frame = cur_swd;
                pfe->coarse_lo = (u16)(pa >> 8); pfe->coarse_hi = (u16)(pb >> 8);
                s_last_prog = pfe;
            }
            if (out_fp_uc) *out_fp_uc = fp_uc;
            if (out_fp_size) *out_fp_size = fp_size;
            return pg;
        }
    }
    u64 vp_hash = fixed ? 0 : eng_hash_bytes(vp_uc, vp_instrs * 16u, 0x56505247ull);
    u64 fp_hash = fixed ? 0 : eng_hash_bytes(fp_uc, fp_size, 0x46505247ull);

    if (memo && pg->vp_hash == vp_hash && pg->fp_hash == fp_hash) {
        pg->fp_ver_frame = g.frames;
        if (out_fp_uc) *out_fp_uc = fp_uc;
        if (out_fp_size) *out_fp_size = fp_size;
        return pg;
    }

    if (memo) {
        /* Re-scan microcode lengths from scratch after an in-place rewrite
         * rather than carrying forward the evicted entry's sizes. */
        resident = eng_program_fetch(NULL, &vp_uc, &vp_instrs, &fp_uc, &fp_size);
        fixed = !resident;
        vp_hash = fixed ? 0 : eng_hash_bytes(vp_uc, vp_instrs * 16u, 0x56505247ull);
        fp_hash = fixed ? 0 : eng_hash_bytes(fp_uc, fp_size, 0x46505247ull);
    }

    s_prog_epoch++;
    s_last_prog = NULL;
    s_last_pipe = NULL;
    if (!s_prog_epoch) {
        memset(s_prog_fast, 0, sizeof s_prog_fast);
        memset(s_pipe_fast, 0, sizeof s_pipe_fast);
        s_prog_epoch = 1u;
    }
    if (g.last_fp_prog == pg) g.last_fp_prog = NULL;
    if (g.last_vp_prog == pg) g.last_vp_prog = NULL;
    memset(pg, 0, sizeof *pg);
    pg->id = id;
    pg->vp_gen = vp_gen;
    pg->vp_start = vp_start;
    pg->fp_reg = fp_reg;
    pg->vp_hash = vp_hash;
    pg->fp_hash = fp_hash;
    pg->fp_ver_frame = g.frames;
    pg->fixed = (u8)fixed;
    pg->fp_ctrl = fp_ctrl;
    pg->cube_mask = cube_mask;
    pg->vtex_mask = vtex_mask;
    if (fixed) {
        rsx_vertex_layout_plan_init(&pg->layout, 0xFFFFu);
        if (out_fp_uc) *out_fp_uc = NULL;
        if (out_fp_size) *out_fp_size = 0;
        return pg;
    }
    pg->vp_instrs = vp_instrs;
    pg->fp_size = fp_size;
    pg->vp_const_packed = (u8)(eng_vp_pack_use() ? 1 : 0);
    pg->vp_const_sparse = 0;
    pg->vp_const_count = 0;
    pg->vp_const_blk_mask = 0;
    pg->vp_cuse_valid = 0;
    if (eng_vp_sparse_use() || g.upscale_hooks) {
        rsx_vp_const_map cmap;
        if (rsx_vp_analyze_consts(vp_uc, vp_instrs * 16u, &cmap)) {
            memset(pg->vp_cuse, 0, sizeof pg->vp_cuse);
            for (u32 i = 0; i < cmap.count; i++) {
                const u32 slot_i = cmap.slots[i];
                if (slot_i < 512u) pg->vp_cuse[slot_i >> 5] |= 1u << (slot_i & 31u);
            }
            pg->vp_cuse_valid = 1;
            if (eng_vp_sparse_use() && cmap.count <= ENG_VP_SPARSE_MAX) {
                pg->vp_const_sparse = 1;
                pg->vp_const_count = cmap.count;
                u8 bm = 0;
                for (u32 i = 0; i < cmap.count; i++) {
                    pg->vp_const_list[i] = cmap.slots[i];
                    bm |= (u8)(1u << (cmap.slots[i] >> 6));
                }
                pg->vp_const_blk_mask = bm;
                pg->vp_const_slots = cmap.count ? cmap.count : 1u;
            }
        } else if (g.upscale_hooks) {
            /* Indexed-constant VPs fail sparse remap, but still read static
             * camera/object constants (c256..c266). Scan non-indexed reads. */
            memset(pg->vp_cuse, 0, sizeof pg->vp_cuse);
            for (u32 off = 0; off + 16u <= vp_instrs * 16u; off += 16u) {
                u32 d1, d2, d3;
                memcpy(&d1, vp_uc + off + 4u, 4u);
                memcpy(&d2, vp_uc + off + 8u, 4u);
                memcpy(&d3, vp_uc + off + 12u, 4u);
                const u32 src0 = ((d2 >> 23) & 0x1FFu) | ((d1 & 0xFFu) << 9);
                const u32 src1 = (d2 >> 6) & 0x1FFFFu;
                const u32 src2 = ((d3 >> 21) & 0x7FFu) | ((d2 & 0x3Fu) << 11);
                if (((src0 & 3u) == 3u || (src1 & 3u) == 3u || (src2 & 3u) == 3u) &&
                    !((d3 >> 1) & 1u)) {
                    const u32 ci = ((d1 >> 12) & 0x3FFu) & 511u;
                    pg->vp_cuse[ci >> 5] |= 1u << (ci & 31u);
                }
                if (d3 & 1u) break;
            }
            pg->vp_cuse_valid = 1;
        }
    }
    if (!pg->vp_const_sparse) {
        pg->vp_const_slots = pg->vp_const_packed
            ? (u16)rsx_vp_analyze_const_slots(vp_uc, vp_instrs * 16u)
            : (u16)RSX_DSP_NUM_CONSTANTS;
    }
    pg->fp_struct_hash = rsx_fp_structural_hash(fp_uc, fp_size,
                                               1469598103934665603ull);
    if (!pg->fp_struct_hash) { pg->id = 0; return NULL; }
    /* The layout the active vertex program actually reads. An uncertain
     * analysis falls back to all sixteen registers
     * (rsx_live_draw.c:3625-3643). */
    u32 mask = 0xFFFFu;
    rsx_vp_input_analysis analysis = { 0xFFFFu, 0 };
    if (rsx_vp_analyze_inputs(vp_uc, vp_instrs * 16u, &analysis) == (int)vp_instrs &&
        analysis.exact && analysis.input_mask)
        mask = analysis.input_mask;
    rsx_vertex_layout_plan_init(&pg->layout, mask);
    if (out_fp_uc) *out_fp_uc = fp_uc;
    if (out_fp_size) *out_fp_size = fp_size;
    return pg;
}

/* What a draw runs before a title has loaded programs of its own.
 *
 * The reference engine has no fallback and drops such a draw, because a title
 * always has both programs resident by the time it draws anything. A general
 * toolkit does not get to assume that: the host harness's fixed-function modes
 * are exactly this case, and so is the first frame of a title that clears and
 * flips before its first SET_TRANSFORM_PROGRAM. It is written in the same HLSL
 * shape the decompilers emit -- ATTRn inputs, the COLOR0/TEXCOORD varyings, the
 * VPConst block and the viewport epilogue -- so it goes through the one
 * pipeline_create the backend already implements rather than needing a second
 * entry point for a built-in shader.
 *
 * Vertex constant rows 0..3 are the transform when the guest programmed them
 * and identity when it did not: an all-zero matrix would collapse every vertex
 * onto the origin. */
static const char* const kEngFixedVS =
"struct VSInput {\n"
"    float4 a0:ATTR0;  float4 a1:ATTR1;  float4 a2:ATTR2;  float4 a3:ATTR3;\n"
"    float4 a4:ATTR4;  float4 a5:ATTR5;  float4 a6:ATTR6;  float4 a7:ATTR7;\n"
"    float4 a8:ATTR8;  float4 a9:ATTR9;  float4 a10:ATTR10; float4 a11:ATTR11;\n"
"    float4 a12:ATTR12; float4 a13:ATTR13; float4 a14:ATTR14; float4 a15:ATTR15;\n"
"};\n"
"struct VSOutput {\n"
"    float4 pos:SV_Position; float4 col0:COLOR0; float4 col1:COLOR1;\n"
"    float4 fog:FOG;\n"
"    float4 t0:TEXCOORD0; float4 t1:TEXCOORD1; float4 t2:TEXCOORD2; float4 t3:TEXCOORD3;\n"
"    float4 t4:TEXCOORD4; float4 t5:TEXCOORD5; float4 t6:TEXCOORD6; float4 t7:TEXCOORD7;\n"
"};\n"
"cbuffer VPConst : register(b0) {\n"
"    float4 vp_c[512];\n"
"    float4 vp_posscale;\n"
"    float4 vp_posoffset;\n"
"};\n"
"VSOutput main(VSInput input) {\n"
"    float4 _p = float4(dot(vp_c[0], input.a0), dot(vp_c[1], input.a0),\n"
"                       dot(vp_c[2], input.a0), dot(vp_c[3], input.a0));\n"
"    float _m = dot(abs(vp_c[0]), 1.0) + dot(abs(vp_c[1]), 1.0)\n"
"             + dot(abs(vp_c[2]), 1.0) + dot(abs(vp_c[3]), 1.0);\n"
"    if (_m == 0.0) _p = input.a0;\n"
"    VSOutput Out;\n"
"    Out.pos = float4(_p.xyz * vp_posscale.xyz + _p.w * vp_posoffset.xyz, _p.w);\n"
"    Out.col0 = input.a3; Out.col1 = float4(0,0,0,1); Out.fog = (float4)0;\n"
"    Out.t0 = input.a8; Out.t1 = (float4)0; Out.t2 = (float4)0; Out.t3 = (float4)0;\n"
"    Out.t4 = (float4)0; Out.t5 = (float4)0; Out.t6 = (float4)0; Out.t7 = (float4)0;\n"
"    return Out;\n"
"}\n";

static const char* const kEngFixedPS =
"struct PSInput {\n"
"    float4 position : SV_POSITION; float4 col0 : COLOR0; float4 col1 : COLOR1;\n"
"    float4 fog : FOG;\n"
"    float4 tc0:TEXCOORD0; float4 tc1:TEXCOORD1; float4 tc2:TEXCOORD2; float4 tc3:TEXCOORD3;\n"
"    float4 tc4:TEXCOORD4; float4 tc5:TEXCOORD5; float4 tc6:TEXCOORD6; float4 tc7:TEXCOORD7;\n"
"};\n"
"float4 main(PSInput input) : SV_TARGET { return input.col0; }\n";

static int eng_trace(void);
static u32 eng_trace_after(void);
static int eng_stats_on(void);
static double eng_now_ms(void);

/* Phase clock for [ENGTIME]. The FIFO walker is one thread, so one static is
 * enough; a phase accumulates from the previous boundary and resets it. */
static double s_phase_ms;

static void eng_phase_mark(double* acc)
{
    if (!acc) return;
    const double now = eng_now_ms();
    *acc += now - s_phase_ms;
    s_phase_ms = now;
}

/* The pipeline for this draw, built once per distinct key. A negative result
 * is cached too (handle 0), so a program pair that will not translate is not
 * retried on every draw of every frame.
 *
 * Both halves of the lookup are memoized: the program identity (once per
 * program, see eng_program_get) and the pipeline itself, in a hash index
 * instead of a scan of up to 8192 keys per draw. */
/* RSX TEXTURE_ADDRESS value transforms per unit 0..15 (2 bits each):
 * bit 0 = unsigned_remap BIASED (1 -> c.rgb * 2.0 - 1.0, bx2 G-buffer normals),
 * bit 1 = gamma != 0 (sRGB -> linear on c.rgb). Depth units (0x90..0x93) are
 * excluded. Mirrors rsx_d3d12_backend.c v89 TEX_XF. */
static u32 eng_tex_xf_key(void)
{
    if (!g.tex_xf_on) return 0;
    if (eng_m2d_use()) return s_tex_xf_key;
    u32 key = 0;
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) {
        const rsx_dsp_texture* t = &s_units[u];
        if (!t->enabled) continue;
        const u32 base_fmt = t->format & RSX_TEX_FMT_BASE_MASK & ~(u32)RSX_TEX_FMT_UNNORM;
        if (base_fmt >= 0x90u && base_fmt <= 0x93u) continue;
        const u32 ur = (t->wrap >> 12) & 0xFu;
        const u32 ga = (t->wrap >> 20) & 0xFu;
        const u32 mode = (ur == 1u ? 1u : 0u) | (ga ? 2u : 0u);
        key |= mode << (2u * u);
    }
    return key;
}

static const char kEngTxfFn[] =
    "float4 rsx_txf(float4 c, int m) {\n"
    "    if ((m & 2) != 0) {\n"
    "        float3 lo = c.rgb / 12.92;\n"
    "        float3 hi = pow(max((c.rgb + 0.055) / 1.055, (float3)0.0), (float3)2.4);\n"
    "        c.r = (c.r <= 0.04045) ? lo.r : hi.r;\n"
    "        c.g = (c.g <= 0.04045) ? lo.g : hi.g;\n"
    "        c.b = (c.b <= 0.04045) ? lo.b : hi.b;\n"
    "    }\n"
    "    if ((m & 1) != 0) c.rgb = c.rgb * 2.0 - 1.0;\n"
    "    return c;\n"
    "}\n";

static int eng_fp_txf_rewrite(char* hlsl, size_t cap, u32 texk)
{
    int patched = 0;
    static const char* const meth[3] = { "Sample(", "SampleBias(", "SampleLevel(" };
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) {
        const u32 m = (texk >> (2u * u)) & 3u;
        if (!m) continue;
        for (int form = 0; form < 2; form++) {
            for (int mi = 0; mi < 3; mi++) {
                char nd[48];
                snprintf(nd, sizeof nd, form ? "rsx_tex%u.%s" : "rsx_tex[%u].%s", u, meth[mi]);
                const size_t ndlen = strlen(nd);
                char* from = hlsl;
                for (;;) {
                    char* at = strstr(from, nd);
                    if (!at) break;
                    char* q = at + ndlen - 1u; /* '(' */
                    int dep = 0;
                    for (; *q; q++) {
                        if (*q == '(') dep++;
                        else if (*q == ')') { if (--dep == 0) break; }
                    }
                    if (!*q) break;
                    char* end = q + 1;
                    char trail[16];
                    snprintf(trail, sizeof trail, ", %u)", m);
                    const size_t hl = 8u; /* "rsx_txf(" */
                    const size_t tl = strlen(trail);
                    const size_t used = strlen(hlsl);
                    if (used + hl + tl + 1u > cap) return -1;
                    memmove(end + tl, end, strlen(end) + 1u);
                    memcpy(end, trail, tl);
                    memmove(at + hl, at, strlen(at) + 1u);
                    memcpy(at, "rsx_txf(", hl);
                    from = end + hl + tl;
                    patched++;
                }
            }
        }
    }
    if (patched) {
        const size_t fl = sizeof(kEngTxfFn) - 1u;
        const size_t used = strlen(hlsl);
        if (used + fl + 1u > cap) return -1;
        memmove(hlsl + fl, hlsl, used + 1u);
        memcpy(hlsl, kEngTxfFn, fl);
    }
    return patched;
}

/* Mirrors rsx_d3d12_backend.c v170 SUN_NDL_CLAMP: clamp N.L >= 0 on sun
 * cascade FPs so back-facing surfaces with biased shadow compare do not
 * subtract sun light. */
static int eng_fp_wscale_rewrite(char* hlsl, size_t cap)
{
    static const char from[] = "input.position";
    static const char to[]   = "float4(input .position.xy * fp_alpha.yz, input .position.zw)";
    const size_t fl = sizeof(from) - 1u;
    const size_t tl = sizeof(to) - 1u;
    int n = 0;
    char* p = hlsl;
    while ((p = strstr(p, from)) != NULL) {
        const size_t used = strlen(hlsl);
        if (used - fl + tl + 1u > cap) return -1;
        memmove(p + tl, p + fl, strlen(p + fl) + 1u);
        memcpy(p, to, tl);
        p += tl;
        n++;
    }
    return n;
}

static int eng_fp_sun_ndl_clamp(char* hlsl, size_t cap)
{
    static const char from[] = "(((r[2]).xyzw) * ((h[0]).xyzw)); h[4].w = _v.w;";
    static const char to[]   = "(((r[2]).xyzw) * max((h[0]).xyzw, 0.0)); h[4].w = _v.w;";
    if (!strstr(hlsl, from) || !strstr(hlsl, "dot(((h[4]).xyzw).xyz, (-("))
        return 0;
    const size_t fl = sizeof(from) - 1u;
    const size_t tl = sizeof(to) - 1u;
    int n = 0;
    char* p = hlsl;
    while ((p = strstr(p, from)) != NULL) {
        const size_t used = strlen(hlsl);
        if (used - fl + tl + 1u > cap) return -1;
        memmove(p + tl, p + fl, strlen(p + fl) + 1u);
        memcpy(p, to, tl);
        p += tl;
        n++;
    }
    return n;
}

/* whompay/inFAMOUSreborn #89 (DEPTH_BLUR_SMOOTH / DEPTH_BLUR=2): FNV-1a code
 * hash matching fp_scan() in rsx_d3d12_backend.c. The distance blur shader
 * (code_h == 0x29F3C342u) upsamples two 320x180 blurred copies on t2/t3;
 * cubic B-spline (4 bilinear taps) eliminates high-res blockiness. Per-unit
 * wrappers access rsx_tex/rsx_samp directly (glslang SPIR-V 1.0 safe). */
static u32 eng_fp_code_hash(const u8* uc, u32 max_bytes)
{
    if (!uc || max_bytes < 16u) return 0u;
    u32 off = 0u, k = 0u, ch = 2166136261u;
    while (off + 16u <= max_bytes) {
        const u32 w0 = rsx_fp_read_word(uc + off + 0u);
        const u32 w1 = rsx_fp_read_word(uc + off + 4u);
        const u32 w2 = rsx_fp_read_word(uc + off + 8u);
        const u32 w3 = rsx_fp_read_word(uc + off + 12u);
        for (u32 i = 0; i < 16u; i++) { ch ^= uc[off + i]; ch *= 16777619u; }
        off += 16u;
        if ((w1 & 3u) == 2u || (w2 & 3u) == 2u || (w3 & 3u) == 2u) {
            if (off + 16u > max_bytes) break;
            off += 16u;
            k++;
        }
        if (w0 & 1u) {
            ch ^= k; ch *= 16777619u;
            return ch ? ch : 1u;
        }
    }
    return 0u;
}

static int eng_hlsl_replace_all(char* hlsl, size_t cap, const char* from, const char* to)
{
    const size_t fl = strlen(from);
    const size_t tl = strlen(to);
    if (!fl) return 0;
    int n = 0;
    char* p = hlsl;
    while ((p = strstr(p, from)) != NULL) {
        const size_t used = strlen(hlsl);
        if (used - fl + tl + 1u > cap) return -1;
        memmove(p + tl, p + fl, strlen(p + fl) + 1u);
        memcpy(p, to, tl);
        p += tl;
        n++;
    }
    return n;
}

static int eng_fp_depth_blur_rewrite(char* hlsl, size_t cap, const u8* fp_uc,
                                     u32 fp_size, u32 cube_mask)
{
    if (eng_fp_code_hash(fp_uc, fp_size) != 0x29F3C342u) return 0;
    static int mode = -1, smooth = -1;
    if (mode < 0) {
        const char* e = getenv("DEPTH_BLUR");
        mode = (e && *e) ? atoi(e) : 1;
        e = getenv("DEPTH_BLUR_SMOOTH");
        smooth = (e && *e == '0') ? 0 : 1;
    }
    if (mode != 2 && !smooth) return 0;
    const int split = (cube_mask != 0u);
    const char* t2 = split ? "rsx_tex2" : "rsx_tex[2]";
    const char* t3 = split ? "rsx_tex3" : "rsx_tex[3]";
    char helpers[4096];
    const int hw = snprintf(helpers, sizeof helpers,
        "float4 rsx_dof_k(float4 k) {\n"
        "    if (k.y <= 0.0 || k.w <= 0.0) return k;\n"
        "    float sA = max(-k.x / k.y, 1500.0), eA = max((1.0 - k.x) / k.y, 8000.0);\n"
        "    float sB = max(-k.z / k.w, 8000.0), eB = max((1.0 - k.z) / k.w, 15000.0);\n"
        "    float cA = 1.0 / max(eA - sA, 1.0), cB = 1.0 / max(eB - sB, 1.0);\n"
        "    return float4(-sA * cA, cA, -sB * cB, cB);\n"
        "}\n"
        "float4 rsx_dof_s(float4 k) { k.x = min(k.x, 0.5); return k; }\n"
        "float4 rsx_dof_up2(float2 uv) {\n"
        "    uint W, H; %s.GetDimensions(W, H); float2 ts = float2(float(W), float(H));\n"
        "    float2 p = uv * ts - 0.5, i = floor(p), f = p - i, f2 = f * f, f3 = f2 * f;\n"
        "    float2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0, w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;\n"
        "    float2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0, w3 = f3 / 6.0;\n"
        "    float2 g0 = w0 + w1, g1 = w2 + w3;\n"
        "    float2 c0 = (i - 0.5 + w1 / g0) / ts, c1 = (i + 1.5 + w3 / g1) / ts;\n"
        "    return g0.y * (g0.x * %s.Sample(rsx_samp[2], float2(c0.x, c0.y)) + g1.x * %s.Sample(rsx_samp[2], float2(c1.x, c0.y)))\n"
        "         + g1.y * (g0.x * %s.Sample(rsx_samp[2], float2(c0.x, c1.y)) + g1.x * %s.Sample(rsx_samp[2], float2(c1.x, c1.y)));\n"
        "}\n"
        "float4 rsx_dof_up3(float2 uv) {\n"
        "    uint W, H; %s.GetDimensions(W, H); float2 ts = float2(float(W), float(H));\n"
        "    float2 p = uv * ts - 0.5, i = floor(p), f = p - i, f2 = f * f, f3 = f2 * f;\n"
        "    float2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0, w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;\n"
        "    float2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0, w3 = f3 / 6.0;\n"
        "    float2 g0 = w0 + w1, g1 = w2 + w3;\n"
        "    float2 c0 = (i - 0.5 + w1 / g0) / ts, c1 = (i + 1.5 + w3 / g1) / ts;\n"
        "    return g0.y * (g0.x * %s.Sample(rsx_samp[3], float2(c0.x, c0.y)) + g1.x * %s.Sample(rsx_samp[3], float2(c1.x, c0.y)))\n"
        "         + g1.y * (g0.x * %s.Sample(rsx_samp[3], float2(c0.x, c1.y)) + g1.x * %s.Sample(rsx_samp[3], float2(c1.x, c1.y)));\n"
        "}\n",
        t2, t2, t2, t2, t2, t3, t3, t3, t3, t3);
    if (hw < 0 || (size_t)hw >= sizeof helpers) return -1;
    char* ins = strstr(hlsl, "\nstruct PSOutput {");
    char* ins_main = strstr(hlsl, "\nfloat4 main(");
    if (ins_main && (!ins || ins_main < ins)) ins = ins_main;
    if (!ins) return 0;
    ins++;
    const size_t hl = (size_t)hw;
    if (strlen(hlsl) + hl + 1u > cap) return -1;
    memmove(ins + hl, ins, strlen(ins) + 1u);
    memcpy(ins, helpers, hl);
    if (mode == 2) {
        if (eng_hlsl_replace_all(hlsl, cap, "(fp_constants[2])", "(rsx_dof_k(fp_constants[2]))") < 0) return -1;
        if (eng_hlsl_replace_all(hlsl, cap, "(fp_constants[3])", "(rsx_dof_k(fp_constants[3]))") < 0) return -1;
        if (eng_hlsl_replace_all(hlsl, cap, "(fp_constants[4])", "(rsx_dof_s(fp_constants[4]))") < 0) return -1;
        if (eng_hlsl_replace_all(hlsl, cap, "(fp_constants[5])", "(rsx_dof_s(fp_constants[5]))") < 0) return -1;
    }
    if (smooth) {
        char from2[64], from3[64];
        snprintf(from2, sizeof from2, "%s.Sample(rsx_samp[2], ", t2);
        snprintf(from3, sizeof from3, "%s.Sample(rsx_samp[3], ", t3);
        if (eng_hlsl_replace_all(hlsl, cap, from2, "rsx_dof_up2(") < 0) return -1;
        if (eng_hlsl_replace_all(hlsl, cap, from3, "rsx_dof_up3(") < 0) return -1;
    }
    static int logs = 0;
    if (logs < 4 && logs++ < 4)
        fprintf(stderr, "[DEPTHBLUR] fp code 29F3C342: %s%s\n",
                mode == 2 ? "powered-area range" : "game range",
                smooth ? ", B-spline upsample" : "");
    return 1;
}

/* Shadow-map depth compare rewrite (mirrors rsx_d3d12_backend.c v34/v148).
 * Emits rsx_shadow_eval plus one per-unit rsx_shadow_uN wrapper right before
 * main() so rsx_tex/rsx_samp globals are accessed directly without passing
 * opaque image/sampler array elements as function arguments (glslang SPIR-V
 * 1.0 safe). sh_key packs 4 bits per unit 0..15: bit 3 = active, bits 0..2 =
 * RSX TEXTURE_ADDRESS zfunc (0..7). */
static const char kEngShadowEvalFn[] =
    "float4 rsx_shadow_eval(float4 d, float2 ts, float3 p, int zf) {\n"
    "    float2 f = frac(p.xy * ts - 0.5);\n"
    "#if RSX_SHRP\n"
    "    float3 dx = ddx(p), dy = ddy(p);\n"
    "    float det = dx.x * dy.y - dx.y * dy.x;\n"
    "    float2 g = (abs(det) > 1e-24) ? float2(dx.z * dy.y - dx.y * dy.z, dx.x * dy.z - dy.x * dx.z) / det : float2(0.0, 0.0);\n"
    "    float2 o = f / ts;\n"
    "    float4 du = float4(-o.x, 1.0 / ts.x - o.x, 1.0 / ts.x - o.x, -o.x);\n"
    "    float4 dv = float4(1.0 / ts.y - o.y, 1.0 / ts.y - o.y, -o.y, -o.y);\n"
    "    float4 rp = g.x * du + g.y * dv;\n"
    "    if (any(abs(rp) > (float4)RSX_SHRPMAX) || any(isnan(rp))) rp = float4(0.0, 0.0, 0.0, 0.0);\n"
    "    float4 z = p.zzzz + rp - RSX_SHBIAS;\n"
    "#else\n"
    "    float zs = max(abs(ddx(p.z)), abs(ddy(p.z)));\n"
    "    float4 z = p.zzzz - (RSX_SHBIAS + RSX_SHSLOPE * min(zs, 0.01));\n"
    "#endif\n"
    "    float4 r = float4(0.0, 0.0, 0.0, 0.0);\n"
    "    if (zf == 1) r = (float4)(d < z); else if (zf == 2) r = (float4)(d == z);\n"
    "    else if (zf == 3) r = (float4)(d <= z); else if (zf == 4) r = (float4)(d > z);\n"
    "    else if (zf == 5) r = (float4)(d != z); else if (zf == 6) r = (float4)(d >= z);\n"
    "    else if (zf == 7) r = float4(1.0, 1.0, 1.0, 1.0);\n"
    "    float lit = lerp(lerp(r.w, r.z, f.x), lerp(r.x, r.y, f.x), f.y);\n"
    "    return lit.xxxx;\n"
    "}\n";

static int eng_fp_shadow_rewrite(char* hlsl, size_t cap, u64 sh_key)
{
    int patched = 0;
    u32 patched_mask = 0;
    u8 unit_split[RSX_DSP_NUM_TEXTURES];
    memset(unit_split, 0, sizeof unit_split);
    static const int sw[8] = { 0, 4, 2, 6, 1, 5, 3, 7 };

    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) {
        const u32 nib = (u32)((sh_key >> (u * 4u)) & 0xFu);
        if (!(nib & 8u)) continue;
        int zf = (int)(nib & 7u);
        if (g.shadow_flip) zf = sw[zf];
        char needles[2][48];
        snprintf(needles[0], sizeof needles[0], "rsx_tex[%u].Sample(rsx_samp[%u], ", u, u);
        snprintf(needles[1], sizeof needles[1], "rsx_tex%u.Sample(rsx_samp[%u], ", u, u);
        for (int ni = 0; ni < 2; ni++) {
            const size_t nlen = strlen(needles[ni]);
            char* from = hlsl;
            for (;;) {
                char* at = strstr(from, needles[ni]);
                if (!at) break;
                char* a = at + nlen;
                if (*a != '(') { from = a; continue; }
                char* q = a;
                int dep = 0;
                for (; *q; q++) {
                    if (*q == '(') dep++;
                    else if (*q == ')') { if (--dep == 0) break; }
                }
                if (!*q) break;
                char* g1e = q + 1;
                int proj = 0;
                char* tail = NULL;
                if (strncmp(g1e, ".xy)", 4) == 0) {
                    tail = g1e + 4;
                } else if (strncmp(g1e, ".xy / (", 7) == 0) {
                    char* r = g1e + 6;
                    dep = 0;
                    for (; *r; r++) {
                        if (*r == '(') dep++;
                        else if (*r == ')') { if (--dep == 0) break; }
                    }
                    if (!*r || strncmp(r + 1, ".w)", 3) != 0) { from = g1e; continue; }
                    proj = 1;
                    tail = r + 4;
                } else {
                    from = g1e;
                    continue;
                }
                char head[40], trail[24], rbuf[1024];
                snprintf(head, sizeof head, "rsx_shadow_u%u(", u);
                snprintf(trail, sizeof trail, ", %d, %d)", proj, zf);
                const size_t hl = strlen(head), gl = (size_t)(g1e - a), tl = strlen(trail);
                const size_t newlen = hl + gl + tl, oldlen = (size_t)(tail - at);
                const size_t used = strlen(hlsl);
                if (newlen >= sizeof rbuf) { from = tail; continue; }
                if (used - oldlen + newlen + 1u > cap) return -1;
                memcpy(rbuf, head, hl);
                memcpy(rbuf + hl, a, gl);
                memcpy(rbuf + hl + gl, trail, tl);
                rbuf[newlen] = '\0';
                memmove(at + newlen, tail, strlen(tail) + 1u);
                memcpy(at, rbuf, newlen);
                from = at + newlen;
                patched++;
                patched_mask |= 1u << u;
                unit_split[u] = (u8)ni;
            }
        }
    }
    if (!patched) return 0;

    char hdr[8192];
    size_t pos = 0;
    if (g.shadow_rp_on) {
        pos += (size_t)snprintf(hdr + pos, sizeof hdr - pos,
            "#define RSX_SHRP 1\n#define RSX_SHBIAS %.7f\n#define RSX_SHRPMAX %.7f\n",
            g.shadow_bias, g.shadow_rp_max);
    } else {
        pos += (size_t)snprintf(hdr + pos, sizeof hdr - pos,
            "#define RSX_SHRP 0\n#define RSX_SHBIAS %.7f\n#define RSX_SHSLOPE %.4f\n",
            g.shadow_bias, g.shadow_slope);
    }
    const size_t el = sizeof(kEngShadowEvalFn) - 1u;
    if (pos + el >= sizeof hdr) return -1;
    memcpy(hdr + pos, kEngShadowEvalFn, el);
    pos += el;
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) {
        if (!((patched_mask >> u) & 1u)) continue;
        char tname[24];
        snprintf(tname, sizeof tname, unit_split[u] ? "rsx_tex%u" : "rsx_tex[%u]", u);
        const int w = snprintf(hdr + pos, sizeof hdr - pos,
            "float4 rsx_shadow_u%u(float4 c, int proj, int zf) {\n"
            "    float3 p = (proj != 0) ? (c.xyz / c.w) : c.xyz;\n"
            "    uint W, H; %s.GetDimensions(W, H);\n"
            "    float4 d = %s.GatherRed(rsx_samp[%u], p.xy);\n"
            "    return rsx_shadow_eval(d, float2(float(W), float(H)), p, zf);\n"
            "}\n",
            u, tname, tname, u);
        if (w < 0 || pos + (size_t)w >= sizeof hdr) return -1;
        pos += (size_t)w;
    }
    /* Insert right before main() (or struct PSOutput) so rsx_tex/rsx_samp are
     * already declared in scope. */
    char* ins = strstr(hlsl, "\nstruct PSOutput {");
    char* ins_main = strstr(hlsl, "\nfloat4 main(");
    if (ins_main && (!ins || ins_main < ins)) ins = ins_main;
    if (ins) ins++;   /* place at start of the line */
    else ins = hlsl;
    const size_t used = strlen(hlsl);
    if (used + pos + 1u > cap) return -1;
    memmove(ins + pos, ins, strlen(ins) + 1u);
    memcpy(ins, hdr, pos);
    return patched;
}

static u32 eng_pipeline_get(eng_program* pg, const u8* fp_uc, u32 fp_size,
                            const rsx_be_render_state* rs, rsx_be_format rt_fmt,
                            u32 rt_count, u32 tex_xf_key, u64 sh_key,
                            int* out_fixed)
{
    *out_fixed = 1;
    if (s_ef.phase < 0) eng_refresh_flags(ENG_AB_PHASE());
    const int stats = s_ef.stats;
    if (!pg) return 0;
    const u32 pg_slot = (u32)(pg - g.programs);
    const u64 rsh = s_render_state.hash;
    const int m2d = s_ef.m2d;
    eng_pipe_fast_ent* pfe = s_last_pipe;
    if (m2d && pfe && pfe->valid && pfe->epoch == s_prog_epoch &&
        pfe->pg == pg && pfe->rs_hash == rsh &&
        pfe->rt_fmt == rt_fmt && pfe->rt_count == rt_count &&
        pfe->tex_xf_key == tex_xf_key && pfe->sh_key == sh_key) {
        if (stats) g.st_plookups++;
        *out_fixed = (int)pfe->fixed;
        return pfe->handle;
    }
    const u32 pf_idx = ((pg_slot * 2654435761u) ^ (u32)(rsh ^ (rsh >> 32)) ^
                        ((u32)rt_fmt << 7) ^ (rt_count << 12) ^ (tex_xf_key * 131u) ^
                        (u32)(sh_key ^ (sh_key >> 32))) & (ENG_PIPE_FAST_SLOTS - 1u);
    pfe = &s_pipe_fast[pf_idx];
    if (m2d && pfe->valid && pfe->epoch == s_prog_epoch &&
        pfe->pg == pg && pfe->rs_hash == rsh &&
        pfe->rt_fmt == rt_fmt && pfe->rt_count == rt_count &&
        pfe->tex_xf_key == tex_xf_key && pfe->sh_key == sh_key) {
        if (stats) g.st_plookups++;
        s_last_pipe = pfe;
        *out_fixed = (int)pfe->fixed;
        return pfe->handle;
    }
    const rsx_vertex_layout_plan* layout = &pg->layout;
    /* No resident program pair means the built-in one; see kEngFixedVS. */
    const int fixed = pg->fixed;

    /* Identity: the memoized program identity (which covers the vertex
     * program's own bytes, the fragment program's STRUCTURE -- its inline
     * constants are hoisted into the buffered block, so a constant change must
     * not be a new pipeline --, the export-width bit of SHADER_CONTROL, the
     * cube and vertex-texture masks), the input layout, the target format and
     * count the pass has, and the structural render state. */
    u64 key = 0xCBF29CE484222325ull;
    if (fixed) {
        key ^= 0x4E464958ull;
        key *= 0x9E3779B97F4A7C15ull;
    } else {
        key ^= pg->vp_hash;
        key *= 0x9E3779B97F4A7C15ull;
        key ^= pg->fp_struct_hash;
        key *= 0xBF58476D1CE4E5B9ull;
        const u32 fp_ctrl_key = pg->fp_ctrl & 0x4Eu;
        const u32 vpk = pg->vp_const_sparse ? (2u | ((u32)pg->vp_const_count << 2))
                      : (pg->vp_const_packed ? (1u | ((u32)pg->vp_const_slots << 1)) : 0u);
        key ^= ((u64)fp_ctrl_key << 48) | ((u64)pg->cube_mask << 16) |
               ((u64)pg->vtex_mask << 32) | (u64)vpk;
        key *= 0x94D049BB133111EBull;
        key ^= ((u64)tex_xf_key << 32) ^ sh_key;
        key *= 0x9E3779B97F4A7C15ull;
    }
    key ^= ((u64)layout->mask << 32) | ((u64)layout->stride << 16) |
           ((u64)(u32)rt_fmt << 4) | (u64)rt_count;
    key *= 0xBF58476D1CE4E5B9ull;
    key ^= s_render_state.hash;
    key ^= key >> 33;
    key *= 0xFF51AFD7ED558CCDull;
    key ^= key >> 33;

    if (stats) g.st_plookups++;
    {   u32 at = (u32)(key & (ENG_PIPELINE_BUCKETS - 1u));
        for (u32 walked = 0; walked < ENG_PIPELINE_BUCKETS; walked++) {
            const u32 slot = g.pipeline_bucket[at];
            if (!slot) break;
            if (stats) g.st_pprobes++;
            if (g.pipelines[slot - 1u].key == key) {
                *out_fixed = g.pipelines[slot - 1u].fixed;
                pfe->valid = 1;
                pfe->epoch = s_prog_epoch;
                pfe->pg = pg;
                pfe->rs_hash = rsh;
                pfe->rt_fmt = rt_fmt;
                pfe->rt_count = rt_count;
                pfe->tex_xf_key = tex_xf_key;
                pfe->sh_key = sh_key;
                pfe->handle = g.pipelines[slot - 1u].handle;
                pfe->fixed = g.pipelines[slot - 1u].fixed;
                return g.pipelines[slot - 1u].handle;
            }
            at = (at + 1u) & (ENG_PIPELINE_BUCKETS - 1u);
        }
    }
    if (g.n_pipelines >= ENG_MAX_PIPELINES) {
        if (stats) g.st_pipe_full++;   /* the draw is dropped; a soak run sees it */
        return 0;
    }

    const double consts_t0 = stats ? eng_now_ms() : 0.0;
    if (pg->const_cached) {
        g.fp_constants.count = pg->const_count;
        if (pg->const_count)
            memcpy(g.fp_constants.values, pg->const_values,
                   (size_t)pg->const_count * 16u);
    } else {
        if (!fixed && rsx_fp_collect_constants(fp_uc, fp_size, &g.fp_constants) < 0)
            return 0;
        if (fixed) g.fp_constants.count = 0;
        if (g.fp_constants.count <= ENG_PROG_CONST_MAX) {
            pg->const_count = g.fp_constants.count;
            if (g.fp_constants.count)
                memcpy(pg->const_values, g.fp_constants.values,
                       (size_t)g.fp_constants.count * 16u);
            pg->const_cached = 1;
        }
    }
    if (stats) g.st_ms_consts += eng_now_ms() - consts_t0;

    u32 handle = 0;
    int vi = 1, fi = 1;
    u32 nconst = 0;
    if (fixed) {
        snprintf(s_vs_hlsl, sizeof s_vs_hlsl, "%s", kEngFixedVS);
        snprintf(s_ps_hlsl, sizeof s_ps_hlsl, "%s", kEngFixedPS);
    } else {
        if (pg->vp_const_sparse) {
            rsx_vp_const_map cmap;
            memset(&cmap, 0, sizeof cmap);
            cmap.count = pg->vp_const_count;
            for (u32 i = 0; i < cmap.count; i++) {
                cmap.slots[i] = pg->vp_const_list[i];
                cmap.remap[cmap.slots[i]] = (u16)i;
            }
            vi = rsx_vp_decompile_compact_sparse(
                (const u8*)(g.rsx.vp + rsx_dsp_vp_start(&g.rsx) * 4),
                pg->vp_instrs * 16u, pg->vtex_mask, layout->mask,
                &cmap, s_vs_hlsl, sizeof s_vs_hlsl);
        } else {
            vi = pg->vp_const_packed
                ? rsx_vp_decompile_compact_packed(
                    (const u8*)(g.rsx.vp + rsx_dsp_vp_start(&g.rsx) * 4),
                    pg->vp_instrs * 16u, pg->vtex_mask, layout->mask,
                    pg->vp_const_slots, s_vs_hlsl, sizeof s_vs_hlsl)
                : rsx_vp_decompile_compact_ex(
                    (const u8*)(g.rsx.vp + rsx_dsp_vp_start(&g.rsx) * 4),
                    pg->vp_instrs * 16u, pg->vtex_mask, layout->mask, s_vs_hlsl,
                    sizeof s_vs_hlsl);
        }
        fi = rsx_fp_decompile_buffered_ex(fp_uc, fp_size, pg->fp_ctrl,
                                          pg->cube_mask, s_ps_hlsl,
                                          sizeof s_ps_hlsl, &nconst);
        if (fi > 0 && nconst != g.fp_constants.count) fi = -1;
        if (fi > 0 && rs->alpha_test_enable &&
            rsx_fp_apply_alpha_test_buffered(s_ps_hlsl, sizeof s_ps_hlsl,
                                             rs->alpha_func) < 0)
            fi = -1;
        if (fi > 0 && g.sun_ndl_clamp_on &&
            eng_fp_sun_ndl_clamp(s_ps_hlsl, sizeof s_ps_hlsl) < 0)
            fi = -1;
        if (fi > 0 &&
            eng_fp_depth_blur_rewrite(s_ps_hlsl, sizeof s_ps_hlsl, fp_uc, fp_size, pg->cube_mask) < 0)
            fi = -1;
        if (fi > 0 && sh_key &&
            eng_fp_shadow_rewrite(s_ps_hlsl, sizeof s_ps_hlsl, sh_key) < 0)
            fi = -1;
        if (fi > 0 && tex_xf_key &&
            eng_fp_txf_rewrite(s_ps_hlsl, sizeof s_ps_hlsl, tex_xf_key) < 0)
            fi = -1;
        if (fi > 0 && g.res_scaled &&
            eng_fp_wscale_rewrite(s_ps_hlsl, sizeof s_ps_hlsl) < 0)
            fi = -1;
    }
    if (vi > 0 && fi > 0)
        handle = g.be->pipeline_create(g.be->user, s_vs_hlsl, s_ps_hlsl, rs,
                                       layout, layout->stride, rt_fmt, rt_count);
    { static u32 logs = 0;
      if ((logs < 32 && logs++ < 32) || (eng_trace() && g.frames >= eng_trace_after()))
        fprintf(stderr, "[rsx engine] pipeline %016llx: %s vp %d, fp %d,"
                        " %u constants -> %s\n",
                (unsigned long long)key, fixed ? "built-in" : "guest",
                vi, fi, nconst, handle ? "ok" : "FAILED (draw dropped)"); }

    g.pipelines[g.n_pipelines].key = key;
    g.pipelines[g.n_pipelines].handle = handle;
    g.pipelines[g.n_pipelines].fixed = (u8)fixed;
    {   u32 at = (u32)(key % ENG_PIPELINE_BUCKETS);
        while (g.pipeline_bucket[at]) at = (at + 1u) % ENG_PIPELINE_BUCKETS;
        g.pipeline_bucket[at] = g.n_pipelines + 1u;
    }
    g.n_pipelines++;
    *out_fixed = fixed;
    return handle;
}

/* ---- draw accumulation --------------------------------------------------- */

static void dc_reset(void)
{
    dc.n_arr = dc.n_idx = dc.n_packets = 0;
    dc.n_refs = dc.n_source_refs = dc.n_verts = dc.n_cuts = 0;
    dc.refs_remapped = 0;
    dc.fetch_ok = 1;
    dc.inl = NULL;
    dc.inl_bytes = 0;
    dc.n_imm = 0;
    dc.verts_cache = NULL;
    dc.indices_cache = NULL;
    dc.vc_key = 0;
    dc.vc_ent = ENG_VC_NONE;
    dc.cur_vid = 0;
    dc.cur_iid = 0;
    dc.cur_gcache = NULL;
    dc.mc_hit_ent = NULL;
    g_rsx_eng_cur_gcache = NULL;
    dc.mc_hit = 0;
    dc.mc_elig = 0;
    dc.mc_was_cold = 0;
    dc.mc_seen_hit = 0;
    dc.mc_indexed = 0;
    dc.mc_n_draw = 0;
    dc.mc_snap_frame = 0;
    dc.mc_desc_lo = 0;
    dc.mc_desc_hi = 0;
    dc.mc_desc_bytes = 0;
}

/* The method that completes attribute 0 in each SET_VERTEX_DATA* form. */
static int eng_imm_provokes(u32 m)
{
    return m == 0x1884u /* 2F */ || m == 0x1508u /* 3F */ || m == 0x1C0Cu /* 4F */ ||
           m == 0x1900u /* 2S */ || m == 0x1940u /* 4UB */ || m == 0x1984u /* 4S */ ||
           m == 0x1E40u /* 1F */;
}

static void dc_imm_push(void)
{
    if (dc.n_imm >= dc.cap_imm) {
        const u32 next = dc.cap_imm ? dc.cap_imm * 2u : 64u;
        void* p = realloc(dc.imm, (size_t)next * sizeof(*dc.imm));
        if (!p) return;
        dc.imm = (float (*)[RSX_DSP_NUM_VERTEX_ATTR][4])p;
        dc.cap_imm = next;
    }
    for (u32 attr = 0; attr < RSX_DSP_NUM_VERTEX_ATTR; attr++)
        rsx_dsp_vertex_default(&g.rsx, attr, dc.imm[dc.n_imm][attr]);
    dc.n_imm++;
}

static int dc_reserve_verts(u64 bytes);

/* ---- converted-vertex cache (ENG_VCACHE) --------------------------------
 *
 * A group's converted vertices are a pure function of three things: the fetch
 * plan (which attributes, from which offsets and strides, decoded as what),
 * the reference list (which elements, in which order) and the guest bytes
 * those references read. Key all three and a group whose key is already in
 * the arena hands the backend last frame's converted copy instead of decoding
 * every attribute of every vertex again -- fetch was 50 ms of the preload's
 * 149 ms walker frame for 1.16 M vertices.
 *
 * This is the D3D12 cache's policy (rsx_d3d12_backend.c:13924-14000) carried
 * into the shared engine, where every backend gets it: the same key shape
 * (plan + index range + a hash of the raw guest bytes), the same per-size-class
 * LRU with a second chance for anything this frame touched, and the same rule
 * that a group whose plan cannot be hashed exactly is not cached at all. What
 * a hit saves differs: that cache is VRAM-resident and removes the upload too,
 * while this one removes the conversion and leaves the backend's ring copy
 * (~13 MB/frame at an 86 % hit rate).
 */
static int s_vc_cfg = -1, s_vc_phase = -1;

/* ENG_VCACHE=1 turns it on. ENG_VCACHE_PHASE=<0|1> uses it only on that A/B
 * parity, so its fps contribution is the difference between two phases of one
 * run (the arena is kept and filled on both, only the lookup is skipped).
 * ENG_VCACHE_MB sets the arena size. */
static int eng_vcache_cfg(void)
{
    if (s_vc_cfg >= 0) return s_vc_cfg;
    s_vc_cfg = 0;
    {
        const char* e = getenv("ENG_VCACHE");
        if (!e || *e != '1') {
            fprintf(stderr, "[ENGVC] off (ENG_VCACHE=1 caches converted vertices)\n");
            return 0;
        }
    }
    { const char* p = getenv("ENG_VCACHE_PHASE"); if (p && *p) s_vc_phase = atoi(p) & 1; }
    long mb = 512;
    { const char* e = getenv("ENG_VCACHE_MB"); if (e && atol(e) >= 32) mb = atol(e); }
    /* A block offset is a u32, so the arena cannot exceed 4 GB. */
    if (mb > 3072) mb = 3072;
    g.vc_arena = (u8*)malloc((size_t)mb << 20);
    g.vc_ents = (eng_vcent*)calloc(ENG_VC_ENTS, sizeof(eng_vcent));
    g.vc_ht = (eng_vcslot*)malloc((size_t)ENG_VC_BUCKETS * sizeof(eng_vcslot));
    g.vc_seen = (u64*)calloc(131072u, sizeof(u64));
    if (!g.vc_arena || !g.vc_ents || !g.vc_ht || !g.vc_seen) {
        free(g.vc_arena); free(g.vc_ents); free(g.vc_ht); free(g.vc_seen);
        g.vc_arena = NULL; g.vc_ents = NULL; g.vc_ht = NULL; g.vc_seen = NULL;
        fprintf(stderr, "[ENGVC] could not allocate the arena -- off\n");
        return 0;
    }
    g.vc_arena_bytes = (u64)mb << 20;
    for (u32 i = 0; i < ENG_VC_BUCKETS; i++) g.vc_ht[i].ent = ENG_VC_NONE;
    for (u32 c = 0; c < ENG_VC_CLASSES; c++) g.vc_head[c] = g.vc_tail[c] = -1;
    s_vc_cfg = 1;
    fprintf(stderr, "[ENGVC] converted-vertex cache ON: %ld MB arena, %u entries "
                    "(ENG_VCACHE_PHASE phase parity %d)\n", mb, ENG_VC_ENTS, s_vc_phase);
    return 1;
}

static int eng_vcache_use(void)
{
    if (!eng_vcache_cfg()) return 0;
    return s_vc_phase < 0 || (ENG_AB_PHASE() & 1) == s_vc_phase;
}

static inline u32 eng_vc_slot(u64 key)
{
    return (u32)((key * 0x9E3779B97F4A7C15ull) >> 24) & (ENG_VC_BUCKETS - 1u);
}

static u32 eng_vc_find(u64 key)
{
    u32 at = eng_vc_slot(key);
    for (u32 walked = 0; walked < ENG_VC_BUCKETS; walked++) {
        const eng_vcslot* s = &g.vc_ht[at];
        if (s->ent == ENG_VC_NONE) return ENG_VC_NONE;
        if (s->ent != ENG_VC_NONE - 1u && s->key == key) return s->ent;
        at = (at + 1u) & (ENG_VC_BUCKETS - 1u);
    }
    return ENG_VC_NONE;
}

static void eng_vc_ht_rebuild(void)
{
    for (u32 i = 0; i < ENG_VC_BUCKETS; i++) g.vc_ht[i].ent = ENG_VC_NONE;
    g.vc_tombs = 0;
    for (u32 i = 0; i < g.vc_nents; i++) {
        u32 at = eng_vc_slot(g.vc_ents[i].key);
        while (g.vc_ht[at].ent != ENG_VC_NONE) at = (at + 1u) & (ENG_VC_BUCKETS - 1u);
        g.vc_ht[at].key = g.vc_ents[i].key;
        g.vc_ht[at].ent = i;
    }
}

static void eng_vc_ht_insert(u64 key, u32 ent)
{
    if (g.vc_tombs > ENG_VC_BUCKETS / 4u) eng_vc_ht_rebuild();
    u32 at = eng_vc_slot(key), tomb = ENG_VC_BUCKETS;
    for (;;) {
        eng_vcslot* s = &g.vc_ht[at];
        if (s->ent == ENG_VC_NONE) {
            eng_vcslot* where = (tomb != ENG_VC_BUCKETS) ? &g.vc_ht[tomb] : s;
            where->key = key;
            where->ent = ent;
            return;
        }
        if (s->ent == ENG_VC_NONE - 1u) {
            if (tomb == ENG_VC_BUCKETS) tomb = at;
        } else if (s->key == key) {
            s->ent = ent;   /* the same key cannot be live twice */
            return;
        }
        at = (at + 1u) & (ENG_VC_BUCKETS - 1u);
    }
}

static void eng_vc_ht_remove(u64 key)
{
    u32 at = eng_vc_slot(key);
    for (u32 walked = 0; walked < ENG_VC_BUCKETS; walked++) {
        eng_vcslot* s = &g.vc_ht[at];
        if (s->ent == ENG_VC_NONE) return;
        if (s->ent != ENG_VC_NONE - 1u && s->key == key) {
            s->ent = ENG_VC_NONE - 1u;
            g.vc_tombs++;
            return;
        }
        at = (at + 1u) & (ENG_VC_BUCKETS - 1u);
    }
}

static void eng_vc_lru_unlink(int e)
{
    eng_vcent* x = &g.vc_ents[e];
    const u32 c = x->cls;
    if (x->prev >= 0) g.vc_ents[x->prev].next = x->next; else g.vc_head[c] = x->next;
    if (x->next >= 0) g.vc_ents[x->next].prev = x->prev; else g.vc_tail[c] = x->prev;
    x->prev = x->next = -1;
}

static void eng_vc_lru_front(int e)
{
    eng_vcent* x = &g.vc_ents[e];
    const u32 c = x->cls;
    x->prev = -1;
    x->next = g.vc_head[c];
    if (g.vc_head[c] >= 0) g.vc_ents[g.vc_head[c]].prev = e; else g.vc_tail[c] = e;
    g.vc_head[c] = e;
}

/* A block for `bytes` out of the smallest class that holds it: bump-allocate
 * until the arena is exhausted, then evict this class's least recently used
 * entry that the frame being recorded has not touched. An entry used since it
 * took its block gets a second chance, which is the D3D12 cache's CLOCK
 * eviction: approximate LRU without walking a list on every hit. -1 means
 * every entry of the class is live this frame. */
static int eng_vc_alloc(u32 bytes, u32* off, u32* cls_out)
{
    u32 cls = 0;
    while (cls < ENG_VC_CLASSES && (1024u << cls) < bytes) cls++;
    if (cls >= ENG_VC_CLASSES) return -1;
    const u32 block = 1024u << cls;
    *cls_out = cls;
    if (g.vc_bump + block <= g.vc_arena_bytes && g.vc_nents < ENG_VC_ENTS) {
        *off = (u32)g.vc_bump;
        g.vc_bump += block;
        return (int)g.vc_nents++;
    }
    for (u32 tries = 0; tries < 256; tries++) {
        const int t = g.vc_tail[cls];
        if (t < 0) return -1;
        eng_vcent* x = &g.vc_ents[t];
        const u32 age = (u32)(g.frames - x->last_frame);
        if (age > 4u || (age > 1u && x->last_frame <= x->placed)) {
            eng_vc_ht_remove(x->key);
            eng_vc_lru_unlink(t);
            *off = x->off;
            g.vc_evict++;
            g.vc_evict_gen++;
            return t;
        }
        x->placed = g.frames;
        g.vc_evict_gen++;
        eng_vc_lru_unlink(t);
        eng_vc_lru_front(t);
    }
    return -1;
}

/* Fold the plan, the reference list and the guest bytes those references read
 * into one key. Returns 0 when the plan cannot be hashed exactly: an enabled
 * attribute the resolver could not map as one contiguous span is read element
 * by element, and a key that does not see those bytes could hand back another
 * mesh's vertices. */
static int eng_vcache_key(const rsx_vertex_fetch_plan* plan,
                          const rsx_vertex_ref* refs, u32 n, u64* out)
{
    const rsx_vertex_layout_plan* l = &plan->layout;
    if (!n || !l->count || !l->stride) return 0;

    /* Collect and validate all active attribute spans FIRST, merge overlapping
     * interleaved ranges, and enforce ENG_VC_HASH_MAX BEFORE hashing a single
     * byte. Previously unkeyed draws (>4 MB) hashed the full span across every
     * attribute redundantly before discarding the key. */
    typedef struct { const u8* ptr; u64 len; } vc_span;
    vc_span sp[RSX_DSP_NUM_VERTEX_ATTR];
    u32 nsp = 0;
    u32 mn = 0xFFFFFFFFu, mx = 0;
    int have_minmax = 0;
    const int uniform_spans = plan->uniform_element && !plan->inline_data;
    for (u32 slot = 0; slot < l->count; slot++) {
        const rsx_vertex_fetch_attr* f = &plan->attr[l->attrs[slot]];
        if (!f->desc.type || !f->desc.size) continue;   /* a default: in sig */
        if (!f->stable_base || f->stable_last < f->stable_first) return 0;
        const u64 len64 = (u64)(f->stable_last - f->stable_first) * (u64)f->stride +
                          (u64)f->elem_size;
        if (!len64 || len64 > ENG_VC_HASH_MAX) return 0;
        if (uniform_spans && !have_minmax) {
            mn = f->stable_first;
            mx = f->stable_last;
            have_minmax = 1;
        }
        sp[nsp].ptr = f->stable_base;
        sp[nsp].len = len64;
        nsp++;
    }

    /* Dense enough to be worth hashing: the sample below reads whole ranges,
     * so a list spread over a large element space would hash far more than it
     * converts. */
    if (!have_minmax) {
        for (u32 i = 0; i < n; i++) {
            const u32 element = rsx_vertex_element_index(refs[i].vertex_id,
                                                        refs[i].base_index, 1, 0);
            if (element < mn) mn = element;
            if (element > mx) mx = element;
        }
    }
    if (mx < mn) return 0;
    const u64 span_el = (u64)(mx - mn) + 1ull;
    const u64 max_el = eng_m2d_use() ? (u64)n * 4ull : (u64)n;
    if (span_el > max_el) return 0;

    for (u32 i = 1; i < nsp; i++) {
        vc_span tmp = sp[i];
        u32 j = i;
        while (j > 0 && (uintptr_t)sp[j - 1].ptr > (uintptr_t)tmp.ptr) {
            sp[j] = sp[j - 1];
            j--;
        }
        sp[j] = tmp;
    }
    u32 msp = 0;
    u64 total_len = 0;
    for (u32 i = 0; i < nsp; i++) {
        if (msp > 0) {
            const uintptr_t prev_start = (uintptr_t)sp[msp - 1].ptr;
            const uintptr_t prev_end   = prev_start + (uintptr_t)sp[msp - 1].len;
            const uintptr_t cur_start  = (uintptr_t)sp[i].ptr;
            const uintptr_t cur_end    = cur_start + (uintptr_t)sp[i].len;
            if (cur_start <= prev_end) {
                if (cur_end > prev_end)
                    sp[msp - 1].len = (u64)(cur_end - prev_start);
                continue;
            }
        }
        sp[msp++] = sp[i];
    }
    for (u32 i = 0; i < msp; i++) total_len += sp[i].len;
    if (total_len > ENG_VC_HASH_MAX) return 0;

    /* Everything outside guest memory that decides the converted bytes: the
     * attribute list, where each array sits and what it holds, and the
     * per-attribute defaults the decode falls back to. */
    u32 sig[8 + RSX_DSP_NUM_VERTEX_ATTR * 12];
    u32 ns = 0;
    sig[ns++] = l->mask;
    sig[ns++] = l->stride;
    sig[ns++] = l->count;
    sig[ns++] = plan->base_offset;
    sig[ns++] = plan->divider_mask;
    sig[ns++] = (u32)plan->uniform_element | ((plan->inline_data ? 1u : 0u) << 8);
    sig[ns++] = plan->inline_stride;
    sig[ns++] = plan->inline_bytes;
    for (u32 slot = 0; slot < l->count; slot++) {
        const u32 attr = l->attrs[slot];
        const rsx_vertex_fetch_attr* f = &plan->attr[attr];
        sig[ns++] = attr;
        sig[ns++] = f->desc.type | (f->desc.size << 8) |
                    (f->desc.location << 16) | (f->desc.frequency << 24);
        sig[ns++] = f->desc.offset;
        sig[ns++] = f->desc.stride;
        sig[ns++] = f->elem_size;
        sig[ns++] = f->stride;
        sig[ns++] = f->stable_first;
        sig[ns++] = f->stable_last;
        memcpy(&sig[ns], f->default_value, sizeof f->default_value);
        ns += 4;
    }
    u64 key = eng_hash_bytes(refs, n * (u32)sizeof(*refs), 0x9E3779B97F4A7C15ull);
    key = eng_hash_bytes(sig, ns * 4u, key);

    for (u32 i = 0; i < msp; i++) {
        const u32 len = (u32)sp[i].len;
        key = eng_hash_bytes(&len, sizeof len, key);
        key = eng_hash_bytes(sp[i].ptr, len, key);
    }
    g.vc_hash_bytes += total_len;
    *out = key ? key : 1u;
    return 1;
}

static const u8* eng_vcache_hit(u64 key, u32* out_ent)
{
    const u32 e = eng_vc_find(key);
    if (e == ENG_VC_NONE) return NULL;
    eng_vcent* x = &g.vc_ents[e];
    x->last_frame = g.frames;
    if (out_ent) *out_ent = e;
    return g.vc_arena + x->off;
}

/* A full arena, or a size class whose entries are all in use this frame, drops
 * the store: the draw is unaffected, only the next frame's hit is lost. */
static int eng_vc_admit2(u64 key)
{
    if (!g.vc_seen || !s_mc_swd || !eng_m2d_use()) return 1;
    const u32 set = (u32)((key * 0x9E3779B97F4A7C15ull) >> 16) & 65535u;
    u64* slot = &g.vc_seen[set * 2u];
    if (slot[0] == key || slot[1] == key) return 1;
    slot[1] = slot[0];
    slot[0] = key;
    return 0;
}

static int eng_vcache_store(u64 key, const u8* verts, u32 bytes)
{
    u32 off = 0, cls = 0;
    const int e = eng_vc_alloc(bytes, &off, &cls);
    if (e < 0) return -1;
    eng_vcent* x = &g.vc_ents[e];
    memset(x, 0, sizeof *x);
    x->prev = x->next = -1;
    x->key = key;
    x->serial = ++g.vc_serial;
    x->off = off;
    x->bytes = bytes;
    x->last_frame = x->placed = g.frames;
    x->cls = (u8)cls;
    memcpy(g.vc_arena + off, verts, bytes);
    eng_vc_lru_front(e);
    eng_vc_ht_insert(key, (u32)e);
    return e;
}

/* ---- ENG_MESH_CACHE: SPU page-dirty map + fast mesh replay -------------- */
#if defined(__GNUC__) || defined(__clang__)
extern void (*g_vc_put_note)(u32 ea, u32 size) __attribute__((weak));
#endif


static void eng_swd_note_spu_put(u32 ea, u32 size)
{
    u32* sw = __atomic_load_n(&s_swd_gen, __ATOMIC_RELAXED);
    if (!sw || !size) return;
    const u32 f = __atomic_load_n(&s_swd_frame, __ATOMIC_RELAXED);
    const u32 start_pg = ea >> 12;
    const u64 end_ea = (u64)ea + (u64)size - 1u;
    u32 end_pg = (u32)(end_ea >> 12);
    if (end_pg >= (1u << 20)) end_pg = (1u << 20) - 1u;
    for (u32 pg = start_pg; pg <= end_pg; pg++) {
        if (__atomic_load_n(&sw[pg], __ATOMIC_RELAXED) != f)
            __atomic_store_n(&sw[pg], f, __ATOMIC_RELAXED);
    }
    const u32 ca = start_pg >> 8, cb = end_pg >> 8;
    for (u32 c = ca; c <= cb; c++) {
        if (__atomic_load_n(&s_swd_coarse[c], __ATOMIC_RELAXED) != f)
            __atomic_store_n(&s_swd_coarse[c], f, __ATOMIC_RELAXED);
    }
}

static int s_mc_ver_explicit = 0;
static int eng_mc_cfg(void)
{
    if (s_mc_cfg >= 0) return s_mc_cfg;
    s_mc_cfg = 0;
    if (!eng_vcache_cfg()) return 0;
    { const char* e = getenv("ENG_MESH_CACHE"); if (e && *e == '0') {
        fprintf(stderr, "[ENGMC] off (ENG_MESH_CACHE=0)\n");
        return 0;
    } }
    { const char* p = getenv("ENG_MESH_CACHE_PHASE"); if (p && *p) s_mc_phase = atoi(p) & 1; }
    { const char* sp = getenv("ENG_MESH_SWD_PHASE"); if (sp && *sp) s_mc_swd_phase = atoi(sp) & 1; }
    { const char* v = getenv("ENG_MESH_CACHE_VERIFY");
      if (v && atoi(v) >= 1) {
          int n = atoi(v);
          int p2 = 1;
          while (p2 < n && p2 < 256) p2 <<= 1;
          s_mc_ver = p2;
          s_mc_ver_explicit = 1;
      } }
    int want_swd = 0;
#if defined(__GNUC__) || defined(__clang__)
    if (&g_vc_put_note) want_swd = 1;
#endif
    { const char* e = getenv("ENG_MESH_SWD"); if (e && *e == '0') want_swd = 0; }
    g.mc_ents = NULL;
    g.mc_tags = NULL;
#if defined(_WIN32)
    g.mc_ents = (eng_mcent*)calloc(ENG_MC_ENTS, sizeof(eng_mcent));
    g.mc_tags = (eng_mctag*)calloc(ENG_MC_ENTS, sizeof(eng_mctag));
#else
    if (posix_memalign((void**)&g.mc_ents, 64, ENG_MC_ENTS * sizeof(eng_mcent)) == 0)
        memset(g.mc_ents, 0, ENG_MC_ENTS * sizeof(eng_mcent));
    else g.mc_ents = NULL;
    if (posix_memalign((void**)&g.mc_tags, 64, ENG_MC_ENTS * sizeof(eng_mctag)) == 0)
        memset(g.mc_tags, 0, ENG_MC_ENTS * sizeof(eng_mctag));
    else g.mc_tags = NULL;
#endif
    if (!g.mc_ents || !g.mc_tags) {
        free(g.mc_ents); free(g.mc_tags);
        g.mc_ents = NULL; g.mc_tags = NULL;
        fprintf(stderr, "[ENGMC] allocation failed -- off\n");
        return 0;
    }
    g.mc_seen = (u64*)calloc(131072u, sizeof(u64));
    s_mc_swd = 0;
    if (want_swd) {
        u32* sw = (u32*)calloc(1u << 20, sizeof(u32));
        u8*  ns = (u8*)calloc(1u << 20, sizeof(u8));
        if (sw && ns) {
            memset(s_swd_coarse, 0, sizeof s_swd_coarse);
            s_swd_nonspu = ns;
            __atomic_store_n(&s_swd_frame, g.frames + 1u, __ATOMIC_RELEASE);
            __atomic_store_n(&s_swd_gen, sw, __ATOMIC_RELEASE);
#if defined(__GNUC__) || defined(__clang__)
            g_vc_put_note = eng_swd_note_spu_put;
#endif
            s_mc_swd = 1;
        } else {
            free(sw); free(ns);
        }
    }
    s_mc_cfg = 1;
    fprintf(stderr, "[ENGMC] fast mesh cache ON: %u entries (%u-way), swd=%d, verify every %d frames "
                    "(ENG_MESH_CACHE_PHASE %d, ENG_MESH_SWD_PHASE %d)\n",
            ENG_MC_ENTS, ENG_MC_WAYS, s_mc_swd, s_mc_ver, s_mc_phase, s_mc_swd_phase);
    return 1;
}

static int eng_mc_use(void)
{
    if (!eng_mc_cfg()) return 0;
    if (!eng_vcache_use()) return 0;
    if (s_guest_reader || !vm_base) return 0;
    return s_mc_phase < 0 || (ENG_AB_PHASE() & 1) == s_mc_phase;
}

static u64 eng_mc_hash_spans(const eng_mc_span* spans, u32 n_spans)
{
    u64 h = 0x517CC1B727220A95ull;
    for (u32 i = 0; i < n_spans; i++)
        h = eng_hash_bytes(vm_base + spans[i].ea, spans[i].len, h);
    return h;
}

static int eng_mc_spans_clean(const eng_mcent* e)
{
    const u32 vgen = e->verified_frame;
    if (e->coarse_single) {
        const u32 cg = __atomic_load_n(&s_swd_coarse[e->coarse_lo], __ATOMIC_RELAXED);
        if (!cg || cg + 1u < vgen) return 1;
    } else if (e->coarse_lo <= e->coarse_hi && e->coarse_hi < 4096u) {
        int any_dirty = 0;
        for (u32 c = e->coarse_lo; c <= e->coarse_hi; c++) {
            const u32 cg = __atomic_load_n(&s_swd_coarse[c], __ATOMIC_RELAXED);
            if (cg && cg + 1u >= vgen) { any_dirty = 1; break; }
        }
        if (!any_dirty) return 1;
    }
    const u32* sw = s_swd_gen;
    const eng_mc_span* sp = eng_mc_spans_ptr(e);
    for (u32 i = 0; i < e->n_spans; i++) {
        const u32 a = sp[i].ea >> 12;
        const u32 b = (u32)(((u64)sp[i].ea + sp[i].len - 1u) >> 12);
        if (b >= (1u << 20)) return 0;
        const u32 ca = a >> 8, cb = b >> 8;
        for (u32 c = ca; c <= cb; c++) {
            const u32 cg = __atomic_load_n(&s_swd_coarse[c], __ATOMIC_RELAXED);
            if (cg && cg + 1u >= vgen) {
                const u32 p0 = (c == ca) ? a : (c << 8);
                const u32 p1 = (c == cb) ? b : ((c << 8) + 255u);
                for (u32 pg = p0; pg <= p1; pg++) {
                    const u32 sg = __atomic_load_n(&sw[pg], __ATOMIC_RELAXED);
                    if (sg && sg + 1u >= vgen) return 0;
                }
            }
        }
    }
    return 1;
}

static int eng_mc_spans_nonspu(const eng_mcent* e)
{
    const u8* ns = s_swd_nonspu;
    const eng_mc_span* sp = eng_mc_spans_ptr(e);
    for (u32 i = 0; i < e->n_spans; i++) {
        const u32 a = sp[i].ea >> 12;
        const u32 b = (u32)(((u64)sp[i].ea + sp[i].len - 1u) >> 12);
        if (b >= (1u << 20)) return 1;
        for (u32 pg = a; pg <= b; pg++)
            if (ns[pg]) return 1;
    }
    return 0;
}

static void eng_mc_mark_nonspu(const eng_mcent* e)
{
    u8* ns = s_swd_nonspu;
    const eng_mc_span* sp = eng_mc_spans_ptr(e);
    for (u32 i = 0; i < e->n_spans; i++) {
        const u32 a = sp[i].ea >> 12;
        const u32 b = (u32)(((u64)sp[i].ea + sp[i].len - 1u) >> 12);
        if (b >= (1u << 20)) continue;
        for (u32 pg = a; pg <= b; pg++) {
            if (!ns[pg]) { ns[pg] = 1; g.mc_nonspu_pages++; }
        }
    }
}

static int s_vattr_hash_valid;
static u32 s_vattr_hash_gen, s_vattr_hash_mask, s_vattr_hash_bytes;
static u64 s_vattr_h0, s_vattr_h1;

static int eng_mc_build_desc(u32 prim, const rsx_vertex_layout_plan* layout,
                             int allow_remap, u16* out_bytes,
                             u64* out_lo, u64* out_hi)
{
    if (dc.inl_bytes || !layout->count || !layout->stride) return 0;
    int is_idx = 0;
    u32 nb = 0;
    const eng_batch* batches = NULL;
    if (dc.n_idx > 0 && dc.n_arr == 0) {
        is_idx = 1; nb = dc.n_idx; batches = dc.idx;
    } else if (dc.n_arr > 0 && dc.n_idx == 0) {
        is_idx = 0; nb = dc.n_arr; batches = dc.arr;
    } else {
        return 0;
    }
    const u32* regs = g.rsx.regs;
    u32 ia_offset = 0, base_index = 0, restart_val = 0;
    u32 ia_flags = 0;
    if (is_idx) {
        ia_offset = regs[0x181Cu >> 2];
        const u32 ia_fmt = regs[0x1820u >> 2];
        const int is_u32 = ((ia_fmt >> 4) & 0xFFu) == 0u;
        const u32 ia_loc = (ia_fmt & 0xFu) ? RSX_LOCATION_MAIN : RSX_LOCATION_LOCAL;
        const u32 rval = regs[0x1DB0u >> 2];
        const int restart_en = (regs[0x1DACu >> 2] & 1u) != 0u && (is_u32 || rval <= 0xFFFFu);
        base_index = regs[0x173Cu >> 2];
        restart_val = restart_en ? rval : 0u;
        ia_flags = (is_u32 ? 1u : 0u) | (ia_loc ? 2u : 0u) | (restart_en ? 4u : 0u);
    }
    const int fast_desc = (s_ef.phase >= 0) ? (int)s_ef.m2d : eng_m2d_use();
    if (fast_desc) {
        if (!s_vattr_hash_valid || s_vattr_hash_gen != g.rsx.vtx_attr_gen ||
            s_vattr_hash_mask != layout->mask) {
            const u32* fmt_regs = &regs[0x1740u >> 2];
            const u32* off_regs = &regs[0x1680u >> 2];
            u64 a0 = 0x517CC1B727220A95ull ^ ((u64)layout->mask << 32) ^ layout->count;
            u64 a1 = 0x9E3779B97F4A7C15ull ^ g.rsx.regs[0x1FC0u >> 2];
            u32 ab = 0u;
            for (u32 slot = 0; slot < layout->count; slot++) {
                const u32 attr = layout->attrs[slot];
                const u32 fmt = fmt_regs[attr];
                if ((fmt & 7u) && ((fmt >> 4) & 0xFu)) {
                    const u64 w = (u64)fmt | ((u64)off_regs[attr] << 32);
                    a0 = (a0 ^ w) * 0x9E3779B97F4A7C15ull;
                    a1 = (a1 ^ ((w << 27) | (w >> 37))) * 0xBF58476D1CE4E5B9ull;
                    ab += 8u;
                } else {
                    const u32* def = &regs[(0x1C00u >> 2) + attr * 4u];
                    const u64 w0 = (u64)fmt | ((u64)def[0] << 32);
                    const u64 w1 = (u64)def[1] | ((u64)def[2] << 32);
                    const u64 w2 = (u64)def[3];
                    a0 = (a0 ^ w0) * 0x9E3779B97F4A7C15ull;
                    a1 = (a1 ^ ((w0 << 27) | (w0 >> 37))) * 0xBF58476D1CE4E5B9ull;
                    a0 = (a0 ^ w1) * 0x9E3779B97F4A7C15ull;
                    a1 = (a1 ^ ((w1 << 27) | (w1 >> 37))) * 0xBF58476D1CE4E5B9ull;
                    a0 = (a0 ^ w2) * 0x9E3779B97F4A7C15ull;
                    a1 = (a1 ^ (w2 << 13)) * 0xBF58476D1CE4E5B9ull;
                    ab += 20u;
                }
            }
            s_vattr_h0 = a0; s_vattr_h1 = a1; s_vattr_hash_bytes = ab;
            s_vattr_hash_gen = g.rsx.vtx_attr_gen;
            s_vattr_hash_mask = layout->mask;
            s_vattr_hash_valid = 1;
        }
        const u32 bytes = 32u + nb * 8u + s_vattr_hash_bytes;
        *out_bytes = (u16)bytes;
        const u32 fword = (allow_remap ? 1u : 0u) | ((u32)is_idx << 1) | 16u |
                          (ia_flags << 4) | ((u32)layout->count << 8) | (nb << 16);
        u64 h0 = s_vattr_h0 ^ ((u64)bytes << 32);
        u64 h1 = s_vattr_h1 ^ (u64)nb;
        { const u64 w0 = (u64)prim | ((u64)layout->mask << 32);
          const u64 w1 = (u64)regs[0x1738u >> 2] | ((u64)regs[0x1FC0u >> 2] << 32);
          const u64 w2 = (u64)fword | ((u64)ia_offset << 32);
          const u64 w3 = (u64)base_index | ((u64)restart_val << 32);
          h0 = (h0 ^ w0) * 0x9E3779B97F4A7C15ull;
          h1 = (h1 ^ ((w0 << 27) | (w0 >> 37))) * 0xBF58476D1CE4E5B9ull;
          h0 = (h0 ^ w1) * 0x9E3779B97F4A7C15ull;
          h1 = (h1 ^ ((w1 << 27) | (w1 >> 37))) * 0xBF58476D1CE4E5B9ull;
          h0 = (h0 ^ w2) * 0x9E3779B97F4A7C15ull;
          h1 = (h1 ^ ((w2 << 27) | (w2 >> 37))) * 0xBF58476D1CE4E5B9ull;
          h0 = (h0 ^ w3) * 0x9E3779B97F4A7C15ull;
          h1 = (h1 ^ ((w3 << 27) | (w3 >> 37))) * 0xBF58476D1CE4E5B9ull; }
        if (__builtin_expect(nb == 1u, 1)) {
            const u64 wb = (u64)batches[0].first | ((u64)batches[0].count << 32);
            h0 = (h0 ^ wb) * 0x9E3779B97F4A7C15ull;
            h1 = (h1 ^ ((wb << 27) | (wb >> 37))) * 0xBF58476D1CE4E5B9ull;
        } else {
            for (u32 b = 0; b < nb; b++) {
                const u64 wb = (u64)batches[b].first | ((u64)batches[b].count << 32);
                h0 = (h0 ^ wb) * 0x9E3779B97F4A7C15ull;
                h1 = (h1 ^ ((wb << 27) | (wb >> 37))) * 0xBF58476D1CE4E5B9ull;
            }
        }
        h0 ^= h1 >> 31; h0 *= 0xFF51AFD7ED558CCDull; h0 ^= h0 >> 33;
        h1 ^= h0 >> 29; h1 *= 0xC4CEB9FE1A85EC53ull; h1 ^= h1 >> 33;
        *out_lo = h0 ? h0 : 1ull;
        *out_hi = h1;
        return 1;
    }
    u32 p[12 + ENG_MAX_BATCHES * 2 + RSX_DSP_NUM_VERTEX_ATTR * 6];
    p[0] = prim;
    p[1] = layout->mask;
    p[2] = regs[0x1738u >> 2];
    p[3] = regs[0x1FC0u >> 2];
    p[4] = (allow_remap ? 1u : 0u) | ((u32)is_idx << 1) |
           (ia_flags << 4) | ((u32)layout->count << 8) | (nb << 16);
    p[5] = ia_offset;
    p[6] = base_index;
    p[7] = restart_val;
    u32 nw = 8u;
    for (u32 b = 0; b < nb; b++) {
        p[nw++] = batches[b].first;
        p[nw++] = batches[b].count;
    }
    for (u32 slot = 0; slot < layout->count; slot++) {
        const u32 attr = layout->attrs[slot];
        rsx_dsp_vertex_attr desc;
        rsx_dsp_get_vertex_attr(&g.rsx, attr, &desc);
        p[nw++] = attr | (desc.type << 5) | (desc.size << 9) |
                  (desc.location << 13) | (desc.frequency << 16);
        if (desc.type && desc.size) {
            p[nw++] = desc.offset;
            p[nw++] = desc.stride;
        } else {
            float def[4];
            rsx_dsp_vertex_default(&g.rsx, attr, def);
            memcpy(&p[nw], def, 16);
            nw += 4u;
        }
    }
    const u32 bytes = nw * 4u;
    *out_bytes = (u16)bytes;
    const XXH128_hash_t h128 = XXH3_128bits_withSeed(p, bytes, 0xA0761D6478BD642Full);
    *out_lo = h128.low64 ? h128.low64 : 1ull;
    *out_hi = h128.high64;
    return 1;
}

static int eng_mc_try_lookup(u32 prim, const rsx_vertex_layout_plan* layout,
                             int allow_remap)
{
    if (!((s_ef.phase >= 0 && !s_guest_reader && vm_base != NULL) ? (int)s_ef.mc : eng_mc_use()))
        return 0;
    dc.mc_snap_frame = __atomic_load_n(&s_swd_frame, __ATOMIC_RELAXED);
    if (!eng_mc_build_desc(prim, layout, allow_remap,
                           &dc.mc_desc_bytes, &dc.mc_desc_lo, &dc.mc_desc_hi)) {
        g.mc_misses++;
        g.mc_miss_unelig++;
        return 0;
    }
    dc.mc_elig = 1;
    const int m2e4b = (s_ef.phase >= 0) ? (int)s_ef.m2e4b : eng_m2e4b_use();
    eng_mcent* hit = NULL;
    u32 slot_idx = 0;
    if (m2e4b && s_mc_next_pred < ENG_MC_ENTS &&
        g.mc_tags[s_mc_next_pred].lo == dc.mc_desc_lo &&
        g.mc_tags[s_mc_next_pred].hi == dc.mc_desc_hi) {
        eng_mcent* e = &g.mc_ents[s_mc_next_pred];
        if (e->valid && e->desc_bytes == dc.mc_desc_bytes) {
            hit = e;
            slot_idx = s_mc_next_pred;
        }
    }
    if (!hit) {
        const u32 set = (u32)(dc.mc_desc_lo & (ENG_MC_SETS - 1u));
        const u32 base_idx = set * ENG_MC_WAYS;
        const eng_mctag* tbase = &g.mc_tags[base_idx];
        for (u32 w = 0; w < ENG_MC_WAYS; w++) {
            if (tbase[w].lo == dc.mc_desc_lo && tbase[w].hi == dc.mc_desc_hi) {
                eng_mcent* e = &g.mc_ents[base_idx + w];
                if (e->valid && e->desc_bytes == dc.mc_desc_bytes) {
                    hit = e;
                    slot_idx = base_idx + w;
                    break;
                }
            }
        }
    }
    if (!hit) {
        dc.mc_was_cold = 1;
        g.mc_misses++;
        g.mc_miss_cold++;
        return 0;
    }
    if (m2e4b) {
        if (s_mc_prev_slot < ENG_MC_ENTS && g.mc_ents[s_mc_prev_slot].next_slot != slot_idx)
            g.mc_ents[s_mc_prev_slot].next_slot = slot_idx;
        s_mc_prev_slot = slot_idx;
        const u32 ns = hit->next_slot;
        s_mc_next_pred = ns;
        if (ns < ENG_MC_ENTS) {
            __builtin_prefetch(&g.mc_tags[ns], 0, 1);
            __builtin_prefetch(&g.mc_ents[ns], 0, 1);
        }
    }
    const int m2e4 = (s_ef.phase >= 0) ? (int)s_ef.m2e4 : eng_m2e4_use();
    const u32 lru_mask = g.vc_evict ? 3u : 15u;
    const int need_vc_lru = !m2e4 || g.vc_evict || g.vc_bump > (g.vc_arena_bytes >> 1);
    if (!hit->vc_promoted || hit->vc_evict_gen != g.vc_evict_gen ||
        (need_vc_lru && ((g.frames & lru_mask) == (slot_idx & lru_mask)))) {
        if (hit->vc_ent >= ENG_VC_ENTS) {
            g.mc_misses++;
            g.mc_miss_cold++;
            return 0;
        }
        eng_vcent* vx = &g.vc_ents[hit->vc_ent];
        if (vx->key != hit->vc_key || vx->off != hit->verts_off) {
            hit->valid = 0;
            g.mc_tags[slot_idx].lo = 0u;
            g.mc_misses++;
            g.mc_miss_cold++;
            return 0;
        }
        vx->last_frame = g.frames;
        hit->vc_evict_gen = g.vc_evict_gen;
        hit->vc_promoted = 1;
    }
    const int use_swd = (s_ef.phase >= 0) ? (int)s_ef.mc_swd
                                          : (s_mc_swd && (s_mc_swd_phase < 0 || (ENG_AB_PHASE() & 1) == s_mc_swd_phase));
    int ahash = !use_swd || hit->always_hash;
    if (!ahash && hit->nonspu_checked_gen != (u32)g.mc_nonspu_pages) {
        if (eng_mc_spans_nonspu(hit)) {
            hit->always_hash = 1;
            ahash = 1;
        } else {
            hit->nonspu_checked_gen = (u32)g.mc_nonspu_pages;
        }
    }
    const int dirty = use_swd ? !eng_mc_spans_clean(hit) : 0;
    const u32 cur_f = dc.mc_snap_frame;
    const u32 mc_ver = (!s_mc_ver_explicit && s_ef.m2f) ? 256u : (u32)s_mc_ver;
    const int verify = (u32)(cur_f - hit->verified_frame) >= mc_ver ||
                       (((cur_f & (mc_ver - 1u)) == (slot_idx & (mc_ver - 1u))) &&
                        hit->verified_frame != cur_f);
    if (ahash || dirty || verify) {
        const u64 h = eng_mc_hash_spans(eng_mc_spans_ptr(hit), hit->n_spans);
        if (h != hit->mem_hash) {
            if (!dirty && !ahash) {
                hit->always_hash = 1;
                eng_mc_mark_nonspu(hit);
                g.mc_miss_ver++;
            } else {
                g.mc_miss_dirty++;
            }
            g.mc_misses++;
            return 0;
        }
        hit->verified_frame = cur_f;
        if (ahash) g.mc_hits_ahash++;
        else g.mc_hits_ver++;
    } else {
        g.mc_hits_swd++;
    }
    if (m2e4b) {
        const u16 f16 = (u16)g.frames;
        if ((u16)(f16 - hit->last_use_16) >= 8u)
            hit->last_use_16 = f16;
    } else if (!m2e4) {
        if (hit->last_use_frame != g.frames) hit->last_use_frame = g.frames;
    } else if ((u32)(g.frames - hit->last_use_frame) >= 8u) {
        hit->last_use_frame = g.frames;
    }
    dc.mc_hit_ent = hit;
    dc.cur_gcache = &hit->gcache;
    g_rsx_eng_cur_gcache = &hit->gcache;
    dc.n_verts = hit->n_verts;
    dc.n_refs = hit->n_verts;
    dc.mc_n_draw = hit->n_draw;
    dc.mc_indexed = hit->indexed;
    if (m2e4b && g_rsx_vk_gcache_gen != 0u &&
        hit->gcache.v_gen == g_rsx_vk_gcache_gen &&
        (!hit->indexed || hit->gcache.i_gen == g_rsx_vk_gcache_gen) &&
        !(s_ef.phase >= 0 ? (int)s_ef.trace : eng_trace())) {
        /* Hot single-cache-line path: VK gcache already has VB/IB in VRAM; avoid
         * touching eng_mcent cache line 2 unless VK falls back to !batched. */
        dc.verts_cache = g.vc_arena;
        dc.indices_cache = hit->indexed ? (const u32*)g.vc_arena : NULL;
        dc.cur_vid = 0u;
        dc.cur_iid = 0u;
        dc.n_source_refs = hit->n_verts;
    } else {
        dc.verts_cache = g.vc_arena + hit->verts_off;
        dc.indices_cache = hit->indexed ? hit->idx_buf : NULL;
        dc.cur_vid = hit->vc_serial;
        dc.cur_iid = hit->indexed ? hit->idx_serial : 0u;
        dc.n_source_refs = hit->n_source_refs;
    }
    dc.mc_hit = 1;
    dc.fetch_ok = 1;
    g.mc_hits++;
    g.vc_hits++;
    g.vc_hit_verts += hit->n_verts;
    return 1;
}

static int eng_mc_collect_spans(eng_mc_span* out, u32* out_n)
{
    eng_mc_span raw[ENG_MAX_BATCHES + RSX_DSP_NUM_VERTEX_ATTR];
    u32 n = 0;
    u64 total_bytes = 0;
    if (dc.n_idx) {
        rsx_dsp_index_array ia;
        rsx_dsp_get_index_array(&g.rsx, &ia);
        const u32 esz = ia.is_u32 ? 4u : 2u;
        const u32 ia_loc = ia.location ? RSX_LOCATION_MAIN : RSX_LOCATION_LOCAL;
        for (u32 b = 0; b < dc.n_idx; b++) {
            const u64 start = (u64)ia.offset + (u64)dc.idx[b].first * esz;
            const u64 bytes = (u64)dc.idx[b].count * esz;
            if (!bytes) continue;
            if (start > 0xFFFFFFFFull || bytes > 0xFFFFFFFFull ||
                start + bytes > 0x100000000ull)
                return 0;
            const u8* ptr = eng_guest_ptr(NULL, ia_loc, (u32)start, (u32)bytes);
            if (!ptr || ptr < vm_base) return 0;
            const size_t ea = (size_t)(ptr - vm_base);
            if (ea + bytes > 0x100000000ull) return 0;
            raw[n].ea = (u32)ea;
            raw[n].len = (u32)bytes;
            n++;
        }
    }
    const rsx_vertex_layout_plan* l = &dc.fetch_plan.layout;
    for (u32 slot = 0; slot < l->count; slot++) {
        const rsx_vertex_fetch_attr* f = &dc.fetch_plan.attr[l->attrs[slot]];
        if (!f->desc.type || !f->desc.size) continue;
        if (!f->stable_base || f->stable_last < f->stable_first) return 0;
        if (f->stable_base < vm_base) return 0;
        const u32 len = (f->stable_last - f->stable_first) * f->stride + f->elem_size;
        if (!len) return 0;
        const size_t ea = (size_t)(f->stable_base - vm_base);
        if (ea + len > 0x100000000ull) return 0;
        raw[n].ea = (u32)ea;
        raw[n].len = len;
        n++;
    }
    for (u32 i = 1; i < n; i++) {
        eng_mc_span tmp = raw[i];
        u32 j = i;
        while (j > 0 && raw[j - 1].ea > tmp.ea) { raw[j] = raw[j - 1]; j--; }
        raw[j] = tmp;
    }
    u32 m = 0;
    for (u32 i = 0; i < n; i++) {
        if (m > 0) {
            const u64 prev_end = (u64)out[m - 1].ea + out[m - 1].len;
            const u64 cur_end = (u64)raw[i].ea + raw[i].len;
            if ((u64)raw[i].ea <= prev_end) {
                if (cur_end > prev_end) out[m - 1].len = (u32)(cur_end - out[m - 1].ea);
                continue;
            }
        }
        if (m >= ENG_MC_MAX_SPANS) {
            /* Merge small gaps (same or adjacent page) if a multi-batch draw has many runs. */
            u32 best = 0;
            u64 best_gap = 0xFFFFFFFFFFFFFFFFull;
            for (u32 k = 1; k < m; k++) {
                const u64 gap = (u64)out[k].ea - ((u64)out[k - 1].ea + out[k - 1].len);
                if (gap < best_gap) { best_gap = gap; best = k; }
            }
            if (best_gap > 4096u) return 0;
            const u64 end_k = (u64)out[best].ea + out[best].len;
            out[best - 1].len = (u32)(end_k - out[best - 1].ea);
            for (u32 k = best; k + 1 < m; k++) out[k] = out[k + 1];
            m--;
        }
        out[m++] = raw[i];
    }
    for (u32 i = 0; i < m; i++) total_bytes += out[i].len;
    if (total_bytes > ENG_VC_HASH_MAX) return 0;
    *out_n = m;
    return 1;
}

static void eng_mc_store(int indexed, u32 n_draw, const u32* indices)
{
    if (!dc.mc_elig || dc.mc_hit || dc.vc_ent == ENG_VC_NONE || !dc.vc_key) return;
    if (!dc.n_verts || !n_draw || !dc.fetch_ok) return;
    eng_mc_span spans[ENG_MC_MAX_SPANS];
    u32 n_spans = 0;
    if (!eng_mc_collect_spans(spans, &n_spans)) return;
    const u32 set = (u32)(dc.mc_desc_lo & (ENG_MC_SETS - 1u));
    const u32 base_idx = set * ENG_MC_WAYS;
    eng_mcent* base = &g.mc_ents[base_idx];
    const eng_mctag* tbase = &g.mc_tags[base_idx];
    eng_mcent* dst = NULL;
    for (u32 w = 0; w < ENG_MC_WAYS; w++) {
        eng_mcent* e = &base[w];
        if (e->valid && tbase[w].lo == dc.mc_desc_lo &&
            tbase[w].hi == dc.mc_desc_hi && e->desc_bytes == dc.mc_desc_bytes) {
            dst = e;
            break;
        }
    }
    if (!dst) {
        for (u32 w = 0; w < ENG_MC_WAYS; w++)
            if (!base[w].valid) { dst = &base[w]; break; }
    }
    if (!dst) {
        dst = &base[0];
        const u16 f16 = (u16)g.frames;
        u16 best_age = (u16)(f16 - dst->last_use_16);
        for (u32 w = 1; w < ENG_MC_WAYS; w++) {
            const u16 age = (u16)(f16 - base[w].last_use_16);
            if (age > best_age) { best_age = age; dst = &base[w]; }
        }
        dst->always_hash = 0;
    }
    if (indexed) {
        const u32 ibytes = n_draw * 4u;
        if (dst->idx_cap < ibytes) {
            u32* nb = (u32*)realloc(dst->idx_buf, ibytes);
            if (!nb) return;
            dst->idx_buf = nb;
            dst->idx_cap = ibytes;
        }
        memcpy(dst->idx_buf, indices, ibytes);
        dst->idx_serial = ++g.mc_idx_serial;
    } else {
        dst->idx_serial = 0u;
    }
    if (n_spans > ENG_MC_INLINE_SPANS) {
        eng_mc_span* es = (eng_mc_span*)realloc(dst->extra_spans, n_spans * sizeof(eng_mc_span));
        if (!es) return;
        dst->extra_spans = es;
        memcpy(dst->extra_spans, spans, n_spans * sizeof(eng_mc_span));
    } else {
        if (dst->extra_spans) { free(dst->extra_spans); dst->extra_spans = NULL; }
        if (n_spans) memcpy(dst->spans, spans, n_spans * sizeof(eng_mc_span));
    }
    u32 clo = 0xFFFFu, chi = 0u;
    for (u32 i = 0; i < n_spans; i++) {
        const u32 ca = spans[i].ea >> 20;
        const u32 cb = (u32)(((u64)spans[i].ea + spans[i].len - 1u) >> 20);
        if (ca < clo) clo = ca;
        if (cb > chi) chi = cb;
    }
    dst->coarse_lo = (u16)(clo < 4096u ? clo : 4096u);
    dst->coarse_hi = (u16)(chi < 4096u ? chi : 4096u);
    dst->coarse_single = (u8)(n_spans > 0 && clo == chi && clo < 4096u);
    const eng_vcent* vx = &g.vc_ents[dc.vc_ent];
    /* Keep cold-miss draws on the per-frame staging ring so 1-frame dynamic
     * meshes do not pollute the persistent ReBAR GPU vertex/index caches.
     * Static meshes promote to ReBAR on their second frame (mc_hit). */
    dc.cur_vid = 0u;
    dc.cur_iid = 0u;
    dc.cur_gcache = NULL;
    memset(&dst->gcache, 0, sizeof dst->gcache);
    dst->desc_bytes = dc.mc_desc_bytes;
    dst->n_spans = (u16)n_spans;
    dst->mem_hash = eng_mc_hash_spans(spans, n_spans);
    dst->vc_key = dc.vc_key;
    dst->vc_serial = vx->serial;
    dst->vc_ent = dc.vc_ent;
    dst->vc_evict_gen = g.vc_evict_gen;
    dst->vc_promoted = 0;
    dst->verts_off = vx->off;
    dst->verts_bytes = vx->bytes;
    dst->n_verts = dc.n_verts;
    dst->n_source_refs = dc.n_source_refs;
    dst->n_draw = n_draw;
    dst->indexed = (u8)(indexed != 0);
    dst->verified_frame = dc.mc_snap_frame;
    dst->last_use_frame = g.frames;
    dst->last_use_16 = (u16)g.frames;
    dst->next_slot = 0xFFFFFFFFu;
    dst->nonspu_checked_gen = 0u;
    dst->valid = 1;
    const u32 slot_idx = (u32)(dst - g.mc_ents);
    g.mc_tags[slot_idx].lo = dc.mc_desc_lo;
    g.mc_tags[slot_idx].hi = dc.mc_desc_hi;
}

/* Immediate vertices are already decoded: pack only the attributes the guest
 * vertex program reads, without a guest-memory fetch or deduplication. */
static void dc_imm_build(const rsx_vertex_layout_plan* layout)
{
    dc.fetch_ok = 0;
    if (!dc_reserve_verts((u64)dc.n_imm * layout->stride)) return;
    for (u32 v = 0; v < dc.n_imm; v++)
        for (u32 slot = 0; slot < layout->count; slot++) {
            const u32 attr = layout->attrs[slot];
            float* value = (float*)(dc.verts + (u64)v * layout->stride + slot * 16u);
            memcpy(value, dc.imm[v][attr], 16);
        }
    dc.layout = *layout;
    dc.n_refs = dc.n_source_refs = dc.n_verts = dc.n_imm;
    dc.refs_remapped = 0;
    dc.n_cuts = 0;
    dc.fetch_ok = 1;
}

static int dc_reserve_verts(u64 bytes)
{
    if (bytes <= dc.verts_cap) return 1;
    u64 next = dc.verts_cap ? dc.verts_cap : (1u << 20);
    while (next < bytes) next *= 2u;
    u8* v = (u8*)realloc(dc.verts, (size_t)next);
    if (!v) return 0;
    dc.verts = v;
    dc.verts_cap = next;
    return 1;
}

static int dc_reserve_indices(u32 count)
{
    if (count <= g.index_cap) return 1;
    u32 next = g.index_cap ? g.index_cap : 4096u;
    while (next < count) next *= 2u;
    u32* p = (u32*)realloc(g.indices, (size_t)next * sizeof(u32));
    if (!p) return 0;
    g.indices = p;
    g.index_cap = next;
    return 1;
}

/* Turn this group's batches into a reference list, recording a CUT rather
 * than fetching a phantom vertex wherever the restart sentinel appears, then
 * collapse repeated references and fetch each unique vertex once.
 * rsx_live_draw.c's fetch_batches_hoisted (4165-4292). */
static int eng_mc_admit2(u64 lo, u64 hi)
{
    if (!g.mc_seen || !s_mc_swd) return 1;
    const u64 k = lo ^ (hi * 0x9E3779B97F4A7C15ull);
    const u64 tag = k ? k : 1ull;
    const u32 set = (u32)((tag * 0xBF58476D1CE4E5B9ull) >> 16) & 65535u;
    u64* slot = &g.mc_seen[set * 2u];
    if (slot[0] == tag || slot[1] == tag) return 1;
    slot[1] = slot[0];
    slot[0] = tag;
    return 0;
}

static void dc_fetch(u32 prim, const rsx_vertex_layout_plan* layout, int allow_remap, int mc_already_missed)
{
    const int timing = eng_stats_on();
    const double t_fetch = timing ? eng_now_ms() : 0.0;
    if (!mc_already_missed && eng_mc_try_lookup(prim, layout, allow_remap)) {
        if (timing) g.st_ms_vckey += eng_now_ms() - t_fetch;
        return;
    }
    {
        u64 need_refs = 0;
        for (u32 b = 0; b < dc.n_arr; b++) need_refs += dc.arr[b].count;
        for (u32 b = 0; b < dc.n_idx; b++) need_refs += dc.idx[b].count;
        if (need_refs > dc.cap_refs) {
            if (need_refs > 0x10000000ull) { dc.fetch_ok = 0; return; }
            u32 next = dc.cap_refs ? dc.cap_refs : 4096u;
            while ((u64)next < need_refs) next *= 2u;
            rsx_vertex_ref* r = (rsx_vertex_ref*)realloc(dc.refs, (size_t)next * sizeof(*r));
            if (!r) { dc.fetch_ok = 0; return; }
            dc.refs = r;
            dc.cap_refs = next;
        }
    }
    for (u32 b = 0; b < dc.n_arr && dc.fetch_ok; b++) {
        const u32 first = dc.arr[b].first, count = dc.arr[b].count;
        rsx_vertex_ref* dst = &dc.refs[dc.n_refs];
        for (u32 i = 0; i < count; i++) {
            dst[i].vertex_id = first + i;
            dst[i].base_index = 0;
        }
        dc.n_refs += count;
    }

    if (dc.n_idx && dc.fetch_ok) {
        const u32 base_index = rsx_dsp_vertex_data_base_index(&g.rsx);
        rsx_dsp_index_array ia;
        rsx_dsp_get_index_array(&g.rsx, &ia);
        const u32 esz = ia.is_u32 ? 4u : 2u;
        const int restart_en = rsx_dsp_restart_index_enabled(&g.rsx, ia.is_u32);
        const u32 restart_val = rsx_dsp_restart_index(&g.rsx);
        /* The index array's location field is a DMA selector, not the
         * rsx_dsp location enum: 0 selects local memory. */
        const u32 ia_loc = ia.location ? RSX_LOCATION_MAIN : RSX_LOCATION_LOCAL;

        for (u32 b = 0; b < dc.n_idx && dc.fetch_ok; b++) {
            const u32 first = dc.idx[b].first, count = dc.idx[b].count;
            const u64 start = (u64)ia.offset + (u64)first * esz;
            const u64 bytes = (u64)count * esz;
            const u8* run = NULL;
            if (count && start <= 0xFFFFFFFFull && bytes <= 0xFFFFFFFFull &&
                start + bytes <= 0x100000000ull)
                run = eng_guest_ptr(NULL, ia_loc, (u32)start, (u32)bytes);
            if (run && !restart_en) {
                rsx_vertex_ref* dst = &dc.refs[dc.n_refs];
                if (ia.is_u32) {
                    for (u32 i = 0; i < count; i++) {
                        u32 w;
                        memcpy(&w, run + (size_t)i * 4u, 4);
                        dst[i].vertex_id = __builtin_bswap32(w);
                        dst[i].base_index = base_index;
                    }
                } else {
                    for (u32 i = 0; i < count; i++) {
                        dst[i].vertex_id = (u32)((run[(size_t)i * 2u] << 8) | run[(size_t)i * 2u + 1u]);
                        dst[i].base_index = base_index;
                    }
                }
                dc.n_refs += count;
            } else {
                for (u32 i = 0; i < count && dc.fetch_ok; i++) {
                    const u8* p = run ? run + (size_t)i * esz
                        : eng_guest_ptr(NULL, ia_loc, ia.offset + (first + i) * esz, esz);
                    if (!p) { dc.fetch_ok = 0; break; }
                    const u32 index = ia.is_u32
                        ? (((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | p[3])
                        : (u32)((p[0] << 8) | p[1]);
                    if (restart_en && index == restart_val) {
                        if (!rsx_restart_cut_push(&dc.cuts, &dc.n_cuts, &dc.cap_cuts,
                                                  dc.n_refs))
                            dc.fetch_ok = 0;
                        continue;
                    }
                    dc.refs[dc.n_refs].vertex_id = index;
                    dc.refs[dc.n_refs].base_index = base_index;
                    dc.n_refs++;
                }
            }
        }
    }
    if (!dc.fetch_ok) return;

    dc.n_source_refs = dc.n_refs;
    /* Only a topology this engine rebuilds through an index buffer may share
     * repeated references. A line strip or a point list is drawn in the order
     * it arrived, so collapsing duplicates would reorder it. A single
     * contiguous DRAW_ARRAYS batch is already strictly unique. */
    if (allow_remap && dc.n_refs > 1 && !(dc.n_idx == 0 && dc.n_arr == 1)) {
        u32 unique = dc.n_refs;
        if (rsx_vertex_remap_build(&dc.ref_remap, dc.refs, dc.n_refs, &unique)) {
            dc.refs_remapped = unique < dc.n_refs;
            dc.n_refs = unique;
        }
    }

    if (timing) { g.st_ms_vcrefs += eng_now_ms() - t_fetch; }
    double t_stage = timing ? eng_now_ms() : 0.0;
    dc.layout = *layout;
    rsx_vertex_fetch_plan_init(&dc.fetch_plan, &g.rsx, layout, eng_guest_ptr, NULL);
    rsx_vertex_fetch_plan_set_inline(&dc.fetch_plan, &g.rsx, dc.inl, dc.inl_bytes);
    rsx_vertex_fetch_plan_prepare(&dc.fetch_plan, dc.refs, dc.n_refs);

    /* Converted-vertex cache: the plan is prepared by now, so the key can see
     * the spans the fetch would read. A hit skips the conversion entirely and
     * hands the backend the block stored for this exact plan, reference list
     * and guest content. */
    const int vc = eng_vcache_use();
    u64 vc_key = 0;   /* 0 = this group is not cacheable */
    int skip_vc = 0;
    if (vc && dc.n_refs && s_ef.m2f && dc.mc_elig && dc.mc_was_cold) {
        if (!eng_mc_admit2(dc.mc_desc_lo, dc.mc_desc_hi)) {
            skip_vc = 1;
        } else {
            dc.mc_seen_hit = 1;
        }
    }
    if (vc && dc.n_refs && !skip_vc) {
        if (eng_vcache_key(&dc.fetch_plan, dc.refs, dc.n_refs, &vc_key)) {
            u32 hit_ent = ENG_VC_NONE;
            const u8* hit = eng_vcache_hit(vc_key, &hit_ent);
            if (hit) {
                dc.verts_cache = hit;
                dc.vc_key = vc_key;
                dc.vc_ent = hit_ent;
                dc.cur_vid = g.vc_ents[hit_ent].serial;
                dc.n_verts = dc.n_refs;
                g.vc_hits++;
                g.vc_hit_verts += dc.n_refs;
                if (timing) g.st_ms_vckey += eng_now_ms() - t_stage;
                return;
            }
        }
    }

    if (timing) { const double now = eng_now_ms();
                  g.st_ms_vckey += now - t_stage; t_stage = now; }

    if (!dc_reserve_verts((u64)dc.n_refs * layout->stride)) {
        dc.fetch_ok = 0;
        return;
    }
    if (!rsx_vertex_fetch_all(&dc.fetch_plan, dc.refs, dc.n_refs, dc.verts)) {
        dc.fetch_ok = 0;
        return;
    }
    if (timing) { const double now = eng_now_ms();
                  g.st_ms_vcconvert += now - t_stage; }
    dc.n_verts = dc.n_refs;
    if (vc) {
        g.vc_misses++;
        g.vc_miss_verts += dc.n_refs;
        const u64 bytes = (u64)dc.n_refs * layout->stride;
        if (!vc_key) {
            if (!skip_vc) g.vc_unkeyed++;
        } else if (bytes <= (1024ull << (ENG_VC_CLASSES - 1u)) &&
                   (dc.mc_seen_hit || eng_vc_admit2(vc_key))) {
            const int stored = eng_vcache_store(vc_key, dc.verts, (u32)bytes);
            if (stored >= 0) { dc.vc_key = vc_key; dc.vc_ent = (u32)stored; dc.cur_vid = 0u; }
        }
    }
}

/* ---- sink ---------------------------------------------------------------- */



static void sink_begin(void* user, const rsx_dispatch* r, u32 prim)
{
    (void)user; (void)r; (void)prim;
    dc_reset();
}

static void sink_draw_arrays(void* user, const rsx_dispatch* r, u32 first, u32 count)
{
    (void)user; (void)r;
    dc.n_packets++;
    if (dc.n_arr >= ENG_MAX_BATCHES) return;
    dc.arr[dc.n_arr].first = first;
    dc.arr[dc.n_arr].count = count;
    dc.n_arr++;
}

/* INLINE_ARRAY delivers one finished stream rather than a (first, count)
 * range; queue it as an ordinary consecutive batch and let the fetch plan
 * read the vertices out of it (rsx_vertex_fetch_plan_set_inline). */
static void sink_inline_array(void* user, const rsx_dispatch* r,
                              const u8* data, u32 bytes)
{
    (void)user;
    const u32 stride = rsx_vertex_inline_layout(r, NULL);
    if (!stride || bytes < stride) return;
    dc.inl = data;
    dc.inl_bytes = bytes;
    dc.n_packets++;
    if (dc.n_arr >= ENG_MAX_BATCHES) return;
    dc.arr[dc.n_arr].first = 0;
    dc.arr[dc.n_arr].count = bytes / stride;
    dc.n_arr++;
}

static void sink_draw_index(void* user, const rsx_dispatch* r, u32 first, u32 count)
{
    (void)user; (void)r;
    dc.n_packets++;
    if (dc.n_idx >= ENG_MAX_BATCHES) return;
    dc.idx[dc.n_idx].first = first;
    dc.idx[dc.n_idx].count = count;
    dc.n_idx++;
}

/* ENG_FEEDBACK=0 drops the copies an in-place pass samples its own target
 * through and reads guest memory instead (the pre-fix behaviour);
 * ENG_FEEDBACK_PHASE=<0|1> does that only on one A/B parity, so the copy's
 * cost and its effect on the picture can be read from the phases of ONE run. */
static int eng_feedback_use(void)
{
    static int cfg = -1, phase = -1;
    if (cfg < 0) {
        const char* e = getenv("ENG_FEEDBACK");
        const char* p = getenv("ENG_FEEDBACK_PHASE");
        cfg = (e && *e == '0') ? 0 : 1;
        if (p && *p) phase = atoi(p) & 1;
    }
    if (!cfg) return 0;
    return phase < 0 || (ENG_AB_PHASE() & 1) == phase;
}

/* ENG_FEEDBACK_RUN=<N> (fallback FEEDBACK_RUN, default 0 = copy before every
 * feedback draw, exact): consecutive feedback draws into the same target
 * (inFamous G-buffer decals: ~170/frame) refresh the snapshot only every N
 * draws. Any non-feedback draw or clear in between breaks the run.
 * ENG_FEEDBACK_RUN_PHASE=<0|1> (or FEEDBACK_RUN_PHASE) enables in-run A/B. */
static u32 eng_feedback_run_max(void)
{
    static int cfg = -1, phase = -1;
    if (cfg < 0) {
        const char* e = getenv("ENG_FEEDBACK_RUN");
        if (!e || !*e) e = getenv("FEEDBACK_RUN");
        const char* p = getenv("ENG_FEEDBACK_RUN_PHASE");
        if (!p || !*p) p = getenv("FEEDBACK_RUN_PHASE");
        cfg = (e && *e) ? atoi(e) : 0;
        if (cfg < 0) cfg = 0;
        if (p && *p) phase = atoi(p) & 1;
        if (cfg > 1)
            fprintf(stderr, "[ENG] feedback snapshot run batching: every %d consecutive draw(s), A/B parity %d\n",
                    cfg, phase);
    }
    if (cfg <= 1) return 0u;
    if (phase >= 0 && (ENG_AB_PHASE() & 1) != phase) return 0u;
    return (u32)cfg;
}

/* Which texture units this draw binds, and from where: a unit naming a
 * registered colour surface samples that live target rather than guest
 * memory (every post-process, shadow and reflection in a PS3 title is a
 * render-to-texture read back through a texture unit), a unit naming one of
 * the targets THIS draw writes samples a copy of it taken just before the
 * draw, a unit naming a tracked zeta in DEPTH24_D8 samples its snapshot, and
 * everything else is a guest upload (rsx_live_draw.c:5986-6062). */
static u32 eng_shadow_lit_tex(void)
{
    if (g.shadow_lit_tex) return g.shadow_lit_tex;
    if (!g.be->texture_create || !g.be->texture_upload) return 0;
    const u32 h = g.be->texture_create(g.be->user, RSX_BE_FMT_R8G8B8A8,
                                       1, 1, 1, 1, 0xAAE4u, RSX_TEX_FMT_A8R8G8B8);
    if (!h) return 0;
    static const u8 white[4] = { 0xFFu, 0xFFu, 0xFFu, 0xFFu };
    g.be->texture_upload(g.be->user, h, 0, 0, 1, 1, white, 4, 1);
    g.shadow_lit_tex = h;
    return h;
}

/* sp00nznet/ps3recomp #814: PS3RECOMP_ENG_FEEDBACK_PER_PASS (default 1 after
 * Deck 15 W +0.51 fps / BC-250 +1.53 fps A/B win; =0 disables): own-target
 * samples in one unchanged pass (colour set + zeta) share one snapshot instead
 * of copying the full target per draw. */
static int eng_feedback_per_pass(void)
{
    static int on = -1;
    if (__builtin_expect(on < 0, 0)) {
        const char* e = getenv("PS3RECOMP_ENG_FEEDBACK_PER_PASS");
        on = (e && *e && atoi(e) == 0) ? 0 : 1;
        if (on) fprintf(stderr, "[rsx engine] PS3RECOMP_ENG_FEEDBACK_PER_PASS=1 (one own-target snapshot per pass)\n");
    }
    return on;
}

static void eng_pass_note(const u32* targets, u32 n_targets)
{
    rsx_dsp_surface sf;
    rsx_dsp_get_surface(&g.rsx, &sf);
    int same = g.pass_serial && g.pass_n == n_targets &&
               g.pass_zloc == sf.zeta_location && g.pass_zoff == sf.zeta_offset;
    for (u32 k = 0; same && k < n_targets; k++) same = (g.pass_targets[k] == targets[k]);
    if (same) return;
    g.pass_serial++;
    g.pass_n = n_targets;
    memcpy(g.pass_targets, targets, n_targets * sizeof(u32));
    g.pass_zloc = sf.zeta_location;
    g.pass_zoff = sf.zeta_offset;
}

static int eng_feedback_shared(const eng_surface* s)
{
    return eng_feedback_per_pass() && s->feedback && s->feedback_pass &&
           s->feedback_pass == g.pass_serial;
}

static u32 sink_bind_textures(const u32* target_slots, u32 n_targets,
                              u32 current_zslot,
                              u32 textures[RSX_BE_MAX_TEXTURES],
                              u64* out_sh_key, u8* out_had_dynamic,
                              u32 only_units_mask)
{
    u32 mask = 0;
    u64 sh_key = 0;
    u8 had_dynamic = 0;
    eng_pass_note(target_slots, n_targets);
    const int fast_tex = (s_ef.phase >= 0) ? (int)s_ef.m2c : eng_m2c_use();
    const int m2d = (s_ef.phase >= 0) ? (int)s_ef.m2d : eng_m2d_use();
    const int m2e4b = (s_ef.phase >= 0) ? (int)s_ef.m2e4b : eng_m2e4b_use();
    const int partial = (only_units_mask != 0xFFFFFFFFu);
    u32 iter = partial ? only_units_mask : (m2d ? s_enabled_tex_mask : 0xFFFFu);
    while (iter) {
        const u32 u = (u32)__builtin_ctz(iter);
        iter &= iter - 1u;
        const rsx_dsp_texture* t = &s_units[u];   /* eng_units_decode, once a draw */
        if (!t->enabled) continue;
        mask |= 1u << u;
        g.cur_sh_mask &= ~(1u << u);
        eng_unit_cache* uc = &s_ucache[u];
        if (fast_tex && uc->valid) {
            if (uc->kind == 4u && uc->frame == g.frames &&
                uc->surface_gen == g.surface_gen && uc->tex_gen == g.tex_gen) {
                textures[u] = uc->handle;
                s_tunit[u] = 4;
                continue;
            }
            if (uc->kind == 1u && uc->surface_gen == g.surface_gen && uc->slot < g.n_surfaces) {
                int is_own = 0;
                for (u32 k = 0; k < n_targets; k++)
                    if (target_slots[k] == uc->slot) { is_own = 1; break; }
                if (!is_own) {
                    textures[u] = uc->handle;
                    s_tunit[u] = 1;
                    continue;
                }
            }
            if (uc->kind == 3u && uc->surface_gen == g.surface_gen &&
                uc->slot != current_zslot && uc->slot < g.n_zdepths &&
                g.zdepths[uc->slot].snapshot_valid && g.zdepths[uc->slot].snapshot == uc->handle) {
                textures[u] = uc->handle;
                sh_key |= ((u64)uc->sh_nibble) << (u * 4u);
                g.cur_sh_nibble[u] = uc->sh_nibble;
                g.cur_sh_mask |= 1u << u;
                s_tunit[u] = 3;
                continue;
            }
        }
        if (m2d) {
            const u32 tf_base = eng_tex_fast_idx(t->offset, t->format, t->width, t->height, t->remap);
            const u32 tf_ways = m2e4b ? ENG_TEX_FAST_WAYS : 1u;
            eng_tex_fast_ent* tf_hit_ent = NULL;
            for (u32 w = 0; w < tf_ways; w++) {
                eng_tex_fast_ent* tfe = &s_tex_fast[tf_base + w];
                if (tfe->valid &&
                    tfe->surface_gen == g.surface_gen && tfe->tex_gen == g.tex_gen &&
                    tfe->offset == t->offset && tfe->location == t->location &&
                    tfe->format == t->format && tfe->width == t->width &&
                    tfe->height == t->height && tfe->levels == t->mipmaps &&
                    tfe->pitch == t->pitch && tfe->remap == (t->remap & 0xFFFFu) &&
                    tfe->cubemap == (u8)(t->cubemap != 0)) {
                    int tf_hit = (tfe->frame == g.frames);
                    if (!tf_hit && (s_ef.phase >= 0 ? (int)s_ef.m2e4 : eng_m2e4_use()) && s_mc_swd && tfe->swd_ok) {
                        const u32 tver = s_ef.m2f ? 256u : 64u;
                        if ((u32)(g.frames - tfe->full_hash_frame) < tver &&
                            (g.frames & (tver - 1u)) != ((tf_base + w) & (tver - 1u))) {
                            int clean = 1;
                            const u32 vgen = tfe->swd_ver_frame;
                            for (u32 c = tfe->coarse_lo; c <= tfe->coarse_hi; c++) {
                                const u32 cg = __atomic_load_n(&s_swd_coarse[c], __ATOMIC_RELAXED);
                                if (cg && cg + 1u >= vgen) { clean = 0; break; }
                            }
                            if (clean) {
                                tfe->frame = g.frames;
                                tf_hit = 1;
                            }
                        }
                    }
                    if (tf_hit) { tf_hit_ent = tfe; break; }
                }
            }
            if (tf_hit_ent) {
                textures[u] = tf_hit_ent->handle;
                s_tunit[u] = 4;
                uc->handle = tf_hit_ent->handle;
                uc->frame = g.frames;
                uc->surface_gen = g.surface_gen;
                uc->tex_gen = g.tex_gen;
                uc->kind = 4u;
                uc->valid = 1;
                continue;
            }
        }

        const u32 base_fmt = t->format & RSX_TEX_FMT_BASE_MASK & ~(u32)RSX_TEX_FMT_UNNORM;
        const int is_depth_fmt = (base_fmt >= 0x90u && base_fmt <= 0x93u);
        if (is_depth_fmt && !t->cubemap) {
            if (g.shadows_on) {
                const u32 z = eng_zdepth_find_sized(t->location, t->offset,
                                                    t->width, t->height);
                if (z < ENG_MAX_ZDEPTHS && current_zslot != z) {
                    const u32 snap = eng_zdepth_snapshot(z);
                    if (snap) {
                        textures[u] = snap;
                        const u32 zf = (t->wrap >> 28) & 7u;
                        const u8 nib = (u8)(8u | zf);
                        sh_key |= ((u64)nib) << (u * 4u);
                        g.cur_sh_nibble[u] = nib;
                        g.cur_sh_mask |= 1u << u;
                        s_tunit[u] = 3;
                        uc->handle = snap;
                        uc->slot = (u16)z;
                        uc->surface_gen = g.surface_gen;
                        uc->sh_nibble = nib;
                        uc->kind = 3u;
                        uc->valid = 1;
                        continue;
                    }
                    had_dynamic = 1;
                }
            }
            if (g.shadow_lit_on) {
                const u32 lit = eng_shadow_lit_tex();
                if (lit) {
                    textures[u] = lit;
                    s_tunit[u] = 3;
                    continue;
                }
            }
            had_dynamic = 1;
        }
        int sampled = is_depth_fmt ? -1
            : (int)eng_surface_find_sized(t->location, t->offset, t->width,
                                          t->height);
        if (sampled < 0 || sampled == (int)ENG_MAX_SURFACES || !g.surfaces[sampled].handle)
            sampled = -1;
        /* A zeta surface sampled as A8R8G8B8 (0x85): deferred lighting reads
         * 24-bit depth (.wxy) and 8-bit stencil (.z) out of the depth target.
         * A colour target that has actually had colour drawn into it wins;
         * an address with no colour target (or only a depth-only cwritten=0
         * target, like half-res 0x0F340000) resolves to the converted zeta. */
        if (!is_depth_fmt && !t->cubemap && base_fmt == RSX_TEX_FMT_A8R8G8B8 &&
            g.zconv_color_on &&
            (sampled < 0 || (g.zs_cwritten_on && !g.surfaces[sampled].cwritten))) {
            const u32 z = eng_zdepth_find_sized(t->location, t->offset,
                                                t->width, t->height);
            if (z < ENG_MAX_ZDEPTHS) {
                if (z == current_zslot) had_dynamic = 1;
                const u32 csnap = eng_zdepth_color_snapshot(z);
                if (csnap) {
                    const u32 view = g.be->surface_view
                        ? g.be->surface_view(g.be->user, csnap,
                                             t->remap & 0xFFFFu, t->format)
                        : 0;
                    textures[u] = view ? view : csnap;
                    s_tunit[u] = 3;
                    continue;
                }
                had_dynamic = 1;
            }
        }
        /* A unit naming the very target this draw writes. Reading the image
         * the same draw renders into is undefined on every host API, and an
         * MRT set writes more than one attachment, so what it gets is a copy
         * of the target (color_snapshot) or, without one, the guest's bytes
         * at the address. */
        int own = -1;
        if (sampled >= 0)
            for (u32 k = 0; k < n_targets; k++)
                if (target_slots[k] == (u32)sampled) own = (int)target_slots[k];
        if (own >= 0) {
            eng_surface* s = &g.surfaces[own];
            /* The backend owns the copy and refills it in command order, the
             * way depth_snapshot does, so the engine neither holds a second
             * one nor releases it. */
            const int shared = eng_feedback_shared(s);
            if (!shared) had_dynamic = 1;
            const u32 fbn = eng_feedback_run_max();
            const u32 seq = g.draw_seq;
            const int same_draw = (s->feedback_frame == g.frames && s->feedback_last_draw == seq);
            int run_skip = shared;
            if (!run_skip && fbn > 1u && s->feedback && s->feedback_frame == g.frames) {
                if (same_draw) {
                    run_skip = 1;
                } else if (s->feedback_last_draw + 1u == seq && s->feedback_runlen + 1u < fbn) {
                    run_skip = 1;
                    s->feedback_runlen++;
                }
            }
            if (eng_feedback_use() && g.be->color_snapshot && !run_skip &&
                (!m2d || !s->feedback_valid || !s->feedback)) {
                const u32 copy = g.be->color_snapshot(g.be->user, s->handle);
                if (copy) {
                    s->feedback = copy;
                    s->feedback_valid = 1;
                    s->feedback_pass = g.pass_serial;
                    g.st_feedback++;
                }
                if (!same_draw) s->feedback_runlen = 0u;
            }
            s->feedback_last_draw = seq;
            s->feedback_frame = g.frames;
            if (s->feedback) {
                const u32 view = g.be->surface_view
                    ? g.be->surface_view(g.be->user, s->feedback,
                                         t->remap & 0xFFFFu, t->format)
                    : 0;
                textures[u] = view ? view : s->feedback;
                s_tunit[u] = 2;
                if (textures[u]) continue;
            }
            sampled = -1;
        }
        if (sampled >= 0) {
            u32 h = 0;
            const u32 rmp = t->remap & 0xFFFFu;
            if (m2e4b) {
                const u32 rvi = (((u32)sampled * 31u) ^ (t->format * 17u) ^ rmp) & (ENG_RT_VIEW_CACHE - 1u);
                eng_rt_view_ent* rve = &s_rt_view_cache[rvi];
                if (rve->valid && rve->surface_gen == g.surface_gen &&
                    rve->slot == (u32)sampled && rve->format == t->format && rve->remap == rmp) {
                    h = rve->handle;
                } else {
                    const u32 view = g.be->surface_view
                        ? g.be->surface_view(g.be->user, g.surfaces[sampled].handle, rmp, t->format)
                        : 0;
                    h = view ? view : g.surfaces[sampled].handle;
                    if (h) {
                        rve->surface_gen = g.surface_gen;
                        rve->slot = (u32)sampled;
                        rve->format = t->format;
                        rve->remap = rmp;
                        rve->handle = h;
                        rve->valid = 1;
                    }
                }
            } else {
                const u32 view = g.be->surface_view
                    ? g.be->surface_view(g.be->user, g.surfaces[sampled].handle, rmp, t->format)
                    : 0;
                h = view ? view : g.surfaces[sampled].handle;
            }
            textures[u] = h;
            s_tunit[u] = 1;
            if (textures[u]) {
                uc->handle = textures[u];
                uc->slot = (u16)sampled;
                uc->surface_gen = g.surface_gen;
                uc->kind = 1u;
                uc->valid = 1;
            }
            continue;
        }
        textures[u] = eng_texture_slot(t->location, t->offset, t->format,
                                       t->width, t->height, t->mipmaps, t->pitch,
                                       (int)t->cubemap, t->remap & 0xFFFFu);
        s_tunit[u] = 4;
        if (textures[u]) {
            uc->handle = textures[u];
            uc->frame = g.frames;
            uc->surface_gen = g.surface_gen;
            uc->tex_gen = g.tex_gen;
            uc->kind = 4u;
            uc->valid = 1;
            if (m2d && !is_depth_fmt) {
                const u32 tf_base = eng_tex_fast_idx(t->offset, t->format, t->width, t->height, t->remap);
                u32 dst_w = 0;
                if (m2e4b) {
                    u32 match_w = ENG_TEX_FAST_WAYS, empty_w = ENG_TEX_FAST_WAYS, oldest_w = 0;
                    u32 oldest_fr = s_tex_fast[tf_base].frame;
                    const u32 rmp16 = t->remap & 0xFFFFu;
                    for (u32 w = 0; w < ENG_TEX_FAST_WAYS; w++) {
                        eng_tex_fast_ent* ew = &s_tex_fast[tf_base + w];
                        if (ew->valid && ew->offset == t->offset && ew->format == t->format &&
                            ew->width == t->width && ew->height == t->height && ew->remap == rmp16) {
                            match_w = w; break;
                        }
                        if (!ew->valid && empty_w == ENG_TEX_FAST_WAYS) empty_w = w;
                        if (ew->valid && ew->frame < oldest_fr) { oldest_fr = ew->frame; oldest_w = w; }
                    }
                    if (match_w < ENG_TEX_FAST_WAYS) dst_w = match_w;
                    else if (empty_w < ENG_TEX_FAST_WAYS) dst_w = empty_w;
                    else dst_w = oldest_w;
                }
                eng_tex_fast_ent* tfe = &s_tex_fast[tf_base + dst_w];
                tfe->location = t->location; tfe->offset = t->offset;
                tfe->format = t->format; tfe->width = t->width;
                tfe->height = t->height; tfe->levels = t->mipmaps;
                tfe->pitch = t->pitch; tfe->remap = t->remap & 0xFFFFu;
                tfe->cubemap = (u8)(t->cubemap != 0);
                tfe->frame = g.frames; tfe->surface_gen = g.surface_gen;
                tfe->tex_gen = g.tex_gen; tfe->handle = textures[u];
                tfe->swd_ok = s_last_tex_swd_ok;
                tfe->swd_ver_frame = s_last_tex_swd_ver;
                tfe->full_hash_frame = s_last_tex_full_frame;
                tfe->coarse_lo = s_last_tex_clo;
                tfe->coarse_hi = s_last_tex_chi;
                tfe->valid = 1;
            }
        } else {
            had_dynamic = 1;
        }
    }
    if (out_sh_key) *out_sh_key = sh_key;
    if (out_had_dynamic) *out_had_dynamic = had_dynamic;
    return mask;
}

static u32 sink_bind_vertex_textures(
    u32 vtex_mask, u32 textures[RSX_BE_MAX_VERTEX_TEXTURES],
    rsx_be_sampler_desc samplers[RSX_BE_MAX_VERTEX_TEXTURES])
{
    u32 bound = 0;
    for (u32 u = 0; u < RSX_BE_MAX_VERTEX_TEXTURES; u++) {
        if (!((vtex_mask >> u) & 1u)) continue;
        const rsx_dsp_vertex_texture vt = s_vunits[u];
        eng_decode_sampler(vt.filter, vt.wrap, vt.control0, vt.border_color,
                           &samplers[u]);
        /* A vertex unit has no crossbar, so its remap is the identity. */
        textures[u] = eng_texture_slot(vt.location, vt.offset, vt.format,
                                       vt.width, vt.height, vt.mipmaps,
                                       vt.pitch, 0, 0xAAE4u);
        if (textures[u]) bound |= 1u << u;
    }
    return bound;
}

static int eng_trace(void)
{
    static int on = -1;
    if (on < 0) on = getenv("PS3RECOMP_ENG_TRACE") ? 1 : 0;
    return on;
}

static u32 eng_trace_after(void)
{
    static u32 after;
    static int read;
    if (!read) {
        const char* e = getenv("PS3RECOMP_ENG_TRACE_AFTER");
        after = e ? (u32)atoi(e) : 0u;
        read = 1;
    }
    return after;
}

rsx_slowframe_counters g_rsx_sf;
double g_rsx_fifo_prof_ms = 0.0;

int rsx_slowframe_init(void)
{
    static int inited = 0;
    if (!inited) {
        inited = 1;
        const char* e = getenv("SLOWFRAME_MS");
        if (e && *e) {
            double v = atof(e);
            if (v > 0.0) {
                g_rsx_sf.enabled = 1;
                g_rsx_sf.threshold_ms = v;
                const char* md = getenv("SLOWFRAME_MIN_DRAWS");
                g_rsx_sf.min_draws = (md && *md) ? (u32)atoi(md) : 0u;
                fprintf(stderr, "[SLOWFRAME] enabled: threshold=%.1f ms min_draws=%u (per-frame spike log + 300-frame [FRAMEBREAK])\n",
                        v, g_rsx_sf.min_draws);
            }
        }
    }
    return g_rsx_sf.enabled;
}

/* PS3RECOMP_ENG_STATS=1 also turns on the [ENGTIME] phase split. Read once:
 * the walker asks this per BEGIN_END group. */
static int eng_stats_on(void)
{
    static int on = -1;
    if (on < 0) on = (getenv("PS3RECOMP_ENG_STATS") || rsx_slowframe_init()) ? 1 : 0;
    return on;
}

static double eng_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* One BEGIN_END group: fetch, textures, pipeline, binds, draw. The [ENGTIME]
 * split times the five phases inside; the wrapper below owns the group total
 * so every early return still counts. */
/* Same-state consecutive draw fast cache (ENG_M2F): >90% of draws in a frame
 * share identical (prim, program, pipeline, targets, textures, FP constants,
 * viewport, scissor, stencil, depth bounds) with the previous draw and change
 * ONLY the mesh and VP constants. */
static struct {
    u8           valid;
    u8           rebuild;
    u8           fixed;
    rsx_topology topology;
    u32          prim;
    u32          frame;
    u32          vp_gen, vp_start, fp_reg, fp_ctrl;
    u32          pipe_gen, surf_gen, vp_geom_gen, dyn_gen;
    u32          surface_gen, tex_gen, clear_gen, n_zdepths;
    u32          tex_state_gen, prog_epoch;
    eng_program* prog;
    u64          pass_serial;
    u8           upscale_q;
    u32          zslot;
    float        xf[8];
    rsx_upscale_draw_info upscale_di;
} s_draw_fast;

static inline u64 eng_rdtsc(void)
{
#if defined(__x86_64__) || defined(_M_X64)
    u32 lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((u64)hi << 32) | lo;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64)ts.tv_sec * 1000000000ull + (u64)ts.tv_nsec;
#endif
}

static struct {
    int on;
    u32 fr0;
    double ms0;
    u64 t0;
    u64 fast_hits, slow_hits;
    u64 miss_tex, miss_prog, miss_pipe, miss_surf, miss_dyn, miss_mc, miss_other;
    u64 cy_fast_mc, cy_fast_vp, cy_fast_vk;
    u64 cy_slow_prog, cy_slow_fetch, cy_slow_tex, cy_slow_rest, cy_slow_vk;
} s_m2f_p = { .on = -1 };

static void sink_end_impl(void* user, const rsx_dispatch* r)
{
    (void)user; (void)r;
    { const int cur_ph = ENG_AB_PHASE();
      if (__builtin_expect(s_ef.phase != cur_ph, 0)) eng_refresh_flags(cur_ph); }
    if (__builtin_expect(s_m2f_p.on < 0, 0)) {
        const char* e = getenv("ENG_M2F_PROF");
        s_m2f_p.on = (e && *e == '1') ? 1 : 0;
    }
    const int mprof = s_m2f_p.on;
    static u32 trace_groups, trace_logs;
    const int trace = eng_trace() && trace_logs < 64 && g.frames >= eng_trace_after();
    if (trace) trace_logs++;
    const int timing = s_ef.stats;
    const int m2f = s_ef.m2f;
    if (timing) s_phase_ms = eng_now_ms();
    if (trace)
        fprintf(stderr, "[rsx engine] group %u: packets=%u arr=%u idx=%u prim=%u inline=%u immediate=%u\n",
                trace_groups, dc.n_packets, dc.n_arr, dc.n_idx, g.rsx.current_primitive,
                dc.inl_bytes, dc.n_imm);
    trace_groups++;
    const int immediate = !dc.n_packets && dc.n_imm;
    if (!g.ready || (!dc.n_packets && !immediate)) return;

    const u32 prim = g.rsx.current_primitive;
    int mc_already_missed = 0;

    /* ENG_M2F same-state fast path (including tex-only material switches):
     * skips all redundant program/surface/pipeline/FP/RT/dyn checks when only
     * mesh, VP constants, and/or cached guest textures changed. */
    if (__builtin_expect(m2f && !timing && !trace && !immediate && s_draw_fast.valid &&
                         g.bind_dedup_valid && g.tex_bind_valid && !g.last_tex_had_dynamic &&
                         g.last_vtex_mask == 0u && s_units_seeded &&
                         g.rsx.vtex_dirty_mask == 0u &&
                         prim == s_draw_fast.prim &&
                         g.frames == s_draw_fast.frame &&
                         g.rsx.vp_gen == s_draw_fast.vp_gen &&
                         rsx_dsp_vp_start(&g.rsx) == s_draw_fast.vp_start &&
                         rsx_dsp_reg(&g.rsx, 0x08E4) == s_draw_fast.fp_reg &&
                         rsx_dsp_shader_control(&g.rsx) == s_draw_fast.fp_ctrl &&
                         g.rsx.pipe_gen == s_draw_fast.pipe_gen &&
                         g.rsx.surf_gen == s_draw_fast.surf_gen &&
                         g.rsx.vp_geom_gen == s_draw_fast.vp_geom_gen &&
                         g.rsx.dyn_gen == s_draw_fast.dyn_gen &&
                         g.surface_gen == s_draw_fast.surface_gen &&
                         g.tex_gen == s_draw_fast.tex_gen &&
                         g.clear_gen == s_draw_fast.clear_gen &&
                         g.n_zdepths == s_draw_fast.n_zdepths &&
                         s_tex_state_gen == s_draw_fast.tex_state_gen &&
                         s_prog_epoch == s_draw_fast.prog_epoch &&
                         g.pass_serial == s_draw_fast.pass_serial, 1)) {
        int fast_tex_ok = 1;
        if (g.rsx.tex_dirty_mask != 0u) {
            const u32 prev_cube = s_cube_mask;
            const u32 prev_txf = s_tex_xf_key;
            eng_units_decode();
            if (s_cube_mask != prev_cube || s_tex_xf_key != prev_txf || !s_ef.m2d || !s_ef.m2e4b) {
                fast_tex_ok = 0;
            } else {
                const u32 en_mask = s_enabled_tex_mask;
                u32 todo = en_mask & (g.cur_tex_resolve_dirty | ~g.cur_tex_valid_mask);
                while (todo) {
                    const u32 u = (u32)__builtin_ctz(todo);
                    todo &= todo - 1u;
                    const rsx_dsp_texture* t = &s_units[u];
                    const u32 base_fmt = t->format & RSX_TEX_FMT_BASE_MASK & ~(u32)RSX_TEX_FMT_UNNORM;
                    if (base_fmt >= 0x90u && base_fmt <= 0x93u) { fast_tex_ok = 0; break; }
                    const u32 tf_base = eng_tex_fast_idx(t->offset, t->format, t->width, t->height, t->remap);
                    const u32 rmp16 = t->remap & 0xFFFFu;
                    const u8 cm = (u8)(t->cubemap != 0);
                    eng_tex_fast_ent* hit_e = NULL;
                    for (u32 w = 0; w < ENG_TEX_FAST_WAYS; w++) {
                        eng_tex_fast_ent* tfe = &s_tex_fast[tf_base + w];
                        if (tfe->valid &&
                            tfe->surface_gen == g.surface_gen && tfe->tex_gen == g.tex_gen &&
                            tfe->offset == t->offset && tfe->location == t->location &&
                            tfe->format == t->format && tfe->width == t->width &&
                            tfe->height == t->height && tfe->levels == t->mipmaps &&
                            tfe->pitch == t->pitch && tfe->remap == rmp16 && tfe->cubemap == cm) {
                            int ok = (tfe->frame == g.frames);
                            if (!ok && s_ef.m2e4 && s_mc_swd && tfe->swd_ok &&
                                (u32)(g.frames - tfe->full_hash_frame) < 256u &&
                                (g.frames & 255u) != ((tf_base + w) & 255u)) {
                                int clean = 1;
                                const u32 vgen = tfe->swd_ver_frame;
                                for (u32 c = tfe->coarse_lo; c <= tfe->coarse_hi; c++) {
                                    const u32 cg = __atomic_load_n(&s_swd_coarse[c], __ATOMIC_RELAXED);
                                    if (cg && cg + 1u >= vgen) { clean = 0; break; }
                                }
                                if (clean) { tfe->frame = g.frames; ok = 1; }
                            }
                            if (ok) { hit_e = tfe; break; }
                        }
                    }
                    if (!hit_e) { fast_tex_ok = 0; break; }
                    const u32 bit = 1u << u;
                    g.cur_textures[u] = hit_e->handle;
                    s_tunit[u] = 4;
                    g.cur_sh_mask &= ~bit;
                    eng_unit_cache* uc = &s_ucache[u];
                    uc->handle = hit_e->handle;
                    uc->frame = g.frames;
                    uc->surface_gen = g.surface_gen;
                    uc->tex_gen = g.tex_gen;
                    uc->kind = 4u;
                    uc->valid = 1;
                    g.cur_tex_resolve_dirty &= ~bit;
                    g.cur_tex_valid_mask |= bit;
                }
                if (fast_tex_ok) {
                    u64 cur_sh = 0;
                    u32 shm = g.cur_sh_mask & en_mask;
                    while (shm) {
                        const u32 u = (u32)__builtin_ctz(shm);
                        shm &= shm - 1u;
                        cur_sh |= ((u64)g.cur_sh_nibble[u]) << (u * 4u);
                    }
                    if (cur_sh != g.last_tex_sh_key) {
                        fast_tex_ok = 0;
                    } else {
                        g.last_tex_state_gen = s_tex_state_gen;
                        s_draw_fast.tex_state_gen = s_tex_state_gen;
                        int tex_chg = (g.last_tex_mask != en_mask);
                        int samp_chg = tex_chg;
                        u32 tm = en_mask;
                        while (tm) {
                            const u32 u = (u32)__builtin_ctz(tm);
                            tm &= tm - 1u;
                            if (g.last_textures[u] != g.cur_textures[u]) tex_chg = 1;
                            if (g.last_samp_gen[u] != s_unit_samp_gen[u]) { tex_chg = 1; samp_chg = 1; }
                        }
                        if (tex_chg) {
                            g.be->bind_textures(g.be->user, g.cur_textures,
                                                samp_chg ? s_unit_samp : NULL, en_mask);
                            g.last_tex_mask = en_mask;
                            u32 cm2 = en_mask;
                            while (cm2) {
                                const u32 u = (u32)__builtin_ctz(cm2);
                                cm2 &= cm2 - 1u;
                                g.last_textures[u] = g.cur_textures[u];
                                g.last_samp_gen[u] = s_unit_samp_gen[u];
                            }
                        }
                    }
                }
            }
        }
        if (__builtin_expect(fast_tex_ok, 1)) {
        const u64 pt0 = mprof ? eng_rdtsc() : 0;
        eng_program* const prog = s_draw_fast.prog;
        const rsx_vertex_layout_plan* const layout = &prog->layout;
        if (__builtin_expect(eng_mc_try_lookup(prim, layout, s_draw_fast.rebuild), 1)) {
            const u64 pt1 = mprof ? eng_rdtsc() : 0;
            if (__builtin_expect(mprof, 0)) s_m2f_p.cy_fast_mc += pt1 - pt0;
            const int indexed = dc.mc_indexed;
            const u32 n_draw = dc.mc_n_draw;
            if (__builtin_expect(dc.n_verts != 0u && n_draw != 0u, 1)) {
                if (__builtin_expect(g.upscale_hooks, 0)) {
                    if (!s_draw_fast.upscale_q && s_draw_fast.upscale_di.rt_off == 0u &&
                        !s_draw_fast.upscale_di.blend && !s_draw_fast.upscale_di.depth_test &&
                        n_draw >= 3u && n_draw <= 6u) {
                        goto fast_bail_upscale;
                    }
                    rsx_upscale_draw_info udi = s_draw_fast.upscale_di;
                    udi.rec_index = g.upscale_rec_idx++;
                    udi.vcount = n_draw;
                    udi.consts = (const float*)g.rsx.constants;
                    (void)rsx_upscale_note_draw(&udi, (int)s_draw_fast.upscale_q);
                }
                /* VP constants: xf is unchanged (vp_geom_gen & surf_gen unchanged),
                 * prog is unchanged, so only check touched VP constant block gens. */
                if (prog->vp_const_sparse) {
                    u32 bm = prog->vp_const_blk_mask;
                    int blk_chg = (!g.vp_cb_valid || !g.last_vp_sparse || g.last_vp_prog != prog);
                    if (!blk_chg) {
                        while (bm) {
                            const u32 b = (u32)__builtin_ctz(bm);
                            bm &= bm - 1u;
                            if (g.last_vp_block_gen[b] != g.rsx.vp_const_block_gen[b]) {
                                blk_chg = 1; break;
                            }
                        }
                    }
                    if (blk_chg) {
                        const u32 vp_slots = prog->vp_const_count ? (u32)prog->vp_const_count : 1u;
                        const u32 vp_bytes = (vp_slots + 2u) * 16u;
                        int diff = (!g.vp_cb_valid || !g.last_vp_sparse || g.last_vp_bytes != vp_bytes);
                        if (prog->vp_const_count) {
                            u32* dst = (u32*)(g.vp_cb + 32u);
                            const u16* lst = prog->vp_const_list;
                            const u32 cnt = prog->vp_const_count;
#if defined(__AVX2__)
                            if (!diff) {
                                __m128i vd = _mm_setzero_si128();
                                for (u32 i = 0; i < cnt; i++) {
                                    const __m128i nv = _mm_loadu_si128((const __m128i*)g.rsx.constants[lst[i]]);
                                    const __m128i ov = _mm_loadu_si128((const __m128i*)(dst + i * 4u));
                                    vd = _mm_or_si128(vd, _mm_xor_si128(nv, ov));
                                    _mm_storeu_si128((__m128i*)(dst + i * 4u), nv);
                                }
                                if (!_mm_testz_si128(vd, vd)) diff = 1;
                            } else {
                                memcpy(g.vp_cb, s_draw_fast.xf, 32u);
                                for (u32 i = 0; i < cnt; i++)
                                    _mm_storeu_si128((__m128i*)(dst + i * 4u),
                                                     _mm_loadu_si128((const __m128i*)g.rsx.constants[lst[i]]));
                            }
#else
                            if (diff) memcpy(g.vp_cb, s_draw_fast.xf, 32u);
                            for (u32 i = 0; i < cnt; i++) {
                                if (!diff && memcmp(dst + i * 4u, g.rsx.constants[lst[i]], 16u) != 0) diff = 1;
                                memcpy(dst + i * 4u, g.rsx.constants[lst[i]], 16u);
                            }
#endif
                        } else if (diff) {
                            memcpy(g.vp_cb, s_draw_fast.xf, 32u);
                            memset(g.vp_cb + 32u, 0, 16u);
                        }
                        if (diff) {
                            g.last_vp_bytes = vp_bytes;
                            g.last_vp_sparse = 1;
                            g.last_vp_packed = prog->vp_const_packed;
                            g.vp_cb_valid = 1;
                            g.be->bind_vs_constants(g.be->user, g.vp_cb, vp_bytes);
                        }
                        g.last_vp_prog = prog;
                        u32 bm2 = prog->vp_const_blk_mask;
                        while (bm2) {
                            const u32 b = (u32)__builtin_ctz(bm2);
                            bm2 &= bm2 - 1u;
                            g.last_vp_block_gen[b] = g.rsx.vp_const_block_gen[b];
                        }
                    }
                } else {
                    const int vp_packed = prog->vp_const_packed;
                    const u32 vp_slots = vp_packed ? (u32)prog->vp_const_slots : (u32)RSX_DSP_NUM_CONSTANTS;
                    const u32 vp_bytes = (vp_slots + 2u) * 16u;
                    const u32 vp_nblks = (vp_slots + 63u) >> 6;
                    int vp_chg = !g.vp_cb_valid || g.last_vp_sparse || g.last_vp_packed != vp_packed;
                    if (!vp_chg) {
                        if (vp_packed) {
                            if (g.last_vp_bytes < vp_bytes) vp_chg = 1;
                            else {
                                for (u32 b = 0; b < vp_nblks; b++)
                                    if (g.last_vp_block_gen[b] != g.rsx.vp_const_block_gen[b]) { vp_chg = 1; break; }
                            }
                        } else if (g.last_vp_const_gen != g.rsx.vp_const_gen || g.last_vp_bytes != vp_bytes) {
                            vp_chg = 1;
                        }
                    }
                    if (vp_chg) {
                        const void* vp_ptr = g.vp_cb;
                        if (vp_packed) {
                            memcpy(g.rsx.vp_xf, s_draw_fast.xf, 32u);
                            vp_ptr = g.rsx.vp_xf;
                        } else {
                            memcpy(g.vp_cb, g.rsx.constants, RSX_DSP_NUM_CONSTANTS * 16u);
                            memcpy(g.vp_cb + RSX_DSP_NUM_CONSTANTS * 16u, s_draw_fast.xf, 32u);
                        }
                        g.last_vp_const_gen = g.rsx.vp_const_gen;
                        for (u32 b = 0; b < vp_nblks; b++)
                            g.last_vp_block_gen[b] = g.rsx.vp_const_block_gen[b];
                        g.last_vp_packed = vp_packed;
                        g.last_vp_sparse = 0;
                        g.last_vp_prog = NULL;
                        g.last_vp_bytes = vp_bytes;
                        g.vp_cb_valid = 1;
                        g.be->bind_vs_constants(g.be->user, vp_ptr, vp_bytes);
                    }
                }
                const u64 pt2 = mprof ? eng_rdtsc() : 0;
                if (__builtin_expect(mprof, 0)) s_m2f_p.cy_fast_vp += pt2 - pt1;
                const u32 uploaded = indexed ? dc.n_verts : n_draw;
                const void* draw_verts = dc.verts_cache ? (const void*)dc.verts_cache : (const void*)dc.verts;
                g.be->draw(g.be->user, s_draw_fast.topology, draw_verts, uploaded, layout->stride,
                           indexed ? dc.indices_cache : NULL, indexed ? n_draw : 0u);
                if (__builtin_expect(g.upscale_hooks && s_draw_fast.upscale_q, 0)) {
                    g.upscale_scene_zslot = s_draw_fast.zslot;
                    g.upscale_depth_captured = 0;
                }
                if (__builtin_expect(mprof, 0)) {
                    s_m2f_p.cy_fast_vk += eng_rdtsc() - pt2;
                    s_m2f_p.fast_hits++;
                }
                g.draw_seq++;
                g.st_draws++;
                g.st_verts += uploaded;
                if (!s_draw_fast.fixed) g.guest_draws++;
                return;
            }
            return;
        }
        if (__builtin_expect(mprof, 0)) s_m2f_p.miss_mc++;
        mc_already_missed = 1;
        } else if (__builtin_expect(mprof, 0)) {
            s_m2f_p.miss_tex++;
        }
    } else if (__builtin_expect(mprof && m2f && !timing && !trace && !immediate, 0)) {
        if (!s_draw_fast.valid) s_m2f_p.miss_other++;
        else if (g.rsx.surf_gen != s_draw_fast.surf_gen || g.surface_gen != s_draw_fast.surface_gen) s_m2f_p.miss_surf++;
        else if (g.rsx.vp_gen != s_draw_fast.vp_gen || rsx_dsp_vp_start(&g.rsx) != s_draw_fast.vp_start ||
                 rsx_dsp_reg(&g.rsx, 0x08E4) != s_draw_fast.fp_reg || rsx_dsp_shader_control(&g.rsx) != s_draw_fast.fp_ctrl) s_m2f_p.miss_prog++;
        else if (g.rsx.pipe_gen != s_draw_fast.pipe_gen) s_m2f_p.miss_pipe++;
        else if (g.rsx.dyn_gen != s_draw_fast.dyn_gen || g.rsx.vp_geom_gen != s_draw_fast.vp_geom_gen) s_m2f_p.miss_dyn++;
        else if ((g.rsx.tex_dirty_mask | g.rsx.vtex_dirty_mask) != 0u || s_tex_state_gen != s_draw_fast.tex_state_gen) s_m2f_p.miss_tex++;
        else s_m2f_p.miss_other++;
    }
fast_bail_upscale:;
    const u64 sp0 = mprof ? eng_rdtsc() : 0;
    /* Everything that becomes triangles -- lists, strips, fans, quads, quad
     * strips and polygons -- is REBUILT into one triangle list through an
     * index buffer. A host strip topology cannot express a restart cut, which
     * is the whole reason the reference engine rebuilds them, and rebuilding
     * is also what lets repeated references share an uploaded vertex. Points
     * and lines pass through as the guest issued them, and anything left that
     * would need an expansion this engine does not have is dropped rather
     * than drawn as something else. */
    rsx_topology topology = RSX_TOPOLOGY_TRIANGLES;
    int scratch = 0;
    const int rebuild = eng_topology_rebuild(prim, 0, &scratch) != 0;
    if (!rebuild) {
        if (rsx_primitive_needs_expansion(prim)) return;
        topology = rsx_primitive_topology(prim);
        if (topology == RSX_TOPOLOGY_UNSUPPORTED) return;
    }

    /* The layout the vertex program reads comes from the same memoized
     * program identity as the pipeline, so its input analysis is not redone
     * per draw. The program is resolved once and handed to the pipeline
     * lookup: asking for it twice ran the two microcode hashes twice a draw. */
    const u8* fp_uc = NULL;
    u32 fp_size = 0;
    if (__builtin_expect(!m2f || !s_units_seeded ||
                         ((g.rsx.tex_dirty_mask | g.rsx.vtex_dirty_mask) != 0u), 0))
        eng_units_decode();
    eng_program* prog = NULL;
    if (__builtin_expect(m2f && s_ef.m2d && s_ef.m2e4b && s_mc_swd && s_last_prog != NULL, 1)) {
        const eng_prog_fast_ent* pfe = s_last_prog;
        if (__builtin_expect(pfe->valid && pfe->epoch == s_prog_epoch &&
                             pfe->frame == g.frames &&
                             pfe->swd_checked_frame == g.frames + 1u &&
                             pfe->vp_gen == g.rsx.vp_gen &&
                             pfe->vp_start == rsx_dsp_vp_start(&g.rsx) &&
                             pfe->fp_reg == rsx_dsp_reg(&g.rsx, 0x08E4) &&
                             pfe->fp_ctrl == rsx_dsp_shader_control(&g.rsx) &&
                             pfe->cube_mask == s_cube_mask &&
                             pfe->vtex_mask == s_vtex_mask, 1)) {
            prog = pfe->pg;
            fp_uc = pfe->fp_uc;
            fp_size = pfe->fp_size;
        }
    }
    if (__builtin_expect(!prog, 0)) {
        prog = eng_program_get(&fp_uc, &fp_size);
        if (!prog) return;
    }
    if (timing) eng_phase_mark(&g.st_ms_prog);
    const u64 sp1 = mprof ? eng_rdtsc() : 0;
    if (__builtin_expect(mprof, 0)) s_m2f_p.cy_slow_prog += sp1 - sp0;
    const rsx_vertex_layout_plan* layout = &prog->layout;
    if (immediate) dc_imm_build(layout);
    else dc_fetch(prim, layout, rebuild, mc_already_missed);
    if (timing) eng_phase_mark(&g.st_ms_fetch);
    const u64 sp2 = mprof ? eng_rdtsc() : 0;
    if (__builtin_expect(mprof, 0)) s_m2f_p.cy_slow_fetch += sp2 - sp1;
    if (trace)
        fprintf(stderr, "[rsx engine] group %u fetched: verts=%u refs=%u ok=%d mask=0x%X stride=%u\n",
                trace_groups - 1, dc.n_verts, dc.n_refs, dc.fetch_ok, layout->mask, layout->stride);
    if (!dc.n_verts || !dc.fetch_ok) return;

    int indexed = 0;
    u32 n_draw = dc.n_source_refs;
    if (dc.mc_hit) {
        indexed = dc.mc_indexed;
        n_draw = dc.mc_n_draw;
    } else if (rebuild) {
        if (!eng_topology_rebuild(prim, dc.refs_remapped, &indexed)) return;
        n_draw = indexed
            ? rsx_draw_engine_topology_index_count(prim, dc.n_source_refs,
                                                   dc.cuts, dc.n_cuts)
            : dc.n_source_refs - dc.n_source_refs % 3u;
    }
    if (!n_draw) return;

    const rsx_be_render_state* rsp = eng_render_state();
    const int m2d = s_ef.m2d;

    u32 targets[RSX_BE_MAX_COLOR_TARGETS];
    u32 n_targets = 0;
    rsx_dsp_surface sf;
    u32 zslot = ENG_INVALID;
    if (m2d && g.surf_cache_valid && g.surf_cache_gen == g.rsx.surf_gen &&
        g.surf_cache_surface_gen == g.surface_gen && g.surf_cache_nzdepths == g.n_zdepths) {
        n_targets = g.surf_cache_ntargets;
        for (u32 i = 0; i < n_targets; i++) targets[i] = g.surf_cache_targets[i];
        sf = g.surf_cache_sf;
        zslot = g.surf_cache_zslot;
    } else {
        n_targets = eng_current_target_set(targets);
        if (!n_targets) return;
        rsx_dsp_get_surface(&g.rsx, &sf);
        zslot = eng_zdepth_get(sf.zeta_location, sf.zeta_offset,
                               sf.clip_w, sf.clip_h);
        g.surf_cache_valid = 1;
        g.surf_cache_gen = g.rsx.surf_gen;
        g.surf_cache_surface_gen = g.surface_gen;
        g.surf_cache_nzdepths = g.n_zdepths;
        g.surf_cache_sf = sf;
        g.surf_cache_ntargets = n_targets;
        for (u32 i = 0; i < n_targets; i++) g.surf_cache_targets[i] = targets[i];
        g.surf_cache_zslot = zslot;
    }
    if (trace)
        fprintf(stderr, "[rsx engine] group %u targets=%u draw=%u indexed=%d\n",
                trace_groups - 1, n_targets, n_draw, indexed);
    if (!n_targets) return;
    const u32 target = targets[0];
    /* sp00nznet/ps3recomp #811: an empty scissor (explicitly written zero
     * width/height or wholly outside the target) draws nothing; drop early
     * before fetching/binding textures or taking feedback snapshots. */
    {
        const u32 sh_reg = rsx_dsp_reg(&g.rsx, M_SCISSOR_HORIZONTAL);
        const u32 sv_reg = rsx_dsp_reg(&g.rsx, M_SCISSOR_VERTICAL);
        const u32 gx = sh_reg & 0xFFFFu, gw = sh_reg >> 16;
        const u32 gy = sv_reg & 0xFFFFu, gh = sv_reg >> 16;
        const int written_empty =
            (gw == 0u && g.rsx.seen[M_SCISSOR_HORIZONTAL >> 2]) ||
            (gh == 0u && g.rsx.seen[M_SCISSOR_VERTICAL >> 2]);
        if (written_empty) return;
        if (gw > 0u && gh > 0u) {
            if (gx >= g.surfaces[target].w || gy >= g.surfaces[target].h) return;
        }
    }
    const int m2e4b = s_ef.m2e4b;
    const u32 cur_w = sf.clip_w ? sf.clip_w : g.width;
    const u32 cur_h = sf.clip_h ? sf.clip_h : g.height;
    const float W = (float)cur_w;
    const float H = (float)cur_h;
    float xf[8];
    if (m2e4b && g.vp_cache_valid && g.vp_cache_gen == g.rsx.vp_geom_gen &&
        g.vp_cache_w == cur_w && g.vp_cache_h == cur_h) {
        memcpy(xf, g.vp_cache_xf, sizeof xf);
    } else {
        rsx_dsp_viewport vp;
        rsx_dsp_get_viewport(&g.rsx, &vp);
        xf[0] = 1.0f; xf[1] = 1.0f; xf[2] = 1.0f; xf[3] = 0.0f;
        xf[4] = 0.0f; xf[5] = 0.0f; xf[6] = 0.0f; xf[7] = 0.0f;
        if (vp.scale[0] != 0.0f || vp.translate[0] != 0.0f) {
            const float inv_hw = 2.0f / W;
            const float inv_hh = 2.0f / H;
            xf[0] = vp.scale[0] * inv_hw;
            xf[1] = -(vp.scale[1] * inv_hh);
            xf[2] = vp.scale[2];
            xf[4] = vp.translate[0] * inv_hw - 1.0f;
            xf[5] = -(vp.translate[1] * inv_hh - 1.0f);
            xf[6] = vp.translate[2];
        }
        memcpy(g.vp_cache_xf, xf, sizeof xf);
        g.vp_cache_gen = g.rsx.vp_geom_gen;
        g.vp_cache_w = cur_w;
        g.vp_cache_h = cur_h;
        g.vp_cache_valid = 1;
    }

    /* Textures before any pipeline or target binding: a cache refresh can
     * replace a resource, and the backend is free to submit while doing it. */
    u32* textures = g.cur_textures;
    u32 vtextures[RSX_BE_MAX_VERTEX_TEXTURES];
    rsx_be_sampler_desc vsamplers[RSX_BE_MAX_VERTEX_TEXTURES];
    u64 sh_key = 0;
    u32 tex_mask = 0;
    int tex_fast_skip = 0;
    const int tex_env_same = (m2d && g.tex_bind_valid && !g.last_tex_had_dynamic &&
                              g.last_tex_frame == g.frames &&
                              g.last_tex_surface_gen == g.surface_gen &&
                              g.last_tex_tex_gen == g.tex_gen &&
                              g.last_tex_surf_gen == g.rsx.surf_gen &&
                              g.last_tex_clear_gen == g.clear_gen);
    if (tex_env_same && g.last_tex_state_gen == s_tex_state_gen) {
        tex_mask = g.last_tex_mask;
        sh_key = g.last_tex_sh_key;
        tex_fast_skip = 1;
    } else if (m2e4b && tex_env_same) {
        tex_mask = s_enabled_tex_mask;
        const u32 todo = tex_mask & (g.cur_tex_resolve_dirty | ~g.cur_tex_valid_mask);
        u8 had_dyn = 0;
        if (todo) {
            u64 part_sh = 0;
            sink_bind_textures(targets, n_targets, zslot,
                               textures, &part_sh, &had_dyn, todo);
            g.cur_tex_resolve_dirty &= ~todo;
            g.cur_tex_valid_mask |= todo;
        }
        u32 shm = g.cur_sh_mask & tex_mask;
        while (shm) {
            const u32 u = (u32)__builtin_ctz(shm);
            shm &= shm - 1u;
            sh_key |= ((u64)g.cur_sh_nibble[u]) << (u * 4u);
        }
        g.last_tex_state_gen = s_tex_state_gen;
        g.last_tex_sh_key = sh_key;
        g.last_tex_had_dynamic = had_dyn;
    } else {
        u8 had_dyn = 0;
        g.cur_sh_mask = 0;
        tex_mask = sink_bind_textures(targets, n_targets, zslot,
                                      textures, &sh_key, &had_dyn, 0xFFFFFFFFu);
        g.cur_tex_resolve_dirty = 0;
        g.cur_tex_valid_mask = tex_mask;
        g.last_tex_state_gen = s_tex_state_gen;
        g.last_tex_frame = g.frames;
        g.last_tex_surface_gen = g.surface_gen;
        g.last_tex_tex_gen = g.tex_gen;
        g.last_tex_surf_gen = g.rsx.surf_gen;
        g.last_tex_clear_gen = g.clear_gen;
        g.last_tex_sh_key = sh_key;
        g.last_tex_had_dynamic = had_dyn;
    }
    const u32 want_vtex = eng_vtex_mask();
    u32 vtex_mask = 0;
    if (want_vtex) {
        memset(vtextures, 0, sizeof vtextures);
        memset(vsamplers, 0, sizeof vsamplers);
        vtex_mask = sink_bind_vertex_textures(want_vtex, vtextures, vsamplers);
    }
    if (timing) eng_phase_mark(&g.st_ms_textures);
    const u64 sp3 = mprof ? eng_rdtsc() : 0;
    if (__builtin_expect(mprof, 0)) s_m2f_p.cy_slow_tex += sp3 - sp2;

    /* A guest program pair that will not translate has no fallback: the draw
     * is dropped, as the reference engine drops it. Substituting the built-in
     * program would draw the geometry in the wrong colours, which is harder to
     * see than a missing object and hides the translation failure. */
    int pipeline_is_fixed = 1;
    const u32 tex_xf_key = eng_tex_xf_key();
    const u32 pipeline = eng_pipeline_get(prog, fp_uc, fp_size, rsp,
                                          eng_surface_format(sf.color_format),
                                          n_targets, tex_xf_key, sh_key,
                                          &pipeline_is_fixed);
    if (!pipeline) return;
    if (timing) eng_phase_mark(&g.st_ms_pipeline);

    const u32* draw_indices = NULL;
    if (dc.mc_hit) {
        draw_indices = dc.indices_cache;
    } else {
        if (indexed) {
            if (!dc_reserve_indices(n_draw)) return;
            rsx_draw_engine_write_topology_indices(
                prim, dc.n_source_refs, dc.cuts, dc.n_cuts,
                dc.refs_remapped ? dc.ref_remap.occurrence_to_unique : NULL,
                g.indices);
            draw_indices = g.indices;
        }
        if (dc.mc_elig)
            eng_mc_store(indexed, n_draw, draw_indices);
    }

    /* Optional temporal upscaler draw tracking + Halton(2,3) jitter (off when FSR=0). */
    int upscale_q = 0, upscale_evt = 0;
    rsx_upscale_draw_info upscale_di;
    if (__builtin_expect(g.upscale_hooks, 0)) {
        const u32 sel = (sf.color_target == CELL_GCM_SURFACE_TARGET_1) ? 1u : 0u;
        const int is_disp = eng_is_display_surface(sf.color_location[sel], sf.color_offset[sel]);
        const u32 rt_off = is_disp ? 0u : (sf.color_offset[sel] ? sf.color_offset[sel] : 1u);
        const u32 zeta_off = (zslot != ENG_INVALID)
            ? (g.zdepths[zslot].offset ? g.zdepths[zslot].offset : 1u) : 0u;
        const u32 zw = (zslot != ENG_INVALID) ? g.zdepths[zslot].w : 0u;
        const u32 zh = (zslot != ENG_INVALID) ? g.zdepths[zslot].h : 0u;
        rsx_dsp_viewport uvp;
        rsx_dsp_get_viewport(&g.rsx, &uvp);
        const u32 vp_w = uvp.w ? uvp.w : cur_w;
        const u32 vp_h = uvp.h ? uvp.h : cur_h;
        upscale_q = rsx_upscale_draw_qualifies(
            rsp->depth_test ? 1 : 0, zeta_off, zw, zh, rt_off,
            g.surfaces[target].w, g.surfaces[target].h,
            vp_w, vp_h, g.width, g.height);
        memset(&upscale_di, 0, sizeof upscale_di);
        upscale_di.frame = g.frames;
        upscale_di.rec_index = g.upscale_rec_idx++;
        upscale_di.vs_idx = (int)(prog - g.programs);
        upscale_di.vs_hash = (u32)(prog->vp_hash ^ (prog->vp_hash >> 32));
        upscale_di.cuse = prog->vp_cuse_valid ? prog->vp_cuse : NULL;
        upscale_di.consts = (const float*)g.rsx.constants;
        upscale_di.rt_off = rt_off;
        upscale_di.rt_w = g.surfaces[target].w;
        upscale_di.rt_h = g.surfaces[target].h;
        upscale_di.zeta_off = zeta_off;
        upscale_di.zw = zw;
        upscale_di.zh = zh;
        upscale_di.vp_x = uvp.x;
        upscale_di.vp_y = uvp.y;
        upscale_di.vp_w = vp_w;
        upscale_di.vp_h = vp_h;
        upscale_di.fp_addr = prog->fp_ea;
        upscale_di.vcount = n_draw;
        upscale_di.color_target = sf.color_target;
        upscale_di.mrt = n_targets > 1u ? (n_targets - 1u) : 0u;
        upscale_di.depth_test = rsp->depth_test ? 1 : 0;
        upscale_di.depth_mask = rsp->depth_write ? 1 : 0;
        upscale_di.blend = rsp->blend_enable ? 1 : 0;
        upscale_di.posscale[0] = xf[0]; upscale_di.posscale[1] = xf[1]; upscale_di.posscale[2] = xf[2];
        upscale_di.posoffset[0] = xf[4]; upscale_di.posoffset[1] = xf[5]; upscale_di.posoffset[2] = xf[6];
        upscale_di.guest_w = g.width;
        upscale_di.guest_h = g.height;
        upscale_evt = rsx_upscale_note_draw(&upscale_di, upscale_q);
        if (upscale_q) rsx_upscale_jitter_ndc(&xf[4], &xf[5]);
    }

    /* The transform constant block, then the viewport epilogue the vertex
     * program's tail multiplies by: RSX window coordinates mapped into the
     * host's clip space (rsx_live_draw.c:6143-6156). */
    const int vp_sparse = prog->vp_const_sparse;
    const int vp_packed = prog->vp_const_packed;
    const u32 vp_slots = vp_sparse ? (prog->vp_const_count ? (u32)prog->vp_const_count : 1u)
                       : (vp_packed ? (u32)prog->vp_const_slots : (u32)RSX_DSP_NUM_CONSTANTS);
    const u32 vp_bytes = (vp_slots + 2u) * 16u;
    const void* vp_upload_ptr = g.vp_cb;
    int vp_changed = 0;
    if (vp_sparse) {
        vp_changed = !g.vp_cb_valid || !g.last_vp_sparse || g.last_vp_prog != prog ||
                     memcmp(g.last_vp_xf, xf, sizeof xf) != 0;
        if (!vp_changed) {
            u32 bm = prog->vp_const_blk_mask;
            while (bm) {
                const u32 b = (u32)__builtin_ctz(bm);
                bm &= bm - 1u;
                if (g.last_vp_block_gen[b] != g.rsx.vp_const_block_gen[b]) {
                    vp_changed = 1; break;
                }
            }
        }
        if (vp_changed) {
            if (m2e4b) {
                int diff = (!g.vp_cb_valid || !g.last_vp_sparse || g.last_vp_bytes != vp_bytes);
                if (!diff && memcmp(g.vp_cb, xf, sizeof xf) != 0) diff = 1;
                memcpy(g.vp_cb, xf, sizeof xf);
                if (prog->vp_const_count) {
                    u32* dst = (u32*)(g.vp_cb + 32u);
                    const u16* lst = prog->vp_const_list;
                    const u32 cnt = prog->vp_const_count;
#if defined(__AVX2__)
                    if (!diff) {
                        __m128i vd = _mm_setzero_si128();
                        for (u32 i = 0; i < cnt; i++) {
                            const __m128i nv = _mm_loadu_si128((const __m128i*)g.rsx.constants[lst[i]]);
                            const __m128i ov = _mm_loadu_si128((const __m128i*)(dst + i * 4u));
                            vd = _mm_or_si128(vd, _mm_xor_si128(nv, ov));
                            _mm_storeu_si128((__m128i*)(dst + i * 4u), nv);
                        }
                        if (!_mm_testz_si128(vd, vd)) diff = 1;
                    } else {
                        for (u32 i = 0; i < cnt; i++)
                            _mm_storeu_si128((__m128i*)(dst + i * 4u),
                                             _mm_loadu_si128((const __m128i*)g.rsx.constants[lst[i]]));
                    }
#else
                    for (u32 i = 0; i < cnt; i++) {
                        if (!diff && memcmp(dst + i * 4u, g.rsx.constants[lst[i]], 16u) != 0) diff = 1;
                        memcpy(dst + i * 4u, g.rsx.constants[lst[i]], 16u);
                    }
#endif
                } else {
                    static const u32 z4[4] = { 0, 0, 0, 0 };
                    if (!diff && memcmp(g.vp_cb + 32u, z4, 16u) != 0) diff = 1;
                    memset(g.vp_cb + 32u, 0, 16u);
                }
                if (!diff) {
                    vp_changed = 0;
                } else {
                    g.last_vp_bytes = vp_bytes;
                    g.last_vp_sparse = 1;
                    g.last_vp_packed = vp_packed;
                    g.vp_cb_valid = 1;
                }
            } else {
                memcpy(g.vp_cb, xf, sizeof xf);
                if (prog->vp_const_count) {
                    u32* dst = (u32*)(g.vp_cb + 32u);
                    const u16* lst = prog->vp_const_list;
                    const u32 cnt = prog->vp_const_count;
                    for (u32 i = 0; i < cnt; i++)
                        memcpy(dst + i * 4u, g.rsx.constants[lst[i]], 16u);
                } else {
                    memset(g.vp_cb + 32u, 0, 16u);
                }
                if (g.vp_cb_valid && g.last_vp_sparse && g.last_vp_bytes == vp_bytes &&
                    memcmp(g.last_vp_sparse_data, g.vp_cb, vp_bytes) == 0) {
                    vp_changed = 0;
                } else {
                    memcpy(g.last_vp_sparse_data, g.vp_cb, vp_bytes);
                    g.last_vp_bytes = vp_bytes;
                    g.last_vp_sparse = 1;
                    g.last_vp_packed = vp_packed;
                    g.vp_cb_valid = 1;
                }
            }
            g.last_vp_prog = prog;
            memcpy(g.last_vp_xf, xf, sizeof xf);
            u32 bm = prog->vp_const_blk_mask;
            while (bm) {
                const u32 b = (u32)__builtin_ctz(bm);
                bm &= bm - 1u;
                g.last_vp_block_gen[b] = g.rsx.vp_const_block_gen[b];
            }
        }
    } else {
        const u32 vp_nblks = (vp_slots + 63u) >> 6;
        vp_changed = !g.vp_cb_valid || g.last_vp_sparse || g.last_vp_packed != vp_packed ||
                     memcmp(g.last_vp_xf, xf, sizeof xf) != 0;
        if (!vp_changed) {
            if (vp_packed && m2d) {
                if (g.last_vp_bytes < vp_bytes) vp_changed = 1;
                else {
                    for (u32 b = 0; b < vp_nblks; b++) {
                        if (g.last_vp_block_gen[b] != g.rsx.vp_const_block_gen[b]) {
                            vp_changed = 1; break;
                        }
                    }
                }
            } else if (g.last_vp_const_gen != g.rsx.vp_const_gen || g.last_vp_bytes != vp_bytes) {
                vp_changed = 1;
            }
        }
        if (vp_changed) {
            if (vp_packed) {
                _Static_assert(offsetof(rsx_dispatch, constants) == offsetof(rsx_dispatch, vp_xf) + 32,
                               "vp_xf must immediately precede constants");
                memcpy(g.rsx.vp_xf, xf, sizeof xf);
                vp_upload_ptr = g.rsx.vp_xf;
            } else {
                memcpy(g.vp_cb, g.rsx.constants, RSX_DSP_NUM_CONSTANTS * 16u);
                memcpy(g.vp_cb + RSX_DSP_NUM_CONSTANTS * 16u, xf, sizeof xf);
            }
            g.last_vp_const_gen = g.rsx.vp_const_gen;
            for (u32 b = 0; b < vp_nblks; b++)
                g.last_vp_block_gen[b] = g.rsx.vp_const_block_gen[b];
            g.last_vp_packed = vp_packed;
            g.last_vp_sparse = 0;
            g.last_vp_prog = NULL;
            g.last_vp_bytes = vp_bytes;
            memcpy(g.last_vp_xf, xf, sizeof xf);
            g.vp_cb_valid = 1;
        }
    }

    /* The buffered fragment constants, then fp_alpha: the layout
     * rsx_fp_decompile_buffered_ex compiled the shader against. */
    u32 phys_w = cur_w, phys_h = cur_h;
    float wscale_x = 0.0f, wscale_y = 0.0f;
    if (__builtin_expect(g.res_scaled, 0)) {
        eng_scale_dims(cur_w, cur_h, &phys_w, &phys_h);
        wscale_x = (float)cur_w / (float)phys_w;
        wscale_y = (float)cur_h / (float)phys_h;
    }
    u32 fp_bytes = g.last_fp_bytes;
    int fp_changed = 0;
    if (m2d && g.fp_cb_valid && g.last_fp_prog == prog &&
        g.last_fp_alpha_raw == rsp->alpha_ref_raw &&
        g.last_fp_alpha_fmt == rsp->alpha_ref_format &&
        (!g.res_scaled || (g.last_fp_wscale_x == wscale_x && g.last_fp_wscale_y == wscale_y))) {
        fp_changed = 0;
    } else if (m2e4b && g.fp_cb_valid && prog->const_cached && !prog->fixed &&
               g.last_fp_prog_const_cnt == prog->const_count &&
               g.last_fp_prog_const_hash == prog->const_hash &&
               g.last_fp_alpha_raw == rsp->alpha_ref_raw &&
               g.last_fp_alpha_fmt == rsp->alpha_ref_format &&
               (!g.res_scaled || (g.last_fp_wscale_x == wscale_x && g.last_fp_wscale_y == wscale_y))) {
        g.last_fp_prog = prog;
        fp_changed = 0;
    } else {
        u32 fpc_count = 0;
        const void* fpc_src = NULL;
        u64 fpc_chash = 0;
        int fpc_has_chash = 0;
        if (prog->fixed) {
            fpc_count = 0;
        } else if (prog->const_cached) {
            fpc_count = prog->const_count;
            fpc_src = prog->const_values;
            fpc_chash = prog->const_hash;
            fpc_has_chash = 1;
        } else {
            const double ct0 = timing ? eng_now_ms() : 0.0;
            if (rsx_fp_collect_constants(fp_uc, fp_size, &g.fp_constants) < 0) return;
            if (g.fp_constants.count <= ENG_PROG_CONST_MAX) {
                prog->const_count = g.fp_constants.count;
                if (g.fp_constants.count)
                    memcpy(prog->const_values, g.fp_constants.values,
                           (size_t)g.fp_constants.count * 16u);
                prog->const_hash = g.fp_constants.count
                    ? eng_hash_bytes(prog->const_values, g.fp_constants.count * 16u, 0x243F6A8885A308D3ull)
                    : 1ull;
                prog->const_cached = 1;
                fpc_chash = prog->const_hash;
                fpc_has_chash = 1;
            }
            if (timing) g.st_ms_consts += eng_now_ms() - ct0;
            fpc_count = g.fp_constants.count;
            fpc_src = g.fp_constants.values;
        }
        const u32 nslots = fpc_count ? fpc_count : 1u;
        fp_bytes = (nslots + 1u) * 16u;
        if (g.fp_cb_cap < fp_bytes) {
            u8* n = (u8*)realloc(g.fp_cb, fp_bytes);
            if (!n) return;
            g.fp_cb = n;
            g.fp_cb_cap = fp_bytes;
        }
        if (fpc_count)
            memcpy(g.fp_cb, fpc_src, fpc_count * 16u);
        else
            memset(g.fp_cb, 0, 16u);
        { float* alpha = (float*)(g.fp_cb + nslots * 16u);
          alpha[0] = rsx_fp_alpha_ref(rsp->alpha_ref_raw, rsp->alpha_ref_format);
          alpha[1] = wscale_x; alpha[2] = wscale_y; alpha[3] = 0.0f; }
        const u64 fp_hash = eng_hash_bytes(g.fp_cb, fp_bytes, 0x243F6A8885A308D3ull);
        fp_changed = !g.fp_cb_valid || g.last_fp_bytes != fp_bytes || g.last_fp_hash != fp_hash;
        if (fp_changed) {
            g.last_fp_bytes = fp_bytes;
            g.last_fp_hash = fp_hash;
            g.fp_cb_valid = 1;
        }
        g.last_fp_prog = prog;
        g.last_fp_prog_const_hash = fpc_has_chash ? fpc_chash : 0ull;
        g.last_fp_prog_const_cnt = fpc_has_chash ? fpc_count : 0xFFFFFFFFu;
        g.last_fp_alpha_raw = rsp->alpha_ref_raw;
        g.last_fp_alpha_fmt = rsp->alpha_ref_format;
        g.last_fp_wscale_x = wscale_x;
        g.last_fp_wscale_y = wscale_y;
    }

    /* The depth attachment is cleared the first time it is bound, so a title
     * that never clears its shadow zeta still tests against something. */
    const u32 depth_handle = (zslot != ENG_INVALID) ? g.zdepths[zslot].handle : 0;
    if (zslot != ENG_INVALID && !g.zdepths[zslot].cleared) {
        g.be->clear_depth_stencil(g.be->user, depth_handle,
                                  RSX_BE_CLEAR_DEPTH | RSX_BE_CLEAR_STENCIL,
                                  1.0f, 0);
        g.zdepths[zslot].cleared = 1;
        g.zdepths[zslot].had_write = 0;
        g.clear_gen++;
    }

    u32 handles[RSX_BE_MAX_COLOR_TARGETS];
    for (u32 i = 0; i < n_targets; i++) handles[i] = g.surfaces[targets[i]].handle;
    { int rt_chg = !m2d || !g.bind_dedup_valid || g.last_rt_count != n_targets ||
                   g.last_depth_handle != depth_handle;
      if (!rt_chg) {
          for (u32 i = 0; i < n_targets; i++)
              if (g.last_rt_handles[i] != handles[i]) { rt_chg = 1; break; }
      }
      if (rt_chg) {
          g.be->bind_targets(g.be->user, handles, n_targets, depth_handle);
          g.last_rt_count = n_targets;
          for (u32 i = 0; i < n_targets; i++) g.last_rt_handles[i] = handles[i];
          g.last_depth_handle = depth_handle;
      }
    }
    if (!m2d || !g.bind_dedup_valid || g.last_pipeline != pipeline) {
        g.be->bind_pipeline(g.be->user, pipeline);
        g.last_pipeline = pipeline;
    }
    if (vp_changed) g.be->bind_vs_constants(g.be->user, vp_upload_ptr, vp_bytes);
    if (fp_changed) g.be->bind_ps_constants(g.be->user, g.fp_cb, fp_bytes);
    if (!tex_fast_skip) {
      int tex_chg = !eng_m2c_use() || !g.tex_bind_valid || g.last_tex_mask != tex_mask;
      if (!tex_chg) {
          u32 tm = tex_mask;
          while (tm) {
              const u32 u = (u32)__builtin_ctz(tm);
              tm &= tm - 1u;
              if (g.last_textures[u] != textures[u] ||
                  g.last_samp_gen[u] != s_unit_samp_gen[u]) {
                  tex_chg = 1; break;
              }
          }
      }
      if (tex_chg) {
          g.be->bind_textures(g.be->user, textures, s_unit_samp, tex_mask);
          g.last_tex_mask = tex_mask;
          u32 cm = tex_mask;
          while (cm) {
              const u32 u = (u32)__builtin_ctz(cm);
              cm &= cm - 1u;
              g.last_textures[u] = textures[u];
              g.last_samp_gen[u] = s_unit_samp_gen[u];
          }
          g.tex_bind_valid = 1;
      }
    }
    if (!m2d || !g.bind_dedup_valid || vtex_mask != 0u || g.last_vtex_mask != 0u) {
        g.be->bind_vertex_textures(g.be->user, vtextures, vsamplers, vtex_mask);
        g.last_vtex_mask = vtex_mask;
    }
    const float vpW = (float)phys_w;
    const float vpH = (float)phys_h;
    if (!m2d || !g.bind_dedup_valid || g.last_vp_w != vpW || g.last_vp_h != vpH) {
        g.be->set_viewport(g.be->user, 0.0f, 0.0f, vpW, vpH);
        g.last_vp_w = vpW; g.last_vp_h = vpH;
    }

    /* Dynamic state (scissor, stencil ref, depth bounds): gated by dyn_gen. */
    if (!m2e4b || !g.bind_dedup_valid || g.last_dyn_gen != g.rsx.dyn_gen ||
        g.last_dyn_surf_gen != g.rsx.surf_gen) {
        {
            u32 sx = 0, sy = 0;
            u32 sw = g.surfaces[target].w, sh = g.surfaces[target].h;
            const u32 h = rsx_dsp_reg(&g.rsx, M_SCISSOR_HORIZONTAL);
            const u32 v = rsx_dsp_reg(&g.rsx, M_SCISSOR_VERTICAL);
            const u32 gx = h & 0xFFFFu, gw = h >> 16;
            const u32 gy = v & 0xFFFFu, gh = v >> 16;
            if (gw > 0 && gh > 0) {
                u32 right = sx + sw, bottom = sy + sh;
                if (gx > sx) sx = gx;
                if (gy > sy) sy = gy;
                if (gx + gw < right)  right = gx + gw;
                if (gy + gh < bottom) bottom = gy + gh;
                sw = right > sx ? right - sx : 0;
                sh = bottom > sy ? bottom - sy : 0;
            }
            if (__builtin_expect(g.res_scaled, 0)) {
                u32 spw, sph;
                eng_scale_dims(g.surfaces[target].w, g.surfaces[target].h, &spw, &sph);
                if (spw != g.surfaces[target].w || sph != g.surfaces[target].h) {
                    const float rx = (float)spw / (float)g.surfaces[target].w;
                    const float ry = (float)sph / (float)g.surfaces[target].h;
                    u32 x0 = (u32)((float)sx * rx + 0.5f);
                    u32 y0 = (u32)((float)sy * ry + 0.5f);
                    u32 x1 = (u32)((float)(sx + sw) * rx + 0.5f);
                    u32 y1 = (u32)((float)(sy + sh) * ry + 0.5f);
                    if (x0 > spw) x0 = spw;
                    if (y0 > sph) y0 = sph;
                    if (x1 > spw) x1 = spw;
                    if (y1 > sph) y1 = sph;
                    sx = x0; sy = y0;
                    sw = x1 > x0 ? x1 - x0 : 0u;
                    sh = y1 > y0 ? y1 - y0 : 0u;
                }
            }
            if (!m2d || !g.bind_dedup_valid || g.last_sc[0] != sx || g.last_sc[1] != sy ||
                g.last_sc[2] != sw || g.last_sc[3] != sh) {
                g.be->set_scissor(g.be->user, sx, sy, sw, sh);
                g.last_sc[0] = sx; g.last_sc[1] = sy; g.last_sc[2] = sw; g.last_sc[3] = sh;
            }
        }
        { const u32 sref = rsx_dsp_reg(&g.rsx, M_STENCIL_FUNC_REF) & 0xFFu;
          if (!m2d || !g.bind_dedup_valid || g.last_sref != sref) {
              g.be->set_stencil_ref(g.be->user, sref);
              g.last_sref = sref;
          } }
        if (g.be->set_depth_bounds) {
            const u32 lo = rsx_dsp_reg(&g.rsx, 0x0384);
            const u32 hi = rsx_dsp_reg(&g.rsx, 0x0388);
            if (!m2d || !g.last_db_valid || g.last_db_lo != lo || g.last_db_hi != hi) {
                float min, max;
                memcpy(&min, &lo, sizeof min);
                memcpy(&max, &hi, sizeof max);
                g.be->set_depth_bounds(g.be->user, min, max);
                g.last_db_lo = lo; g.last_db_hi = hi; g.last_db_valid = 1;
            }
        }
        /* sp00nznet/ps3recomp #782: blend constant color for CONSTANT_COLOR/ALPHA. */
        if (g.be->set_blend_color) {
            const u32 c = rsx_dsp_reg(&g.rsx, M_BLEND_COLOR);
            const u32 c2 = rsx_dsp_reg(&g.rsx, M_BLEND_COLOR2);
            const u8 fp16 = (u8)(rsp->rt_fp16 != 0);
            if (!m2d || !g.last_bc_valid || g.last_bc_c != c || g.last_bc_c2 != c2 || g.last_bc_fp16 != fp16) {
                float k[4];
                if (fp16) {
                    k[0] = (float)(c & 0xFFFFu) / 65535.0f;  k[1] = (float)(c >> 16) / 65535.0f;
                    k[2] = (float)(c2 & 0xFFFFu) / 65535.0f; k[3] = (float)(c2 >> 16) / 65535.0f;
                } else {
                    k[0] = (float)((c >> 16) & 0xFFu) / 255.0f; k[1] = (float)((c >> 8) & 0xFFu) / 255.0f;
                    k[2] = (float)(c & 0xFFu) / 255.0f;         k[3] = (float)(c >> 24) / 255.0f;
                }
                g.be->set_blend_color(g.be->user, k);
                g.last_bc_c = c; g.last_bc_c2 = c2; g.last_bc_fp16 = fp16; g.last_bc_valid = 1;
            }
        }
        /* sp00nznet/ps3recomp #702: guest polygon offset (SET_POLY_OFFSET_*). */
        if (g.be->set_depth_bias) {
            const u32 en_reg = (topology == RSX_TOPOLOGY_POINTS) ? M_POLY_OFFSET_POINT
                             : (topology == RSX_TOPOLOGY_LINES || topology == RSX_TOPOLOGY_LINE_STRIP) ? M_POLY_OFFSET_LINE
                             : M_POLY_OFFSET_FILL;
            const u32 en = rsx_dsp_reg(&g.rsx, en_reg) & 1u;
            const u32 sf_raw = en ? rsx_dsp_reg(&g.rsx, M_POLY_OFFSET_FACTOR) : 0u;
            const u32 u_raw  = en ? rsx_dsp_reg(&g.rsx, M_POLY_OFFSET_UNITS)  : 0u;
            const u32 dbits  = (sf.depth_format == 1u) ? 16u : 24u;
            if (!m2d || !g.last_bias_valid || g.last_bias_en != en ||
                g.last_bias_sf != sf_raw || g.last_bias_u != u_raw || g.last_bias_bits != dbits) {
                float slope = 0.0f, units = 0.0f;
                if (en) {
                    memcpy(&slope, &sf_raw, sizeof slope);
                    memcpy(&units, &u_raw, sizeof units);
                }
                g.be->set_depth_bias(g.be->user, slope, units, dbits);
                g.last_bias_en = en; g.last_bias_sf = sf_raw;
                g.last_bias_u = u_raw; g.last_bias_bits = dbits; g.last_bias_valid = 1;
            }
        }
        g.last_dyn_gen = g.rsx.dyn_gen;
        g.last_dyn_surf_gen = g.rsx.surf_gen;
    }
    g.bind_dedup_valid = 1;
    if (timing) eng_phase_mark(&g.st_ms_bind);

    const u32 uploaded = indexed ? dc.n_verts : n_draw;
    /* The cache's block when this group was a hit, the scratch buffer when it
     * was converted here. */
    const void* draw_verts = dc.verts_cache ? (const void*)dc.verts_cache
                                            : (const void*)dc.verts;
    if (trace) {
        const float* v0 = (const float*)draw_verts;
        fprintf(stderr, "[rsx engine] draw prim %u: %u verts stride %u, %u indices, pipeline %u fixed %d, target %u depth %u, tex mask 0x%X, "
                        "xf %.4g %.4g %.4g | %.4g %.4g %.4g, surface %ux%u, first floats %.4g %.4g %.4g %.4g\n",
                prim, uploaded, layout->stride, indexed ? n_draw : 0u, pipeline, pipeline_is_fixed,
                handles[0], depth_handle, tex_mask, xf[0], xf[1], xf[2], xf[4], xf[5], xf[6],
                (unsigned)W, (unsigned)H, v0[0], v0[1], v0[2], v0[3]);
        fprintf(stderr, "[rsx engine]   fp constants %u first %.4g %.4g %.4g %.4g | blend %u sf 0x%X df 0x%X | "
                        "depth test %u write %u func 0x%X | cull %u face 0x%X front 0x%X | mask 0x%X | alpha test %u | stencil %u\n",
                g.fp_constants.count, ((const float*)g.fp_cb)[0], ((const float*)g.fp_cb)[1],
                ((const float*)g.fp_cb)[2], ((const float*)g.fp_cb)[3], rsp->blend_enable,
                rsp->sf_rgb, rsp->df_rgb, rsp->depth_test, rsp->depth_write, rsp->depth_func,
                rsp->cull_enable, rsp->cull_face, rsp->front_face, rsp->color_mask,
                rsp->alpha_test_enable, rsp->stencil_enable);
        fprintf(stderr, "[rsx engine]   targets");
        for (u32 t = 0; t < n_targets; t++) fprintf(stderr, " %u", handles[t]);
        fprintf(stderr, " | textures");
        for (u32 t = 0; t < RSX_BE_MAX_TEXTURES; t++)
            if ((tex_mask >> t) & 1u) fprintf(stderr, " %u:%u", t, textures[t]);
        fprintf(stderr, " | target 0x%08X unit kinds", g.surfaces[target].offset);
        for (u32 t = 0; t < RSX_BE_MAX_TEXTURES; t++)
            if ((tex_mask >> t) & 1u) fprintf(stderr, " %u", s_tunit[t]);
        fprintf(stderr, "\n");
    }
    const u64 sp4 = mprof ? eng_rdtsc() : 0;
    if (__builtin_expect(mprof, 0)) s_m2f_p.cy_slow_rest += sp4 - sp3;
    g.be->draw(g.be->user, topology, draw_verts, uploaded, layout->stride,
               indexed ? draw_indices : NULL, indexed ? n_draw : 0);
    if (__builtin_expect(mprof, 0)) {
        s_m2f_p.cy_slow_vk += eng_rdtsc() - sp4;
        s_m2f_p.slow_hits++;
    }
    if (__builtin_expect(g.upscale_hooks, 0)) {
        if (upscale_q) {
            g.upscale_scene_zslot = zslot;
            g.upscale_depth_captured = 0;
        } else if (upscale_evt & RSX_UPSCALE_EVT_COMPOSITE) {
            if (!g.upscale_depth_captured && g.upscale_scene_zslot != ENG_INVALID &&
                g.be->upscale_capture_depth) {
                g.be->upscale_capture_depth(g.be->user, g.zdepths[g.upscale_scene_zslot].handle);
                g.upscale_depth_captured = 1;
            }
            if (g.be->upscale_capture_color)
                g.be->upscale_capture_color(g.be->user, g.surfaces[target].handle);
        }
    }
    if (timing) eng_phase_mark(&g.st_ms_backend_draw);
    /* Counted only when the draw ran the guest's OWN programs, which is what
     * the test hook means: a draw through the built-in pair is a draw, not
     * evidence that the decompile-translate-compile path worked. */
    g.draw_seq++;
    g.st_draws++;
    g.st_verts += uploaded;
    if (!pipeline_is_fixed) g.guest_draws++;

    if (rsp->color_mask != 0u) {
        if (!g.surfaces[targets[0]].cwritten) { g.surfaces[targets[0]].cwritten = 1; g.surface_gen++; }
        g.surfaces[targets[0]].feedback_valid = 0;
    }
    for (u32 i = 1; i < n_targets; i++) {
        const u32 nib = (rsp->color_mask_mrt >> (i * 4u)) & 0xFu;
        if (nib != 0u || (rsp->color_mask != 0u && rsp->color_mask_mrt == 0u)) {
            if (!g.surfaces[targets[i]].cwritten) { g.surfaces[targets[i]].cwritten = 1; g.surface_gen++; }
            g.surfaces[targets[i]].feedback_valid = 0;
        }
    }
    if (zslot != ENG_INVALID && rsp->depth_test && rsp->depth_write) {
        g.zdepths[zslot].had_write = 1;
        g.zdepths[zslot].snapshot_valid = 0;
        g.zdepths[zslot].color_snapshot_valid = 0;
    } else if (zslot != ENG_INVALID && rsp->stencil_enable && rsp->s_write_mask != 0u) {
        g.zdepths[zslot].color_snapshot_valid = 0;
    }
    if (m2f && !g.last_tex_had_dynamic && g.last_vtex_mask == 0u &&
        !(upscale_evt & RSX_UPSCALE_EVT_COMPOSITE) &&
        g.bind_dedup_valid && g.tex_bind_valid && g.vp_cache_valid && g.surf_cache_valid &&
        s_last_prog && s_last_prog->pg == prog && s_last_prog->frame == g.frames &&
        s_last_prog->swd_checked_frame == g.frames + 1u) {
        s_draw_fast.valid = 1;
        s_draw_fast.pass_serial = g.pass_serial;
        s_draw_fast.rebuild = (u8)rebuild;
        s_draw_fast.fixed = (u8)pipeline_is_fixed;
        s_draw_fast.topology = topology;
        s_draw_fast.prim = prim;
        s_draw_fast.frame = g.frames;
        s_draw_fast.vp_gen = g.rsx.vp_gen;
        s_draw_fast.vp_start = rsx_dsp_vp_start(&g.rsx);
        s_draw_fast.fp_reg = rsx_dsp_reg(&g.rsx, 0x08E4);
        s_draw_fast.fp_ctrl = rsx_dsp_shader_control(&g.rsx);
        s_draw_fast.pipe_gen = g.rsx.pipe_gen;
        s_draw_fast.surf_gen = g.rsx.surf_gen;
        s_draw_fast.vp_geom_gen = g.rsx.vp_geom_gen;
        s_draw_fast.dyn_gen = g.rsx.dyn_gen;
        s_draw_fast.surface_gen = g.surface_gen;
        s_draw_fast.tex_gen = g.tex_gen;
        s_draw_fast.clear_gen = g.clear_gen;
        s_draw_fast.n_zdepths = g.n_zdepths;
        s_draw_fast.tex_state_gen = s_tex_state_gen;
        s_draw_fast.prog_epoch = s_prog_epoch;
        s_draw_fast.prog = prog;
        s_draw_fast.upscale_q = (u8)upscale_q;
        s_draw_fast.zslot = zslot;
        memcpy(s_draw_fast.xf, xf, sizeof xf);
        if (g.upscale_hooks) s_draw_fast.upscale_di = upscale_di;
    } else {
        s_draw_fast.valid = 0;
    }
}

static void sink_end(void* user, const rsx_dispatch* r)
{
    if (!eng_stats_on()) { sink_end_impl(user, r); return; }
    const double t0 = eng_now_ms();
    sink_end_impl(user, r);
    g.st_ms_group += eng_now_ms() - t0;
    g.st_groups++;
}

static void sink_clear(void* user, const rsx_dispatch* r, u32 mask)
{
    (void)user; (void)r;
    if (!g.ready) return;
    u32 targets[RSX_BE_MAX_COLOR_TARGETS];
    const u32 n_targets = eng_current_target_set(targets);
    if (!n_targets) return;
    /* sp00nznet/ps3recomp #811: an empty scissor clips clears too. */
    {
        const u32 sh_reg = rsx_dsp_reg(&g.rsx, M_SCISSOR_HORIZONTAL);
        const u32 sv_reg = rsx_dsp_reg(&g.rsx, M_SCISSOR_VERTICAL);
        const u32 gx = sh_reg & 0xFFFFu, gw = sh_reg >> 16;
        const u32 gy = sv_reg & 0xFFFFu, gh = sv_reg >> 16;
        const int written_empty =
            (gw == 0u && g.rsx.seen[M_SCISSOR_HORIZONTAL >> 2]) ||
            (gh == 0u && g.rsx.seen[M_SCISSOR_VERTICAL >> 2]);
        if (written_empty) return;
        if (gw > 0u && gh > 0u &&
            (gx >= g.surfaces[targets[0]].w || gy >= g.surfaces[targets[0]].h))
            return;
    }
    eng_pass_note(targets, n_targets);

    if (mask & (RSX_CLEAR_COLOR_R | RSX_CLEAR_COLOR_G |
                RSX_CLEAR_COLOR_B | RSX_CLEAR_COLOR_A)) {
        const u32 c = rsx_dsp_clear_color(&g.rsx);
        const float rgba[4] = {
            (float)((c >> 16) & 0xFF) / 255.0f,
            (float)((c >>  8) & 0xFF) / 255.0f,
            (float)( c        & 0xFF) / 255.0f,
            (float)((c >> 24) & 0xFF) / 255.0f,
        };
        /* Every target the set names, not only A: a deferred pass clears its
         * whole G-buffer in one CLEAR_SURFACE, and B, C and D would otherwise
         * keep whatever the last frame left in them. */
        for (u32 i = 0; i < n_targets; i++) {
            g.be->clear_color(g.be->user, g.surfaces[targets[i]].handle, rgba);
            g.surfaces[targets[i]].feedback_valid = 0;
            g.surfaces[targets[i]].feedback_pass = 0;
            g.surfaces[targets[i]].feedback_runlen = 0;
            g.surfaces[targets[i]].feedback_last_draw = 0;
        }
        g.clear_gen++;
    }
    if (mask & (RSX_CLEAR_DEPTH | RSX_CLEAR_STENCIL)) {
        rsx_dsp_surface sf;
        rsx_dsp_get_surface(&g.rsx, &sf);
        const u32 zslot = eng_zdepth_get(sf.zeta_location, sf.zeta_offset,
                                         sf.clip_w, sf.clip_h);
        if (zslot == ENG_INVALID) return;
        u32 flags = 0;
        if (mask & RSX_CLEAR_DEPTH)   flags |= RSX_BE_CLEAR_DEPTH;
        if (mask & RSX_CLEAR_STENCIL) flags |= RSX_BE_CLEAR_STENCIL;
        /* ZSTENCIL_CLEAR_VALUE is Z24S8: depth24 << 8 | stencil8, with the
         * nv40 reset 0xFFFFFF00 seeded by rsx_dispatch_init, so a stream that
         * never writes it still clears to 1.0 / 0. */
        if (__builtin_expect(g.upscale_hooks, 0) && (mask & RSX_CLEAR_DEPTH) &&
            !g.upscale_depth_captured && zslot == g.upscale_scene_zslot &&
            rsx_upscale_rec_qualifying() > 0 && g.be->upscale_capture_depth) {
            g.be->upscale_capture_depth(g.be->user, g.zdepths[zslot].handle);
            g.upscale_depth_captured = 1;
        }
        const u32 zs = rsx_dsp_reg(&g.rsx, M_ZSTENCIL_CLEAR);
        g.be->clear_depth_stencil(g.be->user, g.zdepths[zslot].handle, flags,
                                  zs ? (float)(zs >> 8) / 16777215.0f : 1.0f,
                                  (u8)(zs & 0xFFu));
        g.zdepths[zslot].cleared = 1;
        g.clear_gen++;
        if (mask & (RSX_CLEAR_DEPTH | RSX_CLEAR_STENCIL))
            g.zdepths[zslot].color_snapshot_valid = 0;
        if (mask & RSX_CLEAR_DEPTH) {
            /* A clear invalidates the older published depth image; the next
             * texture consumer resolves the newly written pass exactly once. */
            g.zdepths[zslot].had_write = 0;
            g.zdepths[zslot].snapshot_valid = 0;
        }
    }
}

/* Resolve a flip's buffer id to a REGISTERED surface rather than whatever is
 * currently bound: the live target at that instant is often an offscreen
 * shadow or post-process surface, and copying it presents black despite the
 * scene's draws having executed (rsx_live_draw.c:7557-7570). */
static u32 eng_present_surface(u32 buffer_id)
{
    if (buffer_id < 8 && g.display_buffers[buffer_id].valid) {
        const eng_display_buffer* d = &g.display_buffers[buffer_id];
        for (u32 i = 0; i < g.n_surfaces; i++)
            if (g.surfaces[i].handle && g.surfaces[i].location == d->location &&
                g.surfaces[i].offset == d->offset)
                return i;
    }
    return eng_current_surface();
}

/* Shared LOD_CLASS_SCALE (ported from rsx_d3d12_backend.c so Vulkan gets it):
 * scales the 6 draw-distance tier floats in the EBOOT's 8-float table at
 * 0x0078AD80 (400 750 1500 3500 6000 12000). No-op when unset or 1.0. */
static u32 eng_lod_be32(u32 ea)
{
    u32 v;
    memcpy(&v, vm_base + ea, 4);
    return __builtin_bswap32(v);
}

static void eng_lod_class_tick(void)
{
    static float k = -1.0f;
    static u32 ea = 0u;
    static int gave_up = 0;
    static const u32 sig[8] = {
        0x00000000u, 0x43C80000u, 0x443B8000u, 0x44BB8000u,
        0x455AC000u, 0x45BB8000u, 0x463B8000u, 0x49742400u
    };
    if (__builtin_expect(k < 0.0f, 0)) {
        const char* e = getenv("LOD_CLASS_SCALE");
        k = (e && *e) ? (float)atof(e) : 1.0f;
        if (!(k >= 0.25f && k <= 20.0f)) k = 1.0f;
        if (k != 1.0f)
            fprintf(stderr, "[LODX] draw-distance table x%.2f (LOD_CLASS_SCALE)\n", k);
    }
    if (__builtin_expect(k == 1.0f || gave_up || !vm_base, 1)) return;
    if (ppu_vm_size && ppu_vm_size < 0x009F0000u) { gave_up = 1; return; }
    if (!ea) {
        u32 cand = 0x0078AD80u, ok = 1u;
        for (int i = 0; i < 8 && ok; i++) ok = (eng_lod_be32(cand + 4u * (u32)i) == sig[i]);
        if (!ok) {
            cand = 0u;
            for (u32 a = 0x00700000u; a < 0x009F0000u && !cand; a += 16u) {
                if (eng_lod_be32(a + 4u) != sig[1] || eng_lod_be32(a + 28u) != sig[7]) continue;
                u32 m = 1u;
                for (int i = 0; i < 8 && m; i++) m = (eng_lod_be32(a + 4u * (u32)i) == sig[i]);
                if (m) cand = a;
            }
        }
        if (!cand) {
            gave_up = 1;
            fprintf(stderr, "[LODX] draw-distance table NOT found -- scale inactive\n");
            return;
        }
        ea = cand;
        fprintf(stderr, "[LODX] draw-distance table at 0x%08X: 400 750 1500 3500 6000 12000 -> x%.2f\n", ea, k);
    }
    for (int i = 1; i <= 6; i++) {
        float o;
        memcpy(&o, &sig[i], 4);
        const float v = o * k;
        u32 raw;
        memcpy(&raw, &v, 4);
        if (eng_lod_be32(ea + 4u * (u32)i) != raw) {
            const u32 be = __builtin_bswap32(raw);
            memcpy(vm_base + ea + 4u * (u32)i, &be, 4);
        }
    }
}

static void eng_present(u32 buffer_id)
{
    if (!g.ready) return;
    const u32 target = eng_present_surface(buffer_id);
    if (target == ENG_INVALID) return;
    /* PS3RECOMP_ENG_FRAME_LOG=1: what a flip resolves to, and every registered
     * target with its address, so a black present can be attributed to the
     * wrong target or to a frame that really is black. */
    { static int s_frame_log = -1;
      if (s_frame_log < 0) {
          const char* e = getenv("PS3RECOMP_ENG_FRAME_LOG");
          if (!e) e = getenv("PS3RECOMP_METAL_FRAME_DUMP");
          /* 2 logs every present, which is what a double-buffered title needs:
           * sampling every 120th frame aliases with a two-frame alternation. */
          s_frame_log = e ? ((*e == '2') ? 2 : 1) : 0;
      }
      if (s_frame_log && (s_frame_log == 2 || g.frames % 120 == 0)) {
          const u32 current = eng_current_surface();
          const eng_surface* sel = &g.surfaces[target];
          fprintf(stderr, "[eng frame] frame=%u flip=%u selected=%u (%u:0x%08X handle=%u "
                          "%ux%u) current=%u draws=%u surfaces=%u\n",
                  g.frames, buffer_id, target, sel->location, sel->offset,
                  sel->handle, sel->w, sel->h, current, g.guest_draws, g.n_surfaces);
          for (u32 b = 0; b < 8; b++)
              if (g.display_buffers[b].valid)
                  fprintf(stderr, "[eng frame]   display buffer %u: loc=%u offset=%08X "
                                  "%ux%u pitch=%u\n", b, g.display_buffers[b].location,
                          g.display_buffers[b].offset, g.display_buffers[b].width,
                          g.display_buffers[b].height, g.display_buffers[b].pitch);
          for (u32 i = 0; i < g.n_surfaces; ++i)
              fprintf(stderr, "[eng frame]   surface %u loc=%u offset=%08X %ux%u handle=%u\n",
                      i, g.surfaces[i].location, g.surfaces[i].offset,
                      g.surfaces[i].w, g.surfaces[i].h, g.surfaces[i].handle);
          fflush(stderr);
      } }
    /* One presented frame's numbers. The accumulators are reset on every
     * present, so a line that only prints every 20th frame still describes
     * exactly the frame it names, not a 20-frame window: read the phases as
     * milliseconds of that frame. The backend's [engtime] gpu line carries
     * the same frame number and its fence waits. */
    /* This frame's draws, before the per-frame counters are reset below; the
     * A/B summary at the end of the function reports them. */
    const u32 frame_draws = (u32)g.st_draws;
    static int s_stats = -1;
    if (s_stats < 0) s_stats = getenv("PS3RECOMP_ENG_STATS") ? 1 : 0;
    if (s_stats && g.frames && g.frames % 20 == 0) {
        fprintf(stderr, "[eng stats] frame %u: %llu draws, %llu verts, %llu texture hashes "
                        "(%llu MB, %llu full), %llu uploads, %llu tex lookups/%llu probes, "
                        "%llu pipeline lookups/%llu probes, %llu feedback copies, "
                        "%llu handle-0 binds, %llu pipeline-full\n", g.frames,
                (unsigned long long)g.st_draws, (unsigned long long)g.st_verts,
                (unsigned long long)g.st_hashes,
                (unsigned long long)(g.st_hash_bytes >> 20),
                (unsigned long long)g.st_verifies,
                (unsigned long long)g.st_uploads,
                (unsigned long long)g.st_lookups, (unsigned long long)g.st_probes,
                (unsigned long long)g.st_plookups, (unsigned long long)g.st_pprobes,
                (unsigned long long)g.st_feedback,
                (unsigned long long)g.st_handle0, (unsigned long long)g.st_pipe_full);
        fprintf(stderr, "[engtime] frame %u phase %d: group %.2f ms, fetch %.2f, pipeline %.2f "
                        "(program %.2f, constants %.2f), textures %.2f (hash %.2f), "
                        "bind %.2f, draw %.2f | %llu groups, "
                        "%.0f draws/frame, %.2f us/group\n", g.frames, ENG_AB_PHASE(),
                g.st_ms_group, g.st_ms_fetch, g.st_ms_pipeline,
                g.st_ms_prog, g.st_ms_consts,
                g.st_ms_textures, g.st_ms_hash,
                g.st_ms_bind, g.st_ms_backend_draw,
                (unsigned long long)g.st_groups, (double)g.st_draws,
                g.st_groups ? g.st_ms_group * 1000.0 / (double)g.st_groups : 0.0);
        if (s_vc_cfg > 0) {
            const u64 vc_verts = g.vc_hit_verts + g.vc_miss_verts;
            fprintf(stderr, "[ENGVC] frame %u phase %d: %llu hits / %llu misses "
                            "(%.1f%% of %llu verts), %llu converted, %.1f MB hashed, "
                            "refs %.2f ms, key %.2f, convert %.2f | "
                            "%llu evict, %llu unkeyed, %u MB/%u ents\n", g.frames, ENG_AB_PHASE(),
                    (unsigned long long)g.vc_hits, (unsigned long long)g.vc_misses,
                    vc_verts ? 100.0 * (double)g.vc_hit_verts / (double)vc_verts : 0.0,
                    (unsigned long long)vc_verts,
                    (unsigned long long)g.vc_miss_verts,
                    (double)g.vc_hash_bytes / (1024.0 * 1024.0),
                    g.st_ms_vcrefs, g.st_ms_vckey, g.st_ms_vcconvert,
                    (unsigned long long)g.vc_evict,
                    (unsigned long long)g.vc_unkeyed,
                    (unsigned)(g.vc_bump >> 20), g.vc_nents);
        }
        if (s_mc_cfg > 0) {
            fprintf(stderr, "[ENGMC] frame %u phase %d: %llu hits (%llu swd, %llu ver, %llu ahash) / "
                            "%llu misses (%llu dirty, %llu vermiss, %llu cold, %llu unelig) | "
                            "%llu nonspu-pages\n", g.frames, ENG_AB_PHASE(),
                    (unsigned long long)g.mc_hits, (unsigned long long)g.mc_hits_swd,
                    (unsigned long long)g.mc_hits_ver, (unsigned long long)g.mc_hits_ahash,
                    (unsigned long long)g.mc_misses, (unsigned long long)g.mc_miss_dirty,
                    (unsigned long long)g.mc_miss_ver, (unsigned long long)g.mc_miss_cold,
                    (unsigned long long)g.mc_miss_unelig,
                    (unsigned long long)g.mc_nonspu_pages);
        }
        fflush(stderr);
    }
    /* Snapshot per-frame engine stats before reset so SLOWFRAME_MS can pair
     * them with the exact frame's present/GPU/FIFO/idle wall time below. */
    const int sf_on = g_rsx_sf.enabled;
    const double sf_grp = g.st_ms_group, sf_fetch = g.st_ms_fetch;
    const double sf_pipe = g.st_ms_pipeline, sf_prog = g.st_ms_prog, sf_consts = g.st_ms_consts;
    const double sf_tex = g.st_ms_textures, sf_hash = g.st_ms_hash;
    const double sf_bind = g.st_ms_bind, sf_draw = g.st_ms_backend_draw;
    const double sf_vcrefs = g.st_ms_vcrefs, sf_vckey = g.st_ms_vckey, sf_vcconv = g.st_ms_vcconvert;
    const u64 sf_uploads = g.st_uploads, sf_vc_m = g.vc_misses, sf_mc_m = g.mc_misses;
    const u64 sf_pipe_full = g.st_pipe_full;
    const u32 sf_frame = g.frames;

    /* LRU eviction of 1-frame dynamic meshes in vc_arena is normal steady-state
     * recycling; never wipe the arena or mc_ents mid-scene. */
    g.st_draws = g.st_hashes = g.st_hash_bytes = g.st_uploads = g.st_verts = 0;
    g.st_verifies = 0;
    g.st_feedback = 0;
    g.vc_hits = g.vc_misses = g.vc_hit_verts = g.vc_miss_verts = 0;
    g.vc_hash_bytes = g.vc_evict = g.vc_unkeyed = 0;
    g.st_ms_vcrefs = g.st_ms_vckey = g.st_ms_vcconvert = 0.0;
    g.mc_hits = g.mc_hits_swd = g.mc_hits_ver = g.mc_hits_ahash = 0;
    g.mc_misses = g.mc_miss_dirty = g.mc_miss_ver = g.mc_miss_cold = g.mc_miss_unelig = 0;
    g.st_lookups = g.st_probes = g.st_plookups = g.st_pprobes = 0;
    g.st_handle0 = g.st_pipe_full = 0;
    g.st_groups = 0;
    g.st_ms_group = g.st_ms_fetch = g.st_ms_pipeline = g.st_ms_textures = 0.0;
    g.st_ms_bind = g.st_ms_backend_draw = g.st_ms_hash = 0.0;
    g.st_ms_prog = g.st_ms_consts = 0.0;
    g.last_present_surface = target;
    if (__builtin_expect(g.upscale_hooks, 0)) {
        if (!g.upscale_depth_captured && g.upscale_scene_zslot != ENG_INVALID &&
            rsx_upscale_rec_qualifying() > 0 && g.be->upscale_capture_depth) {
            g.be->upscale_capture_depth(g.be->user, g.zdepths[g.upscale_scene_zslot].handle);
            g.upscale_depth_captured = 1;
        }
        rsx_upscale_handoff(1);
        g.upscale_depth_captured = 0;
        g.upscale_scene_zslot = ENG_INVALID;
        g.upscale_rec_idx = 0;
    }
    const double sf_pres0 = sf_on ? eng_now_ms() : 0.0;
    g.be->present(g.be->user, g.surfaces[target].handle);
    { static int s_fprof = -1;
      static u32 fp_n = 0;
      static double fp_prev = 0.0, fp_wall = 0.0, fp_fifo = 0.0;
      static u64 fp_draws = 0;
      if (__builtin_expect(s_fprof < 0, 0)) {
          const char* e = getenv("RSX_FIFO_PROF");
          s_fprof = (e && *e == '1') ? 1 : 0;
      }
      if (__builtin_expect(s_fprof, 0)) {
          const double now = eng_now_ms();
          const double wms = fp_prev > 0.0 ? (now - fp_prev) : 0.0;
          fp_prev = now;
          const double fms = g_rsx_fifo_prof_ms;
          g_rsx_fifo_prof_ms = 0.0;
          if (frame_draws > 0 && wms > 0.0 && wms < 60000.0) {
              fp_wall += wms; fp_fifo += fms; fp_draws += frame_draws;
              if (++fp_n >= 300u) {
                  fprintf(stderr, "[FIFOPROF] 300f fr=%u ph=%d: fifo=%.2f ms/f | wall=%.2f ms/f (%.2f fps) | draws/f=%.0f\n",
                          g.frames, ENG_AB_PHASE(), fp_fifo / (double)fp_n,
                          fp_wall / (double)fp_n, 1000.0 * (double)fp_n / fp_wall,
                          (double)fp_draws / (double)fp_n);
                  fflush(stderr);
                  fp_n = 0; fp_wall = 0.0; fp_fifo = 0.0; fp_draws = 0;
              }
          }
      } }
    if (__builtin_expect(s_m2f_p.on == 1, 0)) {
        if (!s_m2f_p.t0) { s_m2f_p.t0 = eng_rdtsc(); s_m2f_p.ms0 = eng_now_ms(); s_m2f_p.fr0 = g.frames; }
        const u32 df = g.frames - s_m2f_p.fr0;
        if (df >= 300u) {
            const double dms = eng_now_ms() - s_m2f_p.ms0;
            const u64 dt = eng_rdtsc() - s_m2f_p.t0;
            const double scale = (dt && dms > 0.0) ? (dms / (double)dt / (double)df) : 0.0;
            fprintf(stderr,
                "[M2F-PROF] %uf fr=%u: fast=%llu/f(mc=%.2f vp=%.2f vk=%.2f) slow=%llu/f(prog=%.2f fetch=%.2f tex=%.2f rest=%.2f vk=%.2f) | miss: tex=%llu prog=%llu pipe=%llu surf=%llu dyn=%llu mc=%llu oth=%llu\n",
                df, g.frames,
                (unsigned long long)(s_m2f_p.fast_hits / df),
                s_m2f_p.cy_fast_mc * scale, s_m2f_p.cy_fast_vp * scale, s_m2f_p.cy_fast_vk * scale,
                (unsigned long long)(s_m2f_p.slow_hits / df),
                s_m2f_p.cy_slow_prog * scale, s_m2f_p.cy_slow_fetch * scale,
                s_m2f_p.cy_slow_tex * scale, s_m2f_p.cy_slow_rest * scale, s_m2f_p.cy_slow_vk * scale,
                (unsigned long long)(s_m2f_p.miss_tex / df),
                (unsigned long long)(s_m2f_p.miss_prog / df),
                (unsigned long long)(s_m2f_p.miss_pipe / df),
                (unsigned long long)(s_m2f_p.miss_surf / df),
                (unsigned long long)(s_m2f_p.miss_dyn / df),
                (unsigned long long)(s_m2f_p.miss_mc / df),
                (unsigned long long)(s_m2f_p.miss_other / df));
            fflush(stderr);
            s_m2f_p.t0 = eng_rdtsc(); s_m2f_p.ms0 = eng_now_ms(); s_m2f_p.fr0 = g.frames;
            s_m2f_p.fast_hits = s_m2f_p.slow_hits = 0;
            s_m2f_p.miss_tex = s_m2f_p.miss_prog = s_m2f_p.miss_pipe = s_m2f_p.miss_surf = 0;
            s_m2f_p.miss_dyn = s_m2f_p.miss_mc = s_m2f_p.miss_other = 0;
            s_m2f_p.cy_fast_mc = s_m2f_p.cy_fast_vp = s_m2f_p.cy_fast_vk = 0;
            s_m2f_p.cy_slow_prog = s_m2f_p.cy_slow_fetch = s_m2f_p.cy_slow_tex = 0;
            s_m2f_p.cy_slow_rest = s_m2f_p.cy_slow_vk = 0;
        }
    }
    if (sf_on) {
        static double sf_prev_ms = 0.0;
        static u32 sf_n = 0, sf_logged = 0;
        static double sum_wall, sum_idle, sum_fifo, sum_sema, sum_grp, sum_fother;
        static double sum_pres, sum_pwait, sum_fwait, sum_pump, sum_tick;
        static double sum_fetch, sum_tex, sum_pipe, sum_prog, sum_consts, sum_bind, sum_draw;
        static double sum_gpu;
        static u64 sum_idle_n, sum_sub, sum_pcr, sum_scr, sum_sflush, sum_ffast, sum_fslow, sum_draws;
        const double sf_now = eng_now_ms();
        const double pres_ms = sf_now - sf_pres0;
        const double wall_ms = sf_prev_ms > 0.0 ? (sf_now - sf_prev_ms) : 0.0;
        sf_prev_ms = sf_now;
        const double fother = g_rsx_sf.fifo_ms - g_rsx_sf.sema_ms - sf_grp;
        if (frame_draws > 0 && wall_ms > 0.0 && wall_ms < 60000.0) {
            if (wall_ms >= g_rsx_sf.threshold_ms && frame_draws >= g_rsx_sf.min_draws && sf_logged < 300u) {
                sf_logged++;
                const u64 fw_tot = g_rsx_sf.fifo_fast_words + g_rsx_sf.fifo_slow_words;
                fprintf(stderr,
                    "[SLOWFRAME] fr=%u wall=%.2fms | idle=%.2f(n=%u,max=%.2f) fifo=%.2f(n=%u,max=%.2f sema=%.2f/%uw/%ub grp=%.2f fother=%.2f fast=%.0f%%/%lluk) "
                    "pres=%.2f(pwait=%.2f) | guest: gflip=%.2f tw=%.2f(w%d=%.2f) fs=%.2f(m=%.2f) fios=%.2f(m=%.2f) | "
                    "grp: fetch=%.2f(r=%.2f,k=%.2f,c=%.2f,vc=%llu,mc=%llu) "
                    "tex=%.2f(h=%.2f,up=%llu) pipe=%.2f(prog=%.2f,c=%.2f,pf=%llu) bind=%.2f draw=%.2f | "
                    "vk: gpu=%.2f(fw=%.2f,sub=%u,sf=%u) pcr=%u(%.1fms) scr=%u(%.1fms) | draws=%u\n",
                    sf_frame, wall_ms,
                    g_rsx_sf.idle_ms, g_rsx_sf.idle_calls, g_rsx_sf.idle_max_ms,
                    g_rsx_sf.fifo_ms, g_rsx_sf.fifo_calls, g_rsx_sf.fifo_max_ms,
                    g_rsx_sf.sema_ms, g_rsx_sf.sema_waits, g_rsx_sf.sema_blocks,
                    sf_grp, fother, fw_tot ? 100.0 * (double)g_rsx_sf.fifo_fast_words / (double)fw_tot : 0.0,
                    (unsigned long long)(fw_tot / 1000ull),
                    pres_ms, g_rsx_sf.vk_pres_wait_ms,
                    g_rsx_sf.hz_gflip_max_ms, g_rsx_sf.hz_tw_ms, g_rsx_sf.hz_tw_max_wid, g_rsx_sf.hz_tw_max_ms,
                    g_rsx_sf.hz_fs_ms, g_rsx_sf.hz_fs_main_ms, g_rsx_sf.hz_fios_ms, g_rsx_sf.hz_fios_main_ms,
                    sf_fetch, sf_vcrefs, sf_vckey, sf_vcconv,
                    (unsigned long long)sf_vc_m, (unsigned long long)sf_mc_m,
                    sf_tex, sf_hash, (unsigned long long)sf_uploads,
                    sf_pipe, sf_prog, sf_consts, (unsigned long long)sf_pipe_full,
                    sf_bind, sf_draw,
                    g_rsx_sf.vk_gpu_wait_ms, g_rsx_sf.vk_flush_wait_ms,
                    g_rsx_sf.vk_submits, g_rsx_sf.vk_sync_flushes,
                    g_rsx_sf.vk_pipe_creates, g_rsx_sf.vk_pipe_ms,
                    g_rsx_sf.vk_shader_creates, g_rsx_sf.vk_shader_ms,
                    frame_draws);
                fflush(stderr);
            }
            sum_wall += wall_ms; sum_idle += g_rsx_sf.idle_ms; sum_idle_n += g_rsx_sf.idle_calls;
            sum_fifo += g_rsx_sf.fifo_ms; sum_sema += g_rsx_sf.sema_ms; sum_grp += sf_grp;
            sum_fother += fother; sum_pres += pres_ms; sum_pwait += g_rsx_sf.vk_pres_wait_ms;
            sum_fwait += g_rsx_sf.vk_flush_wait_ms; sum_pump += g_rsx_sf.pump_ms; sum_tick += g_rsx_sf.tick_ms;
            sum_fetch += sf_fetch; sum_tex += sf_tex; sum_pipe += sf_pipe; sum_prog += sf_prog;
            sum_consts += sf_consts; sum_bind += sf_bind; sum_draw += sf_draw;
            sum_gpu += g_rsx_sf.vk_gpu_wait_ms; sum_sub += g_rsx_sf.vk_submits;
            sum_pcr += g_rsx_sf.vk_pipe_creates; sum_scr += g_rsx_sf.vk_shader_creates;
            sum_sflush += g_rsx_sf.vk_sync_flushes;
            sum_ffast += g_rsx_sf.fifo_fast_words; sum_fslow += g_rsx_sf.fifo_slow_words;
            sum_draws += frame_draws;
            if (++sf_n >= 300u) {
                const double inv = 1.0 / (double)sf_n;
                const u64 fw_tot = sum_ffast + sum_fslow;
                fprintf(stderr,
                    "[FRAMEBREAK] 300f fr=%u: wall=%.2f (%.2f fps) | idle=%.2f(n=%.1f) fifo=%.2f(sema=%.2f grp=%.2f fother=%.2f fast=%.1f%% %.0fk/f) "
                    "pres=%.2f(pwait=%.2f) pump=%.2f tick=%.2f | grp: fetch=%.2f tex=%.2f pipe=%.2f(prog=%.2f c=%.2f) bind=%.2f draw=%.2f | "
                    "vk: gpu=%.2f(fw=%.2f sub=%.2f sf=%llu) pcr=%llu scr=%llu draws=%.0f\n",
                    sf_frame, sum_wall * inv, sum_wall > 0.0 ? 1000.0 * sf_n / sum_wall : 0.0,
                    sum_idle * inv, (double)sum_idle_n * inv,
                    sum_fifo * inv, sum_sema * inv, sum_grp * inv, sum_fother * inv,
                    fw_tot ? 100.0 * (double)sum_ffast / (double)fw_tot : 0.0,
                    (double)fw_tot * inv / 1000.0,
                    sum_pres * inv, sum_pwait * inv, sum_pump * inv, sum_tick * inv,
                    sum_fetch * inv, sum_tex * inv, sum_pipe * inv, sum_prog * inv, sum_consts * inv,
                    sum_bind * inv, sum_draw * inv,
                    sum_gpu * inv, sum_fwait * inv, (double)sum_sub * inv, (unsigned long long)sum_sflush,
                    (unsigned long long)sum_pcr, (unsigned long long)sum_scr, (double)sum_draws * inv);
                fflush(stderr);
                sf_n = 0;
                sum_wall = sum_idle = sum_fifo = sum_sema = sum_grp = sum_fother = 0.0;
                sum_pres = sum_pwait = sum_fwait = sum_pump = sum_tick = 0.0;
                sum_fetch = sum_tex = sum_pipe = sum_prog = sum_consts = sum_bind = sum_draw = sum_gpu = 0.0;
                sum_idle_n = sum_sub = sum_pcr = sum_scr = sum_sflush = sum_ffast = sum_fslow = sum_draws = 0;
            }
        }
        g_rsx_sf.idle_ms = g_rsx_sf.idle_max_ms = 0.0; g_rsx_sf.idle_calls = 0;
        g_rsx_sf.tick_ms = 0.0; g_rsx_sf.tick_calls = 0;
        g_rsx_sf.fifo_ms = g_rsx_sf.fifo_max_ms = 0.0; g_rsx_sf.fifo_calls = 0;
        g_rsx_sf.pump_ms = 0.0;
        g_rsx_sf.sema_ms = 0.0; g_rsx_sf.sema_waits = g_rsx_sf.sema_blocks = g_rsx_sf.sema_tmo = 0;
        g_rsx_sf.fifo_fast_words = g_rsx_sf.fifo_slow_words = 0;
        g_rsx_sf.vk_gpu_wait_ms = g_rsx_sf.vk_pres_wait_ms = g_rsx_sf.vk_flush_wait_ms = 0.0;
        g_rsx_sf.vk_pipe_ms = g_rsx_sf.vk_shader_ms = 0.0;
        g_rsx_sf.vk_submits = g_rsx_sf.vk_pipe_creates = g_rsx_sf.vk_shader_creates = 0;
        g_rsx_sf.vk_desc_misses = g_rsx_sf.vk_gvb_misses = g_rsx_sf.vk_gib_misses = 0;
        g_rsx_sf.vk_barriers = g_rsx_sf.vk_rps = g_rsx_sf.vk_sync_flushes = 0;
    }

    /* In-run A/B, per frame: the walker's own frame cadence, grouped by the
     * runtime's phase (ppu_loader.cpp PS3_AB_PHASE). A phase change prints the
     * window it closes, so a knob's effect is the difference between two
     * phases of ONE run: no second boot, no drift in the scene. Printed only
     * when a phase is actually running, so a plain run says nothing. */
    if (eng_stats_on()) {
        static int seen_phase = -1;
        static u32 ab_frames;
        static u64 ab_draws;
        static double ab_ms, ab_last;
        const double now = eng_now_ms();
        const int phase = ENG_AB_PHASE();
        if (ab_last > 0.0 && phase != seen_phase) {
            fprintf(stderr, "[ABFPS] phase %d: %u frames, %.2f ms/frame, %.2f fps, "
                            "%.0f draws/frame, %s\n",
                    seen_phase, ab_frames, ab_frames ? ab_ms / ab_frames : 0.0,
                    ab_ms > 0.0 ? 1000.0 * ab_frames / ab_ms : 0.0,
                    ab_frames ? (double)ab_draws / ab_frames : 0.0,
                    (seen_phase & 1) ? "odd: experiments off" : "even: experiments on");
            fflush(stderr);
            ab_frames = 0; ab_draws = 0; ab_ms = 0.0;
        }
        if (phase != seen_phase) { seen_phase = phase; ab_frames = 0; ab_draws = 0; ab_ms = 0.0; }
        if (ab_last > 0.0) { ab_ms += now - ab_last; ab_frames++; ab_draws += frame_draws; }
        ab_last = now;
    }

    eng_lod_class_tick();
    g.frames++;
    g.pass_serial++;
    __atomic_store_n(&s_swd_frame, g.frames + 1u, __ATOMIC_RELEASE);
    g.last_guest_draws = g.guest_draws;
    g.guest_draws = 0;
}

static void sink_flip(void* user, const rsx_dispatch* r, u32 arg)
{
    (void)user; (void)r;
    g.last_flip_buffer = arg & 7u;
    eng_present(g.last_flip_buffer);
}

/* ---- public API ---------------------------------------------------------- */

void rsx_draw_engine_set_backend(const rsx_draw_backend* backend)
{
    g.be = backend;
}

void rsx_draw_engine_set_default(int on)
{
    g.default_on = on;
}

int rsx_draw_engine_enabled(void)
{
    static int cached = -1;
    static int env_seen = 0;
    static int env_on = 0;
    if (!env_seen) {
        const char* e = getenv("PS3RECOMP_RSX_ENGINE");
        env_seen = 1;
        if (e && *e) env_on = (strcmp(e, "dispatch") == 0) ? 1 : -1;
    }
    /* The environment is authoritative in both directions; without it the
     * backend's own default decides. Cached after the first backend has
     * registered, so the FIFO walker pays one compare. */
    const int on = env_on > 0 ? 1 : (env_on < 0 ? 0 : g.default_on);
    if (!g.be) return 0;
    if (cached < 0) cached = on;
    return cached;
}

void rsx_draw_engine_set_render_size(u32 render_w, u32 render_h)
{
    s_req_render_w = render_w;
    s_req_render_h = render_h;
}

int rsx_draw_engine_init(u32 width, u32 height)
{
    if (!g.be) return -1;
    g.width  = width  ? width  : 1280;
    g.height = height ? height : 720;
    g.render_w = s_req_render_w ? s_req_render_w : g.width;
    g.render_h = s_req_render_h ? s_req_render_h : g.height;
    g.res_scaled = (u8)((g.render_w != g.width || g.render_h != g.height) ? 1 : 0);
    g.upscale_hooks = (u8)(rsx_upscale_hooks_on() ? 1 : 0);
    g.upscale_depth_captured = 0;
    g.upscale_scene_zslot = ENG_INVALID;
    g.upscale_rec_idx = 0;
    g.vp_cb = (u8*)calloc(1, ENG_VP_CB_BYTES);
    if (!g.vp_cb) return -1;
    if (g.be->init && g.be->init(g.be->user, g.width, g.height) != 0) {
        free(g.vp_cb); g.vp_cb = NULL;
        return -1;
    }
    rsx_dispatch_sink sink;
    memset(&sink, 0, sizeof sink);
    sink.clear            = sink_clear;
    sink.begin            = sink_begin;
    sink.end              = sink_end;
    sink.draw_arrays      = sink_draw_arrays;
    sink.draw_index_array = sink_draw_index;
    sink.inline_array     = sink_inline_array;
    sink.flip             = sink_flip;
    rsx_dispatch_init(&g.rsx, &sink);
    s_render_state.gen = 0xFFFFFFFFu;
    s_vattr_hash_valid = 0;
    s_last_prog = NULL;
    s_last_pipe = NULL;
    { const char* e = getenv("TEX_XF");          g.tex_xf_on          = (e && *e == '0') ? 0 : 1; }
    { const char* e = getenv("SUN_NDL_CLAMP");   g.sun_ndl_clamp_on   = (e && *e == '0') ? 0 : 1; }
    { const char* e = getenv("CT_NONE");         g.ct_none_on         = (e && *e == '0') ? 0 : 1; }
    { const char* e = getenv("ZS_CWRITTEN");     g.zs_cwritten_on     = (e && *e == '0') ? 0 : 1; }
    { const char* e = getenv("ZETA_EXACT_SIZE"); g.zeta_exact_size_on = (e && *e == '0') ? 0 : 1; }
    { const char* e = getenv("ZCONV_COLOR");     g.zconv_color_on     = (e && *e == '0') ? 0 : 1; }
    { const char* e = getenv("SHADOWS");         g.shadows_on         = (e && *e == '0') ? 0 : 1; }
    { const char* e = getenv("SHADOW_LIT");      g.shadow_lit_on      = (e && *e == '0') ? 0 : 1; }
    { const char* e = getenv("SHADOW_FLIP");     g.shadow_flip        = (e && *e == '1') ? 1 : 0; }
    { const char* e = getenv("SHADOW_RP");       g.shadow_rp_on       = (e && *e == '0') ? 0 : 1; }
    { const char* e = getenv("SAMP_BORDER");     g.samp_border_on     = (e && *e == '0') ? 0 : 1; }
    if (g.shadow_rp_on) {
        const char* eb = getenv("SHADOW_RP_BIAS");
        const char* em = getenv("SHADOW_RP_MAX");
        g.shadow_bias   = eb ? atof(eb) : 0.0003;
        g.shadow_rp_max = em ? atof(em) : 0.003;
        g.shadow_slope  = 0.0;
    } else {
        const char* eb = getenv("SHADOW_BIAS");
        const char* es = getenv("SHADOW_SLOPE");
        g.shadow_bias   = eb ? atof(eb) : 0.0015;
        g.shadow_rp_max = 0.003;
        g.shadow_slope  = es ? atof(es) : 1.5;
    }
    g.ready = 1;
    fprintf(stderr, "[rsx engine] register-file draw engine up (%ux%u, TEX_XF=%u SUN_NDL_CLAMP=%u CT_NONE=%u ZS_CWRITTEN=%u ZETA_EXACT_SIZE=%u ZCONV_COLOR=%u SHADOWS=%u SHADOW_LIT=%u SHADOW_RP=%u SAMP_BORDER=%u)\n",
            g.width, g.height, g.tex_xf_on, g.sun_ndl_clamp_on,
            g.ct_none_on, g.zs_cwritten_on, g.zeta_exact_size_on,
            g.zconv_color_on, g.shadows_on, g.shadow_lit_on,
            g.shadow_rp_on, g.samp_border_on);
    return 0;
}

void rsx_draw_engine_shutdown(void)
{
    if (!g.be) return;
    if (g.ready && g.be->submit_and_wait)
        g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_SHUTDOWN);
    for (u32 i = 0; i < g.n_textures; i++)
        if (g.textures[i].handle) g.be->texture_release(g.be->user, g.textures[i].handle);
    if (g.shadow_lit_tex) g.be->texture_release(g.be->user, g.shadow_lit_tex);
    for (u32 i = 0; i < g.n_pipelines; i++)
        if (g.pipelines[i].handle) g.be->pipeline_release(g.be->user, g.pipelines[i].handle);
    for (u32 i = 0; i < g.n_surfaces; i++)
        if (g.surfaces[i].handle) g.be->color_target_release(g.be->user, g.surfaces[i].handle);
    for (u32 i = 0; i < g.n_zdepths; i++)
        if (g.zdepths[i].handle) g.be->depth_target_release(g.be->user, g.zdepths[i].handle);
    if (g.ready && g.be->shutdown) g.be->shutdown(g.be->user);

    rsx_draw_engine_set_guest_memory(NULL, NULL);
    free(dc.refs); free(dc.cuts); free(dc.verts); free(dc.imm);
    rsx_vertex_remap_destroy(&dc.ref_remap);
    memset(&dc, 0, sizeof dc);
    free(g.tex_staging); free(g.vp_cb); free(g.fp_cb); free(g.indices);
    free(g.vc_arena); free(g.vc_ents); free(g.vc_ht); free(g.vc_seen);
#if defined(__GNUC__) || defined(__clang__)
    if (&g_vc_put_note && g_vc_put_note == eng_swd_note_spu_put)
        g_vc_put_note = NULL;
#endif
    if (g.mc_ents) {
        for (u32 i = 0; i < ENG_MC_ENTS; i++) {
            free(g.mc_ents[i].idx_buf);
            free(g.mc_ents[i].extra_spans);
        }
        free(g.mc_ents);
        g.mc_ents = NULL;
    }
    free(g.mc_tags); g.mc_tags = NULL;
    free(g.mc_seen); g.mc_seen = NULL;
    { u32* sw = __atomic_exchange_n(&s_swd_gen, NULL, __ATOMIC_ACQ_REL); free(sw); }
    memset(s_swd_coarse, 0, sizeof s_swd_coarse);
    free(s_swd_nonspu); s_swd_nonspu = NULL;
    s_swd_frame = 1u;
    /* The arena is gone with the state, so the next init has to look at the
     * environment again (the engine tests bring it up and down). */
    s_vc_cfg = -1; s_vc_phase = -1;
    s_mc_cfg = -1; s_mc_phase = -1; s_mc_ver = 64;
    s_mc_swd = 0; s_mc_swd_phase = -1;
    s_m2c_cfg = -1; s_m2c_phase = -1;
    s_m2d_cfg = -1; s_m2d_phase = -1;
    s_m2e4_cfg = -1; s_m2e4_phase = -1;
    s_m2e4b_cfg = -1; s_m2e4b_phase = -1;
    s_m2f_cfg = -1; s_m2f_phase = -1;
    s_ef.phase = -999999;
    s_draw_fast.valid = 0;
    s_mc_ver_explicit = 0;
    s_vp_pack_cfg = -1; s_vp_sparse_cfg = -1;
    s_mc_prev_slot = 0xFFFFFFFFu;
    s_mc_next_pred = 0xFFFFFFFFu;
    g_rsx_eng_cur_gcache = NULL;
    memset(s_ucache, 0, sizeof s_ucache);
    memset(s_unit_samp, 0, sizeof s_unit_samp);
    memset(s_unit_samp_gen, 0, sizeof s_unit_samp_gen);
    memset(s_unit_samp_raw, 0, sizeof s_unit_samp_raw);
    memset(s_unit_samp_raw_ok, 0, sizeof s_unit_samp_raw_ok);
    memset(s_rt_view_cache, 0, sizeof s_rt_view_cache);
    s_units_seeded = 0;
    s_enabled_tex_mask = s_cube_mask = s_vtex_mask = s_tex_xf_key = s_tex_state_gen = 0;
    memset(s_prog_fast, 0, sizeof s_prog_fast);
    memset(s_pipe_fast, 0, sizeof s_pipe_fast);
    memset(s_tex_fast, 0, sizeof s_tex_fast);
    s_last_prog = NULL;
    s_last_pipe = NULL;
    s_vattr_hash_valid = 0;
    s_prog_epoch = 1u;
    g.vc_arena = NULL; g.vc_ents = NULL; g.vc_ht = NULL; g.vc_seen = NULL;
    s_req_render_w = s_req_render_h = 0;
    const rsx_draw_backend* be = g.be;
    memset(&g, 0, sizeof g);
    g.be = be;
}

static inline u32 eng_rd_be32(const u8* p)
{
    u32 raw;
    memcpy(&raw, p, 4);
    return __builtin_bswap32(raw);
}

void rsx_draw_engine_method(u32 method, u32 arg)
{
    if (!g.ready) return;
    /* The subchannel is a binding slot, not an engine selector, and a title
     * whose SPU-built command lists bind NV4097 elsewhere would otherwise
     * store its state in the wrong register bank.  Standard NV4097 methods
     * (< 0x2000) strip the subchannel with & 0x1FFC.  Driver methods
     * (0xE9xx, 0xEBxx) encode the subchannel bits as part of the address;
     * preserve them with the wider mask. */
    u32 m = (method >= 0xE000u) ? (method & 0xFFFCu) : (method & 0x1FFCu);
    rsx_dispatch_method(&g.rsx, m, arg);
    if (g.rsx.in_begin_end && eng_imm_provokes(m)) dc_imm_push();
}

void rsx_draw_engine_vp_consts_be(u32 method, const u8* be_words, u32 count)
{
    if (!g.ready || !count) return;
    u32 i0 = 0;
    if (method == 0x1EFCu) {
        g.rsx.regs[0x1EFCu >> 2] = eng_rd_be32(be_words);
        i0 = 1;
    }
    if (i0 >= count) return;
    const u32 ld = g.rsx.regs[0x1EFCu >> 2];
    const u32 r0 = (method + 4u * i0 - 0x1F00u) >> 2;
    const u32 base = ld * 4u + r0;
    if (ld >= RSX_DSP_NUM_CONSTANTS || base >= RSX_DSP_NUM_CONSTANTS * 4u) return;
    u32 n = count - i0;
    if (base + n > RSX_DSP_NUM_CONSTANTS * 4u) n = RSX_DSP_NUM_CONSTANTS * 4u - base;
    u32* dst = &g.rsx.constants[0][0] + base;
    const u8* sp = be_words + 4u * i0;
    u32 k = 0;
    int changed = 0;
#if defined(__AVX2__)
    if (n >= 8u) {
        const __m256i sw = _mm256_setr_epi8(
            3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
            3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
        __m256i diff = _mm256_setzero_si256();
        for (; k + 8u <= n; k += 8u) {
            const __m256i nv = _mm256_shuffle_epi8(
                _mm256_loadu_si256((const __m256i*)(sp + 4u * k)), sw);
            const __m256i ov = _mm256_loadu_si256((const __m256i*)(dst + k));
            diff = _mm256_or_si256(diff, _mm256_xor_si256(nv, ov));
            _mm256_storeu_si256((__m256i*)(dst + k), nv);
        }
        if (!_mm256_testz_si256(diff, diff)) changed = 1;
    }
    if (k + 4u <= n) {
        const __m128i sw16 = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
        const __m128i nv = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(sp + 4u * k)), sw16);
        const __m128i ov = _mm_loadu_si128((const __m128i*)(dst + k));
        const __m128i d4 = _mm_xor_si128(nv, ov);
        _mm_storeu_si128((__m128i*)(dst + k), nv);
        if (!_mm_testz_si128(d4, d4)) changed = 1;
        k += 4u;
    }
#endif
    u32 sdiff = 0u;
    for (; k < n; k++) {
        const u32 d = eng_rd_be32(sp + 4u * k);
        sdiff |= d ^ dst[k];
        dst[k] = d;
    }
    if (changed || sdiff) {
        g.rsx.vp_const_gen++;
        const u32 b0 = base >> 8;
        const u32 b1 = (base + n - 1u) >> 8;
        for (u32 b = b0; b <= b1; b++) g.rsx.vp_const_block_gen[b]++;
    }
}

static u8 s_pkt_lut[2048];
static int s_pkt_lut_inited = 0;

static void eng_pkt_lut_init(void)
{
    if (s_pkt_lut_inited) return;
    memset(s_pkt_lut, 0, sizeof s_pkt_lut);
    s_pkt_lut[0x181Cu >> 2] = 1; /* idx_addr (count == 2) */
    for (u32 m = 0x1680u; m < 0x16C0u; m += 4u) s_pkt_lut[m >> 2] = 2; /* vtx offset */
    for (u32 m = 0x1740u; m < 0x1780u; m += 4u) s_pkt_lut[m >> 2] = 2; /* vtx format */
    for (u32 m = 0x1A00u; m < 0x1A00u + RSX_DSP_NUM_TEXTURES * 0x20u; m += 4u) s_pkt_lut[m >> 2] = 3; /* tex block */
    for (u32 m = 0x1840u; m < 0x1880u; m += 4u) s_pkt_lut[m >> 2] = 4; /* tex ctrl3 */
    for (u32 m = 0x0B80u; m < 0x0C00u; m += 4u) s_pkt_lut[m >> 2] = 5; /* vp prog */
    for (u32 m = 0x0300u; m <= 0x038Cu; m += 4u) s_pkt_lut[m >> 2] = 6; /* blend/alpha/stencil */
    for (u32 m = 0x0A00u; m <= 0x0A7Cu; m += 4u) s_pkt_lut[m >> 2] = 7; /* viewport/depth */
    for (u32 m = 0x1830u; m <= 0x183Cu; m += 4u) s_pkt_lut[m >> 2] = 8; /* cull */
    s_pkt_lut[0x1738u >> 2] = 9;
    s_pkt_lut[0x173Cu >> 2] = 9;
    s_pkt_lut[0x1FC0u >> 2] = 9;
    for (u32 m = 0x0200u; m <= 0x022Cu; m += 4u) s_pkt_lut[m >> 2] = 10; /* surface */
    for (u32 m = 0x0280u; m <= 0x028Cu; m += 4u) s_pkt_lut[m >> 2] = 10; /* surface mrt */
    for (u32 m = 0x08C0u; m <= 0x08E8u; m += 4u) s_pkt_lut[m >> 2] = 11; /* fp/misc pure reg */
    s_pkt_lut[0x1D8Cu >> 2] = 11; s_pkt_lut[0x1D90u >> 2] = 11;
    s_pkt_lut[0x1DACu >> 2] = 11; s_pkt_lut[0x1DB0u >> 2] = 11;
    s_pkt_lut[0x1FF0u >> 2] = 11; s_pkt_lut[0x1FF4u >> 2] = 11;
    s_pkt_lut[0x0100u >> 2] = 12; s_pkt_lut[0x1714u >> 2] = 12;
    s_pkt_lut[0x1D60u >> 2] = 12; s_pkt_lut[0x1EA0u >> 2] = 12;
    s_pkt_lut[0x1EA4u >> 2] = 12; s_pkt_lut[0x1E9Cu >> 2] = 12;
    s_pkt_lut_inited = 1;
}

int rsx_draw_engine_state_packet_fast_be(u32 method, const u8* be_words, u32 count)
{
    if (!g.ready || !count) return 1;
    if (__builtin_expect(!s_pkt_lut_inited, 0)) eng_pkt_lut_init();
    const u32 idx0 = method >> 2;
    if (__builtin_expect(idx0 >= 2048u, 0)) return 0;
    const u8 cls = s_pkt_lut[idx0];
    if (!cls) return 0;
    const u32 last_m = method + 4u * (count - 1u);
    switch (cls) {
    case 1: /* 0x181C count == 2 */
        if (count != 2u) return 0;
        g.rsx.regs[0x181Cu >> 2] = eng_rd_be32(be_words);
        g.rsx.regs[0x1820u >> 2] = eng_rd_be32(be_words + 4u);
        return 1;
    case 2: /* vtx offset 0x1680..0x16BC or format 0x1740..0x177C */
        if (!((method < 0x16C0u && last_m < 0x16C0u) ||
              (method >= 0x1740u && last_m < 0x1780u))) return 0;
        {
            u32 diff = 0u;
            u32 i = 0;
#if defined(__AVX2__)
            if (count >= 4u) {
                const __m128i sw16 = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
                __m128i vd = _mm_setzero_si128();
                for (; i + 4u <= count; i += 4u) {
                    const __m128i nv = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(be_words + 4u * i)), sw16);
                    const __m128i ov = _mm_loadu_si128((const __m128i*)&g.rsx.regs[idx0 + i]);
                    vd = _mm_or_si128(vd, _mm_xor_si128(nv, ov));
                    _mm_storeu_si128((__m128i*)&g.rsx.regs[idx0 + i], nv);
                }
                if (!_mm_testz_si128(vd, vd)) diff = 1u;
            }
#endif
            for (; i < count; i++) {
                const u32 d = eng_rd_be32(be_words + 4u * i);
                diff |= d ^ g.rsx.regs[idx0 + i];
                g.rsx.regs[idx0 + i] = d;
            }
            if (diff) g.rsx.vtx_attr_gen++;
        }
        return 1;
    case 3: /* texture unit block 0x1A00..0x1BFC */
        if (last_m >= 0x1A00u + RSX_DSP_NUM_TEXTURES * 0x20u) return 0;
#if defined(__AVX2__)
        if (count == 8u && (((method - 0x1A00u) & 0x1Fu) == 0u)) {
            const __m256i sw = _mm256_setr_epi8(
                3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
                3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
            const __m256i nv = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i*)be_words), sw);
            const __m256i ov = _mm256_loadu_si256((const __m256i*)&g.rsx.regs[idx0]);
            const u32 eq = (u32)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(nv, ov)));
            const u32 dm = (~eq) & 0xFFu;
            if (dm) {
                const u32 prev_wrap = g.rsx.regs[idx0 + 2u];
                const u32 prev_ctrl = g.rsx.regs[idx0 + 3u];
                _mm256_storeu_si256((__m256i*)&g.rsx.regs[idx0], nv);
                const u32 bit = 1u << ((method - 0x1A00u) >> 5);
                g.rsx.tex_dirty_mask |= bit;
                if (dm & 0xACu) g.rsx.tex_samp_dirty_mask |= bit;
                if ((dm & 0x53u) ||
                    ((dm & 0x08u) && ((prev_ctrl ^ g.rsx.regs[idx0 + 3u]) & 0x80000000u)) ||
                    ((dm & 0x04u) && ((prev_wrap ^ g.rsx.regs[idx0 + 2u]) & 0x70F0F000u)))
                    g.rsx.tex_img_dirty_mask |= bit;
            }
            return 1;
        }
#endif
        {
            u32 tmask = 0u, smask = 0u, imask = 0u;
            for (u32 i = 0; i < count; i++) {
                const u32 d = eng_rd_be32(be_words + 4u * i);
                const u32 prev = g.rsx.regs[idx0 + i];
                if (prev != d) {
                    g.rsx.regs[idx0 + i] = d;
                    const u32 off = (method + 4u * i) - 0x1A00u;
                    const u32 bit = 1u << (off >> 5);
                    tmask |= bit;
                    const u32 r = off & 0x1Fu;
                    if (r == 0x08u || r == 0x0Cu || r == 0x14u || r == 0x1Cu) smask |= bit;
                    if (r == 0x00u || r == 0x04u || r == 0x10u || r == 0x18u ||
                        (r == 0x0Cu && ((prev ^ d) & 0x80000000u)) ||
                        (r == 0x08u && ((prev ^ d) & 0x70F0F000u)))
                        imask |= bit;
                }
            }
            g.rsx.tex_dirty_mask |= tmask;
            g.rsx.tex_samp_dirty_mask |= smask;
            g.rsx.tex_img_dirty_mask |= imask;
        }
        return 1;
    case 4: /* tex ctrl3 0x1840..0x187C */
        if (last_m >= 0x1880u) return 0;
        {
            u32 tmask = 0u;
            const u32 u0 = (method - 0x1840u) >> 2;
            for (u32 i = 0; i < count; i++) {
                const u32 d = eng_rd_be32(be_words + 4u * i);
                if (g.rsx.regs[idx0 + i] != d) {
                    g.rsx.regs[idx0 + i] = d;
                    tmask |= 1u << (u0 + i);
                }
            }
            g.rsx.tex_dirty_mask |= tmask;
            g.rsx.tex_img_dirty_mask |= tmask;
        }
        return 1;
    case 5: /* vp prog 0x0B80..0x0BFC */
        if (last_m >= 0x0C00u) return 0;
        {
            u32* load = &g.rsx.regs[0x1E9Cu >> 2];
            int changed = 0;
            for (u32 i = 0; i < count; i++) {
                const u32 m = method + 4u * i;
                const u32 lane = ((m - 0x0B80u) >> 2) & 3u;
                const u32 d = eng_rd_be32(be_words + 4u * i);
                const u32 pos = (*load) * 4u + lane;
                if (pos < RSX_DSP_VP_WORDS && g.rsx.vp[pos] != d) {
                    g.rsx.vp[pos] = d;
                    changed = 1;
                }
                if (lane == 3u) (*load)++;
            }
            if (changed) g.rsx.vp_gen++;
        }
        return 1;
    case 6: /* 0x0300..0x038C */
        if (last_m > 0x038Cu) return 0;
        {
            int pipe_chg = 0, dyn_chg = 0;
            for (u32 i = 0; i < count; i++) {
                const u32 d = eng_rd_be32(be_words + 4u * i);
                if (g.rsx.regs[idx0 + i] != d) {
                    g.rsx.regs[idx0 + i] = d;
                    const u32 m = method + 4u * i;
                    if (m >= 0x0304u && m <= 0x0388u) pipe_chg = 1;
                    if (m == 0x031Cu || m == 0x0334u || m == 0x037Cu || m == 0x0384u || m == 0x0388u) dyn_chg = 1;
                }
            }
            if (pipe_chg) g.rsx.pipe_gen++;
            if (dyn_chg) g.rsx.dyn_gen++;
        }
        return 1;
    case 7: /* 0x0A00..0x0A7C */
        if (last_m > 0x0A7Cu) return 0;
        {
            int pipe_chg = 0, vp_chg = 0, dyn_chg = 0;
            for (u32 i = 0; i < count; i++) {
                const u32 d = eng_rd_be32(be_words + 4u * i);
                if (g.rsx.regs[idx0 + i] != d) {
                    g.rsx.regs[idx0 + i] = d;
                    const u32 m = method + 4u * i;
                    if (m == 0x0A6Cu || m == 0x0A70u || m == 0x0A74u) pipe_chg = 1;
                    if (m <= 0x0A3Cu) vp_chg = 1;
                    if (m == 0x0A60u || m == 0x0A64u || m == 0x0A68u || m == 0x0A78u || m == 0x0A7Cu) dyn_chg = 1;
                }
            }
            if (pipe_chg) g.rsx.pipe_gen++;
            if (vp_chg) g.rsx.vp_geom_gen++;
            if (dyn_chg) g.rsx.dyn_gen++;
        }
        return 1;
    case 8: /* 0x1830..0x183C */
        if (last_m > 0x183Cu) return 0;
        {
            int pipe_chg = 0;
            for (u32 i = 0; i < count; i++) {
                const u32 d = eng_rd_be32(be_words + 4u * i);
                if (g.rsx.regs[idx0 + i] != d) {
                    g.rsx.regs[idx0 + i] = d;
                    const u32 m = method + 4u * i;
                    if (m == 0x1830u || m == 0x1834u || m == 0x183Cu) pipe_chg = 1;
                }
            }
            if (pipe_chg) g.rsx.pipe_gen++;
        }
        return 1;
    case 9: /* 0x1738..0x173C or 0x1FC0 (count == 1) */
        if (!((method >= 0x1738u && last_m <= 0x173Cu) || (method == 0x1FC0u && count == 1u))) return 0;
        {
            u32 diff = 0u;
            for (u32 i = 0; i < count; i++) {
                const u32 d = eng_rd_be32(be_words + 4u * i);
                diff |= d ^ g.rsx.regs[idx0 + i];
                g.rsx.regs[idx0 + i] = d;
            }
            if (diff) g.rsx.vtx_attr_gen++;
        }
        return 1;
    case 10: /* 0x0200..0x022C or 0x0280..0x028C */
        if (!((method <= 0x022Cu && last_m <= 0x022Cu) ||
              (method >= 0x0280u && last_m <= 0x028Cu))) return 0;
        {
            u32 diff = 0u;
            int pipe_chg = 0;
            for (u32 i = 0; i < count; i++) {
                const u32 d = eng_rd_be32(be_words + 4u * i);
                if (g.rsx.regs[idx0 + i] != d) {
                    g.rsx.regs[idx0 + i] = d;
                    diff = 1u;
                    const u32 m = method + 4u * i;
                    if (m == 0x0208u || m == 0x0220u) pipe_chg = 1;
                }
            }
            if (diff) g.rsx.surf_gen++;
            if (pipe_chg) g.rsx.pipe_gen++;
        }
        return 1;
    case 11: /* pure reg ranges */
        if (!((method >= 0x08C0u && last_m <= 0x08E8u) ||
              (method >= 0x1D8Cu && last_m <= 0x1D90u) ||
              (method >= 0x1DACu && last_m <= 0x1DB0u) ||
              (method >= 0x1FF0u && last_m <= 0x1FF4u))) return 0;
        {
            int dyn_chg = 0;
            for (u32 i = 0; i < count; i++) {
                const u32 d = eng_rd_be32(be_words + 4u * i);
                const u32 m = method + 4u * i;
                if (m == 0x08C0u || m == 0x08C4u) g.rsx.seen[idx0 + i] = 1;
                if (g.rsx.regs[idx0 + i] != d) {
                    g.rsx.regs[idx0 + i] = d;
                    if (m == 0x08C0u || m == 0x08C4u) dyn_chg = 1;
                }
            }
            if (dyn_chg) g.rsx.dyn_gen++;
        }
        return 1;
    case 12: /* count == 1 pure regs */
        if (count != 1u) return 0;
        g.rsx.regs[idx0] = eng_rd_be32(be_words);
        return 1;
    default:
        return 0;
    }
}

void rsx_draw_engine_state_packet_be(u32 method, const u8* be_words, u32 count)
{
    if (!g.ready || !count) return;
    if (rsx_draw_engine_state_packet_fast_be(method, be_words, count)) return;
    for (u32 i = 0; i < count; i++)
        rsx_draw_engine_method(method + 4u * i, eng_rd_be32(be_words + 4u * i));
}

void rsx_draw_engine_draw_batch_be(u32 method, const u8* be_words, u32 count)
{
    if (!g.ready || !count) return;
    u32 last = 0u;
    if (method == 0x1824u) {
        dc.n_packets += count;
        u32 n = dc.n_idx;
        for (u32 i = 0; i < count; i++) {
            last = eng_rd_be32(be_words + 4u * i);
            if (n < ENG_MAX_BATCHES) {
                dc.idx[n].first = last & 0xFFFFFFu;
                dc.idx[n].count = (last >> 24) + 1u;
                n++;
            }
        }
        dc.n_idx = n;
    } else if (method == 0x1814u) {
        dc.n_packets += count;
        u32 n = dc.n_arr;
        for (u32 i = 0; i < count; i++) {
            last = eng_rd_be32(be_words + 4u * i);
            if (n < ENG_MAX_BATCHES) {
                dc.arr[n].first = last & 0xFFFFFFu;
                dc.arr[n].count = (last >> 24) + 1u;
                n++;
            }
        }
        dc.n_arr = n;
    }
    g.rsx.regs[method >> 2] = last;
}

void rsx_draw_engine_begin_end(u32 arg)
{
    if (!g.ready) return;
    g.rsx.regs[0x1808u >> 2] = arg;
    if (arg) {
        g.rsx.in_begin_end = 1;
        g.rsx.current_primitive = arg;
        g.rsx.inline_len = 0;
        dc_reset();
    } else {
        g.rsx.in_begin_end = 0;
        if (g.rsx.inline_len) {
            sink_inline_array(NULL, &g.rsx, g.rsx.inline_data, g.rsx.inline_len);
            g.rsx.inline_len = 0;
        }
        sink_end(NULL, &g.rsx);
    }
}

void rsx_draw_engine_set_display_buffer(u32 buffer_id, u32 location, u32 offset,
                                        u32 pitch, u32 width, u32 height)
{
    if (buffer_id >= 8) return;
    eng_display_buffer* d = &g.display_buffers[buffer_id];
    d->location = location;
    d->offset = offset;
    d->pitch = pitch;
    d->width = width;
    d->height = height;
    d->valid = width && height;
}

void rsx_draw_engine_flush(void)
{
    if (g.ready && g.be->submit_and_wait)
        g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_GUEST_REFERENCE);
}

void rsx_draw_engine_present(void)
{
    /* A host that drives the flip itself gets whichever buffer the guest's
     * own last flip named, and display buffer 0 before there has been one:
     * a runner that both consumes 0xE944 and calls here would otherwise
     * re-present a double-buffered title's OTHER scanout. Presenting twice is
     * harmless either way -- the engine never clears the surface, so a second
     * present just blits the same image again. */
    eng_present(g.last_flip_buffer);
}

void rsx_draw_engine_present_buffer(u32 buffer_id)
{
    g.last_flip_buffer = buffer_id & 7u;
    eng_present(g.last_flip_buffer);
}

void rsx_draw_engine_set_flip_buffer(u32 buffer_id)
{
    g.last_flip_buffer = buffer_id & 7u;
}

u32 rsx_draw_engine_guest_draws(void)
{
    return g.last_guest_draws;
}

u32 rsx_draw_engine_frames(void)
{
    return g.frames;
}

void rsx_draw_engine_vertex_cache_count(u64* hits, u64* misses)
{
    if (hits) *hits = g.vc_hits;
    if (misses) *misses = g.vc_misses;
}

void rsx_draw_engine_current_mesh_ids(u64* v_id, u64* i_id)
{
    if (dc.mc_hit_ent) {
        if (v_id) *v_id = dc.mc_hit_ent->vc_serial;
        if (i_id) *i_id = dc.mc_hit_ent->indexed ? dc.mc_hit_ent->idx_serial : 0u;
        return;
    }
    if (v_id) *v_id = dc.cur_vid;
    if (i_id) *i_id = dc.cur_iid;
}

rsx_eng_gcache_memo* rsx_draw_engine_current_mesh_gcache(void)
{
    return dc.cur_gcache;
}

void rsx_draw_engine_resolve_cur_ptrs(const void** out_verts, const u32** out_indices)
{
    if (dc.mc_hit_ent) {
        if (out_verts) *out_verts = (const void*)(g.vc_arena + dc.mc_hit_ent->verts_off);
        if (out_indices) *out_indices = dc.mc_hit_ent->indexed ? dc.mc_hit_ent->idx_buf : NULL;
    }
}

u32 rsx_draw_engine_readback_center(void)
{
    if (!g.ready || !g.be->readback) return 0;
    const u32 slot = g.last_present_surface;
    if (slot >= g.n_surfaces) return 0;
    const eng_surface* s = &g.surfaces[slot];
    if (!s->handle || s->fmt != RSX_BE_FMT_R8G8B8A8 || !s->w || !s->h) return 0;
    u8 px[4] = { 0, 0, 0, 0 };
    g.be->readback(g.be->user, s->handle, s->w / 2u, s->h / 2u, 1, 1, px, 4);
    /* Decoded rows are R,G,B,A; the guest's own clear colour is A8R8G8B8, and
     * a host comparing the two wants them in the same order. */
    return ((u32)px[3] << 24) | ((u32)px[0] << 16) |
           ((u32)px[1] << 8)  |  (u32)px[2];
}
