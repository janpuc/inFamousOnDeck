/* API-agnostic temporal upscaling & frame-generation tracking shared by the
 * D3D12 (rsx_dlss.cpp) and Vulkan (rsx_vk_fsr.c) backends.
 *
 * Owns:
 *   - configuration parsing (UPSCALER / DLSS / FSR / FSR_MODE / FSR_FG / debug)
 *   - Halton(2,3) sub-pixel jitter sequence and render/output ratio helpers
 *   - main-scene draw qualification and post-process composite detection
 *   - camera matrix voting (c263..c266 + viewport z epilogue)
 *   - static-object inter-frame translation delta voting (c256..c259)
 *   - near/far/vertical-FOV extraction and 4x4 matrix inversion
 *   - motion-vector compute constant packing (44 floats) + camera-cut detection
 *   - optional single-frame vertex-constant probe dump (DLSS_PROBE / FSR_PROBE)
 */
#ifndef RSX_UPSCALE_COMMON_H
#define RSX_UPSCALE_COMMON_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rsx_upscale_cfg {
    int   mode;            /* 0 off, 1 on */
    int   up;              /* 0 Native/DLAA, 1 Quality, 2 Balanced, 3 Performance, 4 Ultra */
    int   jitter;          /* 1 = Halton(2,3) jitter on, 0 = off */
    int   mv;              /* 1 = camera motion vectors on, 0 = zero MV */
    int   mvfix;           /* 1 = depth-validated zero-motion fallback (Cole), 0 = camera only */
    int   debug_mv;        /* 0 off, 1 mv, 2 depth, 3 warp, 4 nowarp, 5 zcheck, 6 hud */
    int   probe;           /* >0: dump one gameplay frame's vertex constants after N frames */
    float mipbias;         /* texture LOD bias when active */
    float jsign;           /* +1 or -1 jitter sign passed to upscaler */
    int   mipbias_set;     /* 1 if explicitly overridden by env */
    int   preset;          /* DLSS preset A=1..O=15 (0 = default) */
    int   backend;         /* 0 DLSS (NGX), 1 AMD FSR */
    float fsr_sharp;       /* FSR RCAS sharpness [0..1] */
    char  fsr_ver[16];     /* optional FSR provider substring filter */
    int   fg;              /* FSR_FG=1: frame generation requested */
    int   min_720p;        /* 1 = clamp upscaler render size >= guest size (D3D12 default) */
} rsx_upscale_cfg;

typedef struct rsx_upscale_draw_info {
    uint32_t frame;                 /* batch/frame number being recorded */
    uint32_t rec_index;             /* draw index in its batch/frame */
    int vs_idx; uint32_t vs_hash;
    const char* vs_src;             /* decompiled HLSL of the vertex program (may be NULL) */
    const uint32_t* cuse;           /* 16 words: 512-bit mask of read VP constants (NULL = unknown) */
    const float* consts;            /* 512 x float4 RSX vertex constants */
    uint32_t rt_off, rt_w, rt_h, zeta_off, zw, zh;
    uint32_t vp_x, vp_y, vp_w, vp_h;
    uint32_t fp_addr, vcount, color_target, mrt;
    int depth_test, depth_mask, blend;
    float posscale[3], posoffset[3];
    uint32_t guest_w, guest_h;      /* guest display size (1280x720) */
} rsx_upscale_draw_info;

typedef struct rsx_upscale_batch {
    float jx, jy;                   /* jitter in render pixels */
    uint32_t qualifying;            /* jittered main-scene draws in the frame */
    uint32_t last_q;                /* record index of the last main-scene draw (0xFFFFFFFF = none) */
    uint32_t composite;             /* record index of the pre-HUD composite draw (0xFFFFFFFF = none) */
    uint32_t zeta;                  /* majority main depth surface (guest offset) */
    int cam_valid;
    float cam[16];                  /* camera-relative world -> D3D/VK clip, row-major */
    int delta_valid;
    float delta[3];                 /* camera movement (cm) since previous presented frame */
} rsx_upscale_batch;

/* Event flags returned by rsx_upscale_note_draw() for immediate-mode backends. */
enum {
    RSX_UPSCALE_EVT_QUALIFIED = 1 << 0,
    RSX_UPSCALE_EVT_COMPOSITE = 1 << 1,
};

/* Config & sizing */
const rsx_upscale_cfg* rsx_upscale_get_cfg(void);
float       rsx_upscale_render_ratio(void);
const char* rsx_upscale_mode_name(int mode_idx, int backend);
const char* rsx_upscale_tag(void);
void        rsx_upscale_set_sizes(uint32_t render_w, uint32_t render_h,
                                  uint32_t out_w, uint32_t out_h);
uint32_t    rsx_upscale_render_w(void);
uint32_t    rsx_upscale_render_h(void);
uint32_t    rsx_upscale_out_w(void);
uint32_t    rsx_upscale_out_h(void);
uint32_t    rsx_upscale_jitter_phases(void);
int         rsx_upscale_is_upscaling(void);

/* Active state controlled by whichever backend (D3D12 / Vulkan) initializes */
void  rsx_upscale_set_active(int active);
int   rsx_upscale_is_active(void);
int   rsx_upscale_hooks_on(void);
int   rsx_upscale_probe_on(void);
float rsx_upscale_mip_bias(void);

/* Per-draw record-side hooks */
int   rsx_upscale_draw_qualifies(int depth_test, uint32_t zeta_off, uint32_t zw, uint32_t zh,
                                 uint32_t rt_off, uint32_t rt_w, uint32_t rt_h,
                                 uint32_t vp_w, uint32_t vp_h,
                                 uint32_t guest_w, uint32_t guest_h);
void  rsx_upscale_jitter_px(uint32_t frame_idx, float* jx, float* jy);
void  rsx_upscale_jitter_ndc(float* dx, float* dy);
int   rsx_upscale_note_draw(const rsx_upscale_draw_info* di, int qualifies);
uint32_t rsx_upscale_rec_qualifying(void);

/* Frame handoff (record -> exec) and exec-side queries */
void rsx_upscale_handoff(int present);
const rsx_upscale_batch* rsx_upscale_exec(void);

/* Math & MV constant packing */
int  rsx_upscale_invert4(const float* m, float* out);
void rsx_upscale_cam_planes(const float* m, float* nearp, float* farp, float* fov);

/* Packs the 44-float MV/debug/merge shader constant block from current exec
 * batch and internal previous-frame camera/jitter history.
 * Returns 1 if motion vectors are valid (mv_on), 0 otherwise.
 * *out_cut is set to 1 on a camera cut / missing camera / missing delta.
 * Call rsx_upscale_commit_history() once the frame succeeds, or
 * rsx_upscale_reset_history() when a frame is skipped (menu/loading/video). */
int  rsx_upscale_pack_mv_consts(float k[44], int hist_ok, int* out_cut);
void rsx_upscale_commit_history(void);
void rsx_upscale_reset_history(void);

#ifdef __cplusplus
}
#endif
#endif /* RSX_UPSCALE_COMMON_H */
