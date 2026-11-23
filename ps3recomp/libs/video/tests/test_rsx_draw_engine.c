/*
 * ps3recomp - the platform-neutral halves of the RSX draw engine
 *
 * Everything here runs with no GPU and no host graphics API: the engine is
 * driven through rsx_draw_engine_method(), exactly as the FIFO walker drives
 * it, against a stub backend that records what it was asked for. That is the
 * only way to check the four pieces whose bugs are silent -- a surface that is
 * reallocated when it should not be, a texture unit that samples guest memory
 * instead of the render target at the same address, a texture that never
 * updates because its content hash is only taken once, and a pipeline that is
 * recompiled every time a constant changes.
 *
 * Builds and runs on Linux as well as macOS; nothing in it is Apple-specific.
 */
#include "rsx_draw_engine.h"
#include "rsx_primitives.h"   /* RSX_PRIMITIVE_* */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        if (cond) {                                                           \
            printf("[PASS] " __VA_ARGS__); printf("\n");                      \
        } else {                                                              \
            printf("[FAIL] " __VA_ARGS__);                                    \
            printf("   (%s:%d)\n", __FILE__, __LINE__);                       \
            g_failures++;                                                     \
        }                                                                     \
    } while (0)

/* ---- the guest memory the engine reads ---------------------------------- */
/*
 * Two number spaces, as on the hardware: an RSX offset means one thing in the
 * IO-mapped main-memory context and another in local video memory. The test's
 * map puts main at the bottom of the buffer and local above it, which is what
 * lets the surface-keying check bind two different objects at one offset.
 */
#define GUEST_BYTES     (16u << 20)
#define GUEST_LOCAL_EA  (8u << 20)

u8* vm_base;
u32 ppu_vm_size;

u32 cellGcmResolveLocated(int local, u32 offset)
{
    return local ? GUEST_LOCAL_EA + offset : offset;
}

u32 cellGcmResolveIO(u32 offset)
{
    return offset < GUEST_LOCAL_EA ? offset : 0;
}

/* ---- the stub backend ---------------------------------------------------- */

#define STUB_MAX_OBJECTS 8192
/* Colour clears kept by surface, so a test can say which targets were hit. */
#define STUB_MAX_CLEARED 8

typedef struct {
    u32 next_handle;
    u8  alive[STUB_MAX_OBJECTS];

    int n_texture_create, n_texture_release, n_texture_upload;
    int fail_next_create;      /* make the next texture_create fail once */
    int n_color_create, n_color_release, n_color_seeded;
    int n_color_snapshot;
    int n_depth_create, n_depth_snapshot, n_depth_color_snapshot;
    int n_pipeline_create;
    int n_draw, n_clear_color, n_clear_ds, n_present;
    u32 cleared[STUB_MAX_CLEARED];

    /* the last draw's bindings */
    u32 bound_tex[RSX_BE_MAX_TEXTURES];
    rsx_be_sampler_desc bound_samplers[RSX_BE_MAX_TEXTURES];
    u32 bound_mask;
    u32 bound_rt[RSX_BE_MAX_COLOR_TARGETS];
    u32 bound_rt_count;
    u32 pipeline_rt_count;
    u32 bound_surface, bound_depth, bound_pipeline;
    u32 draw_vertices, draw_indices, draw_stride;
    float draw_first_vertex[4];   /* ATTR0 of the first vertex handed over */
    float draw_attr3[4];          /* ATTR3, when the full fixed layout is used */
    rsx_topology draw_topology;
    u32 scissor[4];
    u32 stencil_ref;
    u32 last_created_texture, last_created_color, last_snapshot;
    u32 last_created_depth, last_depth_w, last_depth_h;
    u32 last_depth_snap_src, last_depth_snap_w, last_depth_snap_h, last_depth_snap;
    u32 last_dcol_snap_src, last_dcol_snap_w, last_dcol_snap_h, last_dcol_snap;
    u32 last_released;
    int depth_bounds_calls;
    float depth_bounds[2], draw_depth_bounds[2];
    u32 pipeline_depth_bounds;
    u32 pipeline_color_mask, pipeline_color_mask_mrt;
    u32 pipeline_layout_count, pipeline_vs_attrs;
    char last_ps_hlsl[16384];
} StubState;

static StubState stub;

static u32 stub_handle(void)
{
    if (stub.next_handle + 1 >= STUB_MAX_OBJECTS) return 0;
    const u32 h = ++stub.next_handle;
    stub.alive[h] = 1;
    return h;
}

static int stub_init(void* u, u32 w, u32 h) { (void)u; (void)w; (void)h; return 0; }
static void stub_shutdown(void* u) { (void)u; }
static void stub_submit(void* u, u32 reason) { (void)u; (void)reason; }

static u32 stub_texture_create(void* u, rsx_be_format f, u32 w, u32 h, u32 mips,
                               u32 faces, u32 remap, u32 rsx_fmt)
{
    (void)u; (void)f; (void)w; (void)h; (void)mips; (void)faces;
    (void)remap; (void)rsx_fmt;
    if (stub.fail_next_create) { stub.fail_next_create = 0; return 0; }
    stub.n_texture_create++;
    stub.last_created_texture = stub_handle();
    return stub.last_created_texture;
}

static void stub_texture_upload(void* u, u32 t, u32 face, u32 mip, u32 w, u32 h,
                                const void* src, u32 row, u32 rows)
{
    (void)u; (void)t; (void)face; (void)mip; (void)w; (void)h;
    (void)src; (void)row; (void)rows;
    stub.n_texture_upload++;
}

static void stub_release(void* u, u32 handle)
{
    (void)u;
    if (handle && handle < STUB_MAX_OBJECTS) stub.alive[handle] = 0;
    stub.last_released = handle;
}

static void stub_texture_release(void* u, u32 handle)
{ stub.n_texture_release++; stub_release(u, handle); }

static u32 stub_color_create(void* u, rsx_be_format f, u32 w, u32 h,
                             const void* seed, u32 seed_row)
{
    (void)u; (void)f; (void)w; (void)h; (void)seed_row;
    stub.n_color_create++;
    if (seed) stub.n_color_seeded++;
    stub.last_created_color = stub_handle();
    return stub.last_created_color;
}

static void stub_color_release(void* u, u32 handle)
{ stub.n_color_release++; stub_release(u, handle); }

static u32 stub_surface_view(void* u, u32 surface, u32 remap, u32 fmt)
{ (void)u; (void)remap; (void)fmt; return surface; }

/* The colour feedback copy: a target's own sampleable copy, which the engine
 * binds for a draw that reads the target it writes. A fresh handle every call,
 * so a test can tell the copy from the target. */
static u32 stub_color_snapshot(void* u, u32 surface)
{
    (void)u; (void)surface;
    stub.n_color_snapshot++;
    stub.last_snapshot = stub_handle();
    return stub.last_snapshot;
}

static u32 stub_depth_create(void* u, u32 w, u32 h)
{
    (void)u;
    stub.n_depth_create++;
    stub.last_depth_w = w; stub.last_depth_h = h;
    stub.last_created_depth = stub_handle();
    return stub.last_created_depth;
}

static u32 stub_depth_snapshot(void* u, u32 d, u32 w, u32 h)
{
    (void)u;
    stub.n_depth_snapshot++;
    stub.last_depth_snap_src = d;
    stub.last_depth_snap_w = w; stub.last_depth_snap_h = h;
    stub.last_depth_snap = stub_handle();
    return stub.last_depth_snap;
}

static u32 stub_depth_color_snapshot(void* u, u32 d, u32 w, u32 h)
{
    (void)u;
    stub.n_depth_color_snapshot++;
    stub.last_dcol_snap_src = d;
    stub.last_dcol_snap_w = w; stub.last_dcol_snap_h = h;
    stub.last_dcol_snap = stub_handle();
    return stub.last_dcol_snap;
}

static u32 stub_pipeline_create(void* u, const char* vs, const char* ps,
                                const rsx_be_render_state* rs,
                                const rsx_vertex_layout_plan* layout,
                                u32 stride, rsx_be_format rt, u32 rt_count)
{
    (void)u; (void)layout; (void)stride; (void)rt;
    if (!vs || !ps || !*vs || !*ps) return 0;
    stub.n_pipeline_create++;
    stub.pipeline_rt_count = rt_count;
    stub.pipeline_depth_bounds = rs->depth_bounds_test;
    stub.pipeline_color_mask = rs->color_mask;
    stub.pipeline_color_mask_mrt = rs->color_mask_mrt;
    stub.pipeline_layout_count = layout ? layout->count : 0u;
    { u32 na = 0; for (const char* q = vs; (q = strstr(q, ":ATTR")) != NULL; q += 5) na++;
      stub.pipeline_vs_attrs = na; }
    snprintf(stub.last_ps_hlsl, sizeof stub.last_ps_hlsl, "%s", ps);
    return stub_handle();
}

static void stub_pipeline_release(void* u, u32 p) { (void)u; (void)p; }

static void stub_bind_targets(void* u, const u32* s, u32 count, u32 d)
{
    (void)u;
    stub.bound_surface = count ? s[0] : 0;
    stub.bound_rt_count = count;
    for (u32 i = 0; i < RSX_BE_MAX_COLOR_TARGETS; i++)
        stub.bound_rt[i] = (i < count) ? s[i] : 0;
    stub.bound_depth = d;
}
static void stub_bind_pipeline(void* u, u32 p) { (void)u; stub.bound_pipeline = p; }
static void stub_bind_vs(void* u, const void* d, u32 n) { (void)u; (void)d; (void)n; }
static void stub_bind_ps(void* u, const void* d, u32 n) { (void)u; (void)d; (void)n; }

static void stub_bind_textures(void* u, const u32* tex,
                               const rsx_be_sampler_desc* smp, u32 mask)
{
    (void)u;
    memcpy(stub.bound_tex, tex, sizeof stub.bound_tex);
    if (smp) memcpy(stub.bound_samplers, smp, sizeof stub.bound_samplers);
    stub.bound_mask = mask;
}

static void stub_bind_vtextures(void* u, const u32* tex,
                                const rsx_be_sampler_desc* smp, u32 mask)
{ (void)u; (void)tex; (void)smp; (void)mask; }

static void stub_viewport(void* u, float x, float y, float w, float h)
{ (void)u; (void)x; (void)y; (void)w; (void)h; }

static void stub_scissor(void* u, u32 x, u32 y, u32 w, u32 h)
{
    (void)u;
    stub.scissor[0] = x; stub.scissor[1] = y;
    stub.scissor[2] = w; stub.scissor[3] = h;
}

static void stub_stencil_ref(void* u, u32 r) { (void)u; stub.stencil_ref = r; }

static void stub_depth_bounds(void* u, float min, float max)
{
    (void)u;
    stub.depth_bounds_calls++;
    stub.depth_bounds[0] = min;
    stub.depth_bounds[1] = max;
}

static void stub_draw(void* u, rsx_topology topology, const void* verts,
                      u32 count, u32 stride, const u32* indices,
                      u32 index_count)
{
    (void)u; (void)indices;
    stub.n_draw++;
    memcpy(stub.draw_depth_bounds, stub.depth_bounds, sizeof stub.draw_depth_bounds);
    memset(stub.draw_first_vertex, 0, sizeof stub.draw_first_vertex);
    memset(stub.draw_attr3, 0, sizeof stub.draw_attr3);
    if (verts && stride >= sizeof stub.draw_first_vertex)
        memcpy(stub.draw_first_vertex, verts, sizeof stub.draw_first_vertex);
    if (verts && stride >= 64)
        memcpy(stub.draw_attr3, (const u8*)verts + 48, sizeof stub.draw_attr3);
    stub.draw_topology = topology;
    stub.draw_vertices = count;
    stub.draw_stride = stride;
    stub.draw_indices = index_count;
}

static void stub_clear_color(void* u, u32 s, const float rgba[4])
{
    (void)u; (void)rgba;
    if (stub.n_clear_color < (int)STUB_MAX_CLEARED)
        stub.cleared[stub.n_clear_color] = s;
    stub.n_clear_color++;
}

static void stub_clear_ds(void* u, u32 d, u32 f, float z, u8 s)
{ (void)u; (void)d; (void)f; (void)z; (void)s; stub.n_clear_ds++; }

static u32 presented_surface;
static void stub_present(void* u, u32 s) { (void)u; presented_surface = s; stub.n_present++; }

/* One pixel of a surface, in the R,G,B,A order rsx_texture_decode produces:
 * the readback hook has to put those back into the guest's A8R8G8B8. */
static u8 g_stub_pixel[4] = { 0x11, 0x22, 0x33, 0x44 };

static void stub_readback(void* u, u32 s, u32 x, u32 y, u32 w, u32 h,
                          void* out, u32 pitch)
{
    (void)u; (void)s; (void)x; (void)y; (void)pitch;
    if (w == 1 && h == 1) memcpy(out, g_stub_pixel, 4);
}

static const rsx_draw_backend g_stub_backend = {
    .user = NULL,
    .init = stub_init,
    .shutdown = stub_shutdown,
    .submit_and_wait = stub_submit,
    .texture_create = stub_texture_create,
    .texture_upload = stub_texture_upload,
    .texture_release = stub_texture_release,
    .color_target_create = stub_color_create,
    .color_target_release = stub_color_release,
    .surface_view = stub_surface_view,
    .color_snapshot = stub_color_snapshot,
    .depth_target_create = stub_depth_create,
    .depth_target_release = stub_release,
    .depth_snapshot = stub_depth_snapshot,
    .depth_color_snapshot = stub_depth_color_snapshot,
    .pipeline_create = stub_pipeline_create,
    .pipeline_release = stub_pipeline_release,
    .bind_targets = stub_bind_targets,
    .bind_pipeline = stub_bind_pipeline,
    .bind_vs_constants = stub_bind_vs,
    .bind_ps_constants = stub_bind_ps,
    .bind_textures = stub_bind_textures,
    .bind_vertex_textures = stub_bind_vtextures,
    .set_viewport = stub_viewport,
    .set_scissor = stub_scissor,
    .set_stencil_ref = stub_stencil_ref,
    .draw = stub_draw,
    .clear_color = stub_clear_color,
    .clear_depth_stencil = stub_clear_ds,
    .present = stub_present,
    .readback = stub_readback,
    .set_depth_bounds = stub_depth_bounds,
};

/* ---- driving the engine -------------------------------------------------- */

#define M_CONTEXT_DMA_COLOR_B 0x018C
#define M_CONTEXT_DMA_COLOR_A 0x0194
#define M_CONTEXT_DMA_COLOR_C 0x01B4
#define M_CONTEXT_DMA_COLOR_D 0x01B8
#define M_SURFACE_CLIP_H      0x0200
#define M_SURFACE_CLIP_V      0x0204
#define M_SURFACE_FORMAT      0x0208
#define M_COLOR_A_OFFSET      0x0210
#define M_ZETA_OFFSET         0x0214
#define M_COLOR_B_OFFSET      0x0218
#define M_COLOR_TARGET        0x0220
#define M_COLOR_C_OFFSET      0x0288
#define M_COLOR_D_OFFSET      0x028C
#define M_SCISSOR_H           0x08C0
#define M_SCISSOR_V           0x08C4
#define M_VTXBUF_OFFSET       0x1680
#define M_VTXFMT              0x1740
#define M_BEGIN_END           0x1808
#define M_DRAW_ARRAYS         0x1814
#define M_INLINE_ARRAY        0x1818
#define M_TEX_UNIT            0x1A00   /* + unit * 0x20 */
#define M_STENCIL_FUNC_REF    0x0334
#define M_CLEAR_BUFFERS       0x1D94
#define M_VP_UPLOAD_CONST_ID  0x1EFC
#define M_VP_UPLOAD_CONST     0x1F00

#define DMA_LOCAL  0xFEED0000u
#define DMA_MAIN   0xFEED0001u

#define VTX_OFFSET  0x00010000u        /* main memory, well clear of ea 0 */
#define TEX_OFFSET  0x00020000u

static void m(u32 method, u32 arg) { rsx_draw_engine_method(method, arg); }

static void engine_up(void)
{
    memset(&stub, 0, sizeof stub);
    rsx_draw_engine_set_backend(&g_stub_backend);
    if (rsx_draw_engine_init(1280, 720) != 0) {
        printf("[FAIL] engine init\n");
        exit(1);
    }
    /* One 256x256 surface in local memory, and a vertex array in main. */
    m(M_CONTEXT_DMA_COLOR_A, DMA_LOCAL);
    m(M_SURFACE_CLIP_H, 256u << 16);
    m(M_SURFACE_CLIP_V, 256u << 16);
    m(M_COLOR_A_OFFSET, 0x40000u);
    /* type 2 (float32), size 4, stride 16; bit 31 of the offset = main. */
    m(M_VTXFMT, 2u | (4u << 4) | (16u << 8));
    m(M_VTXBUF_OFFSET, 0x80000000u | VTX_OFFSET);
}

static void engine_down(void)
{
    rsx_draw_engine_shutdown();
    rsx_draw_engine_set_backend(NULL);
}

static void draw_triangle(void)
{
    m(M_BEGIN_END, RSX_PRIMITIVE_TRIANGLES);
    m(M_DRAW_ARRAYS, 0u | ((3u - 1u) << 24));
    m(M_BEGIN_END, 0u);
}

/* Bind unit 0 at `offset` in `location` (0 = local, 1 = main), w x h A8R8G8B8
 * linear with one level. */
static void bind_texture_unit_sized(u32 location, u32 offset, u32 w, u32 h)
{
    const u32 dma = (location == RSX_LOCATION_MAIN) ? 2u : 1u;
    m(M_TEX_UNIT + 0x00, offset);
    m(M_TEX_UNIT + 0x04, dma | (2u << 4) | ((0x85u | 0x20u) << 8) | (1u << 16));
    m(M_TEX_UNIT + 0x0C, 0x80000000u);        /* control0: unit enable */
    m(M_TEX_UNIT + 0x10, 0xAAE4u);            /* identity crossbar */
    m(M_TEX_UNIT + 0x18, (w << 16) | h);      /* image rect */
}

static void bind_texture_unit(u32 location, u32 offset)
{
    bind_texture_unit_sized(location, offset, 4u, 4u);
}

/* ---- 1. the surface set -------------------------------------------------- */

static void test_surface_set(void)
{
    printf("-- surface set\n");
    engine_up();

    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(stub.n_color_create == 1 && stub.n_clear_color == 1,
          "a clear registers the bound colour surface (%d create, %d clear)",
          stub.n_color_create, stub.n_clear_color);
    const u32 first = stub.last_created_color;

    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(stub.n_color_create == 1,
          "the same (context, offset, size, format) is the same surface");

    /* A size change is a SECOND target, and the first keeps its contents: a
     * title that alternates between two declarations of one address destroys
     * neither (inFamous renders at two sizes, and stealing the slot was the
     * reason its world was black). */
    m(M_SURFACE_CLIP_H, 512u << 16);
    m(M_SURFACE_CLIP_V, 512u << 16);
    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(stub.n_color_create == 2 && stub.n_color_release == 0 &&
          stub.alive[first] == 1,
          "a size change is a new target and keeps the old one (%d create, "
          "%d release)", stub.n_color_create, stub.n_color_release);

    /* Announcing the first size again finds the first target, not a third. */
    m(M_SURFACE_CLIP_H, 256u << 16);
    m(M_SURFACE_CLIP_V, 256u << 16);
    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(stub.n_color_create == 2,
          "and declaring the old size again reuses the old target (%d)",
          stub.n_color_create);

    /* An offset change is a different surface, and does not disturb the
     * first. */
    m(M_COLOR_A_OFFSET, 0x80000u);
    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(stub.n_color_create == 3 && stub.n_color_release == 0,
          "a different offset is a different surface");

    /* The same offset in the OTHER context DMA is also a different surface:
     * RSX offsets are two overlapping number spaces. */
    m(M_CONTEXT_DMA_COLOR_A, DMA_MAIN);
    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(stub.n_color_create == 4 && stub.n_color_release == 0,
          "the same offset in another context DMA is another surface");

    /* Main memory that the IO table maps is seeded from the guest's own
     * bytes; local memory is not, because nothing guarantees it is backed. */
    CHECK(stub.n_color_seeded == 1,
          "only the main-memory surface was seeded from guest bytes (%d)",
          stub.n_color_seeded);

    /* An implausible clip does not destroy a live target. */
    const int creates = stub.n_color_create;
    m(M_SURFACE_CLIP_H, 60000u << 16);
    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(stub.n_color_create == creates,
          "an implausible surface declaration keeps the existing target");

    engine_down();
}

/* ---- 2. surface aliasing on texture bind --------------------------------- */

static void test_surface_aliasing(void)
{
    printf("-- surface aliasing\n");
    engine_up();

    /* Render into a surface in MAIN memory, then sample a texture unit bound
     * at the same offset in the same context: the unit must read the live
     * target, not the guest bytes behind it. That is what every reflection,
     * shadow composite and post-process pass depends on. */
    m(M_CONTEXT_DMA_COLOR_A, DMA_MAIN);
    m(M_COLOR_A_OFFSET, TEX_OFFSET);
    m(M_CLEAR_BUFFERS, 0xF0u);
    const u32 surface = stub.last_created_color;

    /* Now draw into a DIFFERENT surface while sampling that one. */
    m(M_COLOR_A_OFFSET, 0x90000u);
    bind_texture_unit(RSX_LOCATION_MAIN, TEX_OFFSET);
    draw_triangle();
    CHECK(stub.n_draw == 1, "the draw executed");
    CHECK(stub.bound_mask == 1u, "unit 0 is the only bound unit (0x%X)",
          stub.bound_mask);
    CHECK(stub.bound_tex[0] == surface,
          "the unit samples the registered surface (%u, want %u)",
          stub.bound_tex[0], surface);
    CHECK(stub.n_texture_create == 0,
          "and nothing was uploaded from guest memory (%d uploads)",
          stub.n_texture_create);

    /* A unit naming the surface the pass is drawing INTO must not alias it:
     * reading a target a pass is writing is undefined on every API, so it
     * samples the backend's copy of the target instead -- and the copy is
     * refilled for THIS draw, not taken once. */
    m(M_COLOR_A_OFFSET, TEX_OFFSET);
    const int uploads = stub.n_texture_create;
    draw_triangle();
    CHECK(stub.bound_tex[0] != surface && stub.bound_tex[0] == stub.last_snapshot,
          "a unit naming its own render target samples its copy (%u, want %u)",
          stub.bound_tex[0], stub.last_snapshot);
    CHECK(stub.n_color_snapshot == 1 && stub.n_texture_create == uploads,
          "the backend was asked for one copy and nothing was uploaded "
          "(%d copies, %d uploads)", stub.n_color_snapshot,
          stub.n_texture_create - uploads);

    /* The same offset in the other context DMA is a different object. */
    m(M_COLOR_A_OFFSET, 0x90000u);
    bind_texture_unit(RSX_LOCATION_LOCAL, TEX_OFFSET);
    const int before = stub.n_texture_create;
    draw_triangle();
    CHECK(stub.n_texture_create == before + 1 && stub.bound_tex[0] != surface,
          "a unit in the other context uploads instead of aliasing");

    engine_down();
}

/* ---- 2b. the MRT set ----------------------------------------------------- */

static void test_mrt_set(void)
{
    printf("-- MRT set\n");
    engine_up();

    /* Targets B, C and D alongside the A engine_up declared, then MRT3 --
     * the way a deferred pass declares its G-buffer. */
    m(M_CONTEXT_DMA_COLOR_B, DMA_LOCAL);
    m(M_CONTEXT_DMA_COLOR_C, DMA_LOCAL);
    m(M_CONTEXT_DMA_COLOR_D, DMA_LOCAL);
    m(M_COLOR_B_OFFSET, 0x50000u);
    m(M_COLOR_C_OFFSET, 0x60000u);
    m(M_COLOR_D_OFFSET, 0x70000u);
    m(M_COLOR_TARGET, CELL_GCM_SURFACE_TARGET_MRT3);

    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(stub.n_color_create == 4,
          "an MRT set registers all four targets (%d)", stub.n_color_create);
    CHECK(stub.n_clear_color == 4,
          "and one CLEAR_SURFACE clears every one of them (%d)",
          stub.n_clear_color);
    CHECK(stub.cleared[0] != stub.cleared[1] &&
          stub.cleared[1] != stub.cleared[2] &&
          stub.cleared[2] != stub.cleared[3] &&
          stub.cleared[0] != stub.cleared[3],
          "four distinct surfaces, not target A four times (%u,%u,%u,%u)",
          stub.cleared[0], stub.cleared[1], stub.cleared[2], stub.cleared[3]);

    draw_triangle();
    CHECK(stub.bound_rt_count == 4 && stub.bound_rt[0] == stub.cleared[0],
          "the draw binds all four, A first (%u bound)", stub.bound_rt_count);
    CHECK(stub.pipeline_rt_count == 4,
          "and its pipeline is built for four attachments (%u)",
          stub.pipeline_rt_count);

    /* A texture unit naming a target of the set must not alias it: reading a
     * target the pass writes is undefined, and an MRT set writes more than
     * one, so any of the four gets a copy rather than itself. */
    bind_texture_unit(RSX_LOCATION_LOCAL, 0x60000u);
    const int before = stub.n_texture_create;
    draw_triangle();
    CHECK(stub.bound_tex[0] != stub.bound_rt[2] &&
          stub.bound_tex[0] == stub.last_snapshot &&
          stub.n_texture_create == before,
          "a unit naming target C samples its copy instead of aliasing it "
          "(%u, want %u)", stub.bound_tex[0], stub.last_snapshot);

    /* Back to target A alone: one attachment, and a pipeline of its own,
     * because the attachment count is part of the pipeline key. */
    const int pipelines = stub.n_pipeline_create;
    m(M_COLOR_TARGET, CELL_GCM_SURFACE_TARGET_0);
    draw_triangle();
    CHECK(stub.bound_rt_count == 1 && stub.pipeline_rt_count == 1 &&
          stub.n_pipeline_create == pipelines + 1,
          "target A alone binds one and builds its own pipeline (%u bound, %d)",
          stub.bound_rt_count, stub.n_pipeline_create);

    engine_down();
}

/* ---- 3. the texture cache ------------------------------------------------ */

static void test_texture_cache(void)
{
    printf("-- texture cache\n");
    engine_up();
    bind_texture_unit(RSX_LOCATION_MAIN, TEX_OFFSET);
    memset(vm_base + TEX_OFFSET, 0x11, 4 * 4 * 4);

    draw_triangle();
    CHECK(stub.n_texture_create == 1 && stub.n_texture_upload == 1,
          "the first bind uploads the texture");
    const u32 first = stub.last_created_texture;

    draw_triangle();
    draw_triangle();
    CHECK(stub.n_texture_create == 1,
          "a repeat bind in the same frame reuses it (%d uploads)",
          stub.n_texture_create);

    /* The guest rewrites the image in place -- an animated UI, a video frame,
     * a surface the 2D engine wrote. The cache must notice, and it costs one
     * content hash per entry per frame, not one per draw. */
    rsx_draw_engine_present();
    memset(vm_base + TEX_OFFSET, 0x22, 4 * 4 * 4);
    draw_triangle();
    CHECK(stub.n_texture_create == 2 && stub.alive[first] == 0,
          "changed guest bytes are re-decoded and the old resource retired");

    rsx_draw_engine_present();
    draw_triangle();
    CHECK(stub.n_texture_create == 2,
          "unchanged bytes in a later frame do not re-decode");

    /* Fill the cache past its capacity with distinct descriptors, then come
     * back to the first: it must have been evicted, not still resident, and
     * the most recent one must still be there. Returning nothing instead of
     * evicting is what made a recovered scene render as flat geometry. */
    for (u32 i = 0; i < RSX_DRAW_ENGINE_TEXTURE_CACHE; i++) {
        rsx_draw_engine_present();
        bind_texture_unit(RSX_LOCATION_MAIN, TEX_OFFSET + 0x1000u + i * 0x100u);
        draw_triangle();
    }
    CHECK(stub.n_texture_release >= 1,
          "the full cache evicts rather than giving up (%d evictions)",
          stub.n_texture_release);
    const u32 newest = stub.last_created_texture;
    const int creates = stub.n_texture_create;
    draw_triangle();
    CHECK(stub.n_texture_create == creates && stub.bound_tex[0] == newest,
          "the most recently used entry survived");

    rsx_draw_engine_present();
    bind_texture_unit(RSX_LOCATION_MAIN, TEX_OFFSET);
    draw_triangle();
    CHECK(stub.n_texture_create == creates + 1,
          "the least recently used entry was the one evicted");

    /* A first upload that fails -- the guest span is not mapped yet, or a
     * staging allocation failed -- used to poison the entry: it was inserted
     * with handle 0 and every later bind returned that zero, so the texture
     * never appeared. The next bind retries the upload instead. */
    rsx_draw_engine_present();
    stub.fail_next_create = 1;
    bind_texture_unit_sized(RSX_LOCATION_MAIN, TEX_OFFSET + 0x60000u, 4u, 4u);
    draw_triangle();
    CHECK(stub.bound_tex[0] == 0, "a failed first upload binds no texture");
    draw_triangle();
    CHECK(stub.bound_tex[0] != 0,
          "the next bind retries the upload and binds it");

    engine_down();
}

/* The sparse validation policy: a large texture is sampled at ~1024 spread
 * words every frame, and fully re-hashed at least every ENG_TEX_VERIFY_FRAMES
 * frames, so a change the samples cannot see still lands inside that window.
 * A missed animation is a frozen surface, which is why the window is a rule
 * and not an optimization. */
static void test_texture_sparse_validation(void)
{
    printf("-- sparse texture validation\n");
    engine_up();
    const u32 span = 128u * 128u * 4u;
    const u32 step = span / 1024u;          /* the sampler's stride at this size */
    memset(vm_base + TEX_OFFSET, 0x11, span);
    bind_texture_unit_sized(RSX_LOCATION_MAIN, TEX_OFFSET, 128u, 128u);
    draw_triangle();
    CHECK(stub.n_texture_create == 1 && stub.n_texture_upload == 1,
          "the first bind uploads the large texture");

    /* Byte `step * k + 2` is inside the word the sampler reads at k*step:
     * the samples see it and the next frame re-decodes. */
    rsx_draw_engine_present();
    vm_base[TEX_OFFSET + step * 5u + 2u] ^= 0xFFu;
    draw_triangle();
    CHECK(stub.n_texture_create == 2,
          "a change under a sample is caught on the next frame");

    /* Byte `step * k + 8` sits between two sample words: invisible to the
     * sparse hash, so only the full verification can catch it, and the rule
     * is that it lands inside ENG_TEX_VERIFY_FRAMES frames. */
    const int creates = stub.n_texture_create;
    int caught = -1;
    for (u32 frame = 1; frame <= 8 && caught < 0; frame++) {
        if (frame == 1) vm_base[TEX_OFFSET + step * 5u + 8u] ^= 0xFFu;
        rsx_draw_engine_present();
        draw_triangle();
        if (stub.n_texture_create != creates) caught = (int)frame;
    }
    CHECK(caught >= 1 && caught <= 8,
          "a change between two samples is caught by the full verification "
          "(%d frames after the change, window %u)", caught,
          (unsigned)8u);

    /* A texture that is not bound every frame cannot be verified on the frame
     * its stagger phase comes round, so the age rule has to cover it: the
     * first bind after the window verifies regardless. */
    vm_base[TEX_OFFSET + step * 7u + 8u] ^= 0xFFu;
    const int creates2 = stub.n_texture_create;
    for (u32 frame = 0; frame < 8; frame++) rsx_draw_engine_present();
    draw_triangle();
    CHECK(stub.n_texture_create == creates2 + 1,
          "a texture last seen before the window is verified on its next bind");

    /* And an unchanged texture is not re-decoded by that same verification. */
    const int settled = stub.n_texture_create;
    for (u32 frame = 0; frame < 16; frame++) {
        rsx_draw_engine_present();
        draw_triangle();
    }
    CHECK(stub.n_texture_create == settled,
          "verification alone never re-decodes an unchanged texture");

    engine_down();
}

/* ---- 3b. the converted-vertex cache ------------------------------------- */

/* A group's converted vertices are keyed by the fetch plan, the reference list
 * and the guest bytes those references read, so a repeat draw reuses the
 * converted block and any change in the three converts again. A stale hit is
 * a mesh drawn with last frame's bytes, which is why the key covers all of it
 * rather than a sample. */
static void test_vertex_cache(void)
{
    printf("-- converted-vertex cache\n");
    setenv("ENG_VCACHE", "1", 1);
    setenv("ENG_VCACHE_MB", "32", 1);
    engine_up();
    u64 hits = 0, misses = 0;

    draw_triangle();
    rsx_draw_engine_vertex_cache_count(&hits, &misses);
    CHECK(stub.n_draw == 1 && misses == 1 && hits == 0,
          "the first group is converted (%llu hit, %llu miss)",
          (unsigned long long)hits, (unsigned long long)misses);
    float first[16];
    memcpy(first, stub.draw_first_vertex, sizeof first);

    rsx_draw_engine_present();
    draw_triangle();
    rsx_draw_engine_vertex_cache_count(&hits, &misses);
    CHECK(hits == 1 && misses == 0 && stub.n_draw == 2,
          "the same group next frame comes out of the cache");
    CHECK(memcmp(first, stub.draw_first_vertex, sizeof first) == 0,
          "the cached block carries the same vertices");

    /* A guest write into the vertex array, with the same plan and index range:
     * the key has to see the bytes or the draw keeps the old geometry. */
    rsx_draw_engine_present();
    vm_base[VTX_OFFSET + 1] ^= 0x3F;
    draw_triangle();
    rsx_draw_engine_vertex_cache_count(&hits, &misses);
    CHECK(misses == 1 && hits == 0,
          "changed guest vertex bytes are converted again");
    CHECK(memcmp(first, stub.draw_first_vertex, sizeof first) != 0,
          "and the draw sees them");

    /* A different index range over the same array is a different group. */
    rsx_draw_engine_present();
    m(M_BEGIN_END, RSX_PRIMITIVE_TRIANGLES);
    m(M_DRAW_ARRAYS, 1u | ((3u - 1u) << 24));
    m(M_BEGIN_END, 0u);
    rsx_draw_engine_vertex_cache_count(&hits, &misses);
    CHECK(hits == 0 && misses == 1, "a different index range is a miss");

    /* And the same range again, now with those bytes, is a hit. */
    rsx_draw_engine_present();
    m(M_BEGIN_END, RSX_PRIMITIVE_TRIANGLES);
    m(M_DRAW_ARRAYS, 1u | ((3u - 1u) << 24));
    m(M_BEGIN_END, 0u);
    rsx_draw_engine_vertex_cache_count(&hits, &misses);
    CHECK(hits == 1 && misses == 0, "the same range again is a hit");

    engine_down();
    unsetenv("ENG_VCACHE");
    unsetenv("ENG_VCACHE_MB");
}

/* ---- 4. the pipeline key ------------------------------------------------- */

static void test_pipeline_key(void)
{
    printf("-- pipeline key\n");

    /* The key over the render state directly: what must and must not change
     * it. The alpha REFERENCE lives in the fragment constant block, so a
     * title fading something out must not compile a pipeline per frame. */
    rsx_be_render_state a;
    memset(&a, 0, sizeof a);
    a.alpha_test_enable = 1;
    a.alpha_func = 0x0204;
    a.alpha_ref_raw = 0x40;
    a.depth_func = 0x0201;
    const u64 base = rsx_draw_engine_hash_render_state(&a, 1469598103934665603ull);

    rsx_be_render_state b = a;
    b.alpha_ref_raw = 0xC0;
    CHECK(rsx_draw_engine_hash_render_state(&b, 1469598103934665603ull) == base,
          "the alpha reference is not part of the key");

    b = a; b.alpha_func = 0x0202;
    CHECK(rsx_draw_engine_hash_render_state(&b, 1469598103934665603ull) != base,
          "the alpha compare mode is");

    b = a; b.depth_func = 0x0203;
    CHECK(rsx_draw_engine_hash_render_state(&b, 1469598103934665603ull) != base,
          "so is the depth compare mode");

    b = a; b.color_mask = 0x01010101u;
    CHECK(rsx_draw_engine_hash_render_state(&b, 1469598103934665603ull) != base,
          "so is the colour mask");

    b = a; b.rt_fp16 = 1;
    CHECK(rsx_draw_engine_hash_render_state(&b, 1469598103934665603ull) != base,
          "so is an FP16 render target");

    /* And through the engine: transform constants change every draw of every
     * frame, and must not multiply pipelines. */
    engine_up();
    draw_triangle();
    CHECK(stub.n_pipeline_create == 1, "the first draw builds a pipeline");

    for (u32 i = 0; i < 8; i++) {
        m(M_VP_UPLOAD_CONST_ID, i);
        for (u32 w = 0; w < 4; w++) m(M_VP_UPLOAD_CONST + w * 4, 0x3F800000u + i);
        draw_triangle();
    }
    CHECK(stub.n_pipeline_create == 1,
          "changing transform constants does not (%d pipelines)",
          stub.n_pipeline_create);

    /* The stencil reference is encoder state, not pipeline state. */
    m(M_STENCIL_FUNC_REF, 0x7Fu);
    draw_triangle();
    CHECK(stub.n_pipeline_create == 1 && stub.stencil_ref == 0x7Fu,
          "nor does the stencil reference, which reaches the encoder (%u)",
          stub.stencil_ref);

    engine_down();
}

/* ---- 5. restart cuts and topology expansion ------------------------------ */

static void test_restart_expansion(void)
{
    printf("-- restart cuts\n");

    /* Eight strip vertices with the guest's restart sentinel after the
     * fourth. Without the cut the expansion joins the two runs with two
     * triangles that were never in the mesh -- upstream's "exploded spiky
     * mesh that occluded the scene". */
    const u32 cuts[1] = { 4 };
    u32 idx[64];

    const u32 whole = rsx_draw_engine_topology_index_count(
        RSX_PRIMITIVE_TRIANGLE_STRIP, 8, NULL, 0);
    const u32 cut = rsx_draw_engine_topology_index_count(
        RSX_PRIMITIVE_TRIANGLE_STRIP, 8, cuts, 1);
    CHECK(whole == (8 - 2) * 3, "an uncut strip of 8 is 6 triangles (%u)", whole);
    CHECK(cut == (4 - 2) * 3 * 2,
          "a strip cut in half is two runs of 2 triangles (%u)", cut);

    memset(idx, 0xFF, sizeof idx);
    rsx_draw_engine_write_topology_indices(RSX_PRIMITIVE_TRIANGLE_STRIP, 8,
                                           cuts, 1, NULL, idx);
    int spans = 0;
    for (u32 t = 0; t < cut / 3; t++) {
        int below = 0, above = 0;
        for (u32 k = 0; k < 3; k++)
            if (idx[t * 3 + k] < 4) below++; else above++;
        if (below && above) spans = 1;
    }
    CHECK(!spans, "no triangle spans the cut");

    /* Winding alternates per triangle within a segment, and each segment
     * starts over: a strip's second triangle swaps its first two vertices. */
    CHECK(idx[0] == 0 && idx[1] == 1 && idx[2] == 2,
          "strip triangle 0 is 0,1,2 (%u,%u,%u)", idx[0], idx[1], idx[2]);
    CHECK(idx[3] == 2 && idx[4] == 1 && idx[5] == 3,
          "strip triangle 1 swaps to 2,1,3 (%u,%u,%u)", idx[3], idx[4], idx[5]);
    CHECK(idx[6] == 4 && idx[7] == 5 && idx[8] == 6,
          "the segment after the cut starts its winding again (%u,%u,%u)",
          idx[6], idx[7], idx[8]);

    /* Fans and quads are cut the same way. */
    const u32 fan = rsx_draw_engine_topology_index_count(
        RSX_PRIMITIVE_TRIANGLE_FAN, 8, cuts, 1);
    CHECK(fan == (4 - 2) * 3 * 2, "a cut fan is two runs of 2 triangles (%u)", fan);
    rsx_draw_engine_write_topology_indices(RSX_PRIMITIVE_TRIANGLE_FAN, 8,
                                           cuts, 1, NULL, idx);
    CHECK(idx[0] == 0 && idx[3] == 0 && idx[6] == 4 && idx[9] == 4,
          "each fan segment fans around its own first vertex");

    const u32 quads = rsx_draw_engine_topology_index_count(
        RSX_PRIMITIVE_QUADS, 8, NULL, 0);
    CHECK(quads == 12, "two quads are four triangles (%u)", quads);

    /* A quad strip advances two vertices per quad, so eight vertices are
     * three quads whole and one quad per half once the strip is cut. */
    const u32 qstrip = rsx_draw_engine_topology_index_count(
        RSX_PRIMITIVE_QUAD_STRIP, 8, NULL, 0);
    const u32 qstrip_cut = rsx_draw_engine_topology_index_count(
        RSX_PRIMITIVE_QUAD_STRIP, 8, cuts, 1);
    CHECK(qstrip == 3 * 6, "an uncut quad strip of 8 is three quads (%u)", qstrip);
    CHECK(qstrip_cut == 2 * 6,
          "a quad strip cut in half is one quad per half (%u)", qstrip_cut);

    memset(idx, 0xFF, sizeof idx);
    rsx_draw_engine_write_topology_indices(RSX_PRIMITIVE_QUAD_STRIP, 8,
                                           cuts, 1, NULL, idx);
    CHECK(idx[0] == 0 && idx[1] == 1 && idx[2] == 2 &&
          idx[3] == 1 && idx[4] == 3 && idx[5] == 2,
          "a quad is split along the diagonal between its pairs "
          "(%u,%u,%u %u,%u,%u)", idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]);
    CHECK(idx[6] == 4 && idx[7] == 5 && idx[8] == 6,
          "the segment after the cut pairs from its own first vertex "
          "(%u,%u,%u)", idx[6], idx[7], idx[8]);
    spans = 0;
    for (u32 t = 0; t < qstrip_cut / 3; t++) {
        int below = 0, above = 0;
        for (u32 k = 0; k < 3; k++)
            if (idx[t * 3 + k] < 4) below++; else above++;
        if (below && above) spans = 1;
    }
    CHECK(!spans, "and no quad spans the cut");

    /* A polygon is a fan around its own first vertex, and a cut starts a
     * second polygon rather than joining the two. */
    const u32 poly = rsx_draw_engine_topology_index_count(
        RSX_PRIMITIVE_POLYGON, 5, NULL, 0);
    const u32 poly_cut = rsx_draw_engine_topology_index_count(
        RSX_PRIMITIVE_POLYGON, 8, cuts, 1);
    CHECK(poly == (5 - 2) * 3, "a pentagon is three triangles (%u)", poly);
    CHECK(poly_cut == (4 - 2) * 3 * 2,
          "a cut polygon is two runs of 2 triangles (%u)", poly_cut);

    memset(idx, 0xFF, sizeof idx);
    rsx_draw_engine_write_topology_indices(RSX_PRIMITIVE_POLYGON, 8,
                                           cuts, 1, NULL, idx);
    CHECK(idx[0] == 0 && idx[1] == 1 && idx[2] == 2 &&
          idx[3] == 0 && idx[4] == 2 && idx[5] == 3,
          "the first polygon fans around vertex 0 (%u,%u,%u %u,%u,%u)",
          idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]);
    CHECK(idx[6] == 4 && idx[9] == 4,
          "the one after the cut fans around vertex 4 (%u, %u)", idx[6], idx[9]);

    /* An occurrence-to-unique map is applied to every emitted index, which is
     * what lets a remapped draw share one uploaded vertex per reference. */
    u32 remap[8];
    for (u32 i = 0; i < 8; i++) remap[i] = 7u - i;
    rsx_draw_engine_write_topology_indices(RSX_PRIMITIVE_TRIANGLES, 6,
                                           NULL, 0, remap, idx);
    CHECK(idx[0] == 7 && idx[5] == 2, "the remap is applied to every index");
}

/* ---- 6. what reaches the backend as an indexed draw ---------------------- */

static void test_indexed_draws(void)
{
    printf("-- indexed draws\n");
    engine_up();

    /* A triangle list of distinct vertices has nothing to share, so it is
     * drawn straight. */
    draw_triangle();
    CHECK(stub.draw_indices == 0 && stub.draw_vertices == 3,
          "a triangle list of distinct vertices is not indexed (%u verts, %u indices)",
          stub.draw_vertices, stub.draw_indices);

    /* A strip is expanded into a triangle list through an INDEX buffer: six
     * vertices become twelve indices over six uploaded vertices, rather than
     * twelve duplicated ones. */
    m(M_BEGIN_END, RSX_PRIMITIVE_TRIANGLE_STRIP);
    m(M_DRAW_ARRAYS, 0u | ((6u - 1u) << 24));
    m(M_BEGIN_END, 0u);
    CHECK(stub.draw_indices == (6 - 2) * 3 && stub.draw_vertices == 6,
          "a strip is indexed, not duplicated (%u verts, %u indices)",
          stub.draw_vertices, stub.draw_indices);

    /* Only the attributes the program reads are fetched, so the stride is the
     * layout's rather than all sixteen registers. The built-in program reads
     * every register, so this is the full one; a guest program that reads two
     * would halve it. */
    CHECK(stub.draw_stride == 16u * 16u,
          "the built-in program's layout is all sixteen attributes (%u bytes)",
          stub.draw_stride);

    /* Quads have no host equivalent and become an indexed triangle list. */
    m(M_BEGIN_END, RSX_PRIMITIVE_QUADS);
    m(M_DRAW_ARRAYS, 0u | ((8u - 1u) << 24));
    m(M_BEGIN_END, 0u);
    CHECK(stub.draw_topology == RSX_TOPOLOGY_TRIANGLES &&
          stub.draw_indices == 12,
          "two quads become twelve triangle indices (%u)", stub.draw_indices);

    /* So do quad strips and polygons, the two the engine used to drop. Six
     * quad-strip vertices are two quads, and a five-vertex polygon is a fan
     * of three triangles. */
    m(M_BEGIN_END, RSX_PRIMITIVE_QUAD_STRIP);
    m(M_DRAW_ARRAYS, 0u | ((6u - 1u) << 24));
    m(M_BEGIN_END, 0u);
    CHECK(stub.draw_topology == RSX_TOPOLOGY_TRIANGLES &&
          stub.draw_indices == 12 && stub.draw_vertices == 6,
          "a quad strip of six becomes twelve indices over six vertices "
          "(%u verts, %u indices)", stub.draw_vertices, stub.draw_indices);

    m(M_BEGIN_END, RSX_PRIMITIVE_POLYGON);
    m(M_DRAW_ARRAYS, 0u | ((5u - 1u) << 24));
    m(M_BEGIN_END, 0u);
    CHECK(stub.draw_topology == RSX_TOPOLOGY_TRIANGLES &&
          stub.draw_indices == 9 && stub.draw_vertices == 5,
          "a five-vertex polygon becomes nine indices (%u verts, %u indices)",
          stub.draw_vertices, stub.draw_indices);

    /* Points and lines are drawn as they are, in the order they arrived: an
     * index buffer would be pointless and collapsing repeats would reorder
     * them. */
    m(M_BEGIN_END, RSX_PRIMITIVE_LINES);
    m(M_DRAW_ARRAYS, 0u | ((4u - 1u) << 24));
    m(M_BEGIN_END, 0u);
    CHECK(stub.draw_topology == RSX_TOPOLOGY_LINES && stub.draw_indices == 0 &&
          stub.draw_vertices == 4,
          "lines pass through unindexed (%u verts, %u indices)",
          stub.draw_vertices, stub.draw_indices);

    m(M_BEGIN_END, RSX_PRIMITIVE_POINTS);
    m(M_DRAW_ARRAYS, 0u | ((2u - 1u) << 24));
    m(M_BEGIN_END, 0u);
    CHECK(stub.draw_topology == RSX_TOPOLOGY_POINTS && stub.draw_vertices == 2,
          "so do points (%u verts)", stub.draw_vertices);

    engine_down();
}

/* ---- 7. the guest scissor ------------------------------------------------ */

static void test_scissor(void)
{
    printf("-- scissor\n");
    engine_up();

    draw_triangle();
    CHECK(stub.scissor[2] == 256 && stub.scissor[3] == 256,
          "an unwritten scissor register is the whole surface (%ux%u)",
          stub.scissor[2], stub.scissor[3]);

    m(M_SCISSOR_H, (64u << 16) | 16u);
    m(M_SCISSOR_V, (32u << 16) | 8u);
    draw_triangle();
    CHECK(stub.scissor[0] == 16 && stub.scissor[1] == 8 &&
          stub.scissor[2] == 64 && stub.scissor[3] == 32,
          "a guest scissor narrows it (%u,%u %ux%u)",
          stub.scissor[0], stub.scissor[1], stub.scissor[2], stub.scissor[3]);

    m(M_SCISSOR_H, (4096u << 16) | 0u);
    m(M_SCISSOR_V, (4096u << 16) | 0u);
    draw_triangle();
    CHECK(stub.scissor[2] == 256 && stub.scissor[3] == 256,
          "a scissor larger than the surface is clipped to it (%ux%u)",
          stub.scissor[2], stub.scissor[3]);

    /* sp00nznet/ps3recomp #811: written zero width/height or out-of-bounds
     * scissor drops draws and skips clears, then a real scissor draws again. */
    int before = stub.n_draw;
    m(M_SCISSOR_H, (0u << 16) | 100u);
    m(M_SCISSOR_V, (32u << 16) | 8u);
    draw_triangle();
    CHECK(stub.n_draw == before, "a written zero-width scissor draws nothing (%d draws)",
          stub.n_draw - before);

    before = stub.n_draw;
    m(M_SCISSOR_H, (64u << 16) | 16u);
    m(M_SCISSOR_V, (0u << 16) | 8u);
    draw_triangle();
    CHECK(stub.n_draw == before, "a written zero-height scissor draws nothing (%d draws)",
          stub.n_draw - before);

    before = stub.n_draw;
    m(M_SCISSOR_H, (10u << 16) | 300u);
    m(M_SCISSOR_V, (32u << 16) | 8u);
    draw_triangle();
    CHECK(stub.n_draw == before, "a scissor outside the target draws nothing (%d draws)",
          stub.n_draw - before);

    before = stub.n_draw;
    m(M_SCISSOR_H, (64u << 16) | 16u);
    m(M_SCISSOR_V, (32u << 16) | 8u);
    draw_triangle();
    CHECK(stub.n_draw == before + 1 && stub.scissor[2] == 64,
          "a real scissor after an empty one draws again (%d draws, %u wide)",
          stub.n_draw - before, stub.scissor[2]);

    engine_down();
}

/* ---- 8. the presented pixel ---------------------------------------------- */

static void test_readback(void)
{
    printf("-- readback\n");
    engine_up();
    m(M_CLEAR_BUFFERS, 0xF0u);
    rsx_draw_engine_present();
    CHECK(stub.n_present == 1, "the flip presented a surface");
    CHECK(rsx_draw_engine_readback_center() == 0x44112233u,
          "the presented pixel comes back as A8R8G8B8 (0x%08X)",
          rsx_draw_engine_readback_center());
    engine_down();
}

static void test_queued_flip_buffer(void)
{
    engine_up();
    draw_triangle();
    u32 first = stub.bound_rt[0];
    rsx_draw_engine_set_display_buffer(0, RSX_LOCATION_LOCAL, 0x40000, 1024, 256, 256);
    m(M_COLOR_A_OFFSET, 0x60000);
    draw_triangle();
    u32 second = stub.bound_rt[0];
    rsx_draw_engine_set_display_buffer(1, RSX_LOCATION_LOCAL, 0x60000, 1024, 256, 256);
    rsx_draw_engine_present_buffer(0);
    CHECK(presented_surface == first, "queued flip selects first buffer over current target");
    rsx_draw_engine_present_buffer(1);
    CHECK(presented_surface == second && first != second, "queued flip selects second buffer");
    rsx_draw_engine_present();
    CHECK(presented_surface == second, "host present retains last queued buffer");
    engine_down();
}

static unsigned custom_reads;
static const u8* custom_guest_read(void* user, u32 location, u32 offset, u32 bytes)
{
    CHECK(user == &custom_reads, "custom memory resolver receives its user data");
    ++custom_reads;
    u32 ea = location == 0 ? GUEST_LOCAL_EA + offset : offset;
    if ((u64)ea + bytes > GUEST_BYTES) return NULL;
    return vm_base + ea;
}

static void test_custom_guest_mapping(void)
{
    engine_up();
    rsx_draw_engine_set_guest_memory(custom_guest_read, &custom_reads);
    custom_reads = 0;
    draw_triangle();
    CHECK(custom_reads > 0, "draw resolves vertices through runner's memory map");
    unsigned calls = custom_reads;
    rsx_draw_engine_set_guest_memory(NULL, NULL);
    draw_triangle();
    CHECK(custom_reads == calls, "clearing resolver restores default mapping");
    engine_down();
}

/* NV4097_INLINE_ARRAY: the vertices arrive IN the pushbuffer.
 *
 * Nothing consumed method 0x1818 at all, so a title that builds geometry this
 * way (Guitar Hero III draws its whole 2D layer as inline QUADS) issued draws
 * that reached the backend with no vertices and were counted as empty groups.
 * The regression this guards is silent by construction -- the draw is simply
 * absent -- so the check is both that a draw happens AND that its vertices
 * came from the stream rather than from the still-bound vertex array. */
static u32 f32_bits(float f) { u32 u; memcpy(&u, &f, 4); return u; }

static void test_depth_bounds(void)
{
    engine_up();
    m(0x0380, 1);
    m(0x0384, f32_bits(0.0f));
    m(0x0388, f32_bits(0.5f));
    draw_triangle();
    CHECK(stub.pipeline_depth_bounds == 1, "depth-bounds enable reaches the pipeline");
    CHECK(stub.depth_bounds_calls == 1 &&
          stub.draw_depth_bounds[0] == 0.0f && stub.draw_depth_bounds[1] == 0.5f,
          "bounds reach the backend before the first draw");

    const int pipelines = stub.n_pipeline_create;
    m(0x0384, f32_bits(0.5f));
    m(0x0388, f32_bits(1.0f));
    draw_triangle();
    CHECK(stub.depth_bounds_calls == 2 &&
          stub.draw_depth_bounds[0] == 0.5f && stub.draw_depth_bounds[1] == 1.0f,
          "cached pipeline receives changed bounds before the next draw");
    CHECK(stub.n_pipeline_create == pipelines, "changing bounds reuses the pipeline");

    m(0x0384, f32_bits(-0.25f));
    m(0x0388, f32_bits(1.25f));
    draw_triangle();
    CHECK(stub.draw_depth_bounds[0] == -0.25f && stub.draw_depth_bounds[1] == 1.25f,
          "the engine leaves API-specific bounds clamping to the backend");

    m(0x0380, 0);
    draw_triangle();
    CHECK(stub.n_pipeline_create == pipelines + 1 && stub.pipeline_depth_bounds == 0,
          "disabling depth bounds selects a different pipeline");
    engine_down();

    /* Older backends have no callback; drawing must still work. */
    engine_up();
    rsx_draw_backend legacy = g_stub_backend;
    legacy.set_depth_bounds = NULL;
    rsx_draw_engine_set_backend(&legacy);
    m(0x0380, 1);
    draw_triangle();
    CHECK(stub.n_draw == 1, "a backend without depth bounds still draws");
    engine_down();
}

static void test_inline_array(void)
{
    engine_up();
    /* ATTR0 alone: float4, stride 16. The array at VTX_OFFSET holds 1.0f in
     * every component (see main), so reading the array instead of the stream
     * is a distinguishable wrong answer. */
    m(M_VTXFMT, 2u | (4u << 4) | (16u << 8));

    const int before = stub.n_draw;
    m(M_BEGIN_END, RSX_PRIMITIVE_TRIANGLES);
    for (u32 v = 0; v < 3; v++) {
        m(M_INLINE_ARRAY, f32_bits(10.0f + (float)v));  /* x: 10, 11, 12 */
        m(M_INLINE_ARRAY, f32_bits(20.0f));
        m(M_INLINE_ARRAY, f32_bits(30.0f));
        m(M_INLINE_ARRAY, f32_bits(1.0f));
    }
    m(M_BEGIN_END, 0u);

    CHECK(stub.n_draw == before + 1, "inline array issues a draw");
    CHECK(stub.draw_vertices == 3, "inline array draws 3 vertices (got %u)",
          stub.draw_vertices);
    CHECK(stub.draw_topology == RSX_TOPOLOGY_TRIANGLES,
          "inline array keeps the primitive type");
    CHECK(stub.draw_first_vertex[0] == 10.0f &&
          stub.draw_first_vertex[1] == 20.0f &&
          stub.draw_first_vertex[2] == 30.0f,
          "inline vertices come from the stream, not the vertex array "
          "(got %g,%g,%g)", stub.draw_first_vertex[0],
          stub.draw_first_vertex[1], stub.draw_first_vertex[2]);
    engine_down();
}

/* Helper: write one RSX FP word (big-endian of 16-bit half-swapped host word). */
static void fp_put_word(u8* p, u32 h)
{
    u32 be = (h << 16) | (h >> 16);
    p[0] = (u8)(be >> 24); p[1] = (u8)(be >> 16);
    p[2] = (u8)(be >> 8);  p[3] = (u8)(be);
}

/* Helper: upload a minimal 1-instruction VP (NOP + END) and an FP in MAIN
 * memory that samples tex0 (TEX) and tex1 (TEX) and adds them into r0. */
static void load_guest_tex_shaders(void)
{
    const u32 FP_OFF = 0x00030000u;
    u8* fp = vm_base + FP_OFF;
    const u32 SWZ_ID = (0u << 9) | (1u << 11) | (2u << 13) | (3u << 15);
    const u32 EXEC_A = (7u << 18) | (0u << 21) | (1u << 23) | (2u << 25) | (3u << 27);
    const u32 MASK_ALL = 0xFu << 9;
    /* instr 0: TEX r0, TC0, tex0 */
    fp_put_word(fp +  0, (0x17u << 24) | MASK_ALL | (0u << 1) | (4u << 13) | (0u << 17));
    fp_put_word(fp +  4, 1u | SWZ_ID | EXEC_A);
    fp_put_word(fp +  8, 0u);
    fp_put_word(fp + 12, 0u);
    /* instr 1: TEX r1, TC0, tex1 */
    fp_put_word(fp + 16, (0x17u << 24) | MASK_ALL | (1u << 1) | (4u << 13) | (1u << 17));
    fp_put_word(fp + 20, 1u | SWZ_ID | EXEC_A);
    fp_put_word(fp + 24, 0u);
    fp_put_word(fp + 28, 0u);
    /* instr 2: ADD r0, r0, r1 (END) */
    fp_put_word(fp + 32, (0x03u << 24) | MASK_ALL | (0u << 1) | 1u);
    fp_put_word(fp + 36, 0u | (0u << 2) | SWZ_ID | EXEC_A);
    fp_put_word(fp + 40, 0u | (1u << 2) | SWZ_ID);
    fp_put_word(fp + 44, 0u);

    m(0x1E9Cu, 0u);          /* VP_UPLOAD_FROM_ID = 0 */
    m(0x0B80u, 0u); m(0x0B84u, 0u); m(0x0B88u, 0u); m(0x0B8Cu, 1u); /* NOP + END */
    m(0x1EA4u, 0u);          /* VP_START_FROM_ID = 0 */
    m(0x08E4u, FP_OFF | 2u); /* SHADER_PROGRAM: MAIN memory */
    m(0x1D60u, 0x40u);       /* SHADER_CONTROL: 32-bit export */
}

static void bind_tex_ex(u32 unit, u32 loc, u32 off, u32 fmt, u32 w, u32 h,
                        u32 addr_word, u32 border_color)
{
    const u32 b = M_TEX_UNIT + unit * 0x20u;
    m(b + 0x00u, off);
    m(b + 0x04u, (loc == RSX_LOCATION_MAIN ? 2u : 1u) | (2u << 4) | (fmt << 8) | (1u << 16));
    m(b + 0x08u, addr_word);
    m(b + 0x0Cu, 1u << 31);
    m(b + 0x10u, 0xAAE4u);
    m(b + 0x14u, (2u << 16) | (2u << 24));
    m(b + 0x18u, (w << 16) | h);
    m(b + 0x1Cu, border_color);
    m(0x1840u + unit * 4u, w * 4u);
}

static void test_tex_xf(void)
{
    printf("-- tex_xf (bx2 / sRGB)\n");
    unsetenv("TEX_XF");
    engine_up();
    load_guest_tex_shaders();

    /* Unit 0: unsigned_remap = 1 (bx2, bits 12..15); Unit 1: gamma = 1 (sRGB, bits 20..23). */
    bind_tex_ex(0, RSX_LOCATION_MAIN, TEX_OFFSET, RSX_TEX_FMT_A8R8G8B8, 4, 4, 1u << 12, 0);
    bind_tex_ex(1, RSX_LOCATION_MAIN, TEX_OFFSET, RSX_TEX_FMT_A8R8G8B8, 4, 4, 1u << 20, 0);
    draw_triangle();
    CHECK(strstr(stub.last_ps_hlsl, "float4 rsx_txf(") != NULL,
          "TEX_XF emits rsx_txf helper");
    CHECK(strstr(stub.last_ps_hlsl, "rsx_txf(rsx_tex[0].Sample(") != NULL &&
          strstr(stub.last_ps_hlsl, ", 1)") != NULL,
          "TEX_XF wraps unit 0 with bx2 mode 1");
    CHECK(strstr(stub.last_ps_hlsl, "rsx_txf(rsx_tex[1].Sample(") != NULL &&
          strstr(stub.last_ps_hlsl, ", 2)") != NULL,
          "TEX_XF wraps unit 1 with sRGB mode 2");

    const int pipes = stub.n_pipeline_create;
    /* Clear transforms on unit 0 and set depth format on unit 1 with bx2 bits: no wrap. */
    bind_tex_ex(0, RSX_LOCATION_MAIN, TEX_OFFSET, RSX_TEX_FMT_A8R8G8B8, 4, 4, 0u, 0);
    bind_tex_ex(1, RSX_LOCATION_MAIN, TEX_OFFSET, RSX_TEX_FMT_DEPTH24_D8, 4, 4, 1u << 12, 0);
    draw_triangle();
    CHECK(stub.n_pipeline_create == pipes + 1,
          "changing TEX_XF bits selects a distinct pipeline");
    CHECK(strstr(stub.last_ps_hlsl, "rsx_txf(") == NULL,
          "depth format units and zero-mode units are not wrapped in rsx_txf");
    engine_down();

    /* Kill switch TEX_XF=0 disables rewriting completely. */
    setenv("TEX_XF", "0", 1);
    engine_up();
    load_guest_tex_shaders();
    bind_tex_ex(0, RSX_LOCATION_MAIN, TEX_OFFSET, RSX_TEX_FMT_A8R8G8B8, 4, 4, 1u << 12, 0);
    bind_tex_ex(1, RSX_LOCATION_MAIN, TEX_OFFSET, RSX_TEX_FMT_A8R8G8B8, 4, 4, 1u << 20, 0);
    draw_triangle();
    CHECK(strstr(stub.last_ps_hlsl, "rsx_txf(") == NULL,
          "TEX_XF=0 kill switch leaves PS HLSL unwrapped");
    engine_down();
    unsetenv("TEX_XF");
}

static void test_ct_none_and_zeta_sizing(void)
{
    printf("-- CT_NONE and exact-size zeta keying\n");
    engine_up();

    /* Default TARGET_0 keeps color_mask active (0x01010101). */
    draw_triangle();
    CHECK(stub.pipeline_color_mask == 0x01010101u,
          "default COLOR_TARGET=1 keeps colour mask on (0x%X)",
          stub.pipeline_color_mask);

    /* Setting COLOR_TARGET = 0 (NONE) masks colour writes off and creates a
     * distinct pipeline. */
    const int pipes_before = stub.n_pipeline_create;
    m(M_COLOR_TARGET, CELL_GCM_SURFACE_TARGET_NONE);
    draw_triangle();
    CHECK(stub.n_pipeline_create == pipes_before + 1 &&
          stub.pipeline_color_mask == 0u && stub.pipeline_color_mask_mrt == 0u,
          "COLOR_TARGET=0 zeroes color_mask on a distinct pipeline (0x%X, %d)",
          stub.pipeline_color_mask, stub.n_pipeline_create);

    /* Exact-size zeta keying: same zeta address at 512x288 and 1024x1024
     * creates two unpadded depth targets without destroying the first. */
    m(0x0198, DMA_LOCAL);                /* CONTEXT_DMA_ZETA */
    m(M_ZETA_OFFSET, 0x0F340000u);
    m(M_COLOR_A_OFFSET, 0x0F340000u);    /* same offset as colour target */
    m(0x0A74, 1u);                       /* DEPTH_TEST_ENABLE */
    m(0x0A70, 1u);                       /* DEPTH_WRITE_ENABLE */

    m(M_SURFACE_CLIP_H, 512u << 16);
    m(M_SURFACE_CLIP_V, 288u << 16);
    draw_triangle();
    const u32 z512 = stub.last_created_depth;
    CHECK(stub.last_depth_w == 512u && stub.last_depth_h == 288u,
          "512x288 zeta is not padded to 1280x720 (%ux%u)",
          stub.last_depth_w, stub.last_depth_h);

    m(M_SURFACE_CLIP_H, 1024u << 16);
    m(M_SURFACE_CLIP_V, 1024u << 16);
    draw_triangle();
    const u32 z1024 = stub.last_created_depth;
    CHECK(z1024 != z512 && stub.alive[z512] == 1 &&
          stub.last_depth_w == 1024u && stub.last_depth_h == 1024u,
          "1024x1024 zeta at same offset gets its own slot and keeps 512x288 (%ux%u)",
          stub.last_depth_w, stub.last_depth_h);

    /* Sampling 0x0F340000 as DEPTH24_D8 at 1024x1024 while drawing to another
     * zeta must pick z1024's depth snapshot, not the aliased colour surface. */
    m(M_COLOR_TARGET, CELL_GCM_SURFACE_TARGET_0);
    m(M_COLOR_A_OFFSET, 0x00100000u);
    m(M_ZETA_OFFSET, 0x00200000u);
    m(M_SURFACE_CLIP_H, 1280u << 16);
    m(M_SURFACE_CLIP_V, 720u << 16);
    bind_tex_ex(0, RSX_LOCATION_LOCAL, 0x0F340000u, RSX_TEX_FMT_DEPTH24_D8,
                1024u, 1024u, 0u, 0u);
    draw_triangle();
    CHECK(stub.n_depth_snapshot == 1 && stub.last_depth_snap_src == z1024 &&
          stub.last_depth_snap_w == 1024u && stub.last_depth_snap_h == 1024u &&
          stub.bound_tex[0] == stub.last_depth_snap,
          "DEPTH24_D8 texture unit selects exact-size zeta snapshot (%u vs %u, %ux%u)",
          stub.last_depth_snap_src, z1024, stub.last_depth_snap_w, stub.last_depth_snap_h);
    engine_down();

    /* Kill switches CT_NONE=0 and ZETA_EXACT_SIZE=0 restore old behaviour. */
    setenv("CT_NONE", "0", 1);
    setenv("ZETA_EXACT_SIZE", "0", 1);
    engine_up();
    m(M_COLOR_TARGET, CELL_GCM_SURFACE_TARGET_NONE);
    m(0x0198, DMA_LOCAL);
    m(M_ZETA_OFFSET, 0x0F340000u);
    m(M_SURFACE_CLIP_H, 512u << 16);
    m(M_SURFACE_CLIP_V, 288u << 16);
    draw_triangle();
    CHECK(stub.pipeline_color_mask == 0x01010101u,
          "CT_NONE=0 preserves raw color_mask when COLOR_TARGET=0 (0x%X)",
          stub.pipeline_color_mask);
    CHECK(stub.last_depth_w == 1280u && stub.last_depth_h == 720u,
          "ZETA_EXACT_SIZE=0 pads small zeta to canvas size (%ux%u)",
          stub.last_depth_w, stub.last_depth_h);
    engine_down();
    unsetenv("CT_NONE");
    unsetenv("ZETA_EXACT_SIZE");
}

static void test_depth_color_snapshot(void)
{
    printf("-- depth->A8R8G8B8 snapshot (ZCONV_COLOR / ZS_CWRITTEN)\n");
    engine_up();

    /* 1. Depth-only pass at 0x0F340000 (512x288) with COLOR_TARGET=0 creates
     * both an un-written colour surface (cwritten=0) and a depth target. */
    m(0x0198, DMA_LOCAL);
    m(M_COLOR_A_OFFSET, 0x0F340000u);
    m(M_ZETA_OFFSET, 0x0F340000u);
    m(M_SURFACE_CLIP_H, 512u << 16);
    m(M_SURFACE_CLIP_V, 288u << 16);
    m(M_COLOR_TARGET, CELL_GCM_SURFACE_TARGET_NONE);
    m(0x0A74, 1u);   /* DEPTH_TEST_ENABLE */
    m(0x0A70, 1u);   /* DEPTH_WRITE_ENABLE */
    draw_triangle();
    const u32 z_half = stub.last_created_depth;
    const u32 c_half = stub.last_created_color;

    /* 2. Consumer pass reads 0x0F340000 as A8R8G8B8 (0x85) while drawing to
     * a different target/zeta: must pick depth_color_snapshot over cwritten=0. */
    m(M_COLOR_TARGET, CELL_GCM_SURFACE_TARGET_0);
    m(M_COLOR_A_OFFSET, 0x021C0000u);
    m(M_ZETA_OFFSET, 0x01010000u);
    m(M_SURFACE_CLIP_H, 1280u << 16);
    m(M_SURFACE_CLIP_V, 720u << 16);
    bind_tex_ex(0, RSX_LOCATION_LOCAL, 0x0F340000u, RSX_TEX_FMT_A8R8G8B8,
                512u, 288u, 0u, 0u);
    draw_triangle();
    CHECK(stub.n_depth_color_snapshot == 1 && stub.last_dcol_snap_src == z_half &&
          stub.last_dcol_snap_w == 512u && stub.last_dcol_snap_h == 288u &&
          stub.bound_tex[0] == stub.last_dcol_snap && stub.bound_tex[0] != c_half,
          "0x85 read of cwritten=0 aliased zeta binds depth_color_snapshot (%u vs %u)",
          stub.bound_tex[0], stub.last_dcol_snap);
    const u32 first_snap = stub.last_dcol_snap;

    /* 3. A second consumer draw without intervening writes reuses cached snapshot. */
    draw_triangle();
    CHECK(stub.n_depth_color_snapshot == 1 && stub.bound_tex[0] == first_snap,
          "second 0x85 consumer reuses cached depth_color_snapshot (%d)",
          stub.n_depth_color_snapshot);

    /* 3b. Deferred lighting pass: zeta stays bound as the current depth target
     * (read-only depth/bounds test) while sampling its own offset as A8R8G8B8.
     * Because depth_color_snapshot renders into a separate colour image, the
     * consumer still resolves to the snapshot rather than stale guest RAM. */
    m(M_ZETA_OFFSET, 0x0F340000u);
    m(M_SURFACE_CLIP_H, 512u << 16);
    m(M_SURFACE_CLIP_V, 288u << 16);
    m(0x0A74, 0u);   /* DEPTH_TEST_ENABLE = 0 */
    m(0x0A70, 0u);   /* DEPTH_WRITE_ENABLE = 0 */
    bind_tex_ex(0, RSX_LOCATION_LOCAL, 0x0F340000u, RSX_TEX_FMT_A8R8G8B8,
                512u, 288u, 0u, 0u);
    draw_triangle();
    CHECK(stub.n_depth_color_snapshot == 1 && stub.bound_tex[0] == first_snap,
          "0x85 read of currently bound zeta still binds depth_color_snapshot (%u vs %u)",
          stub.bound_tex[0], first_snap);

    /* 4. A stencil-only write to z_half invalidates its colour snapshot. */
    m(M_COLOR_TARGET, CELL_GCM_SURFACE_TARGET_NONE);
    m(M_COLOR_A_OFFSET, 0x0F340000u);
    m(M_ZETA_OFFSET, 0x0F340000u);
    m(M_SURFACE_CLIP_H, 512u << 16);
    m(M_SURFACE_CLIP_V, 288u << 16);
    m(0x0A70, 0u);   /* DEPTH_WRITE_ENABLE = 0 */
    m(0x0328, 1u);   /* STENCIL_TEST_ENABLE = 1 */
    m(0x032C, 0xFFu);/* STENCIL_MASK = 0xFF */
    m(M_TEX_UNIT + 0x0C, 0u);  /* disable unit 0 */
    draw_triangle();

    m(M_COLOR_TARGET, CELL_GCM_SURFACE_TARGET_0);
    m(M_COLOR_A_OFFSET, 0x021C0000u);
    m(M_ZETA_OFFSET, 0x01010000u);
    m(M_SURFACE_CLIP_H, 1280u << 16);
    m(M_SURFACE_CLIP_V, 720u << 16);
    bind_tex_ex(0, RSX_LOCATION_LOCAL, 0x0F340000u, RSX_TEX_FMT_A8R8G8B8,
                512u, 288u, 0u, 0u);
    draw_triangle();
    CHECK(stub.n_depth_color_snapshot == 2 && stub.bound_tex[0] == stub.last_dcol_snap,
          "stencil write invalidates cached depth_color_snapshot (%d)",
          stub.n_depth_color_snapshot);

    /* 5. Once colour IS drawn into 0x0F340000 (cwritten=1), colour target wins. */
    m(M_COLOR_TARGET, CELL_GCM_SURFACE_TARGET_0);
    m(M_COLOR_A_OFFSET, 0x0F340000u);
    m(M_ZETA_OFFSET, 0x01010000u);
    m(M_SURFACE_CLIP_H, 512u << 16);
    m(M_SURFACE_CLIP_V, 288u << 16);
    m(M_TEX_UNIT + 0x0C, 0u);
    draw_triangle();

    m(M_COLOR_A_OFFSET, 0x021C0000u);
    bind_tex_ex(0, RSX_LOCATION_LOCAL, 0x0F340000u, RSX_TEX_FMT_A8R8G8B8,
                512u, 288u, 0u, 0u);
    draw_triangle();
    CHECK(stub.bound_tex[0] == c_half && stub.n_depth_color_snapshot == 2,
          "cwritten=1 colour target wins over zeta at same offset (%u vs %u)",
          stub.bound_tex[0], c_half);
    engine_down();

    /* 6. Kill switch ZCONV_COLOR=0 disables 0x85 zeta conversion completely. */
    setenv("ZCONV_COLOR", "0", 1);
    engine_up();
    m(0x0198, DMA_LOCAL);
    m(M_COLOR_A_OFFSET, 0x01AB8000u);
    m(M_ZETA_OFFSET, 0x01010000u);
    m(0x0A74, 1u); m(0x0A70, 1u);
    draw_triangle();
    m(M_ZETA_OFFSET, 0x02000000u);
    bind_tex_ex(0, RSX_LOCATION_LOCAL, 0x01010000u, RSX_TEX_FMT_A8R8G8B8,
                256u, 256u, 0u, 0u);
    draw_triangle();
    CHECK(stub.n_depth_color_snapshot == 0,
          "ZCONV_COLOR=0 disables 0x85 depth_color_snapshot (%d)",
          stub.n_depth_color_snapshot);
    engine_down();
    unsetenv("ZCONV_COLOR");
}

static void load_guest_shadow_shaders(void)
{
    const u32 FP_OFF = 0x00031000u;
    u8* fp = vm_base + FP_OFF;
    const u32 SWZ_ID = (0u << 9) | (1u << 11) | (2u << 13) | (3u << 15);
    const u32 EXEC_A = (7u << 18) | (0u << 21) | (1u << 23) | (2u << 25) | (3u << 27);
    const u32 MASK_ALL = 0xFu << 9;
    /* instr 0: TEX r0, TC0, tex0 */
    fp_put_word(fp +  0, (0x17u << 24) | MASK_ALL | (0u << 1) | (4u << 13) | (0u << 17));
    fp_put_word(fp +  4, 1u | SWZ_ID | EXEC_A);
    fp_put_word(fp +  8, 0u);
    fp_put_word(fp + 12, 0u);
    /* instr 1: TXP r1, TC0, tex1 */
    fp_put_word(fp + 16, (0x18u << 24) | MASK_ALL | (1u << 1) | (4u << 13) | (1u << 17));
    fp_put_word(fp + 20, 1u | SWZ_ID | EXEC_A);
    fp_put_word(fp + 24, 0u);
    fp_put_word(fp + 28, 0u);
    /* instr 2: ADD r0, r0, r1 (END) */
    fp_put_word(fp + 32, (0x03u << 24) | MASK_ALL | (0u << 1) | 1u);
    fp_put_word(fp + 36, 0u | (0u << 2) | SWZ_ID | EXEC_A);
    fp_put_word(fp + 40, 0u | (1u << 2) | SWZ_ID);
    fp_put_word(fp + 44, 0u);

    m(0x1E9Cu, 0u);
    m(0x0B80u, 0u); m(0x0B84u, 0u); m(0x0B88u, 0u); m(0x0B8Cu, 1u);
    m(0x1EA4u, 0u);
    m(0x08E4u, FP_OFF | 2u);
    m(0x1D60u, 0x40u);
}

static void test_shadow_compare_and_border(void)
{
    printf("-- shadow compare PCF, SHADOW_LIT, and SAMP_BORDER\n");
    engine_up();
    load_guest_shadow_shaders();

    /* Write two shadow-map zetas: 0x0F340000 (1024x1024) and 0x0F440000 (512x512). */
    m(0x0198, DMA_LOCAL);
    m(M_COLOR_TARGET, CELL_GCM_SURFACE_TARGET_NONE);
    m(0x0A74, 1u); m(0x0A70, 1u);
    m(M_ZETA_OFFSET, 0x0F340000u);
    m(M_SURFACE_CLIP_H, 1024u << 16); m(M_SURFACE_CLIP_V, 1024u << 16);
    draw_triangle();
    m(M_ZETA_OFFSET, 0x0F440000u);
    m(M_SURFACE_CLIP_H, 512u << 16); m(M_SURFACE_CLIP_V, 512u << 16);
    draw_triangle();

    /* Lighting pass: unit 0 samples 0x0F340000 (zfunc=1 LESS, border 0x00FFFFFF),
     * unit 1 samples 0x0F440000 (zfunc=3 LEQUAL, projective TXP). */
    m(M_COLOR_TARGET, CELL_GCM_SURFACE_TARGET_0);
    m(M_COLOR_A_OFFSET, 0x021C0000u);
    m(M_ZETA_OFFSET, 0x01010000u);
    m(M_SURFACE_CLIP_H, 1280u << 16); m(M_SURFACE_CLIP_V, 720u << 16);
    bind_tex_ex(0, RSX_LOCATION_LOCAL, 0x0F340000u, RSX_TEX_FMT_DEPTH24_D8,
                1024u, 1024u, (1u << 28) | 4u | (4u << 8), 0x00FFFFFFu);
    bind_tex_ex(1, RSX_LOCATION_LOCAL, 0x0F440000u, RSX_TEX_FMT_DEPTH24_D8,
                512u, 512u, (3u << 28) | 4u | (4u << 8), 0x00FFFFFFu);
    draw_triangle();
    CHECK(strstr(stub.last_ps_hlsl, "#define RSX_SHRP 1") != NULL &&
          strstr(stub.last_ps_hlsl, "float4 rsx_shadow_eval(") != NULL &&
          strstr(stub.last_ps_hlsl, "float4 rsx_shadow_u0(") != NULL &&
          strstr(stub.last_ps_hlsl, "float4 rsx_shadow_u1(") != NULL,
          "shadow rewrite emits RSX_SHRP=1, rsx_shadow_eval, and per-unit wrappers");
    CHECK(strstr(stub.last_ps_hlsl, "rsx_shadow_u0(((input.tc0).xyzw), 0, 1)") != NULL &&
          strstr(stub.last_ps_hlsl, "rsx_shadow_u1(((input.tc0).xyzw), 1, 3)") != NULL,
          "shadow rewrite handles both TEX (proj=0, zf=1) and TXP (proj=1, zf=3)");
    { const char* p_samp = strstr(stub.last_ps_hlsl, "SamplerState rsx_samp");
      const char* p_eval = strstr(stub.last_ps_hlsl, "float4 rsx_shadow_eval");
      const char* p_main = strstr(stub.last_ps_hlsl, "float4 main(");
      CHECK(p_samp && p_eval && p_main && p_samp < p_eval && p_eval < p_main,
            "shadow wrappers are inserted after rsx_samp and before main()"); }
    CHECK(stub.bound_samplers[0].border_color == 0x00FFFFFFu,
          "SAMP_BORDER propagates guest border_color (0x%08X)",
          stub.bound_samplers[0].border_color);

    /* Changing zfunc selects a distinct pipeline. */
    const int pipes = stub.n_pipeline_create;
    bind_tex_ex(0, RSX_LOCATION_LOCAL, 0x0F340000u, RSX_TEX_FMT_DEPTH24_D8,
                1024u, 1024u, (6u << 28) | 4u | (4u << 8), 0x00FFFFFFu);
    draw_triangle();
    CHECK(stub.n_pipeline_create == pipes + 1 &&
          strstr(stub.last_ps_hlsl, "rsx_shadow_u0(((input.tc0).xyzw), 0, 6)") != NULL,
          "changing shadow zfunc builds a distinct pipeline with zf=6");

    /* Unbacked depth address falls back to SHADOW_LIT (1.0 texture) without
     * rewriting unit 0 into rsx_shadow_u0. */
    bind_tex_ex(0, RSX_LOCATION_LOCAL, 0x0D000000u, RSX_TEX_FMT_DEPTH24_D8,
                512u, 512u, (1u << 28), 0u);
    bind_tex_ex(1, RSX_LOCATION_LOCAL, 0x0D010000u, RSX_TEX_FMT_DEPTH24_D8,
                512u, 512u, (1u << 28), 0u);
    draw_triangle();
    CHECK(stub.bound_tex[0] != 0 && stub.bound_tex[0] == stub.bound_tex[1] &&
          strstr(stub.last_ps_hlsl, "rsx_shadow_u0(") == NULL,
          "unbacked depth units share SHADOW_LIT 1.0 texture with no compare rewrite");
    engine_down();

    /* Kill switches: SHADOW_FLIP=1, SHADOW_RP=0, SAMP_BORDER=0. */
    setenv("SHADOW_FLIP", "1", 1);
    setenv("SHADOW_RP", "0", 1);
    setenv("SAMP_BORDER", "0", 1);
    engine_up();
    load_guest_shadow_shaders();
    m(0x0198, DMA_LOCAL);
    m(M_COLOR_TARGET, CELL_GCM_SURFACE_TARGET_NONE);
    m(0x0A74, 1u); m(0x0A70, 1u);
    m(M_ZETA_OFFSET, 0x0F340000u);
    m(M_SURFACE_CLIP_H, 512u << 16); m(M_SURFACE_CLIP_V, 512u << 16);
    draw_triangle();
    m(M_COLOR_TARGET, CELL_GCM_SURFACE_TARGET_0);
    m(M_ZETA_OFFSET, 0x01010000u);
    bind_tex_ex(0, RSX_LOCATION_LOCAL, 0x0F340000u, RSX_TEX_FMT_DEPTH24_D8,
                512u, 512u, (1u << 28) | 4u, 0x00FFFFFFu);
    draw_triangle();
    CHECK(strstr(stub.last_ps_hlsl, "#define RSX_SHRP 0") != NULL &&
          strstr(stub.last_ps_hlsl, "rsx_shadow_u0(((input.tc0).xyzw), 0, 4)") != NULL,
          "SHADOW_FLIP=1 maps LESS(1)->GREATER(4) and SHADOW_RP=0 sets RSX_SHRP=0");
    CHECK(stub.bound_samplers[0].border_color == 0u,
          "SAMP_BORDER=0 zeroes sampler border_color (0x%08X)",
          stub.bound_samplers[0].border_color);
    engine_down();
    unsetenv("SHADOW_FLIP");
    unsetenv("SHADOW_RP");
    unsetenv("SAMP_BORDER");
}

static void test_immediate_data4ub(void)
{
    engine_up();
    m(M_BEGIN_END, RSX_PRIMITIVE_TRIANGLES);
    for (u32 v = 0; v < 3; v++) {
        m(0x194Cu, 0x11223344u);  /* ATTR3: 11 22 33 44 in memory order */
        for (u32 lane = 0; lane < 4; lane++)
            m(0x1C00u + lane * 4, f32_bits(lane == 3 ? 1.0f : (float)v));
    }
    m(M_BEGIN_END, 0u);
    CHECK(stub.n_draw == 1 && stub.draw_vertices == 3,
          "SET_VERTEX_DATA immediate mode emits a triangle");
    const float tolerance = 1.0f / 255.0f;
    CHECK(stub.draw_attr3[0] > 0.06f && stub.draw_attr3[0] < 0.07f &&
          stub.draw_attr3[1] > 0.12f && stub.draw_attr3[1] < 0.14f &&
          stub.draw_attr3[2] > 0.19f && stub.draw_attr3[2] < 0.21f &&
          stub.draw_attr3[3] > 0.25f && stub.draw_attr3[3] < 0.28f,
          "DATA4UB keeps x/y/z/w byte order and nonzero alpha (%.4g %.4g %.4g %.4g)",
          stub.draw_attr3[0], stub.draw_attr3[1], stub.draw_attr3[2], stub.draw_attr3[3]);
    CHECK(stub.draw_attr3[3] > tolerance * 15.0f,
          "menu text is not erased by a reversed alpha byte");
    engine_down();
}

/* Regression: eng_program_get previously XORed (vp_gen << 32) with (fp_reg << 32)
 * before mixing and trusted pg->id alone, so when (vp_gen1 ^ fp_reg1) == (vp_gen2 ^ fp_reg2)
 * a longer VP reused the shorter VP's cached vp_instrs and layout (causing VUID-07904). */
static void test_program_key_no_xor_collision(void)
{
    printf("-- program key no vp_gen^fp_reg XOR collision\n");
    engine_up();

    /* Two identical minimal FPs placed 0x40 bytes apart (fp_reg1 ^ fp_reg2 == 0x40). */
    const u32 FP1 = 0x00032000u;
    const u32 FP2 = FP1 ^ 0x40u;
    const u32 SWZ_ID = (0u << 9) | (1u << 11) | (2u << 13) | (3u << 15);
    const u32 EXEC_A = (7u << 18) | (0u << 21) | (1u << 23) | (2u << 25) | (3u << 27);
    const u32 MASK_ALL = 0xFu << 9;
    for (int k = 0; k < 2; k++) {
        u8* fp = vm_base + (k ? FP2 : FP1);
        fp_put_word(fp +  0, (0x01u << 24) | MASK_ALL | (0u << 1) | 1u); /* MOV r0, r0 (END) */
        fp_put_word(fp +  4, 0u | SWZ_ID | EXEC_A);
        fp_put_word(fp +  8, 0u);
        fp_put_word(fp + 12, 0u);
    }

    /* VP A (1 instr, reads ATTR0 only): VEC MOV o[0], v[0] (END). */
    const u32 vp_mov_v0_d0 = (1u << 30) | (0x3Fu << 15);
    const u32 vp_mov_v0_d1 = (1u << 22) | (0u << 8);      /* vec=MOV, input_src=0 */
    const u32 vp_mov_v0_d2 = (2u << 23);                  /* src0 reg_type=2 (input) */
    const u32 vp_mov_v0_d3 = 1u | (0u << 2) | (0xFu << 13); /* end=1, dst=o0, wmask=xyzw */

    /* VP B (2 instrs, reads ATTR0 and ATTR3):
     *   instr 0: VEC MOV o[1], v[3]
     *   instr 1: VEC MOV o[0], v[0] (END) */
    const u32 vp_mov_v3_d1 = (1u << 22) | (3u << 8);      /* vec=MOV, input_src=3 */
    const u32 vp_mov_v3_d3 = 0u | (1u << 2) | (0xFu << 13); /* end=0, dst=o1, wmask=xyzw */

    /* Upload VP A (advances vp_gen from 0 -> 4). */
    m(0x1E9Cu, 0u);
    m(0x0B80u, vp_mov_v0_d0); m(0x0B84u, vp_mov_v0_d1);
    m(0x0B88u, vp_mov_v0_d2); m(0x0B8Cu, vp_mov_v0_d3);
    m(0x1EA0u, 0u);
    m(0x08E4u, FP1 | 2u);
    m(0x1D60u, 0x40u);
    draw_triangle();
    CHECK(stub.pipeline_layout_count == 1u && stub.pipeline_vs_attrs == 1u,
          "short VP gets 1-attr layout and 1-attr VS (%u / %u)",
          stub.pipeline_layout_count, stub.pipeline_vs_attrs);

    /* Advance vp_gen by exactly 0x40 (64 word changes) so vp_gen2 == vp_gen1 ^ 0x40
     * (4 -> 68: 4 ^ 68 == 0x40 == FP1 ^ FP2), finishing with VP B at slots 0..1. */
    for (u32 step = 0; step < 58u; step++) {
        m(0x1E9Cu, 4u);
        m(0x0B80u, step + 100u);
    }
    /* Now upload VP B (6 words change from VP A/zeros: slot 0 d1/d3, slot 1 d0/d1/d2/d3 -> +6 = +64 total). */
    m(0x1E9Cu, 0u);
    m(0x0B80u, vp_mov_v0_d0); m(0x0B84u, vp_mov_v3_d1);
    m(0x0B88u, vp_mov_v0_d2); m(0x0B8Cu, vp_mov_v3_d3);
    m(0x0B90u, vp_mov_v0_d0); m(0x0B94u, vp_mov_v0_d1);
    m(0x0B98u, vp_mov_v0_d2); m(0x0B9Cu, vp_mov_v0_d3);
    m(0x08E4u, FP2 | 2u);
    draw_triangle();
    CHECK(stub.pipeline_layout_count == 2u && stub.pipeline_vs_attrs == 2u,
          "colliding vp_gen^fp_reg does not alias short VP (layout=%u vs_attrs=%u)",
          stub.pipeline_layout_count, stub.pipeline_vs_attrs);
    engine_down();
}

int main(void)
{
    vm_base = (u8*)calloc(1, GUEST_BYTES);
    if (!vm_base) { printf("[FAIL] guest memory\n"); return 1; }
    ppu_vm_size = GUEST_BYTES;
    /* Three vertices of float4 position, so the fetch has something real to
     * read: the engine treats an unreadable ATTR0 as a failed group. */
    for (u32 i = 0; i < 3 * 4; i++) {
        u8* p = vm_base + VTX_OFFSET + i * 4;
        p[0] = 0x3F; p[1] = 0x80; p[2] = 0; p[3] = 0;    /* 1.0f big-endian */
    }

    test_surface_set();
    test_surface_aliasing();
    test_mrt_set();
    test_texture_cache();
    test_texture_sparse_validation();
    test_vertex_cache();
    test_pipeline_key();
    test_restart_expansion();
    test_indexed_draws();
    test_scissor();
    test_readback();
    test_custom_guest_mapping();
    test_queued_flip_buffer();
    test_inline_array();
    test_immediate_data4ub();
    test_depth_bounds();
    test_tex_xf();
    test_ct_none_and_zeta_sizing();
    test_depth_color_snapshot();
    test_shadow_compare_and_border();
    test_program_key_no_xor_collision();

    free(vm_base);
    printf(g_failures ? "\n%d check(s) FAILED\n" : "\nall checks passed\n",
           g_failures);
    return g_failures ? 1 : 0;
}
