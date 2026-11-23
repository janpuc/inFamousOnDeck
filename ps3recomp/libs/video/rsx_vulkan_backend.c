/*
 * ps3recomp - Vulkan RSX Backend. See rsx_vulkan_backend.h.
 *
 * The file, in order:
 *   loader and device      libvulkan by dlopen, Vulkan 1.0 entry points only;
 *                          device choice, BC textures when the device has them
 *   memory, images, submit helpers
 *   vtable fallback path   the null backend's fixed-function contract:
 *                          position = attribute 0, flat colour = attribute 3
 *                          of each triangle's first vertex, texcoord0 =
 *                          attribute 8, fetched and expanded through the
 *                          shared rsx_fetch_attrib(); unit 0 decoded through
 *                          rsx_texture_decode(); depth test, no blend/stencil
 *   vtable guest programs  the same path running the decompilers' HLSL, with
 *                          whole mip chains and per-unit samplers
 *   optional window        SDL3 + swapchain; a present blits the frame into it
 *   draw engine backend    rsx_draw_backend: targets, clears, present and
 *                          readback (E1), pipelines and draws (E2), textures,
 *                          surface views and samplers (E3), depth snapshots
 *                          (E4) -- the stage names in the commit history
 *   public entry points    init picks the path, shutdown undoes everything
 *
 * Deliberately synchronous: every clear, draw and present is one one-shot
 * command buffer, submitted and waited on. Slow and trivially correct; it is
 * what lets the engine's submit_and_wait and staging reuse cost nothing, and
 * batching is the first thing to change once correctness is settled.
 *
 * Images stay in VK_IMAGE_LAYOUT_GENERAL for their whole life: legal for
 * transfers, sampling and as attachments, so there is no layout bookkeeping.
 *
 * Shaders: libs/video/vulkan/rsx_vk_fallback.{vert,frag}, embedded as SPIR-V.
 * Regenerate after editing them with:
 *   glslangValidator -V --target-env vulkan1.0 --vn rsx_vk_fallback_vert_spv \
 *       -o rsx_vk_fallback_vert.spv.h rsx_vk_fallback.vert
 *   glslangValidator -V --target-env vulkan1.0 --vn rsx_vk_fallback_frag_spv \
 *       -o rsx_vk_fallback_frag.spv.h rsx_vk_fallback.frag
 * then `expand -t 4` both outputs: glslang indents with tabs, the project
 * with spaces.
 */

#ifndef _WIN32

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "rsx_vulkan_backend.h"
#include "rsx_vertex_fetch.h"
#include "rsx_texture_layout.h"
#include "rsx_vp_decompiler.h"
#include "rsx_fp_decompiler.h"
#include "rsx_shader_spirv.h"
#include "rsx_draw_engine.h"
#include "rsx_upscale_common.h"
#include "rsx_vk_fsr.h"
#ifdef PS3RECOMP_VK_FFX
#include "ffx_vk_vulkan_compat.h"
#include "ffx_vk_framegeneration_presenter.h"
#endif
#define XXH_INLINE_ALL
#include "xxhash.h"
#include "vulkan/rsx_vk_fallback_vert.spv.h"
#include "vulkan/rsx_vk_fallback_frag.spv.h"
#include "vulkan/rsx_vk_uiov_vert.spv.h"
#include "vulkan/rsx_vk_uiov_frag.spv.h"
#include "rsx_ui_trainer.h"   /* uiov_vk_*: the in-game overlay batch (rsx_ui_overlay.c) */

/* SDL3 provides the native Vulkan window and, with the same library, the
 * non-Windows cellPad and cellAudio backends. A build without SDL3 keeps every
 * offscreen path and reports the window as unavailable. */
#if defined(PS3RECOMP_SDL3_NATIVE)
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#define VK_WINDOW_SDL 1
#else
#define VK_WINDOW_SDL 0
#endif

#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __linux__
#include <linux/futex.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

/* ---------------------------------------------------------------------------
 * Run-time loaded entry points (Vulkan 1.0 core only)
 * -------------------------------------------------------------------------*/
#define VK_GLOBAL_FNS(X) X(vkCreateInstance)
#define VK_INSTANCE_FNS(X) \
    X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) \
    X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceFormatProperties) \
    X(vkGetPhysicalDeviceFeatures) \
    X(vkCreateDevice) X(vkGetDeviceProcAddr) X(vkEnumerateDeviceExtensionProperties)
#define VK_DEVICE_FNS(X) \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkDeviceWaitIdle) \
    X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements) X(vkBindImageMemory) \
    X(vkCreateImageView) X(vkDestroyImageView) \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory) X(vkUnmapMemory) \
    X(vkInvalidateMappedMemoryRanges) \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer) \
    X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences) X(vkQueueSubmit) \
    X(vkCreateRenderPass) X(vkDestroyRenderPass) X(vkCreateFramebuffer) X(vkDestroyFramebuffer) \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) \
    X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) \
    X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkResetDescriptorPool) \
    X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) \
    X(vkCreateGraphicsPipelines) X(vkDestroyPipeline) \
    X(vkCreateSampler) X(vkDestroySampler) \
    X(vkCmdPipelineBarrier) X(vkCmdClearColorImage) X(vkCmdClearDepthStencilImage) \
    X(vkCmdCopyImageToBuffer) X(vkCmdCopyBufferToImage) X(vkCmdCopyImage) \
    X(vkCmdBeginRenderPass) X(vkCmdEndRenderPass) X(vkCmdBindPipeline) \
    X(vkCmdBindVertexBuffers) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) \
    X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdDraw) \
    X(vkCmdSetStencilReference) X(vkCmdSetDepthBounds) \
    X(vkCmdSetDepthBias) X(vkCmdSetBlendConstants) \
    X(vkCmdBindIndexBuffer) X(vkCmdDrawIndexed) X(vkCmdDrawIndexedIndirect) \
    X(vkCreateSemaphore) X(vkDestroySemaphore) X(vkCmdBlitImage)
/* Window-system entry points, loaded only when a window was requested, so a
 * headless run never depends on a driver that can present. */
#define VK_WSI_INSTANCE_FNS(X) \
    X(vkDestroySurfaceKHR) X(vkGetPhysicalDeviceSurfaceSupportKHR) \
    X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) X(vkGetPhysicalDeviceSurfaceFormatsKHR) \
    X(vkGetPhysicalDeviceSurfacePresentModesKHR)
#define VK_WSI_DEVICE_FNS(X) \
    X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) X(vkGetSwapchainImagesKHR) \
    X(vkAcquireNextImageKHR) X(vkQueuePresentKHR)

#define VK_DECLARE(name) static PFN_##name p##name;
static PFN_vkGetInstanceProcAddr pvkGetInstanceProcAddr;
VK_GLOBAL_FNS(VK_DECLARE)
VK_INSTANCE_FNS(VK_DECLARE)
VK_DEVICE_FNS(VK_DECLARE)
VK_WSI_INSTANCE_FNS(VK_DECLARE)
VK_WSI_DEVICE_FNS(VK_DECLARE)

/* ---------------------------------------------------------------------------
 * Backend state
 * -------------------------------------------------------------------------*/
typedef struct vk_vertex {
    float pos[4];
    float col[4];
    float uv[2];
} vk_vertex;

typedef struct vk_buffer {
    VkBuffer       buf;
    VkDeviceMemory mem;
    void*          ptr;
    VkDeviceSize   size;
    int            coherent;
} vk_buffer;

typedef struct vk_image {
    VkImage        img;
    VkDeviceMemory mem;
    VkImageView    view;
} vk_image;

#define VK_SAMPLER_CACHE 32
#define VK_PIPELINE_SLOTS (2 * 2 * 8)   /* ztest x zwrite x compare op */

typedef struct vk_state {
    void*            lib;
    VkInstance       instance;
    VkPhysicalDevice phys;
    VkDevice         device;
    VkQueue          queue;
    u32              queue_family;
    VkPhysicalDeviceMemoryProperties memprops;

    u32              width, height;
    vk_image         color;           /* R8G8B8A8_UNORM                 */
    vk_image         depth;           /* D32_SFLOAT or fallback         */
    VkFormat         depth_format;
    VkImageAspectFlags depth_aspect;  /* DEPTH, plus STENCIL if it has one  */

    VkRenderPass     render_pass;
    VkFramebuffer    framebuffer;
    VkShaderModule   vs, fs;
    VkDescriptorSetLayout set_layout;
    VkDescriptorPool desc_pool;
    VkDescriptorSet  desc_set;
    VkPipelineLayout pipe_layout;
    VkPipeline       pipelines[VK_PIPELINE_SLOTS];
    VkSampler        sampler;

    vk_image         dummy;           /* 1x1 magenta: "nothing bound"    */
    vk_image         tex[RSX_MAX_TEXTURES];   /* decoded guest textures  */
    int              tex_ready[RSX_MAX_TEXTURES];
    /* Guest path only: each unit's sampler, built from its wrap, filter and
     * LOD registers (the fallback path keeps `sampler`, the null backend's
     * nearest + repeat). Cached by register value; VK_NULL_HANDLE = use
     * `sampler`. */
    VkSampler        tex_sampler[RSX_MAX_TEXTURES];
    struct { u64 key; VkSampler s; } samp_cache[VK_SAMPLER_CACHE];
    u32              samp_count;

    vk_buffer        vertices;        /* expanded triangle list          */
    vk_vertex*       cpu_verts;
    size_t           cpu_cap, cpu_count;

    vk_buffer        readback;
    vk_buffer        fg_readback;     /* headless -fg.ppm dump (lazily allocated) */
    int              fg_readback_ready;
    VkCommandPool    pool;
    VkCommandBuffer  cmd;
    VkCommandBuffer  sync_cmd;
    VkFence          fence;

    /* A present records and submits on its own pool, command buffer and
     * fences, so its second half can run on a host thread of its own while
     * the FIFO walker records the next frame (rsx_vulkan_backend.h,
     * present_begin/_finish). The queue is shared, and Vulkan wants every
     * vkQueueSubmit / vkQueuePresentKHR / vkDeviceWaitIdle on it externally
     * synchronised: queue_mu. present_mu keeps the present's own objects to
     * one present at a time, which matters when a flip in the command stream
     * presents on the walker's thread while a host thread finishes the last. */
    VkCommandPool    present_pool;
    VkCommandBuffer  present_cmd;
    VkFence          present_fence;   /* the copy out of the presented surface */
    VkFence          window_fence;    /* the blit into the swapchain image     */
    int              present_inflight;/* copy submitted, not yet finished      */
    int              present_split;   /* inside rsx_vulkan_backend_present_begin */
    int              present_order;   /* the walker's next command buffer waits */
    u32              pending_draws;   /* guest draws of the in-flight present   */
    pthread_mutex_t  queue_mu;
    pthread_mutex_t  present_mu;

    /* Optional window (PS3RECOMP_VK_WINDOW=1). Rendering stays offscreen
     * either way -- readback and the tests are unchanged -- and a present
     * additionally blits the frame into the window's swapchain. */
    int              eng_active;      /* the register-file engine drives us */
    u32              guest_w, guest_h;  /* PS3 guest display size (1280x720) */
    u32              render_w, render_h;/* internal render size (RES_SCALE / FSR) */
    int              bc_ok;           /* textureCompressionBC enabled       */
    int              depth_bounds_ok; /* depthBounds enabled                */
    int              depth_clamp_ok;  /* depthClamp enabled                 */
    int              indep_blend_ok;  /* independentBlend: per-target MRT state */
    int              multi_draw_indirect_ok; /* multiDrawIndirect + drawIndirectFirstInstance */
    int              aniso_ok;        /* samplerAnisotropy enabled          */
    float            max_aniso;       /* limits.maxSamplerAnisotropy        */
    u32              max_dim;         /* maxImageDimension2D of the device  */
    int              windowed;
    void*            window;          /* SDL_Window*, only with VK_WINDOW_SDL */
    const char*      inst_exts[16];
    u32              inst_ext_count;
    VkSurfaceKHR     surface;
    VkSwapchainKHR   swapchain;
    VkExtent2D       sc_extent;
    u32              sc_count;
    VkImage          sc_images[8];
    VkSemaphore      sc_done[8];      /* per image: the present waits on it */
    VkSemaphore      sem_acquire;
    VkSemaphore      sem_acquire_real;/* second acquire semaphore for FSR FG pair */
    int              fg_was_active;   /* tracks WSI FG transitions for quiescence */
    u64              fg_window_presents;
    VkFormat         sc_format;       /* swapchain format */
    int              sc_overlay_ok;   /* images take a colour attachment + blend: the overlay pass can draw */
    int              sc_dump_ok;      /* PS3RECOMP_VK_OVERLAY_DUMP: images also allow TRANSFER_SRC */
    double           hold_seconds;

    const rsx_state* state;           /* live RSX state, for draws       */
    u32              clear_argb;
    u32              last_center;
    u32              presented;
    const char*      dump_path;
    u32              dump_every;       /* write every nth presented frame */
    u32              dump_frames[8];   /* ... or exactly these presented frames */
    u32              dump_n_frames;

    /* Present/FPS telemetry. A frame sample is recorded only for a present
     * carrying new guest draws, not for a duplicate host present. */
    u64              draw_total;
    u64              rendered_frames;
    u32              last_draws;
    u64              last_present_ms;
    u64              window_presents;
    u64              window_present_failures;
    double           frame_prev_ms;
    double           frame_t0;
    double           frame_times[300];
    u64              frame_draw_sum;
    u32              frame_count;
} vk_state;

static vk_state s_vk;

#define VK_LOG(...)  fprintf(stderr, "[RSX vulkan] " __VA_ARGS__)
#define VK_CHECK(call, what)                                               \
    do { VkResult r_ = (call);                                             \
         if (r_ != VK_SUCCESS) {                                           \
             VK_LOG("%s failed (VkResult %d)\n", (what), (int)r_);         \
             return -1; } } while (0)

static double vk_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

/* PS3RECOMP_ENG_STATS=1: the GPU half of the walker/GPU split. `gpu_wait_ms`
 * is every fence wait summed, `submits` how many submissions they were, so the
 * [engtime] gpu line says whether a frame's wall time is the walker's own work
 * or its wait on the device. Enable the knob before the device exists: read
 * once here, not per submit. */
static struct {
    double gpu_wait_ms;
    u64 submits;
    u64 barriers;
    u64 rps;
    u64 desc_hits, desc_misses;
    u64 gvb_hits, gvb_misses, gib_hits, gib_misses;
    u64 ubo_reuses;
    u64 ubo_bytes;
    u64 desc_binds;
    u64 push_consts;
    u64 ind_calls, ind_draws, dir_draws;
} s_engtime;
static int s_eng_stats;

/* The port's in-run A/B phase (ppu_loader.cpp), printed beside the timings. */
extern volatile int g_ab_phase;

void rsx_vulkan_backend_engtime(u64* gpu_wait_ms, u64* submits)
{
    if (gpu_wait_ms) *gpu_wait_ms = (u64)s_engtime.gpu_wait_ms;
    if (submits) *submits = s_engtime.submits;
}

static int vk_double_cmp(const void* a, const void* b)
{
    const double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}

static void vk_frame_telemetry(u32 draws)
{
    const char* pct = getenv("FRAMEPCT");
    if (!draws || (pct && *pct == '0')) return;
    const double now = vk_now_ms();
    if (!s_vk.frame_prev_ms) {
        s_vk.frame_prev_ms = s_vk.frame_t0 = now;
        return;
    }
    const double elapsed = now - s_vk.frame_prev_ms;
    s_vk.frame_prev_ms = now;
    if (elapsed <= 0.0 || elapsed > 60000.0) return;

    s_vk.frame_times[s_vk.frame_count++] = elapsed;
    s_vk.frame_draw_sum += draws;
    if (s_vk.frame_count < 300) return;

    double sorted[300], wall = 0.0;
    unsigned n20 = 0, n25 = 0, n33 = 0;
    memcpy(sorted, s_vk.frame_times, sizeof sorted);
    qsort(sorted, 300, sizeof sorted[0], vk_double_cmp);
    for (unsigned i = 0; i < 300; i++) {
        wall += sorted[i];
        n20 += sorted[i] > 20.0;
        n25 += sorted[i] > 25.0;
        n33 += sorted[i] > 33.4;
    }
    const double low1 = 3000.0 / (sorted[297] + sorted[298] + sorted[299]);
    fprintf(stderr, "[FRAMEPCT] t=%.1f 300 presented frames: avg %.2f fps, p50 %.2f ms, p99 %.2f ms, "
                    "1%% low %.1f fps, 0.1%% low %.1f fps, >20ms %u, >25ms %u, >33ms %u, "
                    "worst %.1f ms | mean draws %.1f\n",
            (now - s_vk.frame_t0) / 1000.0, 300000.0 / wall, sorted[149], sorted[296], low1,
            1000.0 / sorted[299], n20, n25, n33, sorted[299],
            (double)s_vk.frame_draw_sum / 300.0);
    fflush(stderr);
    s_vk.frame_count = 0;
    s_vk.frame_draw_sum = 0;
}

void rsx_vulkan_backend_heartbeat(void)
{
    static int enabled = -1;
    static double t0, last;
    static u32 flips0, presented0, draws0;
    static u64 frames0, window0;
    if (enabled < 0) {
        const char* e = getenv("HEARTBEAT");
        enabled = !(e && *e == '0');
    }
    if (!enabled || !s_vk.device) return;
    const double now = vk_now_ms();
    if (!t0) {
        t0 = last = now;
        return;
    }
    if (now - last < 10000.0) return;
    last = now;
    extern unsigned cellGcm_flip_request_count(void);
    const u32 flips = cellGcm_flip_request_count();
    char age[48];
    if (s_vk.last_present_ms)
        snprintf(age, sizeof age, "%llu ms ago",
                 (unsigned long long)(now - (double)s_vk.last_present_ms));
    else
        snprintf(age, sizeof age, "never");
    fprintf(stderr, "[HEARTBEAT] t=%.0fs flips %u (+%u) presents %u (+%u) frames +%llu (+%u draws) | "
                    "last present %s, last frame %u draws | window presents %llu (+%llu), failures %llu\n",
            (now - t0) / 1000.0, flips, flips - flips0, s_vk.presented,
            s_vk.presented - presented0, (unsigned long long)(s_vk.rendered_frames - frames0),
            (u32)(s_vk.draw_total - draws0), age, s_vk.last_draws,
            (unsigned long long)s_vk.window_presents,
            (unsigned long long)(s_vk.window_presents - window0),
            (unsigned long long)s_vk.window_present_failures);
    fflush(stderr);
    flips0 = flips;
    presented0 = s_vk.presented;
    draws0 = s_vk.draw_total;
    frames0 = s_vk.rendered_frames;
    window0 = s_vk.window_presents;
}

/* ---------------------------------------------------------------------------
 * Loading
 * -------------------------------------------------------------------------*/
static int vk_load_library(void)
{
    static const char* names[] = { "libvulkan.so.1", "libvulkan.so" };
    for (size_t i = 0; i < sizeof names / sizeof names[0] && !s_vk.lib; i++)
        s_vk.lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
    if (!s_vk.lib) {
        VK_LOG("no Vulkan loader found (dlopen libvulkan.so.1: %s)\n", dlerror());
        return -1;
    }
    pvkGetInstanceProcAddr =
        (PFN_vkGetInstanceProcAddr)dlsym(s_vk.lib, "vkGetInstanceProcAddr");
    if (!pvkGetInstanceProcAddr) { VK_LOG("libvulkan has no vkGetInstanceProcAddr\n"); return -1; }
#define LOAD_G(name) \
    if (!(p##name = (PFN_##name)pvkGetInstanceProcAddr(VK_NULL_HANDLE, #name))) \
        { VK_LOG("missing global function " #name "\n"); return -1; }
    VK_GLOBAL_FNS(LOAD_G)
    return 0;
}

static int vk_load_instance_functions(void)
{
#define LOAD_I(name) \
    if (!(p##name = (PFN_##name)pvkGetInstanceProcAddr(s_vk.instance, #name))) \
        { VK_LOG("missing instance function " #name "\n"); return -1; }
    VK_INSTANCE_FNS(LOAD_I)
    return 0;
}

static int vk_load_wsi_instance_functions(void)
{
    VK_WSI_INSTANCE_FNS(LOAD_I)
    return 0;
}

static int vk_load_device_functions(void)
{
#define LOAD_D(name) \
    if (!(p##name = (PFN_##name)pvkGetDeviceProcAddr(s_vk.device, #name))) \
        { VK_LOG("missing device function " #name "\n"); return -1; }
    VK_DEVICE_FNS(LOAD_D)
    return 0;
}

static int vk_load_wsi_device_functions(void)
{
    VK_WSI_DEVICE_FNS(LOAD_D)
    return 0;
}

/* ---------------------------------------------------------------------------
 * Device selection: PS3RECOMP_VK_DEVICE if set, else the first discrete GPU,
 * else integrated, else anything that is not a CPU rasteriser, else whatever
 * is there (llvmpipe/lavapipe on a GPU-less CI runner).
 * -------------------------------------------------------------------------*/
static int vk_device_rank(VkPhysicalDeviceType t)
{
    switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 4;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:            return 0;
    default:                                     return 1;
    }
}

static int vk_find_graphics_queue(VkPhysicalDevice pd, u32* family)
{
    u32 n = 0;
    pvkGetPhysicalDeviceQueueFamilyProperties(pd, &n, NULL);
    if (n == 0 || n > 64) return -1;
    VkQueueFamilyProperties props[64];
    pvkGetPhysicalDeviceQueueFamilyProperties(pd, &n, props);
    for (u32 i = 0; i < n; i++) {
        if (!(props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
        if (s_vk.windowed) {           /* the same queue must also present */
            VkBool32 ok = VK_FALSE;
            if (pvkGetPhysicalDeviceSurfaceSupportKHR(pd, i, s_vk.surface, &ok) != VK_SUCCESS || !ok)
                continue;
        }
        *family = i; return 0;
    }
    return -1;
}

static int vk_has_device_extension(VkPhysicalDevice pd, const char* name)
{
    u32 n = 0;
    if (pvkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL) != VK_SUCCESS || n == 0) return 0;
    VkExtensionProperties* e = (VkExtensionProperties*)malloc(n * sizeof *e);
    if (!e) return 0;
    int found = 0;
    if (pvkEnumerateDeviceExtensionProperties(pd, NULL, &n, e) == VK_SUCCESS)
        for (u32 i = 0; i < n && !found; i++) found = strcmp(e[i].extensionName, name) == 0;
    free(e);
    return found;
}

static int vk_pick_device(void)
{
    u32 n = 0;
    VK_CHECK(pvkEnumeratePhysicalDevices(s_vk.instance, &n, NULL), "vkEnumeratePhysicalDevices");
    if (n == 0) { VK_LOG("no Vulkan physical devices\n"); return -1; }
    if (n > 16) n = 16;
    VkPhysicalDevice devs[16];
    VK_CHECK(pvkEnumeratePhysicalDevices(s_vk.instance, &n, devs), "vkEnumeratePhysicalDevices");

    int best = -1, best_rank = -1;
    const char* forced = getenv("PS3RECOMP_VK_DEVICE");
    for (u32 i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        pvkGetPhysicalDeviceProperties(devs[i], &p);
        u32 fam;
        int usable = vk_find_graphics_queue(devs[i], &fam) == 0 &&
                     (!s_vk.windowed ||
                      vk_has_device_extension(devs[i], VK_KHR_SWAPCHAIN_EXTENSION_NAME));
        VK_LOG("device %u: %s (type %d, api %u.%u.%u)%s\n", i, p.deviceName,
               (int)p.deviceType, VK_API_VERSION_MAJOR(p.apiVersion),
               VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion),
               usable ? "" : (s_vk.windowed ? " -- cannot present to the window"
                                            : " -- no graphics queue"));
        if (!usable) continue;
        if (forced) {
            if ((u32)atoi(forced) == i) { best = (int)i; break; }
            continue;
        }
        int rank = vk_device_rank(p.deviceType);
        if (rank > best_rank) { best = (int)i; best_rank = rank; }
    }
    if (best < 0) {
        if (forced) VK_LOG("PS3RECOMP_VK_DEVICE=%s is not a usable device\n", forced);
        else        VK_LOG("no device with a graphics queue\n");
        return -1;
    }
    s_vk.phys = devs[best];
    vk_find_graphics_queue(s_vk.phys, &s_vk.queue_family);
    pvkGetPhysicalDeviceMemoryProperties(s_vk.phys, &s_vk.memprops);
    VkPhysicalDeviceProperties p;
    pvkGetPhysicalDeviceProperties(s_vk.phys, &p);
    /* Vulkan guarantees 4096; the engine's targets and textures may use what
     * the device really has (a title's 4608-wide target on RADV). */
    s_vk.max_dim = p.limits.maxImageDimension2D >= 4096u ? p.limits.maxImageDimension2D : 4096u;
    s_vk.max_aniso = p.limits.maxSamplerAnisotropy >= 1.0f ? p.limits.maxSamplerAnisotropy : 1.0f;
    VK_LOG("using device %d: %s\n", best, p.deviceName);
    return 0;
}

/* ---------------------------------------------------------------------------
 * Memory, images, buffers
 * -------------------------------------------------------------------------*/
static int vk_find_memory_type(u32 type_bits, VkMemoryPropertyFlags want, u32* out)
{
    for (u32 i = 0; i < s_vk.memprops.memoryTypeCount; i++)
        if ((type_bits & (1u << i)) &&
            (s_vk.memprops.memoryTypes[i].propertyFlags & want) == want) {
            *out = i; return 0;
        }
    return -1;
}

static int vk_create_image_levels(vk_image* im, VkFormat fmt, u32 w, u32 h, u32 levels,
                                  VkImageUsageFlags usage, VkImageAspectFlags aspect)
{
    if (!levels) levels = 1;
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
        .format = fmt, .extent = { w, h, 1 }, .mipLevels = levels, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VK_CHECK(pvkCreateImage(s_vk.device, &ici, NULL, &im->img), "vkCreateImage");
    VkMemoryRequirements req;
    pvkGetImageMemoryRequirements(s_vk.device, im->img, &req);
    u32 type;
    if (vk_find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &type) &&
        vk_find_memory_type(req.memoryTypeBits, 0, &type)) {
        VK_LOG("no memory type for an image\n"); return -1;
    }
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = req.size, .memoryTypeIndex = type };
    VK_CHECK(pvkAllocateMemory(s_vk.device, &mai, NULL, &im->mem), "vkAllocateMemory(image)");
    VK_CHECK(pvkBindImageMemory(s_vk.device, im->img, im->mem, 0), "vkBindImageMemory");
    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im->img,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
        .subresourceRange = { aspect, 0, levels, 0, 1 },
    };
    VK_CHECK(pvkCreateImageView(s_vk.device, &vci, NULL, &im->view), "vkCreateImageView");
    return 0;
}

static int vk_create_image(vk_image* im, VkFormat fmt, u32 w, u32 h,
                           VkImageUsageFlags usage, VkImageAspectFlags aspect)
{
    return vk_create_image_levels(im, fmt, w, h, 1, usage, aspect);
}

static void vk_present_wait_gpu(void);

/* VK_DRAW_BATCH (vk_eng_draw): draws recorded into s_vk.cmd and not yet
 * submitted. Anything that records other work, destroys an object or reads
 * results flushes them first, so everything else keeps the invariant that
 * all earlier work has completed. */
static u32 s_eb_pending;
static void vk_eng_flush(void);       /* sync: flush open batch + wait all in-flight */
static void vk_eng_flush_async(void); /* async when M2a active, else sync */
static int vk_eb_active(void);       /* batching decided on and on */
static int vk_eb_async_use(void);    /* M2a async batching/vcache active this phase */
static int vk_eb_open_cmd(void);     /* begin current async slot command buffer */
static u32 vk_eb_max(void);
static void vk_eb_recorded(void);   /* one more command in the open batch */
static void vk_eb_release(void);    /* engine shutdown: rings and pool */
static int vk_eb_upload(VkImage img, VkBufferImageCopy* region, const void* src, size_t tight,
                        u32 rows, u32 row_bytes);   /* -2: not batched, do it the old way */
static int vk_eb_any_inflight(void);
static int vk_eb_retire_image(vk_image* im);
static void vk_eb_inval_fast_rp(void);
static int vk_rec_use_batch(void);
static int vk_rec_ensure_open(void);
static void vk_rec_wait_submitted(void);
static void vk_rec_emit_clear_color(VkImage img, const float rgba[4]);
static void vk_rec_emit_clear_depth(VkImage img, VkImageAspectFlags aspect, float depth_value, u8 stencil);
static void vk_rec_emit_img_init(VkImage img, VkImageAspectFlags aspect);
static void vk_rec_emit_color_snap(VkImage src, VkImage dst, u32 w, u32 h);
static void vk_rec_emit_depth_snap(VkImage zimg, VkImage timg, VkBuffer stbuf, u32 w, u32 h);
static void vk_rec_emit_zconv(VkFramebuffer fb, VkDescriptorSet set, VkImage cimg, u32 w, u32 h);

static void vk_destroy_image(vk_image* im)
{
    if (s_eb_pending || vk_eb_any_inflight()) vk_eng_flush();
    /* Never under a present's copy that may still read it. */
    if (im->img) vk_present_wait_gpu();
    if (im->view) pvkDestroyImageView(s_vk.device, im->view, NULL);
    if (im->img)  pvkDestroyImage(s_vk.device, im->img, NULL);
    if (im->mem)  pvkFreeMemory(s_vk.device, im->mem, NULL);
    memset(im, 0, sizeof *im);
}

static int vk_create_host_buffer(vk_buffer* b, VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                               .size = size, .usage = usage,
                               .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    VK_CHECK(pvkCreateBuffer(s_vk.device, &bci, NULL, &b->buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    pvkGetBufferMemoryRequirements(s_vk.device, b->buf, &req);
    u32 type;
    const VkMemoryPropertyFlags vis = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const VkMemoryPropertyFlags coh = vis | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    /* A buffer the GPU only copies into is one the CPU reads back, and the
     * first coherent type is usually write-combined and uncached (RADV's
     * GTT): there the readback memcpy was 82% of the FIFO thread's time in
     * Twisted Metal. Take a cached type for it when there is one; reads
     * invalidate, so a non-coherent one is fine. */
    const VkMemoryPropertyFlags cached = vis | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    const int readback = usage == VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (readback && vk_find_memory_type(req.memoryTypeBits, cached | coh, &type) == 0) b->coherent = 1;
    else if (readback && vk_find_memory_type(req.memoryTypeBits, cached, &type) == 0) b->coherent = 0;
    else if (vk_find_memory_type(req.memoryTypeBits, coh, &type) == 0) b->coherent = 1;
    else if (vk_find_memory_type(req.memoryTypeBits, vis, &type) == 0) b->coherent = 0;
    else { VK_LOG("no host-visible memory for a buffer\n"); return -1; }
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = req.size, .memoryTypeIndex = type };
    VK_CHECK(pvkAllocateMemory(s_vk.device, &mai, NULL, &b->mem), "vkAllocateMemory(buffer)");
    VK_CHECK(pvkBindBufferMemory(s_vk.device, b->buf, b->mem, 0), "vkBindBufferMemory");
    VK_CHECK(pvkMapMemory(s_vk.device, b->mem, 0, VK_WHOLE_SIZE, 0, &b->ptr), "vkMapMemory");
    b->size = size;
    return 0;
}

/* A buffer only the GPU touches (copies through it stay in VRAM). */
static int vk_create_device_buffer(vk_buffer* b, VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                               .size = size, .usage = usage,
                               .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    VK_CHECK(pvkCreateBuffer(s_vk.device, &bci, NULL, &b->buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    pvkGetBufferMemoryRequirements(s_vk.device, b->buf, &req);
    u32 type;
    if (vk_find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &type) &&
        vk_find_memory_type(req.memoryTypeBits, 0, &type)) { VK_LOG("no memory for a device buffer\n"); return -1; }
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = req.size, .memoryTypeIndex = type };
    VK_CHECK(pvkAllocateMemory(s_vk.device, &mai, NULL, &b->mem), "vkAllocateMemory(device buffer)");
    VK_CHECK(pvkBindBufferMemory(s_vk.device, b->buf, b->mem, 0), "vkBindBufferMemory");
    b->ptr = NULL; b->coherent = 0; b->size = size;
    return 0;
}

static void vk_destroy_buffer(vk_buffer* b);

/* Prefer DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (ReBAR VRAM on RADV APUs),
 * falling back to ordinary host-coherent memory. */
static int vk_create_rebar_buffer(vk_buffer* b, VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                               .size = size, .usage = usage,
                               .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    VK_CHECK(pvkCreateBuffer(s_vk.device, &bci, NULL, &b->buf), "vkCreateBuffer(rebar)");
    VkMemoryRequirements req;
    pvkGetBufferMemoryRequirements(s_vk.device, b->buf, &req);
    u32 type;
    const VkMemoryPropertyFlags coh = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags rebar = coh | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (vk_find_memory_type(req.memoryTypeBits, rebar, &type) == 0) b->coherent = 1;
    else if (vk_find_memory_type(req.memoryTypeBits, coh, &type) == 0) b->coherent = 1;
    else { pvkDestroyBuffer(s_vk.device, b->buf, NULL); b->buf = VK_NULL_HANDLE; return -1; }
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = req.size, .memoryTypeIndex = type };
    if (pvkAllocateMemory(s_vk.device, &mai, NULL, &b->mem) != VK_SUCCESS ||
        pvkBindBufferMemory(s_vk.device, b->buf, b->mem, 0) != VK_SUCCESS ||
        pvkMapMemory(s_vk.device, b->mem, 0, VK_WHOLE_SIZE, 0, &b->ptr) != VK_SUCCESS) {
        vk_destroy_buffer(b);
        return -1;
    }
    b->size = size;
    return 0;
}

static void vk_destroy_buffer(vk_buffer* b)
{
    if (s_eb_pending || vk_eb_any_inflight()) vk_eng_flush();
    if (b->ptr) pvkUnmapMemory(s_vk.device, b->mem);
    if (b->buf) pvkDestroyBuffer(s_vk.device, b->buf, NULL);
    if (b->mem) pvkFreeMemory(s_vk.device, b->mem, NULL);
    memset(b, 0, sizeof *b);
}

/* Host writes to non-coherent memory need a flush; there is no
 * vkFlushMappedMemoryRanges loaded, so writes only ever go to coherent
 * buffers (every desktop/L4T driver exposes one). Reads invalidate. */
static void vk_invalidate(vk_buffer* b)
{
    if (b->coherent) return;
    VkMappedMemoryRange r = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                              .memory = b->mem, .offset = 0, .size = VK_WHOLE_SIZE };
    pvkInvalidateMappedMemoryRanges(s_vk.device, 1, &r);
}

/* ---------------------------------------------------------------------------
 * One-shot command submission
 *
 * The queue is shared with the present (see present_mu in vk_state), so every
 * operation on it goes through these.
 * -------------------------------------------------------------------------*/
static VkResult vk_queue_submit(const VkSubmitInfo* si, VkFence fence)
{
    pthread_mutex_lock(&s_vk.queue_mu);
    const VkResult r = pvkQueueSubmit(s_vk.queue, 1, si, fence);
    pthread_mutex_unlock(&s_vk.queue_mu);
    return r;
}

static VkResult vk_queue_present(const VkPresentInfoKHR* pi)
{
    pthread_mutex_lock(&s_vk.queue_mu);
    const VkResult r = pvkQueuePresentKHR(s_vk.queue, pi);
    pthread_mutex_unlock(&s_vk.queue_mu);
    return r;
}

static void vk_device_idle(void)
{
    if (!s_vk.device || !pvkDeviceWaitIdle) return;
    pthread_mutex_lock(&s_vk.queue_mu);
    pvkDeviceWaitIdle(s_vk.device);
    pthread_mutex_unlock(&s_vk.queue_mu);
}

/* A full execution and memory dependency on everything submitted before
 * this command buffer, the present's copy included. */
static void vk_cmd_after_everything(VkCommandBuffer cb)
{
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                           .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
    pvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

static int vk_begin(void)
{
    if (s_eb_pending || vk_eb_any_inflight()) vk_eng_flush();
    s_vk.cmd = s_vk.sync_cmd;
    VK_CHECK(pvkResetCommandBuffer(s_vk.cmd, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VK_CHECK(pvkBeginCommandBuffer(s_vk.cmd, &bi), "vkBeginCommandBuffer");
    /* A present's copy may still be reading the surface this frame is about
     * to draw over: order everything here after it. Set only by a present,
     * which never runs alongside the walker's recording. */
    if (s_vk.present_order) { vk_cmd_after_everything(s_vk.cmd); s_vk.present_order = 0; }
    return 0;
}

static int vk_submit_and_wait(void)
{
    VK_CHECK(pvkEndCommandBuffer(s_vk.cmd), "vkEndCommandBuffer");
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1, .pCommandBuffers = &s_vk.cmd };
    VK_CHECK(pvkResetFences(s_vk.device, 1, &s_vk.fence), "vkResetFences");
    VK_CHECK(vk_queue_submit(&si, s_vk.fence), "vkQueueSubmit");
    /* The GPU half of [ENGTIME]: how long the walker sits on the fence. Only
     * timed when the knob is on, so a normal run pays one cached int test. */
    if (!s_eng_stats) {
        VK_CHECK(pvkWaitForFences(s_vk.device, 1, &s_vk.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        return 0;
    }
    const double wait0 = vk_now_ms();
    VK_CHECK(pvkWaitForFences(s_vk.device, 1, &s_vk.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    const double dw = vk_now_ms() - wait0;
    s_engtime.gpu_wait_ms += dw;
    if (g_rsx_sf.enabled) g_rsx_sf.vk_flush_wait_ms += dw;
    s_engtime.submits++;
    return 0;
}

/* Order this command buffer's use of an image after everything submitted
 * before it; from UNDEFINED it is also the one-time move to GENERAL. */
static void vk_barrier_image_on(VkCommandBuffer cb, VkImage img, VkImageAspectFlags aspect,
                                VkImageLayout old_layout)
{
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .oldLayout = old_layout, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img,
        .subresourceRange = { aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS },
    };
    pvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
}

static void vk_barrier_image(VkImage img, VkImageAspectFlags aspect, VkImageLayout old_layout)
{
    vk_barrier_image_on(s_vk.cmd, img, aspect, old_layout);
}

/* ---------------------------------------------------------------------------
 * Texture unit 0
 * -------------------------------------------------------------------------*/
static void vk_write_texture_descriptor(void)
{
    VkDescriptorImageInfo ii = { .sampler = s_vk.sampler,
                                 .imageView = s_vk.tex_ready[0] ? s_vk.tex[0].view
                                                                : s_vk.dummy.view,
                                 .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet w = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                               .dstSet = s_vk.desc_set, .dstBinding = 0,
                               .descriptorCount = 1,
                               .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                               .pImageInfo = &ii };
    pvkUpdateDescriptorSets(s_vk.device, 1, &w, 0, NULL);
}

/* Replace *dst with an RGBA8 image of `nlv` levels: level m is lw[m] x lh[m]
 * texels, tightly packed, the levels one after another in `rgba`. */
static int vk_upload_texture_levels(vk_image* dst, const u8* rgba,
                                    const u32* lw, const u32* lh, u32 nlv)
{
    VkBufferImageCopy regions[RSX_MAX_TEXTURE_LEVELS];
    if (!nlv || nlv > RSX_MAX_TEXTURE_LEVELS) return -1;
    VkDeviceSize total = 0;
    for (u32 m = 0; m < nlv; m++) {
        regions[m] = (VkBufferImageCopy){
            .bufferOffset = total,
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1 },
            .imageExtent = { lw[m], lh[m], 1 },
        };
        total += (VkDeviceSize)lw[m] * lh[m] * 4u;
    }
    vk_buffer staging = {0};
    if (vk_create_host_buffer(&staging, total, VK_BUFFER_USAGE_TRANSFER_SRC_BIT) ||
        !staging.coherent) {
        vk_destroy_buffer(&staging); return -1;
    }
    memcpy(staging.ptr, rgba, (size_t)total);

    vk_image fresh = {0};
    if (vk_create_image_levels(&fresh, VK_FORMAT_R8G8B8A8_UNORM, lw[0], lh[0], nlv,
                               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT)) {
        vk_destroy_image(&fresh); vk_destroy_buffer(&staging); return -1;
    }
    int rc = vk_begin();
    if (!rc) {
        vk_barrier_image(fresh.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
        pvkCmdCopyBufferToImage(s_vk.cmd, staging.buf, fresh.img, VK_IMAGE_LAYOUT_GENERAL,
                                nlv, regions);
        rc = vk_submit_and_wait();
    }
    vk_destroy_buffer(&staging);
    if (rc) { vk_destroy_image(&fresh); return -1; }

    vk_destroy_image(dst);          /* idle: every submission is waited on */
    *dst = fresh;
    return 0;
}

static int vk_upload_texture(vk_image* dst, const u8* rgba, u32 w, u32 h)
{
    return vk_upload_texture_levels(dst, rgba, &w, &h, 1);
}

/* NV4097_SET_TEXTURE_ADDRESS wrap field, one per axis -- the table the Metal
 * backend and the live draw engine use: 1 WRAP, 2 MIRROR, 3 CLAMP_TO_EDGE,
 * 4 BORDER, 5 CLAMP, 6..8 MIRROR_ONCE. Vulkan 1.0 has MIRROR_CLAMP_TO_EDGE
 * only behind an extension, so the MIRROR_ONCE family clamps to edge here --
 * exact for texcoords in [0,1], the mirrored half is what is lost. */
static VkSamplerAddressMode vk_gcm_wrap(u32 w)
{
    switch (w & 0xFu) {
    case 1:  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case 2:  return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case 4:  return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;      /* 3, 5..8, unset */
    }
}

/* A unit's sampler from its registers, decoded as the Metal backend's
 * samp_slot_for does: SET_TEXTURE_FILTER min at [18:16] (1 NEAREST,
 * 2 LINEAR, 3..6 the four nearest/linear x nearest/linear-mip pairs), mag at
 * [26:24]; SET_TEXTURE_CONTROL0 max LOD at [18:7], min LOD at [30:19], 4.8
 * fixed point. The LOD bias (FILTER [12:0]) is not applied, as in Metal.
 * VK_NULL_HANDLE on failure: the caller falls back to `sampler`. */
static VkSampler vk_unit_sampler(const rsx_texture_state* t)
{
    const u32 minf = (t->filter >> 16) & 7u, magf = (t->filter >> 24) & 7u;
    const u32 lod  = (t->control0 >> 7) & 0xFFFFFFu;       /* max then min LOD */
    const u64 key  = (u64)minf | ((u64)magf << 3)
                   | ((u64)(t->address & 0x000F0F0Fu) << 6) | ((u64)lod << 26);
    for (u32 i = 0; i < s_vk.samp_count; i++)
        if (s_vk.samp_cache[i].key == key) return s_vk.samp_cache[i].s;
    if (s_vk.samp_count >= VK_SAMPLER_CACHE) return VK_NULL_HANDLE;

    const int mip_present = (minf >= 3);
    const int mip_linear  = (minf == 5 || minf == 6);
    /* A min filter with no mip term samples level 0 only: pin the range. */
    const float min_lod = (float)((t->control0 >> 19) & 0xFFFu) / 256.0f;
    float max_lod = mip_present ? (float)((t->control0 >> 7) & 0xFFFu) / 256.0f : 0.0f;
    if (max_lod < min_lod) max_lod = min_lod;
    VkSamplerCreateInfo sci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = (magf == 2) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .minFilter = (minf == 2 || minf == 4 || minf == 6) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .mipmapMode = mip_linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = vk_gcm_wrap(t->address),
        .addressModeV = vk_gcm_wrap(t->address >> 8),
        .addressModeW = vk_gcm_wrap(t->address >> 16),
        .minLod = min_lod, .maxLod = max_lod,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
    };
    VkSampler smp = VK_NULL_HANDLE;
    if (pvkCreateSampler(s_vk.device, &sci, NULL, &smp) != VK_SUCCESS) return VK_NULL_HANDLE;
    s_vk.samp_cache[s_vk.samp_count].key = key;
    s_vk.samp_cache[s_vk.samp_count].s   = smp;
    s_vk.samp_count++;
    return smp;
}

/* One level of a guest texture into tightly packed RGBA8, with the null
 * backend's decode (rsx_null_backend.c, nullsw_bind_texture): formats that
 * decode to fewer than four bytes are splayed, alpha forced opaque. */
static int vk_decode_level_rgba8(u8* dst, const u8* src, u32 w, u32 h,
                                 const rsx_tex_layout* tl)
{
    const u32 pitch = w * 4u;
    if (tl->fmt == RSX_TEXFMT_R8G8B8A8) {
        rsx_texture_decode(dst, pitch, src, w, h, tl, rsx_texture_argb_is_rgba());
        return 0;
    }
    const u32 srcp = tl->row_bytes;
    u8* tmp = (u8*)malloc((size_t)srcp * h);
    if (!tmp) return -1;
    rsx_texture_decode(tmp, srcp, src, w, h, tl, 0);
    for (u32 y = 0; y < h; y++)
        for (u32 x = 0; x < w; x++) {
            const u8* sp = tmp + (size_t)y * srcp + (size_t)x * tl->bytes_per_texel;
            u8* dp = dst + (size_t)y * pitch + (size_t)x * 4u;
            dp[0] = sp[0];
            dp[1] = tl->bytes_per_texel > 1 ? sp[1] : sp[0];
            dp[2] = tl->bytes_per_texel > 2 ? sp[2] : sp[0];
            dp[3] = 255;
        }
    free(tmp);
    return 0;
}

/* Unit `unit` has nothing usable bound. The fallback path's descriptor only
 * ever shows unit 0, so only that one needs rewriting. */
static void vk_tex_off(u32 unit)
{
    s_vk.tex_ready[unit] = 0;
    s_vk.tex_sampler[unit] = VK_NULL_HANDLE;
    if (unit == 0) vk_write_texture_descriptor();
}

/* Same decode as the headless null backend (rsx_null_backend.c,
 * nullsw_bind_texture), so both backends sample identical texels. */
static void vk_cb_bind_texture(void* ud, u32 unit, const rsx_texture_state* t)
{
    (void)ud;
    if (unit >= RSX_MAX_TEXTURES || !t || !s_vk.device) return;
    if (!(t->control0 & 0x80000000u) || !t->offset) { vk_tex_off(unit); return; }

    u32 w = (t->image_rect >> 16) & 0xFFFFu;
    u32 h =  t->image_rect        & 0xFFFFu;
    if (!w || !h || w > 4096u || h > 4096u) { vk_tex_off(unit); return; }

    u32 ea = cellGcmResolveLocated((t->format & 3u) == 1u, t->offset);
    if (!vm_base || ea == 0xFFFFFFFFu) { vk_tex_off(unit); return; }
    const u32 fmt = (t->format >> 8) & 0xFFu;

    /* The whole mip chain, as the Metal backend reads it: SET_TEXTURE_FORMAT's
     * level count above bit 15 and SET_TEXTURE_CONTROL3's row pitch, laid out
     * by the shared rsx_texture_mip_chain() (which clamps the count to what
     * the dimensions allow). Cube maps are not handled yet: face 0 only. */
    const u32 levels    = (t->format >> 16) & 0xFFFFu;
    const u32 row_pitch = t->control3 & 0xFFFFFu;
    rsx_tex_level lv[RSX_MAX_TEXTURE_LEVELS];
    const u32 nlv = rsx_texture_mip_chain(fmt, w, h, levels, row_pitch, lv);
    if (!nlv || lv[0].tl.compressed) { vk_tex_off(unit); return; }   /* no BC path yet */

    size_t total = 0, off[RSX_MAX_TEXTURE_LEVELS];
    u32 lw[RSX_MAX_TEXTURE_LEVELS], lh[RSX_MAX_TEXTURE_LEVELS];
    for (u32 m = 0; m < nlv; m++) {
        off[m] = total;
        lw[m] = lv[m].w; lh[m] = lv[m].h;
        total += (size_t)lv[m].w * lv[m].h * 4u;
    }
    u8* buf = (u8*)malloc(total);
    if (!buf) { vk_tex_off(unit); return; }
    for (u32 m = 0; m < nlv; m++)
        if (vk_decode_level_rgba8(buf + off[m], vm_base + ea + lv[m].offset,
                                  lv[m].w, lv[m].h, &lv[m].tl)) {
            free(buf); vk_tex_off(unit); return;
        }
    s_vk.tex_ready[unit] = vk_upload_texture_levels(&s_vk.tex[unit], buf, lw, lh, nlv) == 0;
    free(buf);
    s_vk.tex_sampler[unit] = s_vk.tex_ready[unit] ? vk_unit_sampler(t) : VK_NULL_HANDLE;
    if (unit == 0) vk_write_texture_descriptor();
}

/* ---------------------------------------------------------------------------
 * Render pass, framebuffer, pipelines
 * -------------------------------------------------------------------------*/
static int vk_pick_depth_format(void)
{
    /* With stencil first -- titles use it for shadows, reflections and UI
     * masks -- and D32 first within each group: a D32 depth aspect copies out
     * as 32-bit floats, which is what depth snapshots are built from. */
    static const VkFormat candidates[] = {
        VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D32_SFLOAT, VK_FORMAT_X8_D24_UNORM_PACK32, VK_FORMAT_D16_UNORM
    };
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    for (size_t i = 0; i < sizeof candidates / sizeof candidates[0]; i++) {
        VkFormatProperties fp;
        pvkGetPhysicalDeviceFormatProperties(s_vk.phys, candidates[i], &fp);
        if ((fp.optimalTilingFeatures & need) == need) {
            s_vk.depth_format = candidates[i]; goto chosen;
        }
    }
    /* TRANSFER_DST is a 1.1 format feature bit; 1.0 drivers may not report
     * it although clears work. Retry on the attachment bit alone. */
    for (size_t i = 0; i < sizeof candidates / sizeof candidates[0]; i++) {
        VkFormatProperties fp;
        pvkGetPhysicalDeviceFormatProperties(s_vk.phys, candidates[i], &fp);
        if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            s_vk.depth_format = candidates[i]; goto chosen;
        }
    }
    VK_LOG("no usable depth format\n");
    return -1;
chosen:
    s_vk.depth_aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (s_vk.depth_format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
        s_vk.depth_format == VK_FORMAT_D24_UNORM_S8_UINT)
        s_vk.depth_aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    return 0;
}

static int vk_create_render_pass(void)
{
    VkAttachmentDescription att[2] = {
        { .format = VK_FORMAT_R8G8B8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
          .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
          .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
          .initialLayout = VK_IMAGE_LAYOUT_GENERAL, .finalLayout = VK_IMAGE_LAYOUT_GENERAL },
        { .format = s_vk.depth_format, .samples = VK_SAMPLE_COUNT_1_BIT,
          .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD,     /* ignored without stencil */
          .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
          .initialLayout = VK_IMAGE_LAYOUT_GENERAL, .finalLayout = VK_IMAGE_LAYOUT_GENERAL },
    };
    VkAttachmentReference cref = { 0, VK_IMAGE_LAYOUT_GENERAL };
    VkAttachmentReference dref = { 1, VK_IMAGE_LAYOUT_GENERAL };
    VkSubpassDescription sub = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 .colorAttachmentCount = 1, .pColorAttachments = &cref,
                                 .pDepthStencilAttachment = &dref };
    VkRenderPassCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                                   .attachmentCount = 2, .pAttachments = att,
                                   .subpassCount = 1, .pSubpasses = &sub };
    VK_CHECK(pvkCreateRenderPass(s_vk.device, &rci, NULL, &s_vk.render_pass), "vkCreateRenderPass");

    VkImageView views[2] = { s_vk.color.view, s_vk.depth.view };
    VkFramebufferCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                    .renderPass = s_vk.render_pass, .attachmentCount = 2,
                                    .pAttachments = views, .width = s_vk.width,
                                    .height = s_vk.height, .layers = 1 };
    VK_CHECK(pvkCreateFramebuffer(s_vk.device, &fci, NULL, &s_vk.framebuffer), "vkCreateFramebuffer");
    return 0;
}

static int vk_create_shader(const uint32_t* code, size_t bytes, VkShaderModule* out)
{
    VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                    .codeSize = bytes, .pCode = code };
    VK_CHECK(pvkCreateShaderModule(s_vk.device, &ci, NULL, out), "vkCreateShaderModule");
    return 0;
}

static int vk_create_pipeline_objects(void)
{
    if (vk_create_shader(rsx_vk_fallback_vert_spv, sizeof rsx_vk_fallback_vert_spv, &s_vk.vs) ||
        vk_create_shader(rsx_vk_fallback_frag_spv, sizeof rsx_vk_fallback_frag_spv, &s_vk.fs))
        return -1;

    VkDescriptorSetLayoutBinding bind = {
        .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
    VkDescriptorSetLayoutCreateInfo lci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &bind };
    VK_CHECK(pvkCreateDescriptorSetLayout(s_vk.device, &lci, NULL, &s_vk.set_layout),
             "vkCreateDescriptorSetLayout");

    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
    VK_CHECK(pvkCreateDescriptorPool(s_vk.device, &pci, NULL, &s_vk.desc_pool), "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = s_vk.desc_pool,
                                        .descriptorSetCount = 1, .pSetLayouts = &s_vk.set_layout };
    VK_CHECK(pvkAllocateDescriptorSets(s_vk.device, &dai, &s_vk.desc_set), "vkAllocateDescriptorSets");

    VkPushConstantRange pcr = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int32_t) };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1, .pSetLayouts = &s_vk.set_layout,
                                        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    VK_CHECK(pvkCreatePipelineLayout(s_vk.device, &plci, NULL, &s_vk.pipe_layout),
             "vkCreatePipelineLayout");

    /* Nearest + repeat: the null backend's point sample with wrapping. */
    VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
                                .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                .maxLod = 0.0f };
    VK_CHECK(pvkCreateSampler(s_vk.device, &sci, NULL, &s_vk.sampler), "vkCreateSampler");

    /* A 1x1 magenta dummy keeps the descriptor valid before any bind --
     * the same "not bound" colour the null backend's sampler returns. */
    static const u8 magenta[4] = { 0xFF, 0x00, 0xFF, 0xFF };
    if (vk_upload_texture(&s_vk.dummy, magenta, 1, 1)) return -1;
    vk_write_texture_descriptor();
    return 0;
}

/* NV4097 depth function is the GL enum 0x200 NEVER .. 0x207 ALWAYS, which is
 * VkCompareOp 0..7 in the same order. Anything else passes, as in the null
 * and D3D12 backends (including the 1 rsx_state_init seeds). */
static u32 vk_compare_slot(u32 gl_func)
{
    return (gl_func >= 0x200u && gl_func <= 0x207u) ? gl_func - 0x200u
                                                    : (u32)VK_COMPARE_OP_ALWAYS;
}

static VkPipeline vk_get_pipeline(int ztest, int zwrite, u32 cmp)
{
    u32 slot = ((u32)ztest * 2u + (u32)zwrite) * 8u + cmp;
    if (s_vk.pipelines[slot]) return s_vk.pipelines[slot];

    VkPipelineShaderStageCreateInfo stages[2] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = s_vk.vs, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = s_vk.fs, .pName = "main" },
    };
    VkVertexInputBindingDescription vb = { 0, sizeof(vk_vertex), VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription va[3] = {
        { 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(vk_vertex, pos) },
        { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(vk_vertex, col) },
        { 2, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(vk_vertex, uv)  },
    };
    VkPipelineVertexInputStateCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vb,
        .vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = va };
    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkPipelineViewportStateCreateInfo vp = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1 };
    /* No culling: the null backend fills both windings. */
    VkPipelineRasterizationStateCreateInfo rs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkPipelineDepthStencilStateCreateInfo ds = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = ztest ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = zwrite ? VK_TRUE : VK_FALSE,
        .depthCompareOp = (VkCompareOp)cmp };
    VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xF };
    VkPipelineColorBlendStateCreateInfo cb = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &cba };
    VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2, .pDynamicStates = dyn };
    VkGraphicsPipelineCreateInfo gci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = stages, .pVertexInputState = &vi,
        .pInputAssemblyState = &ia, .pViewportState = &vp, .pRasterizationState = &rs,
        .pMultisampleState = &ms, .pDepthStencilState = &ds, .pColorBlendState = &cb,
        .pDynamicState = &dy, .layout = s_vk.pipe_layout, .renderPass = s_vk.render_pass,
        .subpass = 0 };
    VkPipeline p = VK_NULL_HANDLE;
    VkResult r = pvkCreateGraphicsPipelines(s_vk.device, VK_NULL_HANDLE, 1, &gci, NULL, &p);
    if (r != VK_SUCCESS) { VK_LOG("vkCreateGraphicsPipelines failed (%d)\n", (int)r); return VK_NULL_HANDLE; }
    s_vk.pipelines[slot] = p;
    return p;
}

/* ---------------------------------------------------------------------------
 * Draw path
 * -------------------------------------------------------------------------*/
static void vk_cb_track_state(void* ud, const rsx_state* state)
{
    (void)ud; s_vk.state = state;
}

static int vk_push_vertex(const float pos[4], const float col[4], const float uv[4])
{
    if (s_vk.cpu_count == s_vk.cpu_cap) {
        size_t cap = s_vk.cpu_cap ? s_vk.cpu_cap * 2 : 1024;
        vk_vertex* n = (vk_vertex*)realloc(s_vk.cpu_verts, cap * sizeof *n);
        if (!n) return -1;
        s_vk.cpu_verts = n; s_vk.cpu_cap = cap;
    }
    vk_vertex* v = &s_vk.cpu_verts[s_vk.cpu_count++];
    memcpy(v->pos, pos, sizeof v->pos);
    memcpy(v->col, col, sizeof v->col);
    v->uv[0] = uv[0]; v->uv[1] = uv[1];
    return 0;
}

/* One triangle, colour from its first vertex (the null backend's flat shade). */
static void vk_emit_tri(const rsx_state* st, u32 i0, u32 i1, u32 i2)
{
    float p[3][4], uv[3][4], col[4];
    rsx_fetch_attrib(st, 0, i0, p[0]);
    rsx_fetch_attrib(st, 0, i1, p[1]);
    rsx_fetch_attrib(st, 0, i2, p[2]);
    rsx_fetch_attrib(st, 3, i0, col);
    rsx_fetch_attrib(st, 8, i0, uv[0]);
    rsx_fetch_attrib(st, 8, i1, uv[1]);
    rsx_fetch_attrib(st, 8, i2, uv[2]);
    for (int k = 0; k < 3; k++) vk_push_vertex(p[k], col, uv[k]);
}

/* Called once per expanded triangle: the fixed-function path records the
 * null backend's flat-shaded vertex, the guest path all sixteen attributes. */
typedef void (*vk_tri_fn)(const rsx_state* st, u32 i0, u32 i1, u32 i2);

/* Primitive expansion, identical to the null backend's draw_prim(). */
static void vk_expand(const rsx_state* st, u32 prim, u32 first, u32 count, vk_tri_fn emit)
{
    switch (prim) {
    case RSX_PRIMITIVE_TRIANGLES:
        for (u32 i = 0; i + 2 < count; i += 3) emit(st, first + i, first + i + 1, first + i + 2);
        break;
    case RSX_PRIMITIVE_TRIANGLE_STRIP:
        for (u32 i = 0; i + 2 < count; i++)
            emit(st, first + i, first + i + 1 + (i & 1), first + i + 2 - (i & 1));
        break;
    case RSX_PRIMITIVE_TRIANGLE_FAN:
    case RSX_PRIMITIVE_POLYGON:
        for (u32 i = 1; i + 1 < count; i++) emit(st, first, first + i, first + i + 1);
        break;
    case RSX_PRIMITIVE_QUADS:
        for (u32 i = 0; i + 3 < count; i += 4) {
            emit(st, first + i, first + i + 1, first + i + 2);
            emit(st, first + i, first + i + 2, first + i + 3);
        }
        break;
    case RSX_PRIMITIVE_QUAD_STRIP:
        for (u32 i = 0; i + 3 < count; i += 2) {
            emit(st, first + i, first + i + 1, first + i + 3);
            emit(st, first + i, first + i + 3, first + i + 2);
        }
        break;
    default:
        break;   /* points and lines: no coverage, as in the null backend */
    }
}

static int vk_ensure_vertex_buffer(VkDeviceSize bytes)
{
    if (s_vk.vertices.buf && s_vk.vertices.size >= bytes) return 0;
    VkDeviceSize size = s_vk.vertices.size ? s_vk.vertices.size : 64 * 1024;
    while (size < bytes) size *= 2;
    vk_destroy_buffer(&s_vk.vertices);   /* idle: every submission is waited on */
    if (vk_create_host_buffer(&s_vk.vertices, size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) return -1;
    if (!s_vk.vertices.coherent) { VK_LOG("vertex memory is not host-coherent\n"); return -1; }
    return 0;
}

/* ---------------------------------------------------------------------------
 * Guest programs (stage V3)
 *
 * With the HLSL -> SPIR-V translator built and PS3RECOMP_VK_GUEST_PROGRAMS=1,
 * a draw runs the guest's own vertex and fragment programs, as the Metal
 * backend's guest path does and keyed the same way: decompiled to HLSL by the
 * shared decompilers, translated by rsx_shader_spirv.cpp, cached per program.
 * Constants use the Metal/D3D12 layout: VPConst is the 512 transform
 * constants plus the viewport epilogue, PSConstants the fragment program's
 * inline constants plus fp_alpha.
 *
 * Vertices are still fetched and expanded on the CPU through the shared
 * rsx_fetch_attrib(), now all sixteen attributes (a disabled one reads its
 * constant register, as on the hardware). Every RSX vertex format therefore
 * arrives as float4, and the pipeline has one fixed input layout: location n
 * is attribute n, which is what the vertex decompiler declares.
 *
 * A program that needs a cube map or a vertex texture falls back to the
 * fixed-function path, with a message. Not honoured on this path yet: the RSX
 * viewport and scissor (the whole target is used, as on the fallback path),
 * blending, stencil, culling, per-unit sampler state.
 * -------------------------------------------------------------------------*/
#define VK_VP_CACHE      256
#define VK_FP_CACHE      512
#define VK_GPIPE_CACHE   512
#define VK_FP_MAX_BYTES  4096u              /* bound on a fragment program, as Metal/D3D12 */
#define VK_HLSL_BYTES    (256u * 1024u)
#define VK_SPV_WORDS     (256u * 1024u)
#define VK_VPCONST_BYTES ((RSX_MAX_VERTEX_CONSTANTS + 2u) * 16u)
#define VK_GUEST_ATTRS   16u
#define VK_GUEST_FLOATS  (VK_GUEST_ATTRS * 4u)   /* per vertex */

typedef struct { u32 hash; VkShaderModule mod; } vk_vp_entry;
typedef struct { u64 key; VkShaderModule mod; u32 nconst; } vk_fp_entry;
typedef struct { int vs, fs; u32 depth_key; VkPipeline pipe; } vk_gpipe_entry;

static struct {
    int                   on;
    char*                 hlsl;
    u32*                  spv;
    char                  log[4096];
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout      pipe_layout;
    VkDescriptorPool      pool;
    VkDescriptorSet       set;
    vk_buffer             ubo;              /* VPConst at 0, PSConstants at ps_off */
    VkDeviceSize          ps_off;
    VkDeviceSize          max_ubo_range;
    VkDeviceSize          max_ssbo_range;
    int                   ssbo_ok;
    VkDescriptorSetLayout ssbo_const_layout; /* set 0: binding 0 = STORAGE_BUFFER */
    VkDescriptorSetLayout ssbo_tex_layout;   /* set 1: binding 16 = textures[16], 48 = samplers[16] */
    VkPipelineLayout      ssbo_pipe_layout;  /* sets [0, 1] + 8B push constants */
    VkDescriptorPool      ssbo_pool;         /* unbatched const set + tex set */
    VkDescriptorSet       ssbo_const_set;
    VkDescriptorSet       ssbo_tex_set;
    vk_buffer             vertices;
    float*                cpu;
    size_t                cpu_cap, cpu_count;   /* in vertices */
    vk_vp_entry           vp[VK_VP_CACHE];     u32 vp_count;
    vk_fp_entry           fp[VK_FP_CACHE];     u32 fp_count;
    vk_gpipe_entry        pipes[VK_GPIPE_CACHE]; u32 pipe_count;
    rsx_fp_constant_block consts;
    u32                   draws, last_draws;
    u32                   warned;
} s_g;

int rsx_vulkan_backend_guest_programs(void)
{
    const char* v = getenv("PS3RECOMP_VK_GUEST_PROGRAMS");
    return rsx_hlsl_to_spirv_available() && v && v[0] && v[0] != '0';
}

static u32 vk_fnv1a32(const u8* p, u32 n)
{
    u32 h = 2166136261u;
    for (u32 i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

static u64 vk_fnv1a64(const void* d, size_t n, u64 h)
{
    const u8* p = (const u8*)d;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

/* Built from the registers, as the Metal and D3D12 backends do. */
static u32 vk_cube_mask(const rsx_state* st)
{
    u32 m = 0;
    for (u32 u = 0; u < RSX_MAX_TEXTURES; u++) {
        const rsx_texture_state* t = &st->textures[u];
        const u32 w = (t->image_rect >> 16) & 0xFFFFu, h = t->image_rect & 0xFFFFu;
        if ((t->control0 & 0x80000000u) && (t->format & 4u) && w && w == h) m |= 1u << u;
    }
    return m;
}

static u32 vk_vtex_mask(const rsx_state* st)
{
    u32 m = 0;
    for (u32 u = 0; u < RSX_MAX_VERTEX_TEXTURES; u++)
        if (st->vertex_textures[u].control0 & 0x80000000u) m |= 1u << u;
    return m;
}

static VkShaderModule vk_guest_module(const char* what, int stage)
{
    u32 n = 0;
    if (rsx_hlsl_to_spirv(s_g.hlsl, stage, s_g.spv, VK_SPV_WORDS, &n,
                          s_g.log, sizeof s_g.log) != 0) {
        VK_LOG("%s: %s\n", what, s_g.log);
        return VK_NULL_HANDLE;
    }
    VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                    .codeSize = (size_t)n * 4u, .pCode = s_g.spv };
    VkShaderModule m = VK_NULL_HANDLE;
    if (pvkCreateShaderModule(s_vk.device, &ci, NULL, &m) != VK_SUCCESS) {
        VK_LOG("%s: vkCreateShaderModule failed\n", what);
        return VK_NULL_HANDLE;
    }
    return m;
}

/* The vertex program, keyed on its own bytes to its end bit (as Metal). */
static int vk_vp_slot(const rsx_state* st)
{
    if (st->vp_ucode_bytes < 16) return -1;
    u32 vstart = st->transform_program_start * 16u;
    if (vstart >= st->vp_ucode_bytes) vstart = 0;
    const u8* uc = st->vp_ucode + vstart;
    const u32 avail = st->vp_ucode_bytes - vstart;
    const u32 instrs = rsx_vp_program_size_instrs(uc, avail);
    const u32 len = instrs ? instrs * 16u : avail;
    const u32 hash = vk_fnv1a32(uc, len);
    for (u32 i = 0; i < s_g.vp_count; i++)
        if (s_g.vp[i].hash == hash) return s_g.vp[i].mod ? (int)i : -1;
    if (s_g.vp_count >= VK_VP_CACHE) return -1;

    char what[64];
    snprintf(what, sizeof what, "vertex program %08X", hash);
    VkShaderModule m = VK_NULL_HANDLE;
    const int ni = rsx_vp_decompile_ex(uc, len, 0, s_g.hlsl, VK_HLSL_BYTES);
    if (ni <= 0) VK_LOG("%s: decompile failed (%d)\n", what, ni);
    else         m = vk_guest_module(what, RSX_SHADER_STAGE_VERTEX);
    if (s_g.vp_count < 32)
        VK_LOG("%s: %d instrs -> %s\n", what, ni, m ? "ok" : "FAILED (fixed-function instead)");
    const int slot = (int)s_g.vp_count++;
    s_g.vp[slot].hash = hash;
    s_g.vp[slot].mod  = m;
    return m ? slot : -1;
}

/* The fragment program, resolved and keyed as the Metal backend's
 * fp_slot_for: structure (inline constants live in PSConstants), export
 * width, and the alpha test patched into the source. No cube units here --
 * the caller has already sent those to the fixed-function path. */
static int vk_fp_slot(const rsx_state* st, const u8** uc)
{
    if (!vm_base || st->shader_program == 0) return -1;
    const u32 off = cellGcmResolveLocated((st->shader_program & 3u) == 1u,
                                          st->shader_program & ~3u);
    if (off == 0xFFFFFFFFu) return -1;
    *uc = vm_base + off;
    u64 key = rsx_fp_structural_hash(*uc, VK_FP_MAX_BYTES, 1469598103934665603ull);
    if (!key) return -1;                       /* malformed or unterminated */
    const u32 ctrl     = st->shader_control;
    const u32 ctrl_key = ctrl & CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS;
    const u32 alpha_en = (st->alpha_test_enable && st->alpha_func != 0x0207u) ? 1u : 0u;
    const u32 alpha_fn = alpha_en ? st->alpha_func : 0u;
    key = vk_fnv1a64(&ctrl_key, sizeof ctrl_key, key);
    key = vk_fnv1a64(&alpha_en, sizeof alpha_en, key);
    key = vk_fnv1a64(&alpha_fn, sizeof alpha_fn, key);
    for (u32 i = 0; i < s_g.fp_count; i++)
        if (s_g.fp[i].key == key) return s_g.fp[i].mod ? (int)i : -1;
    if (s_g.fp_count >= VK_FP_CACHE) return -1;

    char what[64];
    snprintf(what, sizeof what, "fragment program %016llX", (unsigned long long)key);
    VkShaderModule m = VK_NULL_HANDLE;
    u32 nconst = 0;
    int ni = rsx_fp_decompile_buffered_ex(*uc, VK_FP_MAX_BYTES, ctrl, 0,
                                          s_g.hlsl, VK_HLSL_BYTES, &nconst);
    if (ni > 0 && alpha_en &&
        rsx_fp_apply_alpha_test_buffered(s_g.hlsl, VK_HLSL_BYTES, st->alpha_func) < 0)
        ni = -1;
    if (ni <= 0) VK_LOG("%s: decompile failed (%d)\n", what, ni);
    else         m = vk_guest_module(what, RSX_SHADER_STAGE_FRAGMENT);
    if (s_g.fp_count < 32)
        VK_LOG("%s: %d instrs, %u constants -> %s\n", what, ni, nconst,
               m ? "ok" : "FAILED (fixed-function instead)");
    const int slot = (int)s_g.fp_count++;
    s_g.fp[slot].key    = key;
    s_g.fp[slot].mod    = m;
    s_g.fp[slot].nconst = nconst;
    return m ? slot : -1;
}

static VkPipeline vk_guest_pipeline(int vs, int fs, int ztest, int zwrite, u32 cmp)
{
    const u32 dkey = (u32)ztest | ((u32)zwrite << 1) | (cmp << 2);
    for (u32 i = 0; i < s_g.pipe_count; i++)
        if (s_g.pipes[i].vs == vs && s_g.pipes[i].fs == fs && s_g.pipes[i].depth_key == dkey)
            return s_g.pipes[i].pipe;
    if (s_g.pipe_count >= VK_GPIPE_CACHE) return VK_NULL_HANDLE;

    VkPipelineShaderStageCreateInfo stages[2] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = s_g.vp[vs].mod, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = s_g.fp[fs].mod, .pName = "main" },
    };
    VkVertexInputBindingDescription vb = { 0, VK_GUEST_FLOATS * 4u, VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription va[VK_GUEST_ATTRS];
    for (u32 a = 0; a < VK_GUEST_ATTRS; a++) {
        va[a].location = a;
        va[a].binding  = 0;
        va[a].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        va[a].offset   = a * 16u;
    }
    VkPipelineVertexInputStateCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vb,
        .vertexAttributeDescriptionCount = VK_GUEST_ATTRS, .pVertexAttributeDescriptions = va };
    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkPipelineViewportStateCreateInfo vp = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1 };
    VkPipelineRasterizationStateCreateInfo rs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkPipelineDepthStencilStateCreateInfo ds = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = ztest ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = zwrite ? VK_TRUE : VK_FALSE,
        .depthCompareOp = (VkCompareOp)cmp };
    VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xF };
    VkPipelineColorBlendStateCreateInfo cb = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &cba };
    VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2, .pDynamicStates = dyn };
    VkGraphicsPipelineCreateInfo gci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = stages, .pVertexInputState = &vi,
        .pInputAssemblyState = &ia, .pViewportState = &vp, .pRasterizationState = &rs,
        .pMultisampleState = &ms, .pDepthStencilState = &ds, .pColorBlendState = &cb,
        .pDynamicState = &dy, .layout = s_g.pipe_layout, .renderPass = s_vk.render_pass,
        .subpass = 0 };
    VkPipeline p = VK_NULL_HANDLE;
    VkResult r = pvkCreateGraphicsPipelines(s_vk.device, VK_NULL_HANDLE, 1, &gci, NULL, &p);
    if (r != VK_SUCCESS) {
        VK_LOG("guest pipeline: vkCreateGraphicsPipelines failed (%d)\n", (int)r);
        return VK_NULL_HANDLE;
    }
    vk_gpipe_entry* e = &s_g.pipes[s_g.pipe_count++];
    e->vs = vs; e->fs = fs; e->depth_key = dkey; e->pipe = p;
    return p;
}

/* One vertex: all sixteen attributes as float4, attribute n at float 4n. */
static int vk_guest_push(const rsx_state* st, u32 idx)
{
    if (s_g.cpu_count == s_g.cpu_cap) {
        size_t cap = s_g.cpu_cap ? s_g.cpu_cap * 2 : 1024;
        float* n = (float*)realloc(s_g.cpu, cap * VK_GUEST_FLOATS * sizeof(float));
        if (!n) return -1;
        s_g.cpu = n; s_g.cpu_cap = cap;
    }
    float* v = s_g.cpu + s_g.cpu_count * VK_GUEST_FLOATS;
    for (u32 a = 0; a < VK_GUEST_ATTRS; a++) rsx_fetch_attrib(st, (int)a, idx, v + a * 4u);
    s_g.cpu_count++;
    return 0;
}

static void vk_guest_tri(const rsx_state* st, u32 i0, u32 i1, u32 i2)
{
    vk_guest_push(st, i0);
    vk_guest_push(st, i1);
    vk_guest_push(st, i2);
}

static int vk_guest_init(void)
{
    if (!rsx_vulkan_backend_guest_programs()) return 0;
    s_g.hlsl = (char*)malloc(VK_HLSL_BYTES);
    s_g.spv  = (u32*)malloc(VK_SPV_WORDS * sizeof(u32));
    if (!s_g.hlsl || !s_g.spv) { VK_LOG("guest programs: out of memory\n"); return -1; }

    VkPhysicalDeviceProperties p;
    pvkGetPhysicalDeviceProperties(s_vk.phys, &p);
    s_g.max_ubo_range = p.limits.maxUniformBufferRange;
    s_g.max_ssbo_range = p.limits.maxStorageBufferRange;
    VkDeviceSize align = p.limits.minUniformBufferOffsetAlignment;
    if (align == 0) align = 256;
    if (align < 16u) align = 16u;
    s_g.ps_off = (VK_VPCONST_BYTES + align - 1u) / align * align;
    if (VK_VPCONST_BYTES > s_g.max_ubo_range) {
        VK_LOG("guest programs: maxUniformBufferRange %llu < VPConst\n",
               (unsigned long long)s_g.max_ubo_range);
        return -1;
    }

    const VkShaderStageFlags both = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    /* The two constant blocks are DYNAMIC: a batch gives every draw its own
     * slice of one ring, and the offset moves at bind time. That is what lets
     * a batched draw reuse a descriptor set keyed on its textures alone
     * instead of rewriting nineteen descriptors per draw. */
    VkDescriptorSetLayoutBinding b[4] = {
        { RSX_SPIRV_VPCONST_BINDING, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, both, NULL },
        { RSX_SPIRV_PSCONST_BINDING, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, both, NULL },
        { RSX_SPIRV_TEXTURE_BINDING, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, RSX_MAX_TEXTURES,
          VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { RSX_SPIRV_SAMPLER_BINDING, VK_DESCRIPTOR_TYPE_SAMPLER, RSX_MAX_TEXTURES,
          VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
    };
    VkDescriptorSetLayoutCreateInfo lci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4, .pBindings = b };
    VK_CHECK(pvkCreateDescriptorSetLayout(s_vk.device, &lci, NULL, &s_g.set_layout),
             "vkCreateDescriptorSetLayout(guest)");
    VkDescriptorPoolSize ps[3] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, RSX_MAX_TEXTURES },
        { VK_DESCRIPTOR_TYPE_SAMPLER, RSX_MAX_TEXTURES },
    };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = 1, .poolSizeCount = 3, .pPoolSizes = ps };
    VK_CHECK(pvkCreateDescriptorPool(s_vk.device, &pci, NULL, &s_g.pool),
             "vkCreateDescriptorPool(guest)");
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = s_g.pool, .descriptorSetCount = 1,
                                        .pSetLayouts = &s_g.set_layout };
    VK_CHECK(pvkAllocateDescriptorSets(s_vk.device, &dai, &s_g.set), "vkAllocateDescriptorSets(guest)");
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1, .pSetLayouts = &s_g.set_layout };
    VK_CHECK(pvkCreatePipelineLayout(s_vk.device, &plci, NULL, &s_g.pipe_layout),
             "vkCreatePipelineLayout(guest)");
    if (vk_create_host_buffer(&s_g.ubo,
                              s_g.ps_off + (RSX_FP_MAX_INLINE_CONSTANTS + 1u) * 16u,
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) || !s_g.ubo.coherent) {
        VK_LOG("guest programs: no host-coherent uniform memory\n");
        return -1;
    }
    /* SSBO constant ring layouts: set 0 holds one persistently bound STORAGE_BUFFER
     * covering the constant ring; set 1 holds textures + samplers; 8B push constants
     * carry (vp_off, ps_off) in float4 units. */
    if (p.limits.maxPushConstantsSize >= 8u && s_g.max_ssbo_range >= s_g.ubo.size) {
        VkDescriptorSetLayoutBinding sb0 = {
            RSX_SPIRV_SSBO_BINDING, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, both, NULL
        };
        VkDescriptorSetLayoutCreateInfo slci0 = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1, .pBindings = &sb0
        };
        VkDescriptorSetLayoutBinding sb1[2] = {
            { RSX_SPIRV_TEXTURE_BINDING, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, RSX_MAX_TEXTURES,
              VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
            { RSX_SPIRV_SAMPLER_BINDING, VK_DESCRIPTOR_TYPE_SAMPLER, RSX_MAX_TEXTURES,
              VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        };
        VkDescriptorSetLayoutCreateInfo slci1 = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 2, .pBindings = sb1
        };
        VkDescriptorPoolSize sps[3] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, RSX_MAX_TEXTURES },
            { VK_DESCRIPTOR_TYPE_SAMPLER, RSX_MAX_TEXTURES },
        };
        VkDescriptorPoolCreateInfo spci = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 2, .poolSizeCount = 3, .pPoolSizes = sps
        };
        if (pvkCreateDescriptorSetLayout(s_vk.device, &slci0, NULL, &s_g.ssbo_const_layout) == VK_SUCCESS &&
            pvkCreateDescriptorSetLayout(s_vk.device, &slci1, NULL, &s_g.ssbo_tex_layout) == VK_SUCCESS &&
            pvkCreateDescriptorPool(s_vk.device, &spci, NULL, &s_g.ssbo_pool) == VK_SUCCESS) {
            VkDescriptorSetLayout sets[2] = { s_g.ssbo_const_layout, s_g.ssbo_tex_layout };
            VkPushConstantRange pcr = { both, 0u, 8u };
            VkPipelineLayoutCreateInfo splci = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 2, .pSetLayouts = sets,
                .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr
            };
            VkDescriptorSetAllocateInfo sdai0 = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = s_g.ssbo_pool, .descriptorSetCount = 1,
                .pSetLayouts = &s_g.ssbo_const_layout
            };
            VkDescriptorSetAllocateInfo sdai1 = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = s_g.ssbo_pool, .descriptorSetCount = 1,
                .pSetLayouts = &s_g.ssbo_tex_layout
            };
            if (pvkCreatePipelineLayout(s_vk.device, &splci, NULL, &s_g.ssbo_pipe_layout) == VK_SUCCESS &&
                pvkAllocateDescriptorSets(s_vk.device, &sdai0, &s_g.ssbo_const_set) == VK_SUCCESS &&
                pvkAllocateDescriptorSets(s_vk.device, &sdai1, &s_g.ssbo_tex_set) == VK_SUCCESS) {
                VkDescriptorBufferInfo sbi = { s_g.ubo.buf, 0, s_g.ubo.size };
                VkWriteDescriptorSet sw = {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = s_g.ssbo_const_set,
                    .dstBinding = RSX_SPIRV_SSBO_BINDING,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &sbi
                };
                pvkUpdateDescriptorSets(s_vk.device, 1, &sw, 0, NULL);
                s_g.ssbo_ok = 1;
            }
        }
    }
    s_g.on = 1;
    VK_LOG("guest programs: on (HLSL -> SPIR-V through glslang, ssbo=%d)\n", s_g.ssbo_ok);
    return 0;
}

static void vk_guest_shutdown(void)
{
    if (s_vk.device) {
        for (u32 i = 0; i < s_g.pipe_count; i++)
            if (s_g.pipes[i].pipe) pvkDestroyPipeline(s_vk.device, s_g.pipes[i].pipe, NULL);
        for (u32 i = 0; i < s_g.vp_count; i++)
            if (s_g.vp[i].mod) pvkDestroyShaderModule(s_vk.device, s_g.vp[i].mod, NULL);
        for (u32 i = 0; i < s_g.fp_count; i++)
            if (s_g.fp[i].mod) pvkDestroyShaderModule(s_vk.device, s_g.fp[i].mod, NULL);
        if (s_g.ssbo_pipe_layout)  pvkDestroyPipelineLayout(s_vk.device, s_g.ssbo_pipe_layout, NULL);
        if (s_g.ssbo_pool)         pvkDestroyDescriptorPool(s_vk.device, s_g.ssbo_pool, NULL);
        if (s_g.ssbo_tex_layout)   pvkDestroyDescriptorSetLayout(s_vk.device, s_g.ssbo_tex_layout, NULL);
        if (s_g.ssbo_const_layout) pvkDestroyDescriptorSetLayout(s_vk.device, s_g.ssbo_const_layout, NULL);
        if (s_g.pipe_layout) pvkDestroyPipelineLayout(s_vk.device, s_g.pipe_layout, NULL);
        if (s_g.pool)        pvkDestroyDescriptorPool(s_vk.device, s_g.pool, NULL);
        if (s_g.set_layout)  pvkDestroyDescriptorSetLayout(s_vk.device, s_g.set_layout, NULL);
        vk_destroy_buffer(&s_g.ubo);
        vk_destroy_buffer(&s_g.vertices);
    }
    free(s_g.hlsl);
    free(s_g.spv);
    free(s_g.cpu);
    memset(&s_g, 0, sizeof s_g);
}

static int vk_ensure_buffer(vk_buffer* b, VkDeviceSize bytes, VkBufferUsageFlags usage)
{
    if (b->buf && b->size >= bytes) return 0;
    VkDeviceSize size = b->size ? b->size : 64 * 1024;
    while (size < bytes) size *= 2;
    vk_destroy_buffer(b);                 /* idle: every submission is waited on */
    if (vk_create_host_buffer(b, size, usage)) return -1;
    if (!b->coherent) { VK_LOG("vertex memory is not host-coherent\n"); return -1; }
    return 0;
}


/* Returns 1 when the draw went through the guest programs (or had nothing to
 * draw), 0 when the caller must use the fixed-function path instead. */
static int vk_guest_draw(const rsx_state* st, u32 prim, u32 first, u32 count)
{
    if (!s_g.on) return 0;
    if (vk_cube_mask(st) || vk_vtex_mask(st)) {
        if (!(s_g.warned & 1u)) {
            VK_LOG("guest programs: cube maps and vertex textures are not supported "
                   "yet; those draws use the fixed-function path\n");
            s_g.warned |= 1u;
        }
        return 0;
    }
    const int vs = vk_vp_slot(st);
    if (vs < 0) return 0;
    const u8* fp_uc = NULL;
    const int fs = vk_fp_slot(st, &fp_uc);
    if (fs < 0) return 0;
    /* The constant count must be what the program was compiled against. */
    if (rsx_fp_collect_constants(fp_uc, VK_FP_MAX_BYTES, &s_g.consts) < 0 ||
        s_g.consts.count != s_g.fp[fs].nconst)
        return 0;
    const u32 nslots = s_g.consts.count ? s_g.consts.count : 1u;
    const u32 fp_len = (nslots + 1u) * 16u;
    if (fp_len > s_g.max_ubo_range) return 0;

    const int ztest  = st->depth_test_enable ? 1 : 0;
    const int zwrite = ztest && st->depth_mask;
    VkPipeline pipe = vk_guest_pipeline(vs, fs, ztest, zwrite, vk_compare_slot(st->depth_func));
    if (!pipe) return 0;

    s_g.cpu_count = 0;
    vk_expand(st, prim, first, count, vk_guest_tri);
    if (!s_g.cpu_count) return 1;
    const VkDeviceSize bytes = (VkDeviceSize)s_g.cpu_count * VK_GUEST_FLOATS * sizeof(float);
    if (vk_ensure_buffer(&s_g.vertices, bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) return 0;
    memcpy(s_g.vertices.ptr, s_g.cpu, (size_t)bytes);

    /* VPConst: the 512 transform constants, then the viewport epilogue as the
     * Metal and D3D12 backends write it -- x/y identity, and z remapped from
     * GL clip space to [0, 1] when the guest programmed a z scale. */
    u8* vpc = (u8*)s_g.ubo.ptr;
    memcpy(vpc, st->vertex_constants, RSX_MAX_VERTEX_CONSTANTS * 16u);
    float* vpx = (float*)(vpc + RSX_MAX_VERTEX_CONSTANTS * 16u);
    vpx[0] = vpx[1] = vpx[3] = 1.0f;
    vpx[4] = vpx[5] = vpx[7] = 0.0f;
    if (st->viewport_scale[2] != 0.0f) { vpx[2] = st->viewport_scale[2]; vpx[6] = st->viewport_offset[2]; }
    else                                { vpx[2] = 1.0f;                  vpx[6] = 0.0f; }
    /* PSConstants: inline constants as host-order bit patterns, then fp_alpha. */
    u8* fpc = (u8*)s_g.ubo.ptr + s_g.ps_off;
    memset(fpc, 0, fp_len);
    if (s_g.consts.count) memcpy(fpc, s_g.consts.values, s_g.consts.count * 16u);
    float* alpha = (float*)(fpc + nslots * 16u);
    alpha[0] = rsx_fp_alpha_ref(st->alpha_ref, st->surface_format & 0x1Fu);

    /* Descriptors: rewritten per draw, which is safe because every submission
     * is waited on before the next one is recorded. The constant offsets are
     * dynamic (VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC). */
    VkDescriptorBufferInfo bi[2] = {
        { s_g.ubo.buf, 0, VK_VPCONST_BYTES },
        { s_g.ubo.buf, 0, fp_len },
    };
    VkDescriptorImageInfo ii[RSX_MAX_TEXTURES], si[RSX_MAX_TEXTURES];
    for (u32 u = 0; u < RSX_MAX_TEXTURES; u++) {
        ii[u] = (VkDescriptorImageInfo){ VK_NULL_HANDLE,
                                         s_vk.tex_ready[u] ? s_vk.tex[u].view : s_vk.dummy.view,
                                         VK_IMAGE_LAYOUT_GENERAL };
        si[u] = (VkDescriptorImageInfo){ s_vk.tex_sampler[u] ? s_vk.tex_sampler[u] : s_vk.sampler,
                                         VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
    }
    VkWriteDescriptorSet w[4] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s_g.set,
          .dstBinding = RSX_SPIRV_VPCONST_BINDING, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &bi[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s_g.set,
          .dstBinding = RSX_SPIRV_PSCONST_BINDING, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &bi[1] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s_g.set,
          .dstBinding = RSX_SPIRV_TEXTURE_BINDING, .descriptorCount = RSX_MAX_TEXTURES,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = ii },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s_g.set,
          .dstBinding = RSX_SPIRV_SAMPLER_BINDING, .descriptorCount = RSX_MAX_TEXTURES,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .pImageInfo = si },
    };
    pvkUpdateDescriptorSets(s_vk.device, 4, w, 0, NULL);

    if (vk_begin()) return 0;
    vk_barrier_image(s_vk.color.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    vk_barrier_image(s_vk.depth.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
    VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                  .renderPass = s_vk.render_pass, .framebuffer = s_vk.framebuffer,
                                  .renderArea = { { 0, 0 }, { s_vk.width, s_vk.height } } };
    pvkCmdBeginRenderPass(s_vk.cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    pvkCmdBindPipeline(s_vk.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    VkViewport vpt = { 0.0f, 0.0f, (float)s_vk.width, (float)s_vk.height, 0.0f, 1.0f };
    VkRect2D sc = { { 0, 0 }, { s_vk.width, s_vk.height } };
    pvkCmdSetViewport(s_vk.cmd, 0, 1, &vpt);
    pvkCmdSetScissor(s_vk.cmd, 0, 1, &sc);
    const u32 dyn[2] = { 0u, (u32)s_g.ps_off };
    pvkCmdBindDescriptorSets(s_vk.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_g.pipe_layout,
                             0, 1, &s_g.set, 2, dyn);
    VkDeviceSize off = 0;
    pvkCmdBindVertexBuffers(s_vk.cmd, 0, 1, &s_g.vertices.buf, &off);
    pvkCmdDraw(s_vk.cmd, (u32)s_g.cpu_count, 1, 0, 0);
    pvkCmdEndRenderPass(s_vk.cmd);
    if (vk_submit_and_wait()) return 0;
    s_g.draws++;
    return 1;
}

static void vk_draw(u32 prim, u32 first, u32 count)
{
    const rsx_state* st = s_vk.state;
    if (!st || !s_vk.device || count < 3) return;
    if (vk_guest_draw(st, prim, first, count)) return;

    s_vk.cpu_count = 0;
    vk_expand(st, prim, first, count, vk_emit_tri);
    if (!s_vk.cpu_count) return;

    const VkDeviceSize bytes = (VkDeviceSize)s_vk.cpu_count * sizeof(vk_vertex);
    if (vk_ensure_vertex_buffer(bytes)) return;
    memcpy(s_vk.vertices.ptr, s_vk.cpu_verts, (size_t)bytes);

    /* Depth writes need the test enabled, as on the hardware and in both
     * reference backends. */
    const int ztest  = st->depth_test_enable ? 1 : 0;
    const int zwrite = ztest && st->depth_mask;
    VkPipeline pipe = vk_get_pipeline(ztest, zwrite, vk_compare_slot(st->depth_func));
    if (!pipe) return;
    const int32_t textured = (s_vk.tex_ready[0] && st->vertex_attribs[8].enabled) ? 1 : 0;

    if (vk_begin()) return;
    vk_barrier_image(s_vk.color.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    vk_barrier_image(s_vk.depth.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
    VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                  .renderPass = s_vk.render_pass, .framebuffer = s_vk.framebuffer,
                                  .renderArea = { { 0, 0 }, { s_vk.width, s_vk.height } } };
    pvkCmdBeginRenderPass(s_vk.cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    pvkCmdBindPipeline(s_vk.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    /* Whole target, as the null backend's ndc_to_px maps it. */
    VkViewport vp = { 0.0f, 0.0f, (float)s_vk.width, (float)s_vk.height, 0.0f, 1.0f };
    VkRect2D sc = { { 0, 0 }, { s_vk.width, s_vk.height } };
    pvkCmdSetViewport(s_vk.cmd, 0, 1, &vp);
    pvkCmdSetScissor(s_vk.cmd, 0, 1, &sc);
    pvkCmdBindDescriptorSets(s_vk.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_vk.pipe_layout,
                             0, 1, &s_vk.desc_set, 0, NULL);
    pvkCmdPushConstants(s_vk.cmd, s_vk.pipe_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                        0, sizeof textured, &textured);
    VkDeviceSize off = 0;
    pvkCmdBindVertexBuffers(s_vk.cmd, 0, 1, &s_vk.vertices.buf, &off);
    pvkCmdDraw(s_vk.cmd, (u32)s_vk.cpu_count, 1, 0, 0);
    pvkCmdEndRenderPass(s_vk.cmd);
    vk_submit_and_wait();
}

static void vk_cb_draw_arrays(void* ud, u32 primitive, u32 first, u32 count)
{
    (void)ud;
    vk_draw(primitive, first, count);
}

static void vk_cb_draw_indexed(void* ud, u32 primitive, u32 offset, u32 count)
{
    /* Not read yet, as in the null backend: the harness geometry is
     * non-indexed. Real titles need this; it lands with the index fetch. */
    (void)ud; (void)primitive; (void)offset; (void)count;
}

/* ---------------------------------------------------------------------------
 * Clear and present
 * -------------------------------------------------------------------------*/
static void vk_cb_clear(void* ud, u32 flags, u32 color, float depth, u8 stencil)
{
    (void)ud; (void)stencil;                       /* no stencil buffer yet */
    s_vk.clear_argb = color;
    /* CLEAR_SURFACE bits: 0x01 depth, 0x02 stencil, 0xF0 the colour channels
     * (all or nothing, as in the null backend). */
    const int do_color = (flags & 0xF0u) != 0;
    const int do_depth = (flags & 0x01u) != 0;
    if (!s_vk.device || (!do_color && !do_depth)) return;

    if (vk_begin()) return;
    if (do_color) {
        VkClearColorValue v;
        v.float32[0] = (float)((color >> 16) & 0xFFu) / 255.0f;   /* R */
        v.float32[1] = (float)((color >>  8) & 0xFFu) / 255.0f;   /* G */
        v.float32[2] = (float)( color        & 0xFFu) / 255.0f;   /* B */
        v.float32[3] = (float)((color >> 24) & 0xFFu) / 255.0f;   /* A */
        VkImageSubresourceRange r = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_barrier_image(s_vk.color.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
        pvkCmdClearColorImage(s_vk.cmd, s_vk.color.img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
    }
    if (do_depth) {
        VkClearDepthStencilValue d = { depth, 0 };
        VkImageSubresourceRange r = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        vk_barrier_image(s_vk.depth.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
        pvkCmdClearDepthStencilImage(s_vk.cmd, s_vk.depth.img, VK_IMAGE_LAYOUT_GENERAL, &d, 1, &r);
    }
    vk_submit_and_wait();
}

/* ---------------------------------------------------------------------------
 * Optional window: PS3RECOMP_VK_WINDOW=1
 *
 * An SDL3 window with a Vulkan swapchain. Rendering still targets the
 * offscreen image, so readback and every test are unchanged; a present then
 * also blits that image into the next swapchain image. Any failure on this
 * path drops back to headless with a message instead of failing the run.
 * -------------------------------------------------------------------------*/
static int vk_env_on(const char* name)
{
    const char* v = getenv(name);
    return v && v[0] && v[0] != '0';
}

/* ---------------------------------------------------------------------------
 * In-game overlay pass (system dialogs, trophy toast)
 *
 * libs/video/rsx_ui_overlay.c owns the dialog state, the input and the quad geometry (uiov_vk_frame). This is the
 * Vulkan half: one extra render pass draws those quads into the swapchain image after the frame was blitted into it
 * and before it is presented, so FSR output, a frame-generation pair and the letterbox bars all carry it.
 * It costs nothing while no dialog or toast is up: uiov_vk_frame returns 0, no object below is created and the
 * present path records exactly what it always did. PS3RECOMP_VK_OVERLAY=0 turns it off, and then the title's
 * dialogs are answered automatically as in a headless run.
 * -------------------------------------------------------------------------*/
#define VK_OV_MAX_VERTS 12000u

typedef struct vk_ov {
    int state;                          /* 0 not built, 1 ready, -1 unavailable */
    VkFormat fmt;                       /* the swapchain format the render pass matches */
    VkRenderPass render_pass;
    VkShaderModule vs, fs;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout pipe_layout;
    VkPipeline pipe;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    VkSampler sampler;
    vk_image atlas;                     /* the 1024x1024 ui_font.bin atlas */
    vk_buffer stage;                    /* its upload source, kept so a lost upload can be redone */
    vk_buffer vb;                       /* this frame's quads; the present is synchronous, one buffer is enough */
    int atlas_uploaded;
    VkImageView view[8];                /* per swapchain image, made on first use */
    VkFramebuffer fb[8];
    int batch_ok;                       /* this frame has quads to draw */
    uiov_vk_batch batch;
    u32 frames;                         /* frames the overlay drew */
    vk_buffer dump_buf;                 /* PS3RECOMP_VK_OVERLAY_DUMP readback of the finished swapchain image */
    int dump_pending;
    u32 dump_n;                         /* dumps written so far */
} vk_ov;
static vk_ov s_ov;

/* Raw frees: everything here is only ever used by the present path, so the engine flush vk_destroy_* does is not
 * wanted (and must not run from the present thread). The callers have made the device idle. */
static void vk_ov_free_buffer(vk_buffer* b)
{
    if (b->ptr) pvkUnmapMemory(s_vk.device, b->mem);
    if (b->buf) pvkDestroyBuffer(s_vk.device, b->buf, NULL);
    if (b->mem) pvkFreeMemory(s_vk.device, b->mem, NULL);
    memset(b, 0, sizeof *b);
}

static void vk_ov_free_image(vk_image* im)
{
    if (im->view) pvkDestroyImageView(s_vk.device, im->view, NULL);
    if (im->img)  pvkDestroyImage(s_vk.device, im->img, NULL);
    if (im->mem)  pvkFreeMemory(s_vk.device, im->mem, NULL);
    memset(im, 0, sizeof *im);
}

/* The per-swapchain-image views and framebuffers; made again after every swapchain recreation. */
static void vk_ov_targets_drop(void)
{
    if (!s_vk.device) return;
    for (u32 i = 0; i < 8; i++) {
        if (s_ov.fb[i])   pvkDestroyFramebuffer(s_vk.device, s_ov.fb[i], NULL);
        if (s_ov.view[i]) pvkDestroyImageView(s_vk.device, s_ov.view[i], NULL);
        s_ov.fb[i] = VK_NULL_HANDLE; s_ov.view[i] = VK_NULL_HANDLE;
    }
}

static void vk_ov_destroy(void)
{
    if (s_vk.device && (s_ov.state || s_ov.render_pass || s_ov.vb.buf)) {
        vk_ov_targets_drop();
        if (s_ov.pipe)        pvkDestroyPipeline(s_vk.device, s_ov.pipe, NULL);
        if (s_ov.pipe_layout) pvkDestroyPipelineLayout(s_vk.device, s_ov.pipe_layout, NULL);
        if (s_ov.pool)        pvkDestroyDescriptorPool(s_vk.device, s_ov.pool, NULL);
        if (s_ov.set_layout)  pvkDestroyDescriptorSetLayout(s_vk.device, s_ov.set_layout, NULL);
        if (s_ov.sampler)     pvkDestroySampler(s_vk.device, s_ov.sampler, NULL);
        if (s_ov.vs)          pvkDestroyShaderModule(s_vk.device, s_ov.vs, NULL);
        if (s_ov.fs)          pvkDestroyShaderModule(s_vk.device, s_ov.fs, NULL);
        if (s_ov.render_pass) pvkDestroyRenderPass(s_vk.device, s_ov.render_pass, NULL);
        vk_ov_free_buffer(&s_ov.vb);
        vk_ov_free_buffer(&s_ov.stage);
        vk_ov_free_buffer(&s_ov.dump_buf);
        vk_ov_free_image(&s_ov.atlas);
    }
    memset(&s_ov, 0, sizeof s_ov);
}

/* A swapchain is about to be made with `fmt`: the old image views go, and a render pass for another format too. */
static void vk_ov_swapchain_reset(VkFormat fmt)
{
    vk_ov_targets_drop();
    if (s_ov.state == 1 && s_ov.fmt != fmt) vk_ov_destroy();
}

static int vk_ov_init(void)
{
    const uiov_vk_batch* b = &s_ov.batch;
    if (!b->atlas_rgba || !b->atlas_w || !b->atlas_h) return -1;
    s_ov.fmt = s_vk.sc_format;
    VkAttachmentDescription att = {
        .format = s_ov.fmt, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,   /* where the blit left it */
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,          /* the pass replaces the present transition */
    };
    VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription sub = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 .colorAttachmentCount = 1, .pColorAttachments = &ref };
    VkSubpassDependency dep[2] = {
        { .srcSubpass = VK_SUBPASS_EXTERNAL, .dstSubpass = 0,
          .srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT, .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
          .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT },
        { .srcSubpass = 0, .dstSubpass = VK_SUBPASS_EXTERNAL,
          .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, .dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
          .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, .dstAccessMask = 0 },
    };
    VkRenderPassCreateInfo rpi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                                   .attachmentCount = 1, .pAttachments = &att, .subpassCount = 1, .pSubpasses = &sub,
                                   .dependencyCount = 2, .pDependencies = dep };
    VK_CHECK(pvkCreateRenderPass(s_vk.device, &rpi, NULL, &s_ov.render_pass), "vkCreateRenderPass(overlay)");

    VkShaderModuleCreateInfo smi = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                     .codeSize = sizeof rsx_vk_uiov_vert_spv, .pCode = rsx_vk_uiov_vert_spv };
    VK_CHECK(pvkCreateShaderModule(s_vk.device, &smi, NULL, &s_ov.vs), "vkCreateShaderModule(overlay vs)");
    smi.codeSize = sizeof rsx_vk_uiov_frag_spv; smi.pCode = rsx_vk_uiov_frag_spv;
    VK_CHECK(pvkCreateShaderModule(s_vk.device, &smi, NULL, &s_ov.fs), "vkCreateShaderModule(overlay fs)");

    VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
                                .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0.25f };
    VK_CHECK(pvkCreateSampler(s_vk.device, &sci, NULL, &s_ov.sampler), "vkCreateSampler(overlay)");

    VkDescriptorSetLayoutBinding dlb = { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
    VkDescriptorSetLayoutCreateInfo dli = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .bindingCount = 1, .pBindings = &dlb };
    VK_CHECK(pvkCreateDescriptorSetLayout(s_vk.device, &dli, NULL, &s_ov.set_layout), "vkCreateDescriptorSetLayout(overlay)");
    VkPushConstantRange pcr = { VK_SHADER_STAGE_VERTEX_BIT, 0, 2 * sizeof(float) };
    VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                       .setLayoutCount = 1, .pSetLayouts = &s_ov.set_layout,
                                       .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    VK_CHECK(pvkCreatePipelineLayout(s_vk.device, &pli, NULL, &s_ov.pipe_layout), "vkCreatePipelineLayout(overlay)");

    VkPipelineShaderStageCreateInfo stages[2] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = s_ov.vs, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = s_ov.fs, .pName = "main" } };
    VkVertexInputBindingDescription vbind = { 0, sizeof(uiov_vk_vertex), VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription vattr[3] = {
        { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 }, { 1, 0, VK_FORMAT_R32G32_SFLOAT, 8 },
        { 2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 16 } };
    VkPipelineVertexInputStateCreateInfo vi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vbind,
        .vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = vattr };
    VkPipelineInputAssemblyStateCreateInfo ia = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkPipelineViewportStateCreateInfo vp = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1 };
    VkPipelineRasterizationStateCreateInfo rs = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
    VkPipelineMultisampleStateCreateInfo ms = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkPipelineColorBlendAttachmentState cba = {
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA, .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                          VK_COLOR_COMPONENT_A_BIT };
    VkPipelineColorBlendStateCreateInfo cb = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &cba };
    const VkDynamicState dyn_states[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2, .pDynamicStates = dyn_states };
    VkGraphicsPipelineCreateInfo gpi = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb,
        .pDynamicState = &dyn, .layout = s_ov.pipe_layout, .renderPass = s_ov.render_pass, .subpass = 0 };
    VK_CHECK(pvkCreateGraphicsPipelines(s_vk.device, VK_NULL_HANDLE, 1, &gpi, NULL, &s_ov.pipe),
             "vkCreateGraphicsPipelines(overlay)");

    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
    VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
    VK_CHECK(pvkCreateDescriptorPool(s_vk.device, &dpi, NULL, &s_ov.pool), "vkCreateDescriptorPool(overlay)");
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = s_ov.pool, .descriptorSetCount = 1, .pSetLayouts = &s_ov.set_layout };
    VK_CHECK(pvkAllocateDescriptorSets(s_vk.device, &dai, &s_ov.set), "vkAllocateDescriptorSets(overlay)");

    const VkDeviceSize atlas_bytes = (VkDeviceSize)b->atlas_w * b->atlas_h * 4u;
    if (vk_create_image(&s_ov.atlas, VK_FORMAT_R8G8B8A8_UNORM, b->atlas_w, b->atlas_h,
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT)) return -1;
    if (vk_create_host_buffer(&s_ov.stage, atlas_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return -1;
    memcpy(s_ov.stage.ptr, b->atlas_rgba, (size_t)atlas_bytes);
    if (vk_create_host_buffer(&s_ov.vb, (VkDeviceSize)VK_OV_MAX_VERTS * sizeof(uiov_vk_vertex),
                              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) return -1;

    VkDescriptorImageInfo dii = { .sampler = s_ov.sampler, .imageView = s_ov.atlas.view,
                                  .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet wds = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s_ov.set,
                                 .dstBinding = 0, .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &dii };
    pvkUpdateDescriptorSets(s_vk.device, 1, &wds, 0, NULL);
    s_ov.state = 1;
    VK_LOG("in-game overlay ready (swapchain format %d)\n", (int)s_ov.fmt);
    return 0;
}

/* Once per presented window frame, before any blit slot: ask the overlay what to draw and stage it. */
static void vk_ov_frame(u32 fit_w, u32 fit_h)
{
    s_ov.batch_ok = 0;
    if (s_ov.state < 0 || !s_vk.sc_overlay_ok) return;   /* no pass possible: the overlay stays unavailable and the
                                                           * title's dialogs are answered automatically */
    uiov_vk_batch batch;
    memset(&batch, 0, sizeof batch);
    if (!uiov_vk_frame(fit_w, fit_h, &batch) || !batch.verts || !batch.nverts) return;
    s_ov.batch = batch;
    s_ov.frames++;
    if (s_ov.state == 0 && vk_ov_init() != 0) {
        VK_LOG("in-game overlay unavailable, dialogs fall back to automatic answers\n");
        vk_device_idle();
        vk_ov_destroy();
        s_ov.state = -1;
        return;
    }
    if (s_ov.state != 1) return;
    u32 n = batch.nverts > VK_OV_MAX_VERTS ? VK_OV_MAX_VERTS : batch.nverts;
    n -= n % 3u;
    memcpy(s_ov.vb.ptr, batch.verts, (size_t)n * sizeof(uiov_vk_vertex));
    s_ov.batch.nverts = n;
    s_ov.batch_ok = n != 0;
}

/* Records the overlay into `cb` for swapchain image `idx` (in TRANSFER_DST after the blit). On success the render
 * pass has also moved the image to PRESENT_SRC and the caller skips its own transition. 0 = nothing recorded. */
static int vk_ov_record(VkCommandBuffer cb, u32 idx, u32 fx, u32 fy, u32 fw, u32 fh)
{
    if (!s_ov.batch_ok || s_ov.state != 1 || idx >= s_vk.sc_count || !fw || !fh) return 0;
    if (!s_ov.fb[idx]) {
        VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = s_vk.sc_images[idx],
                                      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = s_vk.sc_format,
                                      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
        if (pvkCreateImageView(s_vk.device, &vci, NULL, &s_ov.view[idx]) != VK_SUCCESS) return 0;
        VkFramebufferCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                        .renderPass = s_ov.render_pass, .attachmentCount = 1, .pAttachments = &s_ov.view[idx],
                                        .width = s_vk.sc_extent.width, .height = s_vk.sc_extent.height, .layers = 1 };
        if (pvkCreateFramebuffer(s_vk.device, &fci, NULL, &s_ov.fb[idx]) != VK_SUCCESS) {
            pvkDestroyImageView(s_vk.device, s_ov.view[idx], NULL);
            s_ov.view[idx] = VK_NULL_HANDLE;
            return 0;
        }
    }
    if (!s_ov.atlas_uploaded) {
        VkImageMemoryBarrier ab = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = s_ov.atlas.img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
        pvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, NULL, 0, NULL, 1, &ab);
        VkBufferImageCopy bic = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                  .imageExtent = { s_ov.batch.atlas_w, s_ov.batch.atlas_h, 1 } };
        pvkCmdCopyBufferToImage(cb, s_ov.stage.buf, s_ov.atlas.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
        ab.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; ab.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        ab.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ab.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        pvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                              0, 0, NULL, 0, NULL, 1, &ab);
        s_ov.atlas_uploaded = 1;
    }
    VkRenderPassBeginInfo rb = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = s_ov.render_pass,
                                 .framebuffer = s_ov.fb[idx], .renderArea = { { 0, 0 }, s_vk.sc_extent } };
    pvkCmdBeginRenderPass(cb, &rb, VK_SUBPASS_CONTENTS_INLINE);
    pvkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s_ov.pipe);
    VkViewport vpt = { (float)fx, (float)fy, (float)fw, (float)fh, 0.0f, 1.0f };
    VkRect2D scr = { { (int32_t)fx, (int32_t)fy }, { fw, fh } };
    pvkCmdSetViewport(cb, 0, 1, &vpt);
    pvkCmdSetScissor(cb, 0, 1, &scr);
    pvkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s_ov.pipe_layout, 0, 1, &s_ov.set, 0, NULL);
    const float scale[2] = { 2.0f / s_ov.batch.vw, 2.0f / s_ov.batch.vh };
    pvkCmdPushConstants(cb, s_ov.pipe_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof scale, scale);
    const VkDeviceSize off = 0;
    pvkCmdBindVertexBuffers(cb, 0, 1, &s_ov.vb.buf, &off);
    pvkCmdDraw(cb, s_ov.batch.nverts, 1, 0, 0);
    pvkCmdEndRenderPass(cb);
    /* PS3RECOMP_VK_OVERLAY_DUMP=<path with %u>: a few overlay frames read back after the pass (headless verification:
     * there is no screenshot of a swapchain image otherwise). Needs the swapchain images to allow TRANSFER_SRC. */
    /* One dump every 1.5 s of overlay time, 12 at most (a frame count would not do: with the title idle behind a dialog
     * the window presents hundreds of duplicate frames a second). */
    static double dump_last_ms;
    int dump_due = 0;
    if (s_vk.sc_dump_ok && s_ov.dump_n < 12) {
        const double t = vk_now_ms();
        if (!dump_last_ms) dump_last_ms = t;
        else if (t - dump_last_ms >= 1500.0) { dump_last_ms = t; dump_due = 1; }
    }
    if (dump_due) {
        const VkDeviceSize bytes = (VkDeviceSize)s_vk.sc_extent.width * s_vk.sc_extent.height * 4u;
        if (!s_ov.dump_buf.buf && vk_create_host_buffer(&s_ov.dump_buf, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return 1;
        VkImageMemoryBarrier db = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = s_vk.sc_images[idx], .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
        pvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, NULL, 0, NULL, 1, &db);
        VkBufferImageCopy bic = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                  .imageExtent = { s_vk.sc_extent.width, s_vk.sc_extent.height, 1 } };
        pvkCmdCopyImageToBuffer(cb, s_vk.sc_images[idx], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s_ov.dump_buf.buf, 1, &bic);
        db.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; db.dstAccessMask = 0;
        db.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; db.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        pvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                              0, 0, NULL, 0, NULL, 1, &db);
        s_ov.dump_pending = 1;
        s_ov.dump_n++;
    }
    return 1;
}

/* After the window submit finished: write the readback of vk_ov_record's dump, if it asked for one. */
static void vk_ov_after_submit(void)
{
    if (!s_ov.dump_pending) return;
    s_ov.dump_pending = 0;
    const char* tmpl = getenv("PS3RECOMP_VK_OVERLAY_DUMP");
    if (!tmpl || !s_ov.dump_buf.ptr) return;
    char path[1024];
    if (snprintf(path, sizeof path, tmpl, s_ov.dump_n) <= 0) return;
    FILE* f = fopen(path, "wb");
    if (!f) { VK_LOG("overlay dump: cannot write %s\n", path); return; }
    const u32 w = s_vk.sc_extent.width, h = s_vk.sc_extent.height;
    const int bgra = s_vk.sc_format == VK_FORMAT_B8G8R8A8_UNORM || s_vk.sc_format == VK_FORMAT_B8G8R8A8_SRGB;
    fprintf(f, "P6\n%u %u\n255\n", w, h);
    const unsigned char* px = (const unsigned char*)s_ov.dump_buf.ptr;
    for (u32 i = 0; i < w * h; i++) {
        const unsigned char rgb[3] = { px[i * 4 + (bgra ? 2 : 0)], px[i * 4 + 1], px[i * 4 + (bgra ? 0 : 2)] };
        fwrite(rgb, 1, 3, f);
    }
    fclose(f);
    VK_LOG("overlay dump: %s (%ux%u)\n", path, w, h);
}

static void vk_window_drop(const char* why)
{
#if VK_WINDOW_SDL
    if (why) VK_LOG("window disabled, continuing headless: %s\n", why);
#else
    (void)why;
#endif
    if (s_vk.device && pvkDeviceWaitIdle) {
        vk_device_idle();
        vk_ov_destroy();
        for (u32 i = 0; i < 8; i++)
            if (s_vk.sc_done[i]) {
                pvkDestroySemaphore(s_vk.device, s_vk.sc_done[i], NULL);
                s_vk.sc_done[i] = VK_NULL_HANDLE;
            }
        if (s_vk.sem_acquire) pvkDestroySemaphore(s_vk.device, s_vk.sem_acquire, NULL);
        if (s_vk.sem_acquire_real) pvkDestroySemaphore(s_vk.device, s_vk.sem_acquire_real, NULL);
        if (s_vk.swapchain && pvkDestroySwapchainKHR)
            pvkDestroySwapchainKHR(s_vk.device, s_vk.swapchain, NULL);
    }
    if (s_vk.surface && s_vk.instance && pvkDestroySurfaceKHR)
        pvkDestroySurfaceKHR(s_vk.instance, s_vk.surface, NULL);
    if (s_vk.window) {
#if VK_WINDOW_SDL
        SDL_DestroyWindow(s_vk.window);
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
#endif
    }
    s_vk.window = NULL;
    s_vk.surface = VK_NULL_HANDLE;
    s_vk.swapchain = VK_NULL_HANDLE;
    s_vk.sem_acquire = VK_NULL_HANDLE;
    s_vk.sem_acquire_real = VK_NULL_HANDLE;
    s_vk.fg_was_active = 0;
    s_vk.sc_count = 0;
    s_vk.windowed = 0;
}

static void vk_window_drop(const char* why);

/* The window's Vulkan surface, and the message to report when it fails. */
static int vk_create_surface(void)
{
#if VK_WINDOW_SDL
    return SDL_Vulkan_CreateSurface(s_vk.window, s_vk.instance, NULL, &s_vk.surface) ? 0 : -1;
#else
    return -1;
#endif
}

static const char* vk_surface_error(void)
{
#if VK_WINDOW_SDL
    return SDL_GetError();
#else
    return "no SDL3";
#endif
}

static void vk_window_open(const char* title)
{
    if (!vk_env_on("PS3RECOMP_VK_WINDOW")) return;
#if !VK_WINDOW_SDL
    (void)title;
    VK_LOG("window: built without SDL3, continuing headless "
           "(install the SDL3 development package and rebuild)\n");
    return;
#else
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO)) {
        VK_LOG("window disabled, continuing headless: SDL video init failed: %s\n", SDL_GetError());
        return;
    }
    /* Resizable and maximizable; F11 or Alt+Enter switches to borderless fullscreen and back. The frame keeps its
     * aspect ratio inside whatever size the window has (vk_window_fit). */
    SDL_WindowFlags flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE;
    /* Fullscreen is the play default. Set PS3RECOMP_FULLSCREEN=0 or
     * PS3RECOMP_VK_WINDOWED=1 for a resizable window. */
    int fullscreen = 1;
    const char* fs_name = "PS3RECOMP_FULLSCREEN";
    const char* fs = getenv(fs_name);
    if (!fs) { fs_name = "PS3RECOMP_VK_FULLSCREEN"; fs = getenv(fs_name); }
    if ((fs && !vk_env_on(fs_name)) || vk_env_on("PS3RECOMP_VK_WINDOWED")) fullscreen = 0;
    if (fullscreen) flags |= SDL_WINDOW_FULLSCREEN;
    s_vk.window = SDL_CreateWindow(title && title[0] ? title : "ps3recomp",
                                   (int)s_vk.width, (int)s_vk.height, flags);
    s_vk.windowed = 1;          /* so vk_window_drop also shuts SDL video down */
    if (!s_vk.window) {
        char why[256];
        snprintf(why, sizeof why, "SDL_CreateWindow failed: %s", SDL_GetError());
        vk_window_drop(why);
        return;
    }
    SDL_SetWindowMinimumSize(s_vk.window, 320, 180);
    if (flags & SDL_WINDOW_FULLSCREEN) SDL_HideCursor();
    Uint32 n = 0;
    const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&n);
    if (!extensions || n == 0 || n > 16) {
        char why[256];
        snprintf(why, sizeof why, "no Vulkan surface extensions (%s)", SDL_GetError());
        vk_window_drop(why);
        return;
    }
    for (Uint32 i = 0; i < n; i++) s_vk.inst_exts[i] = extensions[i];
    s_vk.inst_ext_count = n;
    const char* hold = getenv("PS3RECOMP_VK_HOLD");
    s_vk.hold_seconds = hold ? atof(hold) : 0.0;
    VK_LOG("window: SDL3 video driver '%s', %u surface extension(s), %s\n",
           SDL_GetCurrentVideoDriver(), n, fullscreen ? "fullscreen" : "windowed");
#endif /* VK_WINDOW_SDL */
}

static VkPresentModeKHR vk_present_mode(void)
{
    const char* e = getenv("PS3RECOMP_VK_VSYNC");
    if (!e) e = getenv("PRESENT_VSYNC");
    const int vsync = !(e && *e == '0');
    u32 count = 0;
    VkPresentModeKHR modes[32];
    if (pvkGetPhysicalDeviceSurfacePresentModesKHR(s_vk.phys, s_vk.surface, &count, NULL) == VK_SUCCESS &&
        count && count <= 32) {
        if (pvkGetPhysicalDeviceSurfacePresentModesKHR(s_vk.phys, s_vk.surface, &count, modes) != VK_SUCCESS)
            count = 0;
    } else {
        count = 0;
    }
#ifdef PS3RECOMP_VK_FFX
    if (rsx_vk_fsr_fg_active()) {
        return ffxVkFrameGenerationSelectPresentMode(true, vsync != 0, modes, count);
    }
#endif
    if (vsync) return VK_PRESENT_MODE_FIFO_KHR;
    for (u32 i = 0; i < count; i++)
        if (modes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR) return VK_PRESENT_MODE_IMMEDIATE_KHR;
    VK_LOG("vsync off requested, but immediate present is unavailable; using FIFO\n");
    return VK_PRESENT_MODE_FIFO_KHR;
}

/* 0 on success, 1 when the window has no drawable area right now (the old swapchain is kept), -1 on failure. */
static int vk_create_swapchain(void)
{
    VkSurfaceCapabilitiesKHR caps;
    {
        const VkResult cr = pvkGetPhysicalDeviceSurfaceCapabilitiesKHR(s_vk.phys, s_vk.surface, &caps);
        if (cr == VK_ERROR_SURFACE_LOST_KHR) { VK_LOG("surface lost (capabilities query)\n"); return -2; }
        if (cr != VK_SUCCESS) { VK_LOG("vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed (VkResult %d)\n", (int)cr); return -1; }
    }
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        VK_LOG("swapchain images cannot be blit destinations\n"); return -1;
    }

    u32 nf = 0;
    VK_CHECK(pvkGetPhysicalDeviceSurfaceFormatsKHR(s_vk.phys, s_vk.surface, &nf, NULL),
             "vkGetPhysicalDeviceSurfaceFormatsKHR");
    if (nf == 0) { VK_LOG("surface reports no formats\n"); return -1; }
    if (nf > 64) nf = 64;
    VkSurfaceFormatKHR fmts[64];
    VK_CHECK(pvkGetPhysicalDeviceSurfaceFormatsKHR(s_vk.phys, s_vk.surface, &nf, fmts),
             "vkGetPhysicalDeviceSurfaceFormatsKHR");
    /* UNORM, not SRGB: the offscreen target holds the guest's values as-is. */
    VkSurfaceFormatKHR fmt = fmts[0];
    if (nf == 1 && fmts[0].format == VK_FORMAT_UNDEFINED) {
        fmt.format = VK_FORMAT_B8G8R8A8_UNORM;
    } else {
        static const VkFormat pref[] = { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM };
        int found = 0;
        for (size_t k = 0; k < 2 && !found; k++)
            for (u32 i = 0; i < nf; i++)
                if (fmts[i].format == pref[k]) { fmt = fmts[i]; found = 1; break; }
    }
    VkFormatProperties fp;
    pvkGetPhysicalDeviceFormatProperties(s_vk.phys, fmt.format, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
        VK_LOG("swapchain format %d cannot be a blit destination\n", (int)fmt.format); return -1;
    }

    VkExtent2D ext = caps.currentExtent;
    if (ext.width == 0xFFFFFFFFu) {
        int w = 0, hh = 0;
#if VK_WINDOW_SDL
        SDL_GetWindowSizeInPixels(s_vk.window, &w, &hh);
#endif
        ext.width = (u32)w; ext.height = (u32)hh;
        if (ext.width  < caps.minImageExtent.width)  ext.width  = caps.minImageExtent.width;
        if (ext.height < caps.minImageExtent.height) ext.height = caps.minImageExtent.height;
        if (ext.width  > caps.maxImageExtent.width)  ext.width  = caps.maxImageExtent.width;
        if (ext.height > caps.maxImageExtent.height) ext.height = caps.maxImageExtent.height;
    }
    if (!ext.width || !ext.height) return 1;   /* minimized: no drawable area, try again once it is shown */

    u32 count = caps.minImageCount + 1;
#ifdef PS3RECOMP_VK_FFX
    if (rsx_vk_fsr_fg_active()) {
        const u32 fg_req = ffxVkFrameGenerationRequestedImageCount(
            caps.minImageCount, caps.maxImageCount, true);
        if (fg_req > count) count = fg_req;
        if (count > 8u) count = 8u;
    }
#endif
    if (caps.maxImageCount && count > caps.maxImageCount) count = caps.maxImageCount;

    static const VkCompositeAlphaFlagBitsKHR alphas[] = {
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
    };
    VkCompositeAlphaFlagBitsKHR alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    for (size_t k = 0; k < 4; k++)
        if (caps.supportedCompositeAlpha & alphas[k]) { alpha = alphas[k]; break; }

    /* The overlay pass draws into the swapchain image, so the images must take a colour attachment and the format
     * must blend. Anything less leaves the overlay unavailable (dialogs then answer themselves). */
    const char* ov_env = getenv("PS3RECOMP_VK_OVERLAY");
    const int ov_ok = !(ov_env && ov_env[0] == '0') &&
        (caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) &&
        (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) &&
        (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT);
    const int dump_ok = ov_ok && getenv("PS3RECOMP_VK_OVERLAY_DUMP") && (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    vk_ov_swapchain_reset(fmt.format);
    VkSwapchainKHR old = s_vk.swapchain;
    VkSwapchainCreateInfoKHR sci = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = s_vk.surface, .minImageCount = count,
        .imageFormat = fmt.format, .imageColorSpace = fmt.colorSpace,
        .imageExtent = ext, .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | (ov_ok ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0) |
                       (dump_ok ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0),
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = caps.currentTransform, .compositeAlpha = alpha,
        .presentMode = vk_present_mode(),
        .clipped = VK_TRUE, .oldSwapchain = old,
    };
    VkSwapchainKHR sc = VK_NULL_HANDLE;
    VkResult r = pvkCreateSwapchainKHR(s_vk.device, &sci, NULL, &sc);
    if (old) pvkDestroySwapchainKHR(s_vk.device, old, NULL);
    s_vk.swapchain = VK_NULL_HANDLE;
    if (r != VK_SUCCESS) {
        VK_LOG("vkCreateSwapchainKHR failed (VkResult %d)\n", (int)r);
        return r == VK_ERROR_SURFACE_LOST_KHR ? -2 : -1;
    }
    s_vk.swapchain = sc;
    s_vk.sc_format = fmt.format;
    s_vk.sc_overlay_ok = ov_ok;
    s_vk.sc_dump_ok = dump_ok;

    u32 n = 0;
    VK_CHECK(pvkGetSwapchainImagesKHR(s_vk.device, sc, &n, NULL), "vkGetSwapchainImagesKHR");
    if (n == 0 || n > 8) { VK_LOG("unsupported swapchain image count %u\n", n); return -1; }
    VK_CHECK(pvkGetSwapchainImagesKHR(s_vk.device, sc, &n, s_vk.sc_images), "vkGetSwapchainImagesKHR");

    for (u32 i = 0; i < 8; i++)
        if (s_vk.sc_done[i]) {
            pvkDestroySemaphore(s_vk.device, s_vk.sc_done[i], NULL);
            s_vk.sc_done[i] = VK_NULL_HANDLE;
        }
    VkSemaphoreCreateInfo semi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    for (u32 i = 0; i < n; i++)
        VK_CHECK(pvkCreateSemaphore(s_vk.device, &semi, NULL, &s_vk.sc_done[i]), "vkCreateSemaphore");

    s_vk.sc_extent = ext;
    s_vk.sc_count = n;
    VK_LOG("window swapchain %ux%u, %u images, format %d, present mode %d%s\n",
           ext.width, ext.height, n, (int)fmt.format, (int)sci.presentMode,
           sci.presentMode == VK_PRESENT_MODE_FIFO_KHR ? " (vsync)" : " (vsync off)");
    return 0;
}

/* The window changed size (or fullscreen state) since the swapchain was made: it is recreated before the next acquire.
 * Also set while the window is minimized, when there is no drawable area to make one for. */
static int s_vk_resized;

/* Minimized, or no drawable area: nothing can be presented, and a swapchain cannot be created for it. */
static int vk_window_hidden(void)
{
    int w = 0, h = 0;
#if VK_WINDOW_SDL
    if (SDL_GetWindowFlags(s_vk.window) & SDL_WINDOW_MINIMIZED) return 1;
    SDL_GetWindowSizeInPixels(s_vk.window, &w, &h);
#endif
    return w <= 0 || h <= 0;
}

/* Suspend/resume hardening. A swapchain that went out of date (resize, fullscreen, a Deck waking from sleep) is made
 * again. VK_ERROR_SURFACE_LOST_KHR means the surface itself is dead (the compositor restarted): then the swapchain
 * goes first, then the surface, and both are made again from the window. A failure retries every 250 ms and gives up
 * (headless) only after 10 s, because a resume can take a moment to give the window a drawable surface back. */
static int s_vk_surface_lost;          /* a result said VK_ERROR_SURFACE_LOST_KHR: rebuild the surface too */
static double s_vk_lost_since_ms;      /* first failed recreation of a streak, 0 = none */
static double s_vk_last_try_ms;

/* After a failed acquire or present an acquire semaphore may be left signalled, and acquiring with a signalled
 * semaphore is invalid: the device is idle here, so make both again. */
static int vk_acquire_sems_reset(void)
{
    VkSemaphoreCreateInfo semi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkSemaphore* sems[2] = { &s_vk.sem_acquire, &s_vk.sem_acquire_real };
    for (int i = 0; i < 2; i++) {
        if (*sems[i]) pvkDestroySemaphore(s_vk.device, *sems[i], NULL);
        *sems[i] = VK_NULL_HANDLE;
        if (pvkCreateSemaphore(s_vk.device, &semi, NULL, sems[i]) != VK_SUCCESS) { *sems[i] = VK_NULL_HANDLE; return -1; }
    }
    return 0;
}

static int vk_rebuild_surface(void)
{
    vk_device_idle();
    vk_ov_targets_drop();
    if (s_vk.swapchain) { pvkDestroySwapchainKHR(s_vk.device, s_vk.swapchain, NULL); s_vk.swapchain = VK_NULL_HANDLE; }
    if (s_vk.surface) { pvkDestroySurfaceKHR(s_vk.instance, s_vk.surface, NULL); s_vk.surface = VK_NULL_HANDLE; }
    if (vk_create_surface() != 0) {
        VK_LOG("cannot recreate the window surface: %s\n", vk_surface_error());
        s_vk.surface = VK_NULL_HANDLE;
        return -1;
    }
    VkBool32 ok = VK_FALSE;
    if (pvkGetPhysicalDeviceSurfaceSupportKHR(s_vk.phys, s_vk.queue_family, s_vk.surface, &ok) != VK_SUCCESS || !ok) {
        VK_LOG("the new surface cannot present from queue family %u\n", s_vk.queue_family);
        return -1;
    }
    VK_LOG("window surface recreated\n");
    return 0;
}

static void vk_swapchain_lost(void)
{
    if (vk_window_hidden()) { s_vk_resized = 1; return; }   /* recreated once it is shown again */
    vk_device_idle();
    if (vk_acquire_sems_reset()) { vk_window_drop("the acquire semaphores could not be recreated"); return; }
    int r;
    if (s_vk_surface_lost) {
        s_vk_surface_lost = 0;
        r = vk_rebuild_surface() == 0 ? vk_create_swapchain() : -1;
    } else {
        r = vk_create_swapchain();
        if (r == -2) r = vk_rebuild_surface() == 0 ? vk_create_swapchain() : -1;
    }
    if (r == 0) {
        if (s_vk_lost_since_ms) VK_LOG("swapchain recovered after %.0f ms\n", vk_now_ms() - s_vk_lost_since_ms);
        s_vk_lost_since_ms = 0;
        s_vk_resized = 0;
        return;
    }
    s_vk_resized = 1;      /* 1: no drawable area yet (the surface is 0x0 before SDL reports the window minimized);
                            * <0: try again shortly */
    if (r > 0) return;
    const double now = vk_now_ms();
    if (!s_vk_lost_since_ms) {
        s_vk_lost_since_ms = now;
        VK_LOG("swapchain could not be recreated: retrying every 250 ms for up to 10 s\n");
    } else if (now - s_vk_lost_since_ms > 10000.0) {
        s_vk_lost_since_ms = 0;
        vk_window_drop("the swapchain could not be recreated");
    }
}

/* Where the frame goes in the swapchain image: the largest rectangle of the frame's aspect ratio, centred. The rest
 * of the image is black (letterbox or pillarbox). */
typedef struct { u32 x, y, w, h; } vk_rect;

static vk_rect vk_window_fit(void)
{
    const u32 dw = s_vk.sc_extent.width, dh = s_vk.sc_extent.height;
    vk_rect r = { 0, 0, dw, dh };
    if (!s_vk.width || !s_vk.height) return r;
    if ((u64)dw * s_vk.height > (u64)dh * s_vk.width) {
        r.w = (u32)(((u64)dh * s_vk.width + s_vk.height / 2) / s_vk.height);
        r.x = (dw - r.w) / 2;
    } else {
        r.h = (u32)(((u64)dw * s_vk.height + s_vk.width / 2) / s_vk.width);
        r.y = (dh - r.h) / 2;
    }
    return r;
}

/* Blit one source image (GENERAL layout) into acquired swapchain image `idx`
 * and present it. Returns the VkResult of vkQueuePresentKHR (or submit error). */
static VkResult vk_present_blit_slot(VkImage src_img, u32 idx, VkSemaphore wait_sem)
{
    VkCommandBuffer cb = s_vk.present_cmd;
    if (pvkResetCommandBuffer(cb, 0) != VK_SUCCESS) { VK_LOG("vkResetCommandBuffer (window) failed\n"); return VK_ERROR_UNKNOWN; }
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    if (pvkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) { VK_LOG("vkBeginCommandBuffer (window) failed\n"); return VK_ERROR_UNKNOWN; }
    vk_barrier_image_on(cb, src_img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = s_vk.sc_images[idx], .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    pvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          0, 0, NULL, 0, NULL, 1, &b);
    const vk_rect fit = vk_window_fit();
    if (fit.w != s_vk.sc_extent.width || fit.h != s_vk.sc_extent.height) {
        /* The bars. The clear and the blit write disjoint pixels, so they need no barrier between them. */
        static const VkClearColorValue black = { { 0.0f, 0.0f, 0.0f, 1.0f } };
        pvkCmdClearColorImage(cb, s_vk.sc_images[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1,
                              &b.subresourceRange);
    }
    VkImageBlit blit = {
        .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .srcOffsets = { { 0, 0, 0 }, { (int32_t)s_vk.width, (int32_t)s_vk.height, 1 } },
        .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .dstOffsets = { { (int32_t)fit.x, (int32_t)fit.y, 0 },
                        { (int32_t)(fit.x + fit.w), (int32_t)(fit.y + fit.h), 1 } },
    };
    pvkCmdBlitImage(cb, src_img, VK_IMAGE_LAYOUT_GENERAL,
                    s_vk.sc_images[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &blit, VK_FILTER_LINEAR);
    /* A dialog or toast is up: the overlay pass draws over the blitted frame and itself leaves the image in
     * PRESENT_SRC. Otherwise the plain transition, exactly as before. */
    if (!vk_ov_record(cb, idx, fit.x, fit.y, fit.w, fit.h)) {
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = 0;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        pvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                              0, 0, NULL, 0, NULL, 1, &b);
    }
    if (pvkEndCommandBuffer(cb) != VK_SUCCESS) {
        VK_LOG("vkEndCommandBuffer failed\n");
        s_ov.atlas_uploaded = 0;   /* the upload recorded above never ran */
        return VK_ERROR_UNKNOWN;
    }

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1, .pWaitSemaphores = &wait_sem,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1, .pCommandBuffers = &cb,
        .signalSemaphoreCount = 1, .pSignalSemaphores = &s_vk.sc_done[idx],
    };
    pvkResetFences(s_vk.device, 1, &s_vk.window_fence);
    if (vk_queue_submit(&si, s_vk.window_fence) != VK_SUCCESS) {
        VK_LOG("vkQueueSubmit (window) failed\n");
        s_ov.atlas_uploaded = 0;
        return VK_ERROR_UNKNOWN;
    }
    pvkWaitForFences(s_vk.device, 1, &s_vk.window_fence, VK_TRUE, UINT64_MAX);
    vk_ov_after_submit();

    VkPresentInfoKHR pi = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1, .pWaitSemaphores = &s_vk.sc_done[idx],
        .swapchainCount = 1, .pSwapchains = &s_vk.swapchain, .pImageIndices = &idx,
    };
    return vk_queue_present(&pi);
}

#ifdef PS3RECOMP_VK_FFX
static VkResult vk_fg_acquire_cb(void* ud, VkSemaphore sem, uint32_t* out_idx)
{
    (void)ud;
    return pvkAcquireNextImageKHR(s_vk.device, s_vk.swapchain, UINT64_MAX,
                                  sem, VK_NULL_HANDLE, out_idx);
}
#endif

/* Blit the offscreen target into the next swapchain image and present it.
 * When FSR 3.1.6 Frame Generation dispatched an interpolated frame, uses the
 * two-acquire FIFO generated->real present plan from ffx_vk_framegeneration_presenter.h. */
static void vk_present_window(void)
{
    if (!s_vk.windowed) return;
    /* PS3RECOMP_VK_TEST_LOST=<seconds> (verification aid, off by default): every N s the swapchain is declared stale, and
     * every second time the surface lost too, so the recovery paths run under validation without a real suspend. */
    {
        static int cfg = -1, n;
        static double next;
        if (cfg < 0) { const char* e = getenv("PS3RECOMP_VK_TEST_LOST"); cfg = e ? atoi(e) : 0; }
        if (cfg > 0) {
            const double t = vk_now_ms();
            if (!next) next = t + cfg * 1000.0;
            else if (t >= next) {
                next = t + cfg * 1000.0;
                s_vk_resized = 1;
                if (++n % 2 == 0) s_vk_surface_lost = 1;
                VK_LOG("test: swapchain declared %s (#%d)\n", s_vk_surface_lost ? "surface-lost" : "out of date", n);
            }
        }
    }
    if (!s_vk.swapchain && !s_vk_resized) return;
    if (vk_window_hidden()) return;
    if (s_vk_resized) {
        /* a streak of failed recreations (a resume still giving the window back) is retried every 250 ms */
        const double t = vk_now_ms();
        if (s_vk_lost_since_ms && t - s_vk_last_try_ms < 250.0) return;
        s_vk_last_try_ms = t;
        vk_swapchain_lost();
        if (!s_vk.windowed || s_vk_resized || !s_vk.swapchain) return;
    }
    { const vk_rect ofit = vk_window_fit(); vk_ov_frame(ofit.w, ofit.h); }

#ifdef PS3RECOMP_VK_FFX
    VkImage fg_img = VK_NULL_HANDLE;
    int fg_reset = 0;
    const int fg_disp = rsx_vk_fsr_fg_frame_ready(&fg_img, &fg_reset);
    if (ffxVkFrameGenerationTransitionNeedsQuiescence(s_vk.fg_was_active != 0, fg_disp != 0)) {
        vk_device_idle();
        s_vk.fg_was_active = fg_disp;
    }
    if (fg_disp && fg_img && s_vk.sc_count >= 2u && s_vk.sem_acquire_real) {
        FfxVkFrameGenerationAcquiredPair pair;
        const VkResult ar = ffxVkFrameGenerationAcquirePair(
            vk_fg_acquire_cb, NULL,
            s_vk.sem_acquire, s_vk.sem_acquire_real,
            s_vk.sc_count, &pair);
        if (!pair.generatedImageAcquired) {
            if (ar == VK_ERROR_SURFACE_LOST_KHR) s_vk_surface_lost = 1;
            if (ar == VK_ERROR_OUT_OF_DATE_KHR || ar == VK_ERROR_SURFACE_LOST_KHR) vk_swapchain_lost();
            else { VK_LOG("vkAcquireNextImageKHR (FG) failed (VkResult %d)\n", (int)ar); vk_swapchain_lost(); }
            return;
        }
        FfxVkFrameGenerationPresentPlan plan;
        if (!ffxVkFrameGenerationBuildPresentPlan(&pair, true, fg_reset != 0, &plan)) {
            vk_swapchain_lost();
            return;
        }
        int need_recreate = (ar == VK_ERROR_OUT_OF_DATE_KHR || ar == VK_SUBOPTIMAL_KHR);
        for (u32 s = 0; s < plan.slotCount; s++) {
            const FfxVkFrameGenerationPresentSlot* sl = &plan.slots[s];
            VkImage src = sl->useInterpolatedScene ? fg_img : s_vk.color.img;
            VkResult pr = vk_present_blit_slot(src, sl->imageIndex, sl->imageAvailableSemaphore);
            if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) {
                need_recreate = 1;
            } else if (pr != VK_SUCCESS) {
                s_vk.window_present_failures++;
                VK_LOG("vkQueuePresentKHR (FG slot %u) failed (VkResult %d)\n", s, (int)pr);
                if (pr == VK_ERROR_SURFACE_LOST_KHR) s_vk_surface_lost = 1;
                need_recreate = 1;   /* also resets the acquire semaphores a failed slot may have left signalled */
            } else {
                s_vk.window_presents++;
                if (sl->useInterpolatedScene) s_vk.fg_window_presents++;
                if (s_vk.window_presents == 1)
                    VK_LOG("window present: first frame submitted to the swapchain\n");
                if (s_vk.fg_window_presents == 1 && sl->useInterpolatedScene)
                    VK_LOG("window present: first FSR 3.1.6 generated->real pair submitted to swapchain\n");
            }
        }
        if (need_recreate) vk_swapchain_lost();
        return;
    }
#endif

    u32 idx = 0;
    VkResult r = pvkAcquireNextImageKHR(s_vk.device, s_vk.swapchain, UINT64_MAX,
                                        s_vk.sem_acquire, VK_NULL_HANDLE, &idx);
    if (r == VK_ERROR_SURFACE_LOST_KHR) s_vk_surface_lost = 1;
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR) { vk_swapchain_lost(); return; }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        VK_LOG("vkAcquireNextImageKHR failed (VkResult %d)\n", (int)r);
        vk_swapchain_lost();
        return;
    }

    r = vk_present_blit_slot(s_vk.color.img, idx, s_vk.sem_acquire);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) {
        vk_swapchain_lost();
        return;
    }
    if (r != VK_SUCCESS) {
        s_vk.window_present_failures++;
        VK_LOG("vkQueuePresentKHR failed (VkResult %d)\n", (int)r);
        if (r == VK_ERROR_SURFACE_LOST_KHR) s_vk_surface_lost = 1;
        vk_swapchain_lost();
        return;
    }
    s_vk.window_presents++;
    if (s_vk.window_presents == 1)
        VK_LOG("window present: first frame submitted to the swapchain\n");
}

static void vk_dump_ppm_ex(const unsigned char* rgba, const char* suffix)
{
    /* A %u in the path receives the presented-frame number, allowing an
     * intro sequence without changing the single-file harness behaviour. */
    char named[1024];
    const char* path = s_vk.dump_path;
    if (strstr(path, "%u")) {
        const int n = snprintf(named, sizeof named, path, s_vk.presented);
        if (n <= 0 || n >= (int)sizeof named) {
            VK_LOG("frame dump path is too long\n");
            return;
        }
        path = named;
    }
    char suffixed[1040];
    if (suffix && suffix[0]) {
        const char* dot = strrchr(path, '.');
        int n;
        if (dot)
            n = snprintf(suffixed, sizeof suffixed, "%.*s%s%s", (int)(dot - path), path, suffix, dot);
        else
            n = snprintf(suffixed, sizeof suffixed, "%s%s", path, suffix);
        if (n <= 0 || n >= (int)sizeof suffixed) return;
        path = suffixed;
    }
    FILE* f = fopen(path, "wb");
    if (!f) { VK_LOG("cannot open %s for the frame dump\n", path); return; }
    fprintf(f, "P6\n%u %u\n255\n", s_vk.width, s_vk.height);
    for (size_t i = 0; i < (size_t)s_vk.width * s_vk.height; i++)
        fwrite(rgba + i * 4, 1, 3, f);
    fclose(f);
}

static void vk_dump_ppm(const unsigned char* rgba)
{
    vk_dump_ppm_ex(rgba, NULL);
}

/* ---------------------------------------------------------------------------
 * Present
 *
 * In two halves. The first records, on the present's own command buffer,
 * whatever puts the frame into the offscreen target (the engine's blit of
 * the flipped surface) and the copy of that target into `readback`, and
 * submits it without waiting. The second waits for it, reads the centre
 * pixel and the frame dump back, and blits the target into the window.
 *
 * The first half reads the frame, so it must not run while the walker is
 * recording; the second touches only the target, `readback` and the
 * present's own objects, none of which the engine path draws into, so a host
 * can run it on a thread of its own beside the walker
 * (rsx_vulkan_backend_present_begin / _finish). On the fixed-function path
 * the walker draws into the target itself, so there the second half always
 * follows the first at once.
 * -------------------------------------------------------------------------*/

/* Wait until the GPU is done with a present's copy. Safe from any thread:
 * present_fence is reset only when a present starts, which never runs
 * alongside the walker. */
static void vk_present_wait_gpu(void)
{
    if (__atomic_load_n(&s_vk.present_inflight, __ATOMIC_ACQUIRE) && s_vk.present_fence)
        pvkWaitForFences(s_vk.device, 1, &s_vk.present_fence, VK_TRUE, UINT64_MAX);
}

/* present_mu held. */
static int dump_frame_named(u32 frame)
{
    for (u32 i = 0; i < s_vk.dump_n_frames; i++)
        if (s_vk.dump_frames[i] == frame) return 1;
    return 0;
}

static void vk_present_complete_locked(void)
{
    if (!s_vk.present_inflight) return;
    if (s_eng_stats) {
        const double w0 = vk_now_ms();
        pvkWaitForFences(s_vk.device, 1, &s_vk.present_fence, VK_TRUE, UINT64_MAX);
        const double dw = vk_now_ms() - w0;
        s_engtime.gpu_wait_ms += dw;
        if (g_rsx_sf.enabled) g_rsx_sf.vk_pres_wait_ms += dw;
    } else {
        pvkWaitForFences(s_vk.device, 1, &s_vk.present_fence, VK_TRUE, UINT64_MAX);
    }
    rsx_vk_fsr_retire_frame();
    __atomic_store_n(&s_vk.present_inflight, 0, __ATOMIC_RELEASE);

    vk_invalidate(&s_vk.readback);
    const unsigned char* px = (const unsigned char*)s_vk.readback.ptr;
    size_t c = ((size_t)(s_vk.height / 2) * s_vk.width + s_vk.width / 2) * 4;
    s_vk.last_center = ((u32)px[c] << 16) | ((u32)px[c + 1] << 8) | (u32)px[c + 2];
    s_vk.presented++;
    s_vk.last_draws = s_vk.pending_draws;
    s_vk.draw_total += s_vk.last_draws;
    if (s_vk.last_draws) s_vk.rendered_frames++;
    s_vk.last_present_ms = (u64)vk_now_ms();
    if (s_vk.dump_path && (s_vk.dump_n_frames ? dump_frame_named(s_vk.presented)
                                              : (s_vk.presented % s_vk.dump_every) == 0)) {
        vk_dump_ppm(px);
        if (s_vk.fg_readback_ready && s_vk.fg_readback.ptr) {
            vk_invalidate(&s_vk.fg_readback);
            vk_dump_ppm_ex((const unsigned char*)s_vk.fg_readback.ptr, "-fg");
        }
    }
    s_vk.fg_readback_ready = 0;
    vk_present_window();
    vk_frame_telemetry(s_vk.last_draws);
}

/* present_mu held: a previous present finished, the present's command
 * buffer begun and ordered after everything submitted so far. */
static VkCommandBuffer vk_present_record_begin(void)
{
    vk_present_complete_locked();
    VkCommandBuffer cb = s_vk.present_cmd;
    if (pvkResetCommandBuffer(cb, 0) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    if (pvkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) return VK_NULL_HANDLE;
    vk_cmd_after_everything(cb);
    return cb;
}

/* present_mu held: copy the target out, submit, and finish at once unless a
 * host asked for the halves separately on the engine path. */
static void vk_present_record_end(VkCommandBuffer cb)
{
    const u32 next_pres = s_vk.presented + 1u;
    const int dump_now = s_vk.dump_path &&
        (s_vk.dump_n_frames ? dump_frame_named(next_pres)
                            : (next_pres % s_vk.dump_every) == 0);
    s_vk.fg_readback_ready = 0;
    vk_barrier_image_on(cb, s_vk.color.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    VkBufferImageCopy region = {
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .imageExtent = { s_vk.width, s_vk.height, 1 },
    };
    pvkCmdCopyImageToBuffer(cb, s_vk.color.img, VK_IMAGE_LAYOUT_GENERAL,
                            s_vk.readback.buf, 1, &region);
    VkBufferMemoryBarrier host = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = s_vk.readback.buf, .offset = 0, .size = VK_WHOLE_SIZE,
    };
    pvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                          0, 0, NULL, 1, &host, 0, NULL);
    if (dump_now) {
        VkImage fg_img = VK_NULL_HANDLE;
        int fg_reset = 0;
        if (rsx_vk_fsr_fg_frame_ready(&fg_img, &fg_reset) && !fg_reset && fg_img) {
            const VkDeviceSize sz = (VkDeviceSize)s_vk.width * s_vk.height * 4u;
            if (!s_vk.fg_readback.buf)
                (void)vk_create_host_buffer(&s_vk.fg_readback, sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            if (s_vk.fg_readback.buf) {
                vk_barrier_image_on(cb, fg_img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
                pvkCmdCopyImageToBuffer(cb, fg_img, VK_IMAGE_LAYOUT_GENERAL,
                                        s_vk.fg_readback.buf, 1, &region);
                host.buffer = s_vk.fg_readback.buf;
                pvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                      0, 0, NULL, 1, &host, 0, NULL);
                s_vk.fg_readback_ready = 1;
            }
        }
    }
    if (pvkEndCommandBuffer(cb) != VK_SUCCESS) { VK_LOG("vkEndCommandBuffer (present) failed\n"); return; }
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1, .pCommandBuffers = &cb };
    pvkResetFences(s_vk.device, 1, &s_vk.present_fence);
    if (vk_queue_submit(&si, s_vk.present_fence) != VK_SUCCESS) {
        VK_LOG("vkQueueSubmit (present) failed\n"); return;
    }
    s_vk.pending_draws = rsx_vulkan_backend_guest_draws();
    __atomic_store_n(&s_vk.present_inflight, 1, __ATOMIC_RELEASE);
    s_vk.present_order = 1;
    const int async_pres = s_vk.eng_active && vk_eb_async_use() && !dump_now;
    if (!(s_vk.present_split && s_vk.eng_active) && !async_pres)
        vk_present_complete_locked();
}

static void vk_cb_present(void* ud, u32 buffer_id)
{
    (void)ud; (void)buffer_id;
    if (!s_vk.device) return;
    pthread_mutex_lock(&s_vk.present_mu);
    VkCommandBuffer cb = vk_present_record_begin();
    if (cb) vk_present_record_end(cb);
    pthread_mutex_unlock(&s_vk.present_mu);
    s_g.last_draws = s_g.draws;
    s_g.draws = 0;
}

/* ---------------------------------------------------------------------------
 * Register-file draw engine backend (rsx_draw_engine.h)
 *
 * The path the Metal backend takes: the shared engine walks the register
 * file, owns surfaces, texture caching, vertex compaction and pipeline keys,
 * and drives this backend through rsx_draw_backend. Selected with
 * PS3RECOMP_RSX_ENGINE=dispatch; only one of the two paths is registered at a
 * time, since the FIFO walker feeds both and would record every draw twice.
 *
 * Every operation is still one synchronous submission, so submit_and_wait has
 * nothing left to do by the time the engine calls it -- which is exactly the
 * contract it asks for (staging is free to reuse on return).
 *
 * Stage E1: colour and depth targets, clears, present and readback.
 * Pipelines, draws and textures are the next stages; until then
 * pipeline_create reports 0, which the engine caches as "cannot build".
 * -------------------------------------------------------------------------*/
#define VK_ENG_MAX_OBJ 2048          /* the engine's texture cache is 1024 */

enum { VK_ENG_FREE = 0, VK_ENG_COLOR, VK_ENG_DEPTH, VK_ENG_TEXTURE, VK_ENG_VIEW };

typedef struct vk_eng_obj {
    u8       kind;
    VkFormat fmt;
    u32      bpp;                    /* bytes per texel, 0 for depth     */
    u32      w, h;
    vk_image im;                     /* a VIEW owns only im.view            */
    u32      block;                  /* bytes per 4x4 block, 0 = not BC     */
    u32      surface, remap, rsx_fmt;/* VIEW: what it is a view of          */
    u8       cube;                   /* TEXTURE: six layers, a CUBE view    */
    u8       snapshot;               /* COLOR: 1 feedback copy, 2 zconv target */
    u32      mips;                   /* TEXTURE: levels the image was created with */
    VkImageView   depth_srv, stencil_srv; /* DEPTH: single-aspect sampled views */
    u32           color_snap;        /* DEPTH: A8R8G8B8 D24S8 snapshot handle */
    VkFramebuffer zconv_fb;          /* COLOR (snapshot==2): 1-RT zconv FB  */
} vk_eng_obj;

static vk_eng_obj s_eobj[VK_ENG_MAX_OBJ];
#define VK_ENG_MAX_VIEWS 128u
static u32 s_eview[VK_ENG_MAX_VIEWS];
static u32 s_eview_count;

/* Bumped by every object release: the batched-descriptor cache names views
 * that a release can destroy even when the handle survives in the round-robin
 * table. Declared here because vk_eng_release bumps it and the cache is built
 * far below. */
static u32 s_obj_gen;
static u8  s_obj_in_desc[VK_ENG_MAX_OBJ];
static void vk_eb_desc_invalidate_handle(u32 handle);
static int vk_m2e4_use(void);

/* Defined with the batch state below; a clear records that it wrote an image. */
static void vk_eb_touch(VkImage img);
static void vk_eb_touch_clear(void);
static void vk_eb_end_rp(void);
static void vk_zconv_reset_sets(void);
static void vk_zconv_shutdown(void);

/* Defined with the E2 pipeline state further down. */
static void vk_eng_drop_framebuffers(void);
static void vk_eng2_shutdown(void* user);

static struct {
    u32 rt[RSX_BE_MAX_COLOR_TARGETS], nrt, depth;
} s_eng_bound;

static VkFormat vk_eng_format(rsx_be_format f, u32* bpp)
{
    switch (f) {
    case RSX_BE_FMT_R8:            *bpp = 1;  return VK_FORMAT_R8_UNORM;
    case RSX_BE_FMT_R8G8:          *bpp = 2;  return VK_FORMAT_R8G8_UNORM;
    case RSX_BE_FMT_R8G8B8A8:      *bpp = 4;  return VK_FORMAT_R8G8B8A8_UNORM;
    case RSX_BE_FMT_R16:           *bpp = 2;  return VK_FORMAT_R16_UNORM;
    case RSX_BE_FMT_R16G16:        *bpp = 4;  return VK_FORMAT_R16G16_UNORM;
    case RSX_BE_FMT_R16G16F:       *bpp = 4;  return VK_FORMAT_R16G16_SFLOAT;
    case RSX_BE_FMT_R16G16B16A16F: *bpp = 8;  return VK_FORMAT_R16G16B16A16_SFLOAT;
    case RSX_BE_FMT_R32F:          *bpp = 4;  return VK_FORMAT_R32_SFLOAT;
    case RSX_BE_FMT_R32G32B32A32F: *bpp = 16; return VK_FORMAT_R32G32B32A32_SFLOAT;
    case RSX_BE_FMT_BC1:           *bpp = 8;  return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;  /* per block */
    case RSX_BE_FMT_BC2:           *bpp = 16; return VK_FORMAT_BC2_UNORM_BLOCK;
    case RSX_BE_FMT_BC3:           *bpp = 16; return VK_FORMAT_BC3_UNORM_BLOCK;
    default:                       *bpp = 0;  return VK_FORMAT_UNDEFINED;
    }
}

/* Colour formats the E2 passes render into: RGBA8 for ordinary and display
 * targets, RGBA16F for SET_SURFACE_FORMAT 0xB HDR targets, and R32F / RGBA32F
 * for the float targets a title renders small reductions into. All four are
 * required colour attachment formats in Vulkan 1.0; blending is guaranteed only
 * for the first two (see vk_eng_rt_blendable). */
#define VK_ENG_RT_FORMATS 4
static int vk_eng_rt_index(VkFormat f)
{
    switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM:      return 0;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return 1;
    case VK_FORMAT_R32_SFLOAT:          return 2;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return 3;
    default:                            return -1;
    }
}

/* Whether the device blends into colour format `f` (guaranteed for RGBA8 and
 * RGBA16F, optional for 32-bit float formats). */
static int vk_eng_rt_blendable(VkFormat f)
{
    if (f == VK_FORMAT_R8G8B8A8_UNORM || f == VK_FORMAT_R16G16B16A16_SFLOAT) return 1;
    VkFormatProperties fp;
    pvkGetPhysicalDeviceFormatProperties(s_vk.phys, f, &fp);
    return (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) != 0;
}

/* The largest image side the engine creates: the device's own limit once a
 * device is picked, the 4096 Vulkan guarantees before. */
static u32 vk_max_dim(void) { return s_vk.max_dim ? s_vk.max_dim : 4096u; }

static u32 vk_eng_alloc(void)
{
    /* Hand out slots round-robin rather than lowest-first, so a handle the
     * engine releases is not immediately reused: a late or duplicated release
     * of an old handle then finds a free slot instead of a live object. */
    static u32 cursor;
    for (u32 step = 0; step < VK_ENG_MAX_OBJ - 1; step++) {
        const u32 i = 1 + (cursor + step) % (VK_ENG_MAX_OBJ - 1);
        if (s_eobj[i].kind == VK_ENG_FREE) {
            cursor = i % (VK_ENG_MAX_OBJ - 1);
            return i;
        }
    }
    VK_LOG("engine: object table full (%d)\n", VK_ENG_MAX_OBJ);
    return 0;
}

static vk_eng_obj* vk_eng_get(u32 handle, int kind)
{
    if (!handle || handle >= VK_ENG_MAX_OBJ) return NULL;
    vk_eng_obj* o = &s_eobj[handle];
    return (o->kind == kind) ? o : NULL;
}

/* Host staging kept across readbacks and uploads. One buffer allocated,
 * mapped and freed per call is three driver round trips, and on Windows
 * those took from tens of milliseconds to seconds each under load. Each
 * grows to the largest request and is reused after its submission
 * completes; every user waits for that before returning. */
static vk_buffer s_rb_stage, s_up_stage;
static vk_buffer* vk_stage_get(vk_buffer* b, VkDeviceSize size, VkBufferUsageFlags usage)
{
    if (b->buf && b->size >= size) return b;
    if (b->buf) vk_destroy_buffer(b);
    VkDeviceSize cap = (VkDeviceSize)1 << 20;
    while (cap < size) cap *= 2u;
    if (vk_create_host_buffer(b, cap, usage)) { vk_destroy_buffer(b); return NULL; }
    return b;
}

/* Copy `rows` rows of `w` texels from host memory into a w x h rectangle at
 * (x, y) of one level / layer of an image that is in GENERAL. */
static int vk_eng_upload_region(VkImage img, u32 layer, u32 mip, u32 x, u32 y,
                                u32 w, u32 h, size_t tight, u32 rows,
                                const void* src, u32 row_bytes)
{
    if (tight > row_bytes) tight = row_bytes;
    {   /* VK_DRAW_BATCH: staged in a ring and recorded into the open batch. */
        VkBufferImageCopy region = {
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1 },
            .imageOffset = { (int32_t)x, (int32_t)y, 0 },
            .imageExtent = { w, h, 1 },
        };
        const int b = vk_eb_upload(img, &region, src, tight, rows, row_bytes);
        if (b != -2) return b;
    }
    /* Unbatched: #552's reused upload staging rather than a buffer created and
     * destroyed per region. */
    vk_buffer* const st = vk_stage_get(&s_up_stage, (VkDeviceSize)tight * rows, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    if (!st || !st->coherent) return -1;
    for (u32 y = 0; y < rows; y++)
        memcpy((u8*)st->ptr + (size_t)y * tight, (const u8*)src + (size_t)y * row_bytes, tight);
    int rc = vk_begin();
    if (!rc) {
        vk_barrier_image(img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
        VkBufferImageCopy region = {
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1 },
            .imageOffset = { (int32_t)x, (int32_t)y, 0 },
            .imageExtent = { w, h, 1 },
        };
        pvkCmdCopyBufferToImage(s_vk.cmd, st->buf, img, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        rc = vk_submit_and_wait();
    }
    return rc;
}

/* The same into the whole of a level, from its origin. */
static int vk_eng_upload_rows(VkImage img, u32 layer, u32 mip, u32 w, u32 h,
                              size_t tight, u32 rows, const void* src, u32 row_bytes)
{
    return vk_eng_upload_region(img, layer, mip, 0, 0, w, h, tight, rows, src, row_bytes);
}

static int vk_eng_upload(VkImage img, u32 layer, u32 mip, u32 w, u32 h, u32 bpp,
                         const void* src, u32 row_bytes)
{
    return vk_eng_upload_rows(img, layer, mip, w, h, (size_t)w * bpp, h, src, row_bytes);
}

/* NV4097 TEXTURE_CONTROL1 crossbar as a Vulkan component mapping: the shared
 * rsx_texture_component_remap() gives, in A,R,G,B order, which uploaded
 * channel (0..3 = R,G,B,A) or constant each output takes -- the table the
 * Metal backend's swizzle_sel reads. */
static VkComponentSwizzle vk_remap_sel(u8 sel)
{
    switch (sel) {
    case 0: return VK_COMPONENT_SWIZZLE_R;
    case 1: return VK_COMPONENT_SWIZZLE_G;
    case 2: return VK_COMPONENT_SWIZZLE_B;
    case 3: return VK_COMPONENT_SWIZZLE_A;
    case RSX_REMAP_ONE: return VK_COMPONENT_SWIZZLE_ONE;
    default: return VK_COMPONENT_SWIZZLE_ZERO;
    }
}
static VkComponentMapping vk_remap(u32 remap, u32 rsx_fmt)
{
    u8 sel[4];
    rsx_texture_component_remap(remap, rsx_fmt & 0x9Fu, sel);
    VkComponentMapping m = { vk_remap_sel(sel[1]), vk_remap_sel(sel[2]),
                             vk_remap_sel(sel[3]), vk_remap_sel(sel[0]) };
    return m;
}
static int vk_eng_make_view(VkImage img, VkFormat fmt, u32 levels, VkComponentMapping map,
                            int cube, VkImageView* out)
{
    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
        .viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D,
        .format = fmt, .components = map,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels ? levels : 1, 0, cube ? 6u : 1u },
    };
    return pvkCreateImageView(s_vk.device, &vci, NULL, out) == VK_SUCCESS ? 0 : -1;
}

static int vk_eng_init(void* user, u32 width, u32 height)
{
    (void)user; (void)width; (void)height;
    memset(s_eobj, 0, sizeof s_eobj);
    s_eview_count = 0;
    memset(&s_eng_bound, 0, sizeof s_eng_bound);
    return s_vk.device ? 0 : -1;
}

static void vk_eng_snapshot_recycle(vk_eng_obj* o);

static void vk_eng_invalidate_rt_cache(void);
static void vk_eng_release(void* user, u32 handle)
{
    (void)user;
    if (!handle || handle >= VK_ENG_MAX_OBJ || s_eobj[handle].kind == VK_ENG_FREE) return;
    const int m2e4 = vk_m2e4_use();
    if (m2e4) vk_eb_desc_invalidate_handle(handle);
    else s_obj_gen++;                    /* cached descriptor sets name the views */
    if (s_eobj[handle].kind == VK_ENG_COLOR || s_eobj[handle].kind == VK_ENG_DEPTH)
        vk_eng_invalidate_rt_cache();
    if (s_eobj[handle].kind == VK_ENG_COLOR) {
        u32 w = 0;
        for (u32 k = 0; k < s_eview_count; k++) {
            const u32 vi = s_eview[k];
            if (s_eobj[vi].kind == VK_ENG_VIEW && s_eobj[vi].surface == handle) {
                if (m2e4) vk_eb_desc_invalidate_handle(vi);
                vk_destroy_image(&s_eobj[vi].im);
                memset(&s_eobj[vi], 0, sizeof s_eobj[vi]);
            } else {
                s_eview[w++] = vi;
            }
        }
        s_eview_count = w;
    }
    if (s_eobj[handle].kind == VK_ENG_DEPTH) {
        if (s_eobj[handle].surface) {
            const u32 snap = s_eobj[handle].surface;
            s_eobj[handle].surface = 0;
            if (vk_eng_get(snap, VK_ENG_TEXTURE)) vk_eng_release(user, snap);
        }
        if (s_eobj[handle].color_snap) {
            const u32 csnap = s_eobj[handle].color_snap;
            s_eobj[handle].color_snap = 0;
            if (vk_eng_get(csnap, VK_ENG_COLOR)) vk_eng_release(user, csnap);
        }
        if (s_eobj[handle].depth_srv || s_eobj[handle].stencil_srv) {
            vk_zconv_reset_sets();
            if (s_eb_pending || vk_eb_any_inflight()) vk_eng_flush();
            vk_present_wait_gpu();
            if (s_eobj[handle].depth_srv) pvkDestroyImageView(s_vk.device, s_eobj[handle].depth_srv, NULL);
            if (s_eobj[handle].stencil_srv) pvkDestroyImageView(s_vk.device, s_eobj[handle].stencil_srv, NULL);
            s_eobj[handle].depth_srv = VK_NULL_HANDLE;
            s_eobj[handle].stencil_srv = VK_NULL_HANDLE;
        }
    }
    /* A colour target's feedback copy (see vk_eng_color_snapshot) goes with
     * it: the engine binds the copy but never releases it. */
    if (s_eobj[handle].kind == VK_ENG_COLOR && !s_eobj[handle].snapshot &&
        s_eobj[handle].surface) {
        const u32 copy = s_eobj[handle].surface;
        s_eobj[handle].surface = 0;
        if (vk_eng_get(copy, VK_ENG_COLOR)) vk_eng_release(user, copy);
    }
    if (s_eobj[handle].kind == VK_ENG_COLOR && s_eobj[handle].zconv_fb) {
        if (s_eb_pending || vk_eb_any_inflight()) vk_eng_flush();
        vk_present_wait_gpu();
        pvkDestroyFramebuffer(s_vk.device, s_eobj[handle].zconv_fb, NULL);
        s_eobj[handle].zconv_fb = VK_NULL_HANDLE;
    }
    if (s_eobj[handle].kind == VK_ENG_COLOR && s_eobj[handle].snapshot == 1) {
        /* Never an attachment, so no framebuffer names it; the image goes
         * back to the pool for the next feedback draw. */
        vk_eng_snapshot_recycle(&s_eobj[handle]);
        memset(&s_eobj[handle], 0, sizeof s_eobj[handle]);
        return;
    }
    if (s_eobj[handle].kind == VK_ENG_TEXTURE && vk_eb_active() && vk_eb_async_use() &&
        vk_eb_retire_image(&s_eobj[handle].im)) {
        memset(&s_eobj[handle], 0, sizeof s_eobj[handle]);
        return;
    }
    if (s_eobj[handle].kind == VK_ENG_COLOR || s_eobj[handle].kind == VK_ENG_DEPTH)
        vk_eng_drop_framebuffers();
    vk_destroy_image(&s_eobj[handle].im);      /* idle: submissions are waited on */
    memset(&s_eobj[handle], 0, sizeof s_eobj[handle]);
}

/* The engine releases each object through the call for its kind: colour
 * targets through color_target_release, depth targets through
 * depth_target_release, and textures, views and colour feedback copies
 * through texture_release. Handles are slots that get reused, so a release
 * naming an object of another kind can only be a stale handle, and freeing
 * it would destroy the live object that now owns the slot. It is refused and
 * reported instead. Free slots pass through: vk_eng_release ignores them. */
static int vk_eng_release_kind_ok(u32 handle, int want_color, int want_depth)
{
    if (!handle || handle >= VK_ENG_MAX_OBJ) return 1;
    const vk_eng_obj* o = &s_eobj[handle];
    if (o->kind == VK_ENG_FREE) return 1;
    const int color = o->kind == VK_ENG_COLOR && !o->snapshot, depth = o->kind == VK_ENG_DEPTH;
    const int ok = want_color ? color : want_depth ? depth : (!color && !depth);
    if (!ok) {
        static u32 logs;
        if (logs++ < 16)
            VK_LOG("engine: stale %s release of object %u (kind %d%s, %ux%u) refused\n",
                   want_color ? "colour target" : want_depth ? "depth target" : "texture", handle,
                   (int)o->kind, o->snapshot ? " snapshot" : "", o->w, o->h);
    }
    return ok;
}
static void vk_eng_texture_release(void* user, u32 handle)
{
    if (vk_eng_release_kind_ok(handle, 0, 0)) vk_eng_release(user, handle);
}
static void vk_eng_color_target_release(void* user, u32 handle)
{
    if (vk_eng_release_kind_ok(handle, 1, 0)) vk_eng_release(user, handle);
}
static void vk_eng_depth_target_release(void* user, u32 handle)
{
    if (vk_eng_release_kind_ok(handle, 0, 1)) vk_eng_release(user, handle);
}

static void vk_eng_snapshot_pool_drain(void);

static void vk_eng_shutdown(void* user)
{
    if (s_eb_pending || vk_eb_any_inflight()) vk_eng_flush();
    pthread_mutex_lock(&s_vk.present_mu);
    vk_present_complete_locked();
    pthread_mutex_unlock(&s_vk.present_mu);
    vk_eng2_shutdown(user);
    for (u32 i = 1; i < VK_ENG_MAX_OBJ; i++) vk_eng_release(user, i);
    vk_eng_snapshot_pool_drain();
    vk_eb_release();
    vk_destroy_buffer(&s_rb_stage);
    vk_destroy_buffer(&s_up_stage);
}

static void vk_eng_submit_and_wait(void* user, u32 reason)
{
    (void)user; (void)reason;
    if (s_eb_pending || vk_eb_any_inflight()) vk_eng_flush();
}

static u32 vk_eng_color_target_create(void* user, rsx_be_format fmt, u32 w, u32 h,
                                      const void* seed, u32 seed_row_bytes)
{
    (void)user;
    u32 bpp;
    const VkFormat vf = vk_eng_format(fmt, &bpp);
    /* Only formats the E2 passes are built for (vk_eng_rt_index). */
    if (vk_eng_rt_index(vf) < 0 || !w || !h || w > vk_max_dim() || h > vk_max_dim()) {
        VK_LOG("engine: colour target format %d %ux%u not supported yet\n", (int)fmt, w, h);
        return 0;
    }
    const u32 hd = vk_eng_alloc();
    if (!hd) return 0;
    vk_eng_obj* o = &s_eobj[hd];
    if (vk_create_image(&o->im, vf, w, h,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT)) {
        vk_destroy_image(&o->im); return 0;
    }
    o->kind = VK_ENG_COLOR; o->fmt = vf; o->bpp = bpp; o->w = w; o->h = h;
    /* UNDEFINED -> GENERAL, then the guest's own bytes when the engine could
     * resolve them, else transparent black so the contents are defined. */
    if (vk_begin()) { vk_eng_release(user, hd); return 0; }
    vk_barrier_image(o->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
    VkClearColorValue zero = { .float32 = { 0.0f, 0.0f, 0.0f, 0.0f } };
    VkImageSubresourceRange r = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    pvkCmdClearColorImage(s_vk.cmd, o->im.img, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &r);
    if (vk_submit_and_wait()) { vk_eng_release(user, hd); return 0; }
    if (seed && seed_row_bytes)
        vk_eng_upload(o->im.img, 0, 0, w, h, bpp, seed, seed_row_bytes);
    return hd;
}

static int vk_depth_sampleable(void)
{
    VkFormatProperties fp;
    pvkGetPhysicalDeviceFormatProperties(s_vk.phys, s_vk.depth_format, &fp);
    return (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
}

static u32 vk_eng_depth_target_create(void* user, u32 w, u32 h)
{
    (void)user;
    if (!w || !h || w > vk_max_dim() || h > vk_max_dim()) return 0;
    const u32 hd = vk_eng_alloc();
    if (!hd) return 0;
    vk_eng_obj* o = &s_eobj[hd];
    const int sampleable = vk_depth_sampleable();
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        (sampleable ? VK_IMAGE_USAGE_SAMPLED_BIT : 0u);
    if (vk_create_image(&o->im, s_vk.depth_format, w, h, usage, s_vk.depth_aspect)) {
        vk_destroy_image(&o->im); return 0;
    }
    o->kind = VK_ENG_DEPTH; o->fmt = s_vk.depth_format; o->w = w; o->h = h;
    if (sampleable) {
        /* Vulkan requires sampled views of a combined depth/stencil image to
         * select exactly one aspect bit (depth or stencil, never both). */
        VkImageViewCreateInfo dv = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = o->im.img,
            .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = s_vk.depth_format,
            .subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 },
        };
        if (pvkCreateImageView(s_vk.device, &dv, NULL, &o->depth_srv) != VK_SUCCESS)
            o->depth_srv = VK_NULL_HANDLE;
        if (s_vk.depth_aspect & VK_IMAGE_ASPECT_STENCIL_BIT) {
            VkImageViewCreateInfo sv = dv;
            sv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
            if (pvkCreateImageView(s_vk.device, &sv, NULL, &o->stencil_srv) != VK_SUCCESS)
                o->stencil_srv = VK_NULL_HANDLE;
        }
    }
    /* Far plane, for the reason the display depth starts there. */
    if (vk_begin()) { vk_eng_release(user, hd); return 0; }
    vk_barrier_image(o->im.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_UNDEFINED);
    VkClearDepthStencilValue far_plane = { 1.0f, 0 };
    VkImageSubresourceRange r = { s_vk.depth_aspect, 0, 1, 0, 1 };
    pvkCmdClearDepthStencilImage(s_vk.cmd, o->im.img, VK_IMAGE_LAYOUT_GENERAL, &far_plane, 1, &r);
    if (vk_submit_and_wait()) { vk_eng_release(user, hd); return 0; }
    return hd;
}

static void vk_eng_bind_targets(void* user, const u32* surfaces, u32 count, u32 depth)
{
    (void)user;
    if (count > RSX_BE_MAX_COLOR_TARGETS) count = RSX_BE_MAX_COLOR_TARGETS;
    for (u32 i = 0; i < RSX_BE_MAX_COLOR_TARGETS; i++)
        s_eng_bound.rt[i] = (i < count && surfaces) ? surfaces[i] : 0;
    s_eng_bound.nrt = count;
    s_eng_bound.depth = depth;
    vk_eng_invalidate_rt_cache();
    vk_eb_inval_fast_rp();
}

static void vk_eng_clear_color(void* user, u32 surface, const float rgba[4])
{
    (void)user;
    /* The debug hook reports the last colour asked for, as Metal's does. */
    s_vk.clear_argb = ((u32)(rgba[3] * 255.0f + 0.5f) << 24) |
                      ((u32)(rgba[0] * 255.0f + 0.5f) << 16) |
                      ((u32)(rgba[1] * 255.0f + 0.5f) <<  8) |
                       (u32)(rgba[2] * 255.0f + 0.5f);
    vk_eng_obj* o = vk_eng_get(surface, VK_ENG_COLOR);
    if (!o) return;
    const int batched = vk_eb_active();
    if (batched && vk_rec_use_batch()) {
        if (!s_eb_pending && vk_rec_ensure_open()) return;
        vk_rec_emit_clear_color(o->im.img, rgba);
        vk_eb_recorded();
        return;
    }
    if (batched ? (!s_eb_pending && vk_eb_open_cmd()) : vk_begin()) return;
    if (batched) vk_eb_end_rp();
    VkClearColorValue v;
    memcpy(v.float32, rgba, sizeof v.float32);
    VkImageSubresourceRange r = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vk_barrier_image(o->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    pvkCmdClearColorImage(s_vk.cmd, o->im.img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
    if (batched) { vk_eb_touch(o->im.img); vk_eb_recorded(); }
    else vk_submit_and_wait();
}

static void vk_eng_clear_depth_stencil(void* user, u32 depth, u32 flags,
                                       float depth_value, u8 stencil)
{
    (void)user;
    vk_eng_obj* o = vk_eng_get(depth, VK_ENG_DEPTH);
    VkImageAspectFlags aspect = 0;
    if (flags & RSX_BE_CLEAR_DEPTH) aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (flags & RSX_BE_CLEAR_STENCIL) aspect |= s_vk.depth_aspect & VK_IMAGE_ASPECT_STENCIL_BIT;
    if (!o || !aspect) return;
    const int batched = vk_eb_active();
    if (batched && vk_rec_use_batch()) {
        if (!s_eb_pending && vk_rec_ensure_open()) return;
        vk_rec_emit_clear_depth(o->im.img, aspect, depth_value, stencil);
        vk_eb_recorded();
        return;
    }
    if (batched ? (!s_eb_pending && vk_eb_open_cmd()) : vk_begin()) return;
    if (batched) vk_eb_end_rp();
    VkClearDepthStencilValue d = { depth_value, stencil };
    VkImageSubresourceRange r = { aspect, 0, 1, 0, 1 };
    vk_barrier_image(o->im.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
    pvkCmdClearDepthStencilImage(s_vk.cmd, o->im.img, VK_IMAGE_LAYOUT_GENERAL, &d, 1, &r);
    if (batched) { vk_eb_touch(o->im.img); vk_eb_recorded(); }
    else vk_submit_and_wait();
}

static void vk_eng_dump_surfaces(void);

static u64 vk_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64)ts.tv_sec * 1000000000ull + (u64)ts.tv_nsec;
}

/* Built-in frame-rate limiter (FPS_CAP=<n>, 0 = uncapped). Runs right before
 * present after the frame's draws have been submitted so the GPU can finish
 * while the CPU sleeps, keeping FPS_CAP=30 + FSR_FG=1 on a steady 60 Hz pace. */
static void vk_fps_cap_wait(void)
{
    static int s_init = 0;
    static u64 s_period_ns = 0;
    static u64 s_next_ns = 0;
    if (__builtin_expect(!s_init, 0)) {
        const char* e = getenv("FPS_CAP");
        const double cap = (e && *e) ? atof(e) : 0.0;
        if (cap >= 1.0 && cap <= 1000.0) {
            s_period_ns = (u64)(1000000000.0 / cap + 0.5);
            VK_LOG("FPS_CAP=%.1f (%.2f ms/frame)\n", cap, (double)s_period_ns / 1000000.0);
        }
        s_init = 1;
    }
    if (!s_period_ns) return;
    u64 now = vk_now_ns();
    if (!s_next_ns) {
        s_next_ns = now + s_period_ns;
        return;
    }
    while (now < s_next_ns) {
        const u64 rem = s_next_ns - now;
        if (rem > 1500000ull) {
            const u64 sl = rem - 1200000ull;
            struct timespec ts = { (time_t)(sl / 1000000000ull), (long)(sl % 1000000000ull) };
            nanosleep(&ts, NULL);
        } else {
#if defined(__x86_64__) || defined(__i386__)
            __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
            __asm__ volatile("yield" ::: "memory");
#endif
        }
        now = vk_now_ns();
    }
    s_next_ns += s_period_ns;
    if (now > s_next_ns + s_period_ns)
        s_next_ns = now + s_period_ns;
}

/* The named surface becomes the frame: blit it onto the display image, then
 * the existing present does the readback, the PPM dump and the window. */
static void vk_eng_present(void* user, u32 surface)
{
    (void)user;
    if (!s_vk.device) return;
    if (s_eb_pending) vk_eng_flush_async();   /* submits on s_vk.queue before present blit */
    vk_rec_wait_submitted();                  /* ensure async recorder submitted before present blit */
    vk_fps_cap_wait();
    vk_eng_obj* o = vk_eng_get(surface, VK_ENG_COLOR);
    pthread_mutex_lock(&s_vk.present_mu);
    VkCommandBuffer cb = vk_present_record_begin();
    if (cb) {
        const int fsr_ran = (o && rsx_vk_fsr_active())
            ? rsx_vk_fsr_frame_end(cb, o->im.img, o->fmt, o->w, o->h,
                                   s_vk.color.img, s_vk.width, s_vk.height)
            : 0;
        if (o && !fsr_ran) {
            vk_barrier_image_on(cb, o->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
            vk_barrier_image_on(cb, s_vk.color.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
            VkImageBlit blit = {
                .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                .srcOffsets = { { 0, 0, 0 }, { (int32_t)o->w, (int32_t)o->h, 1 } },
                .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                .dstOffsets = { { 0, 0, 0 }, { (int32_t)s_vk.width, (int32_t)s_vk.height, 1 } },
            };
            pvkCmdBlitImage(cb, o->im.img, VK_IMAGE_LAYOUT_GENERAL,
                            s_vk.color.img, VK_IMAGE_LAYOUT_GENERAL, 1, &blit,
                            (o->w == s_vk.width && o->h == s_vk.height) ? VK_FILTER_NEAREST
                                                                        : VK_FILTER_LINEAR);
        }
        vk_present_record_end(cb);
    }
    pthread_mutex_unlock(&s_vk.present_mu);
    s_g.last_draws = s_g.draws;
    s_g.draws = 0;
    static u32 debug_present;
    static u32 debug_every;
    /* The frames named by PS3RECOMP_VK_DUMP_SURFACES_FRAME (a comma-separated
     * list), or every debug_every-th frame when none are named. */
    static u32 debug_frames[8];
    static u32 debug_n_frames;
    static int debug_frame_read;
    if (!debug_every) {
        const char* e = getenv("PS3RECOMP_VK_DUMP_SURFACES_EVERY");
        debug_every = (e && atoi(e) > 0) ? (u32)atoi(e) : 600u;
    }
    if (!debug_frame_read) {
        const char* e = getenv("PS3RECOMP_VK_DUMP_SURFACES_FRAME");
        while (e && *e && debug_n_frames < 8u) {
            char* end;
            const long v = strtol(e, &end, 10);
            if (end == e) { e++; continue; }
            if (v > 0) debug_frames[debug_n_frames++] = (u32)v;
            e = end;
        }
        debug_frame_read = 1;
    }
    /* Named frames are what a defect report actually needs: the identity of
     * every live target at the moment it was seen, without paying a readback of
     * every target through the whole run. Two of them settle a scene -- the
     * menu and a gameplay frame come from one boot, one run, one binary. */
    int debug_want = 0;
    if (debug_n_frames) {
        for (u32 i = 0; i < debug_n_frames; i++)
            if (rsx_draw_engine_frames() == debug_frames[i]) debug_want = 1;
    } else {
        debug_want = ++debug_present % debug_every == 0;
    }
    if (getenv("PS3RECOMP_VK_DUMP_SURFACES") && debug_want)
        vk_eng_dump_surfaces();

    /* The GPU half of the walker/GPU split: how long the walker waited on a
     * fence while recording the frame the engine just reported on, which is
     * the frame number this carries. Reset every present so the two lines
     * describe the same frame. */
    if (s_eng_stats) {
        const u32 frame = rsx_draw_engine_frames();
        if (frame && frame % 20 == 0) {
            fprintf(stderr, "[engtime] gpu frame %u phase %d: fence wait %.2f ms over %llu "
                            "submissions, %.3f ms/submission, %llu barriers, %llu rps, desc %llu/%llu hits, "
                            "desc-binds %llu push %llu ind %llu(%llu) dir %llu, "
                            "gvb %llu/%llu gib %llu/%llu ubo-reuse %llu ubo-mb %.1f\n",
                    frame, g_ab_phase, s_engtime.gpu_wait_ms,
                    (unsigned long long)s_engtime.submits,
                    s_engtime.submits ? s_engtime.gpu_wait_ms / (double)s_engtime.submits : 0.0,
                    (unsigned long long)s_engtime.barriers,
                    (unsigned long long)s_engtime.rps,
                    (unsigned long long)s_engtime.desc_hits,
                    (unsigned long long)(s_engtime.desc_hits + s_engtime.desc_misses),
                    (unsigned long long)s_engtime.desc_binds,
                    (unsigned long long)s_engtime.push_consts,
                    (unsigned long long)s_engtime.ind_calls,
                    (unsigned long long)s_engtime.ind_draws,
                    (unsigned long long)s_engtime.dir_draws,
                    (unsigned long long)s_engtime.gvb_hits,
                    (unsigned long long)(s_engtime.gvb_hits + s_engtime.gvb_misses),
                    (unsigned long long)s_engtime.gib_hits,
                    (unsigned long long)(s_engtime.gib_hits + s_engtime.gib_misses),
                    (unsigned long long)s_engtime.ubo_reuses,
                    (double)s_engtime.ubo_bytes / (1024.0 * 1024.0));
            fflush(stderr);
        }
        if (g_rsx_sf.enabled) {
            g_rsx_sf.vk_gpu_wait_ms = s_engtime.gpu_wait_ms;
            g_rsx_sf.vk_submits = (u32)s_engtime.submits;
            g_rsx_sf.vk_barriers = (u32)s_engtime.barriers;
            g_rsx_sf.vk_rps = (u32)s_engtime.rps;
            g_rsx_sf.vk_desc_misses = (u32)s_engtime.desc_misses;
            g_rsx_sf.vk_gvb_misses = (u32)s_engtime.gvb_misses;
            g_rsx_sf.vk_gib_misses = (u32)s_engtime.gib_misses;
        }
        s_engtime.gpu_wait_ms = 0.0;
        s_engtime.submits = 0;
        s_engtime.barriers = 0;
        s_engtime.rps = 0;
        s_engtime.desc_hits = 0;
        s_engtime.desc_misses = 0;
        s_engtime.desc_binds = 0;
        s_engtime.push_consts = 0;
        s_engtime.ind_calls = s_engtime.ind_draws = s_engtime.dir_draws = 0;
        s_engtime.gvb_hits = s_engtime.gvb_misses = 0;
        s_engtime.gib_hits = s_engtime.gib_misses = 0;
        s_engtime.ubo_reuses = 0;
        s_engtime.ubo_bytes = 0;
    }
}

/* Pixels a 2D transfer put into guest memory, pushed into the colour target
 * that rectangle belongs to. */
static void vk_eng_color_target_write(void* user, u32 surface, u32 x, u32 y,
                                      u32 w, u32 h, const void* rows, u32 row_bytes)
{
    (void)user;
    vk_eng_obj* o = vk_eng_get(surface, VK_ENG_COLOR);
    if (!o || o->fmt != VK_FORMAT_R8G8B8A8_UNORM) return;   /* rows are R,G,B,A */
    if (!rows || !w || !h || x + w > o->w || y + h > o->h) return;
    vk_eng_upload_region(o->im.img, 0, 0, x, y, w, h, (size_t)w * o->bpp, h, rows, row_bytes);
}

static void vk_eng_readback(void* user, u32 surface, u32 x, u32 y, u32 w, u32 h,
                            void* out, u32 out_pitch)
{
    (void)user;
    vk_eng_obj* o = vk_eng_get(surface, VK_ENG_COLOR);
    if (!o || !out || !out_pitch || !w || !h || x + w > o->w || y + h > o->h) return;
    const size_t tight = (size_t)w * o->bpp;
    vk_buffer* const st = vk_stage_get(&s_rb_stage, (VkDeviceSize)tight * h, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!st) return;
    if (!vk_begin()) {
        vk_barrier_image(o->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
        VkBufferImageCopy region = {
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .imageOffset = { (int32_t)x, (int32_t)y, 0 }, .imageExtent = { w, h, 1 },
        };
        pvkCmdCopyImageToBuffer(s_vk.cmd, o->im.img, VK_IMAGE_LAYOUT_GENERAL, st->buf, 1, &region);
        VkBufferMemoryBarrier hb = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = st->buf, .offset = 0, .size = VK_WHOLE_SIZE,
        };
        pvkCmdPipelineBarrier(s_vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                              0, 0, NULL, 1, &hb, 0, NULL);
        if (!vk_submit_and_wait()) {
            vk_invalidate(st);
            for (u32 row = 0; row < h; row++)
                memcpy((u8*)out + (size_t)row * out_pitch, (const u8*)st->ptr + (size_t)row * tight, tight);
        }
    }
}

/* ---- E3: textures ---------------------------------------------------------
 * The engine decodes every level to host rows (rsx_texture_decode) and hands
 * them over one upload each; the crossbar is a property of the view, so it
 * becomes the image view's component mapping. A cube map (faces == 6) is a
 * six-layer cube-compatible image, each face uploaded into its layer, sampled
 * through a CUBE view; the pipelines that sample one use the split set layout
 * (vk_eng_split_layout). */
static u32 vk_eng_texture_create(void* user, rsx_be_format fmt, u32 w, u32 h,
                                 u32 mips, u32 faces, u32 remap, u32 rsx_fmt)
{
    (void)user;
    u32 bpp;
    const VkFormat vf = vk_eng_format(fmt, &bpp);
    const int bc = (fmt == RSX_BE_FMT_BC1 || fmt == RSX_BE_FMT_BC2 || fmt == RSX_BE_FMT_BC3);
    if (vf == VK_FORMAT_UNDEFINED || !w || !h || w > vk_max_dim() || h > vk_max_dim() ||
        (bc && !s_vk.bc_ok))
        return 0;
    const int cube = (faces == 6);
    if ((faces != 1 && !cube) || (cube && w != h)) return 0;   /* cube faces are square */
    if (!mips) mips = 1;
    const u32 hd = vk_eng_alloc();
    if (!hd) return 0;
    vk_eng_obj* o = &s_eobj[hd];
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0, .imageType = VK_IMAGE_TYPE_2D,
        .format = vf, .extent = { w, h, 1 }, .mipLevels = mips, .arrayLayers = cube ? 6u : 1u,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
    VkMemoryRequirements req;
    u32 type;
    if (pvkCreateImage(s_vk.device, &ici, NULL, &o->im.img) != VK_SUCCESS) goto fail;
    pvkGetImageMemoryRequirements(s_vk.device, o->im.img, &req);
    if (vk_find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &type) &&
        vk_find_memory_type(req.memoryTypeBits, 0, &type)) goto fail;
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = req.size, .memoryTypeIndex = type };
    if (pvkAllocateMemory(s_vk.device, &mai, NULL, &o->im.mem) != VK_SUCCESS ||
        pvkBindImageMemory(s_vk.device, o->im.img, o->im.mem, 0) != VK_SUCCESS ||
        vk_eng_make_view(o->im.img, vf, mips, vk_remap(remap, rsx_fmt), cube, &o->im.view))
        goto fail;
    o->kind = VK_ENG_TEXTURE; o->fmt = vf; o->w = w; o->h = h; o->cube = (u8)cube; o->mips = mips;
    o->bpp = bc ? 0 : bpp; o->block = bc ? bpp : 0;
    /* UNDEFINED -> GENERAL for every level and face before the uploads arrive. */
    if (vk_eb_active() && vk_eb_async_use()) {
        if (vk_rec_use_batch()) {
            if (!s_eb_pending && vk_rec_ensure_open()) goto fail;
            vk_rec_emit_img_init(o->im.img, VK_IMAGE_ASPECT_COLOR_BIT);
            vk_eb_recorded();
        } else {
            if (!s_eb_pending && vk_eb_open_cmd()) goto fail;
            vk_eb_end_rp();
            vk_barrier_image(o->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
            vk_eb_recorded();
        }
    } else {
        if (vk_begin()) goto fail;
        vk_barrier_image(o->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
        if (vk_submit_and_wait()) goto fail;
    }
    return hd;
fail:
    vk_destroy_image(&o->im);
    memset(o, 0, sizeof *o);
    return 0;
}

static void vk_eng_texture_upload(void* user, u32 texture, u32 face, u32 mip, u32 w, u32 h,
                                  const void* src, u32 row_bytes, u32 rows)
{
    (void)user;
    vk_eng_obj* o = vk_eng_get(texture, VK_ENG_TEXTURE);
    if (!o || !src || !row_bytes || !rows || face >= (o->cube ? 6u : 1u) || mip >= o->mips) return;
    /* A BC row is a row of 4x4 blocks; the engine's `rows` already counts them. */
    const size_t tight = o->block ? (size_t)((w + 3u) / 4u) * o->block : (size_t)w * o->bpp;
    vk_eng_upload_rows(o->im.img, face, mip, w, h, tight, rows, src, row_bytes);
}

/* Half to float, for the dumps of the fp16 targets: a debug readout, not the
 * guest's arithmetic, so the decompiler's exact converter is not shared. */
static float vk_dump_half(u16 h)
{
    const u32 sign = (u32)(h & 0x8000u) << 16;
    u32 e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, bits;
    if (e == 0) {
        if (!m) { bits = sign; }
        else { e = 113u; while (!(m & 0x400u)) { m <<= 1; e--; } m &= 0x3FFu;
               bits = sign | (e << 23) | (m << 13); }
    } else if (e == 0x1Fu) {
        bits = sign | 0x7F800000u | (m << 13);
    } else {
        bits = sign | ((e + 112u) << 23) | (m << 13);
    }
    float f; memcpy(&f, &bits, sizeof f);
    return f;
}

static u8 vk_dump_unorm8(float v)
{
    if (!(v == v) || v <= 0.0f) return 0;
    if (v >= 1.0f) return 255;
    return (u8)(v * 255.0f + 0.5f);
}

/* A colour target sampled with a unit's crossbar: an extra view of the same
 * image, cached by (surface, crossbar, format) and released with the target. */
static void vk_eng_dump_surfaces(void)
{
    const char* dir = getenv("PS3RECOMP_VK_DUMP_SURFACES");
    if (!dir) return;
    for (u32 hd = 1; hd < VK_ENG_MAX_OBJ; hd++) {
        vk_eng_obj* o = &s_eobj[hd];
        if (o->kind != VK_ENG_COLOR || o->snapshot || !o->w || !o->h || !o->bpp) continue;
        /* Every colour format the engine can render into, so a pass that
         * targets an HDR (fp16) or float surface is captured too: the world
         * pass of one title writes fp16 and only the composite lands in the
         * 8-bit display target, so an RGBA8-only dump showed the depth of the
         * frame and none of the scene. */
        const char* tag = "";
        switch (o->fmt) {
        case VK_FORMAT_R8G8B8A8_UNORM:       tag = ""; break;
        case VK_FORMAT_R16G16B16A16_SFLOAT:  tag = "-f16"; break;
        case VK_FORMAT_R32_SFLOAT:           tag = "-r32f"; break;
        case VK_FORMAT_R32G32B32A32_SFLOAT:  tag = "-f32"; break;
        default: continue;
        }
        const size_t row = (size_t)o->w * o->bpp, bytes = row * o->h;
        u8* src = (u8*)malloc(bytes);
        if (!src) return;
        vk_eng_readback(NULL, hd, 0, 0, o->w, o->h, src, (u32)row);
        char path[1024];
        /* The handle only names a backend object; [eng frame] ties it to the
         * guest address it was created for. The frame number is in the name
         * because a double-buffered title alternates which target holds the
         * frame, and two dumps at the same frame overwrite each other. */
        snprintf(path, sizeof path, "%s/surface-f%u-%03u-%ux%u%s.ppm", dir,
                 rsx_draw_engine_frames(), hd, o->w, o->h, tag);
        FILE* f = fopen(path, "wb");
        if (f) {
            fprintf(f, "P6\n%u %u\n255\n", o->w, o->h);
            for (u32 y = 0; y < o->h; y++) {
                const u8* p = src + (size_t)y * row;
                for (u32 x = 0; x < o->w; x++, p += o->bpp) {
                    u8 rgb[3];
                    if (o->fmt == VK_FORMAT_R8G8B8A8_UNORM) {
                        rgb[0] = p[0]; rgb[1] = p[1]; rgb[2] = p[2];
                    } else if (o->fmt == VK_FORMAT_R16G16B16A16_SFLOAT) {
                        u16 v[4]; memcpy(v, p, sizeof v);
                        rgb[0] = vk_dump_unorm8(vk_dump_half(v[0]));
                        rgb[1] = vk_dump_unorm8(vk_dump_half(v[1]));
                        rgb[2] = vk_dump_unorm8(vk_dump_half(v[2]));
                    } else if (o->fmt == VK_FORMAT_R32_SFLOAT) {
                        float v; memcpy(&v, p, sizeof v);
                        rgb[0] = rgb[1] = rgb[2] = vk_dump_unorm8(v);
                    } else {
                        float v[4]; memcpy(v, p, sizeof v);
                        rgb[0] = vk_dump_unorm8(v[0]);
                        rgb[1] = vk_dump_unorm8(v[1]);
                        rgb[2] = vk_dump_unorm8(v[2]);
                    }
                    fwrite(rgb, 1, 3, f);
                }
            }
            fclose(f);
        }
        free(src);
    }
}

static u32 vk_eng_surface_view(void* user, u32 surface, u32 remap, u32 rsx_format)
{
    (void)user;
    vk_eng_obj* t = vk_eng_get(surface, VK_ENG_COLOR);
    if (!t) return 0;
    for (u32 k = 0; k < s_eview_count; k++) {
        const u32 i = s_eview[k];
        if (s_eobj[i].kind == VK_ENG_VIEW && s_eobj[i].surface == surface &&
            s_eobj[i].remap == remap && s_eobj[i].rsx_fmt == rsx_format)
            return i;
    }
    if (s_eview_count >= VK_ENG_MAX_VIEWS) {
        for (u32 i = 1; i < VK_ENG_MAX_OBJ; i++)
            if (s_eobj[i].kind == VK_ENG_VIEW && s_eobj[i].surface == surface &&
                s_eobj[i].remap == remap && s_eobj[i].rsx_fmt == rsx_format)
                return i;
    }
    const u32 hd = vk_eng_alloc();
    if (!hd) return 0;
    vk_eng_obj* o = &s_eobj[hd];
    if (vk_eng_make_view(t->im.img, t->fmt, 1, vk_remap(remap, rsx_format), 0, &o->im.view)) {
        memset(o, 0, sizeof *o); return 0;
    }
    o->kind = VK_ENG_VIEW; o->fmt = t->fmt; o->w = t->w; o->h = t->h;
    o->surface = surface; o->remap = remap; o->rsx_fmt = rsx_format;
    if (s_eview_count < VK_ENG_MAX_VIEWS) s_eview[s_eview_count++] = hd;
    return hd;
}
/* ---- colour feedback snapshots ---------------------------------------------
 * A draw that samples a colour target it is also writing (in-place lighting:
 * read the albedo, write the lit colour over it) gets a copy of that target
 * taken just before the draw. The engine asks for one per such draw
 * (rsx_draw_backend.color_snapshot); without it the unit falls back to the
 * guest-memory bytes at the target's address, which hold whatever was last
 * stored there, and a pass that reads-and-writes paints black.
 *
 * One copy image per target, refilled in command order: with draw batching
 * the copy joins the open batch instead of a submission of its own, and the
 * draws that sample the copy follow it in the same buffer, so each of them
 * sees the target as it was just before that draw. A per-call image and a
 * per-call submission cost inFamous 47 ms a frame and 400 submissions; this
 * costs one GPU copy. The copy is a COLOR object marked `snapshot` so the
 * engine can take a crossbar view of it (vk_eng_surface_view) and so the
 * dumps skip it; it is never an attachment. */
#define VK_SNAP_POOL 8
static struct { vk_image im; VkFormat fmt; u32 w, h; } s_snap_pool[VK_SNAP_POOL];
static u32 s_snap_pool_n;
static vk_buffer s_snap_st;         /* depth snapshot bounce buffer (vk_eng_depth_snapshot) */

static void vk_eng_snapshot_recycle(vk_eng_obj* o)
{
    if (s_snap_pool_n < VK_SNAP_POOL && o->im.img) {
        s_snap_pool[s_snap_pool_n].im = o->im;
        s_snap_pool[s_snap_pool_n].fmt = o->fmt;
        s_snap_pool[s_snap_pool_n].w = o->w;
        s_snap_pool[s_snap_pool_n].h = o->h;
        s_snap_pool_n++;
    } else {
        vk_destroy_image(&o->im);
    }
}

static void vk_eng_snapshot_pool_drain(void)
{
    for (u32 i = 0; i < s_snap_pool_n; i++) vk_destroy_image(&s_snap_pool[i].im);
    s_snap_pool_n = 0;
    vk_destroy_buffer(&s_snap_st);
}

/* The target's copy, made or resized on first use, then refilled. The handle
 * lives in the target's `surface` field (unused for a colour target) and goes
 * when the target goes. */
static u32 vk_eng_color_snapshot(void* user, u32 surface)
{
    vk_eng_obj* t = vk_eng_get(surface, VK_ENG_COLOR);
    if (!t || t->snapshot) return 0;
    u32 hd = t->surface;
    vk_eng_obj* o = vk_eng_get(hd, VK_ENG_COLOR);
    if (!o || !o->snapshot || o->fmt != t->fmt || o->w != t->w || o->h != t->h) {
        if (o && o->snapshot) { vk_eng_release(user, hd); hd = 0; t->surface = 0; }
        hd = vk_eng_alloc();
        if (!hd) return 0;
        o = &s_eobj[hd];
        int reused = 0;
        for (u32 i = 0; i < s_snap_pool_n; i++)
            if (s_snap_pool[i].fmt == t->fmt && s_snap_pool[i].w == t->w &&
                s_snap_pool[i].h == t->h) {
                o->im = s_snap_pool[i].im;
                s_snap_pool[i] = s_snap_pool[--s_snap_pool_n];
                reused = 1;
                break;
            }
        if (!reused && vk_create_image(&o->im, t->fmt, t->w, t->h,
                                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                       VK_IMAGE_ASPECT_COLOR_BIT)) {
            vk_destroy_image(&o->im);
            memset(o, 0, sizeof *o);
            return 0;
        }
        o->kind = VK_ENG_COLOR; o->snapshot = 1;
        o->fmt = t->fmt; o->bpp = t->bpp; o->w = t->w; o->h = t->h;
        t = vk_eng_get(surface, VK_ENG_COLOR);        /* same slot; reload for clarity */
        t->surface = hd;
        if (vk_begin()) { vk_eng_release(user, hd); t->surface = 0; return 0; }
        vk_barrier_image(o->im.img, VK_IMAGE_ASPECT_COLOR_BIT,
                         reused ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED);
        if (vk_submit_and_wait()) { vk_eng_release(user, hd); t->surface = 0; return 0; }
        t = vk_eng_get(surface, VK_ENG_COLOR);
        o = vk_eng_get(hd, VK_ENG_COLOR);
        if (!t || !o) return 0;
    }
    /* The refill. In a batch it is one more command in the open buffer; the
     * draws that sample it come after it, and each batched draw starts with a
     * full memory barrier of its own. */
    const int batch = vk_eb_active();
    if (batch && vk_rec_use_batch()) {
        if (!s_eb_pending && vk_rec_ensure_open()) return 0;
        vk_rec_emit_color_snap(t->im.img, o->im.img, t->w, t->h);
        vk_eb_recorded();
        return hd;
    }
    int rc = batch ? (s_eb_pending ? 0 : vk_eb_open_cmd()) : vk_begin();
    if (!rc) {
        if (batch) vk_eb_end_rp();
        VkImageMemoryBarrier pre[2] = {
            {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = t->im.img,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
            },
            {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = o->im.img,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
            },
        };
        pvkCmdPipelineBarrier(s_vk.cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 2, pre);
        VkImageCopy c = {
            .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .extent = { t->w, t->h, 1 },
        };
        pvkCmdCopyImage(s_vk.cmd, t->im.img, VK_IMAGE_LAYOUT_GENERAL,
                        o->im.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        VkImageMemoryBarrier post[2] = {
            {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = o->im.img,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
            },
            {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = t->im.img,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
            },
        };
        pvkCmdPipelineBarrier(s_vk.cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, NULL, 0, NULL, 2, post);
        if (batch) vk_eb_recorded();
        else if (vk_submit_and_wait()) rc = -1;
    }
    return rc ? 0 : hd;
}

/* ---- E4: depth snapshots --------------------------------------------------
 * A depth target read as a texture, which the engine asks for only after a
 * depth-writing draw. Metal resolves it with a fullscreen pass into R32Float;
 * Vulkan can do it with two copies on the GPU, since a D32_SFLOAT depth
 * aspect copies out as 32-bit floats: depth -> buffer -> an R32_SFLOAT
 * texture. Other depth formats would need a conversion pass: they report 0.
 *
 * Each depth target keeps ONE snapshot texture (its handle in `surface`),
 * refilled on every request, so nothing accumulates however often the engine
 * invalidates and asks again; it goes when the depth target goes. */
static u32 vk_eng_depth_snapshot(void* user, u32 depth, u32 w, u32 h)
{
    vk_eng_obj* z = vk_eng_get(depth, VK_ENG_DEPTH);
    if (!z) return 0;
    if (s_vk.depth_format != VK_FORMAT_D32_SFLOAT &&
        s_vk.depth_format != VK_FORMAT_D32_SFLOAT_S8_UINT) {
        static int warned_fmt;
        if (!warned_fmt) { VK_LOG("engine: depth snapshots need D32_SFLOAT, this device uses %d\n",
                                  (int)s_vk.depth_format); warned_fmt = 1; }
        return 0;
    }
    if (w > z->w) w = z->w;
    if (h > z->h) h = z->h;
    if (!w || !h) return 0;

    u32 snap = z->surface;
    vk_eng_obj* t = vk_eng_get(snap, VK_ENG_TEXTURE);
    if (!t || t->w != w || t->h != h) {
        if (t) vk_eng_release(user, snap);
        z->surface = 0;
        snap = vk_eng_alloc();
        if (!snap) return 0;
        t = &s_eobj[snap];
        if (vk_create_image(&t->im, VK_FORMAT_R32_SFLOAT, w, h,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            VK_IMAGE_ASPECT_COLOR_BIT)) {
            vk_destroy_image(&t->im); memset(t, 0, sizeof *t); return 0;
        }
        t->kind = VK_ENG_TEXTURE; t->fmt = VK_FORMAT_R32_SFLOAT; t->bpp = 4; t->w = w; t->h = h;
        z = vk_eng_get(depth, VK_ENG_DEPTH);          /* same slot; reload for clarity */
        z->surface = snap;
        if (vk_begin()) return 0;
        vk_barrier_image(t->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
        if (vk_submit_and_wait()) return 0;
    }

    /* The bounce buffer stays (device-local when the device has it, grown as
     * needed): a fresh host buffer per snapshot cost ~4.8 ms each, 375 times
     * in 20 s of God of War: Ascension's first level. With draw batching the
     * copies join the open batch instead of a submission of their own; the
     * draws that sample the snapshot come after them in the same buffer, and
     * each batched draw starts with a full memory barrier. */
    const VkDeviceSize need = (VkDeviceSize)w * h * 4u;
    if (s_snap_st.size < need) {
        vk_destroy_buffer(&s_snap_st);                   /* flushes a pending batch first */
        if (vk_create_device_buffer(&s_snap_st, need,
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
            vk_destroy_buffer(&s_snap_st); return 0;
        }
    }
    vk_buffer st = s_snap_st;
    const int batch = vk_eb_active();
    if (batch && vk_rec_use_batch()) {
        if (!s_eb_pending && vk_rec_ensure_open()) return 0;
        vk_rec_emit_depth_snap(z->im.img, t->im.img, st.buf, w, h);
        vk_eb_recorded();
        return snap;
    }
    int rc = batch ? (s_eb_pending ? 0 : vk_eb_open_cmd()) : vk_begin();
    if (!rc && batch) { vk_eb_end_rp(); vk_cmd_after_everything(s_vk.cmd); vk_eb_touch_clear(); }
    if (!rc) {
        vk_barrier_image(z->im.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
        vk_barrier_image(t->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
        VkBufferImageCopy out = { .imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 },
                                  .imageExtent = { w, h, 1 } };
        pvkCmdCopyImageToBuffer(s_vk.cmd, z->im.img, VK_IMAGE_LAYOUT_GENERAL, st.buf, 1, &out);
        VkBufferMemoryBarrier bb = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = st.buf, .offset = 0, .size = VK_WHOLE_SIZE };
        pvkCmdPipelineBarrier(s_vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, NULL, 1, &bb, 0, NULL);
        VkBufferImageCopy in = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                 .imageExtent = { w, h, 1 } };
        pvkCmdCopyBufferToImage(s_vk.cmd, st.buf, t->im.img, VK_IMAGE_LAYOUT_GENERAL, 1, &in);
        if (batch) {
            vk_barrier_image(t->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
            vk_eb_recorded();
        } else
            rc = vk_submit_and_wait();
    }
    return rc ? 0 : snap;
}
/* The depth aspect of a D32_SFLOAT target copies out as 32-bit floats, which
 * is what the engine asks for; other depth formats report failure. */
static int vk_eng_depth_readback(void* user, u32 depth, u32 x, u32 y, u32 w, u32 h,
                                 float* out, u32 out_pitch)
{
    (void)user;
    vk_eng_obj* z = vk_eng_get(depth, VK_ENG_DEPTH);
    if (!z || !out || !w || !h || x + w > z->w || y + h > z->h ||
        out_pitch < w * 4u || (s_vk.depth_format != VK_FORMAT_D32_SFLOAT &&
                                  s_vk.depth_format != VK_FORMAT_D32_SFLOAT_S8_UINT))
        return -1;
    vk_buffer* const st = vk_stage_get(&s_rb_stage, (VkDeviceSize)w * h * 4u, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!st) return -1;
    int rc = vk_begin();
    if (!rc) {
        vk_barrier_image(z->im.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
        VkBufferImageCopy region = {
            .imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 },
            .imageOffset = { (int32_t)x, (int32_t)y, 0 }, .imageExtent = { w, h, 1 },
        };
        pvkCmdCopyImageToBuffer(s_vk.cmd, z->im.img, VK_IMAGE_LAYOUT_GENERAL, st->buf, 1, &region);
        VkBufferMemoryBarrier hb = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = st->buf, .offset = 0, .size = VK_WHOLE_SIZE,
        };
        pvkCmdPipelineBarrier(s_vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                              0, 0, NULL, 1, &hb, 0, NULL);
        rc = vk_submit_and_wait();
        if (!rc) {
            vk_invalidate(st);
            for (u32 row = 0; row < h; row++)
                memcpy((u8*)out + (size_t)row * out_pitch,
                       (const u8*)st->ptr + (size_t)row * w * 4u, (size_t)w * 4u);
        }
    }
    return rc ? -1 : 0;
}
/* ---- E2: pipelines and draws ---------------------------------------------
 * A pipeline handle carries the two translated modules and the render state
 * the engine keyed it on. Topology is not part of that key (Metal sets it on
 * the encoder), but Vulkan 1.0 bakes it into the pipeline, so each handle
 * builds its VkPipeline per topology on first use.
 *
 * Bindings are the translator's contract (rsx_shader_spirv.h), and the set
 * layout, pipeline layout, descriptor set and constant UBO are the ones the
 * vtable guest path already creates. Vertex inputs are the engine's compact
 * layout: slot n is a float4 at n*16, at location n, as Metal reads it.
 * Known gaps, stated rather than guessed: vertex-texture units (their
 * samplers s0..s3 would alias the pixel samplers' binding), colour targets
 * other than RGBA8 and RGBA16F, depth-only passes, and stencil (the depth format
 * has no stencil aspect yet, so the test is inert). */
/* As many as the engine keeps (ENG_MAX_PIPELINES in rsx_draw_engine.c). God of
 * War: Ascension selects ~740 distinct fragment programs in its first level on
 * top of the menu's; with 512 slots every later pipeline failed silently and the
 * level drew nothing but its clears. */
#define VK_ENG_MAX_PIPE   8192
#define VK_ENG_TOPOLOGIES 5
#define VK_ENG_FB_CACHE   64

typedef struct vk_eng_pipe {
    int                 used;
    VkShaderModule      vs, fs;
    VkShaderModule      vs_ssbo, fs_ssbo;
    VkShaderModule      vs_inst;
    rsx_be_render_state rs;
    u32                 nslots, stride, rt_count;
    VkFormat            rt_fmt;          /* picks the render pass it is built against */
    u32                 cube_mask;       /* TextureCube units: the split set layout */
    u8                  depth_or_stencil_write;
    VkPipeline          variant[VK_ENG_TOPOLOGIES];
    VkPipeline          variant_ssbo[VK_ENG_TOPOLOGIES];
    VkPipeline          variant_inst[VK_ENG_TOPOLOGIES];
} vk_eng_pipe;

static vk_eng_pipe s_epipe[VK_ENG_MAX_PIPE];

static struct {
    VkRenderPass rp[VK_ENG_RT_FORMATS][RSX_BE_MAX_COLOR_TARGETS + 1]; /* by format, colour count */
    struct { u32 key[RSX_BE_MAX_COLOR_TARGETS + 1]; VkFramebuffer fb; } fb[VK_ENG_FB_CACHE];
    u32          fb_count;
    vk_image     scratch_depth;                          /* when no zeta is bound */
    u32          scratch_w, scratch_h;
    u32          pipeline;
    u32          vs_bytes, ps_bytes;
    int          vs_dirty, ps_dirty, tex_dirty;
    const void*  vs_src;              /* zero-copy pointer for batched VS UBO */
    u8*          ucpu;                /* the constants, in the shared UBO's layout */
    u32          tex[RSX_BE_MAX_TEXTURES];
    VkImage      tex_img[RSX_BE_MAX_TEXTURES];  /* the image each unit samples */
    rsx_be_sampler_desc samp[RSX_BE_MAX_TEXTURES];
    VkSampler    samp_vk[RSX_BE_MAX_TEXTURES];
    u8           samp_idx[RSX_BE_MAX_TEXTURES];
    u32          tex_mask;
    u32          tex_rt_mask;         /* units sampling a color target/view */
    u64          tex_bind_hash;
    u32          tex_bind_gen;
    /* Cached resolved target/depth/RP/FB state (invalidated on target bind or FB drop). */
    u8            rt_cache_ok;
    u32           rt_cache_n, rt_cache_w, rt_cache_h, rt_cache_dkey;
    VkFormat      rt_cache_fmt;
    VkImage       rt_cache_img[RSX_BE_MAX_COLOR_TARGETS];
    VkImage       rt_cache_depth_img;
    VkImageView   rt_cache_dview;
    VkRenderPass  rt_cache_rp;
    VkFramebuffer rt_cache_fb;
    VkFramebuffer last_fb;
    VkRenderPass  last_fb_rp;
    u32           last_fb_n;
    u32           last_fb_rt[RSX_BE_MAX_COLOR_TARGETS];
    u32           last_fb_dkey;
    float        vp[4];
    u32          sc[4];
    int          have_vp, have_sc;
    u8           vp_dirty, sc_dirty, sref_dirty, db_dirty, bias_dirty, bc_dirty;
    u8           dyn_dirty_mask;      /* bit0=vp, bit1=sc, bit2=sref, bit3=db, bit4=bias, bit5=bc */
    u32          stencil_ref;
    float        depth_bounds[2];     /* min, max for the next draw */
    float        depth_bias[2];       /* constant (scaled units), slope */
    float        blend_color[4];      /* RGBA blend constants (#782) */
    vk_buffer    indices;
    u32          warned;
    /* The split set layout (vk_eng_split_layout) and its one set, made the
     * first time a pipeline samples a cube map; the 1x1 cube a cube unit
     * with nothing bound samples. */
    VkDescriptorSetLayout split_layout;
    VkDescriptorPool      split_pool;
    VkDescriptorSet       split_set;
    VkPipelineLayout      split_pipe_layout;
    VkDescriptorSetLayout ssbo_split_tex_layout;
    VkDescriptorPool      ssbo_split_pool;
    VkDescriptorSet       ssbo_split_tex_set;
    VkPipelineLayout      ssbo_split_pipe_layout;
    u32                   dummy_cube;
    u32                   dummy_2d;
} s_e2;

#define VK_ENG_SAMP_CACHE 64
static struct { u64 key; VkSampler s; } s_esamp[VK_ENG_SAMP_CACHE];
static u32 s_esamp_count;

static void vk_eng_invalidate_rt_cache(void) { s_e2.rt_cache_ok = 0; }

static void vk_eng_drop_framebuffers(void)
{
    if (s_eb_pending || vk_eb_any_inflight()) vk_eng_flush();
    for (u32 i = 0; i < s_e2.fb_count; i++)
        if (s_e2.fb[i].fb) pvkDestroyFramebuffer(s_vk.device, s_e2.fb[i].fb, NULL);
    s_e2.fb_count = 0;
    s_e2.last_fb = VK_NULL_HANDLE;
    s_e2.rt_cache_ok = 0;
}

/* The render pass for `n` colour attachments of format `cf` plus depth,
 * LOAD/STORE in GENERAL like the vtable path's: clears are separate
 * operations. An MRT set shares target A's format (engine contract). */
static VkRenderPass vk_eng_render_pass(VkFormat cf, u32 n)
{
    const int fi = vk_eng_rt_index(cf);
    if (fi < 0 || n < 1 || n > RSX_BE_MAX_COLOR_TARGETS) return VK_NULL_HANDLE;
    if (s_e2.rp[fi][n]) return s_e2.rp[fi][n];
    VkAttachmentDescription att[RSX_BE_MAX_COLOR_TARGETS + 1];
    VkAttachmentReference   cref[RSX_BE_MAX_COLOR_TARGETS];
    for (u32 i = 0; i <= n; i++) {
        att[i] = (VkAttachmentDescription){
            .format = (i < n) ? cf : s_vk.depth_format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            /* Stencil lives across draws like depth does (a title writes it in
             * one pass and tests it in the next); ignored for colour and for
             * depth formats without stencil. */
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
            .initialLayout = VK_IMAGE_LAYOUT_GENERAL, .finalLayout = VK_IMAGE_LAYOUT_GENERAL };
        if (i < n) cref[i] = (VkAttachmentReference){ i, VK_IMAGE_LAYOUT_GENERAL };
    }
    VkAttachmentReference dref = { n, VK_IMAGE_LAYOUT_GENERAL };
    VkSubpassDescription sub = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 .colorAttachmentCount = n, .pColorAttachments = cref,
                                 .pDepthStencilAttachment = &dref };
    VkRenderPassCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                                   .attachmentCount = n + 1, .pAttachments = att,
                                   .subpassCount = 1, .pSubpasses = &sub };
    if (pvkCreateRenderPass(s_vk.device, &rci, NULL, &s_e2.rp[fi][n]) != VK_SUCCESS)
        s_e2.rp[fi][n] = VK_NULL_HANDLE;
    return s_e2.rp[fi][n];
}

/* Tables carried from the Metal backend, itself copied from the D3D12 live
 * draw engine a title has shipped on. */
static VkBlendFactor vk_gcm_blend_factor(u32 f)
{
    switch (f) {
    case 0x0000: return VK_BLEND_FACTOR_ZERO;
    case 0x0001: return VK_BLEND_FACTOR_ONE;
    case 0x0300: return VK_BLEND_FACTOR_SRC_COLOR;
    case 0x0301: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 0x0302: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 0x0303: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 0x0304: return VK_BLEND_FACTOR_DST_ALPHA;
    case 0x0305: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 0x0306: return VK_BLEND_FACTOR_DST_COLOR;
    case 0x0307: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 0x0308: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    /* sp00nznet/ps3recomp #782: blend constant factors (vk_eng_set_blend_color). */
    case 0x8001: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 0x8002: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 0x8003: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 0x8004: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default:     return VK_BLEND_FACTOR_ONE;
    }
}
static VkBlendOp vk_gcm_blend_op(u32 e)
{
    switch (e) {
    case 0x8007: return VK_BLEND_OP_MIN;
    case 0x8008: return VK_BLEND_OP_MAX;
    case 0x800A: return VK_BLEND_OP_SUBTRACT;
    case 0x800B: return VK_BLEND_OP_REVERSE_SUBTRACT;
    default:     return VK_BLEND_OP_ADD;                  /* FUNC_ADD 0x8006 */
    }
}
static VkCompareOp vk_gcm_compare(u32 f)
{
    return (f >= 0x0200u && f <= 0x0206u) ? (VkCompareOp)(f - 0x0200u) : VK_COMPARE_OP_ALWAYS;
}
static VkStencilOp vk_gcm_stencil_op(u32 op)
{
    switch (op) {
    case 0x0000: return VK_STENCIL_OP_ZERO;
    case 0x1E01: return VK_STENCIL_OP_REPLACE;
    case 0x1E02: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
    case 0x1E03: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
    case 0x150A: return VK_STENCIL_OP_INVERT;
    case 0x8507: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
    case 0x8508: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
    default:     return VK_STENCIL_OP_KEEP;               /* GL_KEEP 0x1E00 */
    }
}
static VkStencilOpState vk_gcm_stencil_face(u32 func, u32 fail, u32 zfail, u32 zpass,
                                            u32 cmp_mask, u32 write_mask)
{
    VkStencilOpState s = { .failOp = vk_gcm_stencil_op(fail), .passOp = vk_gcm_stencil_op(zpass),
                           .depthFailOp = vk_gcm_stencil_op(zfail),
                           .compareOp = vk_gcm_compare(func),
                           .compareMask = cmp_mask & 0xFFu, .writeMask = write_mask & 0xFFu,
                           .reference = 0 };                /* dynamic */
    return s;
}

static int vk_eng_translate_ex(const char* hlsl, int stage, u32 flags, VkShaderModule* out)
{
    const double t0 = g_rsx_sf.enabled ? vk_now_ms() : 0.0;
    u32 words = 0;
    if (rsx_hlsl_to_spirv_ex(hlsl, stage, flags, s_g.spv, VK_SPV_WORDS, &words, s_g.log, sizeof s_g.log) || !words) {
        VK_LOG("engine: %s translation failed (flags=%u): %s\n",
               stage == RSX_SHADER_STAGE_VERTEX ? "vertex" : "fragment", flags, s_g.log);
        return -1;
    }
    VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                    .codeSize = (size_t)words * 4u, .pCode = s_g.spv };
    int rc = pvkCreateShaderModule(s_vk.device, &ci, NULL, out) == VK_SUCCESS ? 0 : -1;
    if (g_rsx_sf.enabled) {
        g_rsx_sf.vk_shader_creates++;
        g_rsx_sf.vk_shader_ms += vk_now_ms() - t0;
    }
    return rc;
}

static int vk_eng_translate(const char* hlsl, int stage, VkShaderModule* out)
{
    return vk_eng_translate_ex(hlsl, stage, 0u, out);
}

static void vk_eb_inval_binds(void);

/* ---- depth+stencil -> A8R8G8B8 conversion (ZCONV_STENCIL) -----------------
 * Packs RSX D24S8 bytes into R8G8B8A8_UNORM as float4(mid, lo, stencil, hi)
 * so deferred lighting passes that sample a zeta address as format 0x85 read
 * 24-bit depth (.wxy) and 8-bit stencil (.z). ZCONV_STENCIL=0 writes B = 0.
 * In Vulkan, a VK_IMAGE_ASPECT_STENCIL_BIT view presents stencil in R (.x). */
#define VK_ZCONV_MAX_SETS 32u
static struct {
    int                   tried, ready, use_stencil;
    VkRenderPass          rp;
    VkDescriptorSetLayout dsl;
    VkDescriptorPool      pool;
    VkPipelineLayout      pipe_layout;
    VkPipeline            pipe;
    struct { VkImageView dview, sview; VkDescriptorSet set; } sets[VK_ZCONV_MAX_SETS];
    u32                   set_count;
} s_zconv;

static void vk_zconv_reset_sets(void)
{
    if (!s_zconv.pool || !s_zconv.set_count) return;
    if (s_eb_pending || vk_eb_any_inflight()) vk_eng_flush();
    pvkResetDescriptorPool(s_vk.device, s_zconv.pool, 0);
    s_zconv.set_count = 0;
}

static void vk_zconv_shutdown(void)
{
    if (s_eb_pending || vk_eb_any_inflight()) vk_eng_flush();
    for (u32 i = 1; i < VK_ENG_MAX_OBJ; i++)
        if (s_eobj[i].kind == VK_ENG_COLOR && s_eobj[i].zconv_fb) {
            pvkDestroyFramebuffer(s_vk.device, s_eobj[i].zconv_fb, NULL);
            s_eobj[i].zconv_fb = VK_NULL_HANDLE;
        }
    if (s_zconv.pipe)        pvkDestroyPipeline(s_vk.device, s_zconv.pipe, NULL);
    if (s_zconv.pipe_layout) pvkDestroyPipelineLayout(s_vk.device, s_zconv.pipe_layout, NULL);
    if (s_zconv.pool)        pvkDestroyDescriptorPool(s_vk.device, s_zconv.pool, NULL);
    if (s_zconv.dsl)         pvkDestroyDescriptorSetLayout(s_vk.device, s_zconv.dsl, NULL);
    if (s_zconv.rp)          pvkDestroyRenderPass(s_vk.device, s_zconv.rp, NULL);
    memset(&s_zconv, 0, sizeof s_zconv);
}

static int vk_zconv_init(void)
{
    if (s_zconv.ready) return 0;
    if (s_zconv.tried) return -1;
    s_zconv.tried = 1;
    if (!s_g.on || !vk_depth_sampleable()) return -1;
    const char* env_st = getenv("ZCONV_STENCIL");
    s_zconv.use_stencil = ((s_vk.depth_aspect & VK_IMAGE_ASPECT_STENCIL_BIT) != 0) &&
                          !(env_st && *env_st == '0');

    static const char vs_hlsl[] =
        "struct O { float4 p : SV_Position; };\n"
        "O main(uint id : SV_VertexID) {\n"
        "    O o;\n"
        "    float2 t = float2(float((id << 1u) & 2u), float(id & 2u));\n"
        "    o.p = float4(t * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);\n"
        "    return o;\n"
        "}\n";
    static const char ps_hlsl_st[] =
        "Texture2D<float4> zt : register(t0);\n"
        "Texture2D<uint4>  zs : register(t1);\n"
        "float4 main(float4 p : SV_POSITION) : SV_TARGET {\n"
        "    int3 tc = int3(int2(p.xy), 0);\n"
        "    float d = zt.Load(tc).x;\n"
        "    uint v = (uint)round(saturate(d) * 16777215.0);\n"
        "    float hi  = float((v >> 16) & 255u) / 255.0;\n"
        "    float mid = float((v >>  8) & 255u) / 255.0;\n"
        "    float lo  = float( v        & 255u) / 255.0;\n"
        "    float st  = float(zs.Load(tc).x & 255u) / 255.0;\n"
        "    return float4(mid, lo, st, hi);\n"
        "}\n";
    static const char ps_hlsl_nost[] =
        "Texture2D<float4> zt : register(t0);\n"
        "float4 main(float4 p : SV_POSITION) : SV_TARGET {\n"
        "    int3 tc = int3(int2(p.xy), 0);\n"
        "    float d = zt.Load(tc).x;\n"
        "    uint v = (uint)round(saturate(d) * 16777215.0);\n"
        "    float hi  = float((v >> 16) & 255u) / 255.0;\n"
        "    float mid = float((v >>  8) & 255u) / 255.0;\n"
        "    float lo  = float( v        & 255u) / 255.0;\n"
        "    return float4(mid, lo, 0.0, hi);\n"
        "}\n";

    VkShaderModule vs = VK_NULL_HANDLE, ps = VK_NULL_HANDLE;
    if (vk_eng_translate(vs_hlsl, RSX_SHADER_STAGE_VERTEX, &vs) ||
        vk_eng_translate(s_zconv.use_stencil ? ps_hlsl_st : ps_hlsl_nost,
                         RSX_SHADER_STAGE_FRAGMENT, &ps)) {
        if (vs) pvkDestroyShaderModule(s_vk.device, vs, NULL);
        if (ps) pvkDestroyShaderModule(s_vk.device, ps, NULL);
        return -1;
    }
    VkAttachmentDescription att = {
        .format = VK_FORMAT_R8G8B8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_GENERAL, .finalLayout = VK_IMAGE_LAYOUT_GENERAL };
    VkAttachmentReference cref = { 0, VK_IMAGE_LAYOUT_GENERAL };
    VkSubpassDescription sub = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 .colorAttachmentCount = 1, .pColorAttachments = &cref };
    VkRenderPassCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                                   .attachmentCount = 1, .pAttachments = &att,
                                   .subpassCount = 1, .pSubpasses = &sub };
    VkDescriptorSetLayoutBinding b[2] = {
        { RSX_SPIRV_TEXTURE_BINDING, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { RSX_SPIRV_TEXTURE_BINDING + 1u, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
    };
    VkDescriptorSetLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .bindingCount = s_zconv.use_stencil ? 2u : 1u, .pBindings = b };
    VkDescriptorPoolSize psize = { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_ZCONV_MAX_SETS * 2u };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = VK_ZCONV_MAX_SETS, .poolSizeCount = 1, .pPoolSizes = &psize };
    if (pvkCreateRenderPass(s_vk.device, &rci, NULL, &s_zconv.rp) != VK_SUCCESS ||
        pvkCreateDescriptorSetLayout(s_vk.device, &lci, NULL, &s_zconv.dsl) != VK_SUCCESS ||
        pvkCreateDescriptorPool(s_vk.device, &pci, NULL, &s_zconv.pool) != VK_SUCCESS) {
        pvkDestroyShaderModule(s_vk.device, vs, NULL);
        pvkDestroyShaderModule(s_vk.device, ps, NULL);
        vk_zconv_shutdown(); s_zconv.tried = 1;
        return -1;
    }
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1, .pSetLayouts = &s_zconv.dsl };
    if (pvkCreatePipelineLayout(s_vk.device, &plci, NULL, &s_zconv.pipe_layout) != VK_SUCCESS) {
        pvkDestroyShaderModule(s_vk.device, vs, NULL);
        pvkDestroyShaderModule(s_vk.device, ps, NULL);
        vk_zconv_shutdown(); s_zconv.tried = 1;
        return -1;
    }
    VkPipelineShaderStageCreateInfo st[2] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = ps, .pName = "main" } };
    VkPipelineVertexInputStateCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkPipelineViewportStateCreateInfo vps = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1 };
    VkPipelineRasterizationStateCreateInfo ras = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkPipelineColorBlendAttachmentState cba = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT };
    VkPipelineColorBlendStateCreateInfo cb = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &cba };
    VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2, .pDynamicStates = dyn };
    VkGraphicsPipelineCreateInfo gci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &vps, .pRasterizationState = &ras, .pMultisampleState = &ms,
        .pColorBlendState = &cb, .pDynamicState = &dy,
        .layout = s_zconv.pipe_layout, .renderPass = s_zconv.rp, .subpass = 0 };
    VkResult r = pvkCreateGraphicsPipelines(s_vk.device, VK_NULL_HANDLE, 1, &gci, NULL, &s_zconv.pipe);
    pvkDestroyShaderModule(s_vk.device, vs, NULL);
    pvkDestroyShaderModule(s_vk.device, ps, NULL);
    if (r != VK_SUCCESS) {
        VK_LOG("engine: zconv pipeline creation failed\n");
        vk_zconv_shutdown(); s_zconv.tried = 1;
        return -1;
    }
    s_zconv.ready = 1;
    VK_LOG("engine: depth->A8R8G8B8 conversion pipeline ready (ZCONV_STENCIL=%d)\n", s_zconv.use_stencil);
    return 0;
}

static VkDescriptorSet vk_zconv_set_for(VkImageView dview, VkImageView sview)
{
    for (u32 i = 0; i < s_zconv.set_count; i++)
        if (s_zconv.sets[i].dview == dview && s_zconv.sets[i].sview == sview)
            return s_zconv.sets[i].set;
    if (s_zconv.set_count >= VK_ZCONV_MAX_SETS) vk_zconv_reset_sets();
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = s_zconv.pool, .descriptorSetCount = 1,
                                        .pSetLayouts = &s_zconv.dsl };
    if (pvkAllocateDescriptorSets(s_vk.device, &dai, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkDescriptorImageInfo ii[2] = {
        { VK_NULL_HANDLE, dview, VK_IMAGE_LAYOUT_GENERAL },
        { VK_NULL_HANDLE, sview, VK_IMAGE_LAYOUT_GENERAL },
    };
    VkWriteDescriptorSet ws[2];
    ws[0] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                    .dstSet = set, .dstBinding = RSX_SPIRV_TEXTURE_BINDING,
                                    .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                    .pImageInfo = &ii[0] };
    u32 wn = 1;
    if (s_zconv.use_stencil) {
        ws[1] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                        .dstSet = set, .dstBinding = RSX_SPIRV_TEXTURE_BINDING + 1u,
                                        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                        .pImageInfo = &ii[1] };
        wn = 2;
    }
    pvkUpdateDescriptorSets(s_vk.device, wn, ws, 0, NULL);
    s_zconv.sets[s_zconv.set_count].dview = dview;
    s_zconv.sets[s_zconv.set_count].sview = sview;
    s_zconv.sets[s_zconv.set_count].set = set;
    s_zconv.set_count++;
    return set;
}

static u32 vk_eng_depth_color_snapshot(void* user, u32 depth, u32 w, u32 h)
{
    vk_eng_obj* z = vk_eng_get(depth, VK_ENG_DEPTH);
    if (!z || !z->depth_srv) return 0;
    if (vk_zconv_init() != 0) return 0;
    if (s_zconv.use_stencil && !z->stencil_srv) return 0;
    if (w > z->w) w = z->w;
    if (h > z->h) h = z->h;
    if (!w || !h) return 0;

    u32 csnap = z->color_snap;
    vk_eng_obj* c = vk_eng_get(csnap, VK_ENG_COLOR);
    if (!c || c->snapshot != 2 || c->w != w || c->h != h || !c->zconv_fb) {
        if (c) vk_eng_release(user, csnap);
        z = vk_eng_get(depth, VK_ENG_DEPTH);
        if (!z) return 0;
        z->color_snap = 0;
        csnap = vk_eng_alloc();
        if (!csnap) return 0;
        c = &s_eobj[csnap];
        if (vk_create_image(&c->im, VK_FORMAT_R8G8B8A8_UNORM, w, h,
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            VK_IMAGE_ASPECT_COLOR_BIT)) {
            vk_destroy_image(&c->im); memset(c, 0, sizeof *c); return 0;
        }
        VkFramebufferCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                        .renderPass = s_zconv.rp, .attachmentCount = 1,
                                        .pAttachments = &c->im.view, .width = w, .height = h, .layers = 1 };
        if (pvkCreateFramebuffer(s_vk.device, &fci, NULL, &c->zconv_fb) != VK_SUCCESS) {
            vk_destroy_image(&c->im); memset(c, 0, sizeof *c); return 0;
        }
        c->kind = VK_ENG_COLOR; c->snapshot = 2;
        c->fmt = VK_FORMAT_R8G8B8A8_UNORM; c->bpp = 4; c->w = w; c->h = h;
        z = vk_eng_get(depth, VK_ENG_DEPTH);
        z->color_snap = csnap;
        if (vk_begin()) { vk_eng_release(user, csnap); z->color_snap = 0; return 0; }
        vk_barrier_image(c->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
        if (vk_submit_and_wait()) { vk_eng_release(user, csnap); z->color_snap = 0; return 0; }
        z = vk_eng_get(depth, VK_ENG_DEPTH);
        c = vk_eng_get(csnap, VK_ENG_COLOR);
        if (!z || !c || !c->zconv_fb) return 0;
    }
    VkDescriptorSet set = vk_zconv_set_for(z->depth_srv, z->stencil_srv);
    if (!set) return 0;
    const int batch = vk_eb_active();
    if (batch && vk_rec_use_batch()) {
        if (!s_eb_pending && vk_rec_ensure_open()) return 0;
        vk_rec_emit_zconv(c->zconv_fb, set, c->im.img, w, h);
        vk_eb_recorded();
        return csnap;
    }
    int rc = batch ? (s_eb_pending ? 0 : vk_eb_open_cmd()) : vk_begin();
    if (!rc) {
        if (batch) {
            vk_eb_end_rp();
            vk_cmd_after_everything(s_vk.cmd);
            vk_eb_touch_clear();
        } else {
            vk_barrier_image(z->im.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
            vk_barrier_image(c->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
        }
        VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                      .renderPass = s_zconv.rp, .framebuffer = c->zconv_fb,
                                      .renderArea = { { 0, 0 }, { w, h } } };
        pvkCmdBeginRenderPass(s_vk.cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        pvkCmdBindPipeline(s_vk.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_zconv.pipe);
        VkViewport vp = { 0.0f, 0.0f, (float)w, (float)h, 0.0f, 1.0f };
        VkRect2D sc = { { 0, 0 }, { w, h } };
        pvkCmdSetViewport(s_vk.cmd, 0, 1, &vp);
        pvkCmdSetScissor(s_vk.cmd, 0, 1, &sc);
        pvkCmdBindDescriptorSets(s_vk.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 s_zconv.pipe_layout, 0, 1, &set, 0, NULL);
        pvkCmdDraw(s_vk.cmd, 3, 1, 0, 0);
        pvkCmdEndRenderPass(s_vk.cmd);
        if (batch) {
            vk_eb_inval_binds();
            vk_eb_touch(c->im.img);
            vk_eb_recorded();
        } else {
            rc = vk_submit_and_wait();
        }
    }
    return rc ? 0 : csnap;
}

/* A pixel program with a cube unit declares all sixteen units one by one
 * (TextureCube or Texture2D rsx_texN : register(tN)) instead of the
 * Texture2D array, and register(tN) is binding RSX_SPIRV_TEXTURE_BINDING + N
 * (rsx_shader_spirv.h): sixteen single bindings, where the shared layout has
 * one binding of sixteen. Those pipelines use this layout, made on first use
 * so a title without cube maps never builds it. */
static VkPipelineLayout vk_eng_split_layout(void)
{
    if (s_e2.split_pipe_layout) return s_e2.split_pipe_layout;
    const VkShaderStageFlags both = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding b[3 + RSX_MAX_TEXTURES];
    b[0] = (VkDescriptorSetLayoutBinding){ RSX_SPIRV_VPCONST_BINDING, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, both, NULL };
    b[1] = (VkDescriptorSetLayoutBinding){ RSX_SPIRV_PSCONST_BINDING, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, both, NULL };
    b[2] = (VkDescriptorSetLayoutBinding){ RSX_SPIRV_SAMPLER_BINDING, VK_DESCRIPTOR_TYPE_SAMPLER,
                                           RSX_MAX_TEXTURES, VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
    for (u32 u = 0; u < RSX_MAX_TEXTURES; u++)
        b[3 + u] = (VkDescriptorSetLayoutBinding){ RSX_SPIRV_TEXTURE_BINDING + u,
                                                   VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                                                   VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
    VkDescriptorSetLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .bindingCount = 3 + RSX_MAX_TEXTURES, .pBindings = b };
    VkDescriptorPoolSize ps[3] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, RSX_MAX_TEXTURES },
        { VK_DESCRIPTOR_TYPE_SAMPLER, RSX_MAX_TEXTURES },
    };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = 1, .poolSizeCount = 3, .pPoolSizes = ps };
    if (pvkCreateDescriptorSetLayout(s_vk.device, &lci, NULL, &s_e2.split_layout) != VK_SUCCESS ||
        pvkCreateDescriptorPool(s_vk.device, &pci, NULL, &s_e2.split_pool) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = s_e2.split_pool, .descriptorSetCount = 1,
                                        .pSetLayouts = &s_e2.split_layout };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1, .pSetLayouts = &s_e2.split_layout };
    if (pvkAllocateDescriptorSets(s_vk.device, &dai, &s_e2.split_set) != VK_SUCCESS ||
        pvkCreatePipelineLayout(s_vk.device, &plci, NULL, &s_e2.split_pipe_layout) != VK_SUCCESS) {
        s_e2.split_pipe_layout = VK_NULL_HANDLE;
        return VK_NULL_HANDLE;
    }
    return s_e2.split_pipe_layout;
}

static VkPipelineLayout vk_eng_ssbo_split_layout(void)
{
    if (s_e2.ssbo_split_pipe_layout) return s_e2.ssbo_split_pipe_layout;
    if (!s_g.ssbo_ok) return VK_NULL_HANDLE;
    const VkShaderStageFlags both = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding b[1 + RSX_MAX_TEXTURES];
    b[0] = (VkDescriptorSetLayoutBinding){ RSX_SPIRV_SAMPLER_BINDING, VK_DESCRIPTOR_TYPE_SAMPLER,
                                           RSX_MAX_TEXTURES, VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
    for (u32 u = 0; u < RSX_MAX_TEXTURES; u++)
        b[1 + u] = (VkDescriptorSetLayoutBinding){ RSX_SPIRV_TEXTURE_BINDING + u,
                                                   VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                                                   VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
    VkDescriptorSetLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .bindingCount = 1 + RSX_MAX_TEXTURES, .pBindings = b };
    VkDescriptorPoolSize ps[2] = {
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, RSX_MAX_TEXTURES },
        { VK_DESCRIPTOR_TYPE_SAMPLER, RSX_MAX_TEXTURES },
    };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = ps };
    if (pvkCreateDescriptorSetLayout(s_vk.device, &lci, NULL, &s_e2.ssbo_split_tex_layout) != VK_SUCCESS ||
        pvkCreateDescriptorPool(s_vk.device, &pci, NULL, &s_e2.ssbo_split_pool) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = s_e2.ssbo_split_pool, .descriptorSetCount = 1,
                                        .pSetLayouts = &s_e2.ssbo_split_tex_layout };
    VkDescriptorSetLayout sets[2] = { s_g.ssbo_const_layout, s_e2.ssbo_split_tex_layout };
    VkPushConstantRange pcr = { both, 0u, 8u };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 2, .pSetLayouts = sets,
                                        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    if (pvkAllocateDescriptorSets(s_vk.device, &dai, &s_e2.ssbo_split_tex_set) != VK_SUCCESS ||
        pvkCreatePipelineLayout(s_vk.device, &plci, NULL, &s_e2.ssbo_split_pipe_layout) != VK_SUCCESS) {
        s_e2.ssbo_split_pipe_layout = VK_NULL_HANDLE;
        return VK_NULL_HANDLE;
    }
    return s_e2.ssbo_split_pipe_layout;
}

/* VK_CONST_SSBO=0 keeps dynamic UBO rebinds; default 1 uses persistently bound
 * set-0 SSBO + 4B/8B push-constant offsets. VK_CONST_SSBO=a or
 * VK_CONST_SSBO_PHASE=<0|1> runs SSBO on that PS3_AB_PHASE parity only. */
static void vk_const_ssbo_cfg(int* out_en, int* out_phase)
{
    static int cfg = -1, phase = -1;
    if (cfg < 0) {
        const char* e = getenv("VK_CONST_SSBO");
        if (e && (e[0] == 'a' || e[0] == 'A')) { cfg = 1; phase = 0; }
        else { cfg = (e && *e == '0') ? 0 : 1; }
        const char* p = getenv("VK_CONST_SSBO_PHASE");
        if (p && *p) phase = atoi(p) & 1;
    }
    if (out_en) *out_en = cfg;
    if (out_phase) *out_phase = phase;
}

/* VK_INDIRECT_BATCH=0 disables vkCmdDrawIndexedIndirect + firstInstance vp_off
 * coalescing; default 1 enables it when multiDrawIndirect + drawIndirectFirstInstance
 * and SSBO are available. VK_INDIRECT_BATCH=a or VK_INDIRECT_BATCH_PHASE=<0|1>
 * enables in-run A/B on that PS3_AB_PHASE parity. */
static void vk_indirect_batch_cfg(int* out_en, int* out_phase)
{
    static int cfg = -1, phase = -1;
    if (cfg < 0) {
        const char* e = getenv("VK_INDIRECT_BATCH");
        if (e && (e[0] == 'a' || e[0] == 'A')) { cfg = 1; phase = 0; }
        else { cfg = (e && *e == '0') ? 0 : 1; }
        const char* p = getenv("VK_INDIRECT_BATCH_PHASE");
        if (p && *p) phase = atoi(p) & 1;
    }
    if (out_en) *out_en = cfg;
    if (out_phase) *out_phase = phase;
}

/* Which units a pixel program declares as TextureCube. */
static u32 vk_eng_cube_units(const char* ps_hlsl)
{
    u32 mask = 0;
    char decl[40];
    for (u32 u = 0; u < RSX_MAX_TEXTURES; u++) {
        snprintf(decl, sizeof decl, "TextureCube rsx_tex%u ", u);
        if (strstr(ps_hlsl, decl)) mask |= 1u << u;
    }
    return mask;
}

static u32 vk_eng_pipeline_create(void* user, const char* vs_hlsl, const char* ps_hlsl,
                                  const rsx_be_render_state* rs, const rsx_vertex_layout_plan* layout,
                                  u32 vertex_stride, rsx_be_format rt_fmt, u32 rt_count)
{
    (void)user;
    /* Same gate as Metal: no guest programs, no engine pipelines. */
    if (!s_g.on || !vs_hlsl || !ps_hlsl || !rs || !layout || !vertex_stride) return 0;
    u32 rt_bpp;
    const VkFormat cf = vk_eng_format(rt_fmt, &rt_bpp);
    if (vk_eng_rt_index(cf) < 0) {
        if (!(s_e2.warned & 1u)) { VK_LOG("engine: colour format %d not supported yet\n", (int)rt_fmt); s_e2.warned |= 1u; }
        return 0;
    }
    if (!rt_count) rt_count = 1;
    if (rt_count > RSX_BE_MAX_COLOR_TARGETS) rt_count = RSX_BE_MAX_COLOR_TARGETS;
    u32 slot = VK_ENG_MAX_PIPE;
    for (u32 i = 0; i < VK_ENG_MAX_PIPE; i++) if (!s_epipe[i].used) { slot = i; break; }
    if (slot == VK_ENG_MAX_PIPE) {
        if (!(s_e2.warned & 128u)) {
            VK_LOG("engine: pipeline table full (%u), draws will drop\n", (unsigned)VK_ENG_MAX_PIPE);
            s_e2.warned |= 128u;
        }
        return 0;
    }
    vk_eng_pipe* P = &s_epipe[slot];
    memset(P, 0, sizeof *P);
    int ssbo_cfg = 0, ssbo_phase = -1;
    int ind_cfg = 0, ind_phase = -1;
    vk_const_ssbo_cfg(&ssbo_cfg, &ssbo_phase);
    vk_indirect_batch_cfg(&ind_cfg, &ind_phase);
    const int need_ssbo = (ssbo_cfg && s_g.ssbo_ok);
    const int need_ubo  = (!need_ssbo || ssbo_phase >= 0);
    const int need_inst = (need_ssbo && ind_cfg && s_vk.multi_draw_indirect_ok);
    const int need_ssbo_vs = (need_ssbo && (!need_inst || ind_phase >= 0));
    if (need_ubo) {
        if (vk_eng_translate_ex(vs_hlsl, RSX_SHADER_STAGE_VERTEX, 0u, &P->vs)) return 0;
        if (vk_eng_translate_ex(ps_hlsl, RSX_SHADER_STAGE_FRAGMENT, 0u, &P->fs)) {
            pvkDestroyShaderModule(s_vk.device, P->vs, NULL); memset(P, 0, sizeof *P); return 0;
        }
    }
    if (need_ssbo) {
        int ok = (vk_eng_translate_ex(ps_hlsl, RSX_SHADER_STAGE_FRAGMENT, RSX_SPIRV_FLAG_CONST_SSBO, &P->fs_ssbo) == 0);
        if (ok && need_ssbo_vs)
            ok = (vk_eng_translate_ex(vs_hlsl, RSX_SHADER_STAGE_VERTEX, RSX_SPIRV_FLAG_CONST_SSBO, &P->vs_ssbo) == 0);
        if (ok && need_inst)
            ok = (vk_eng_translate_ex(vs_hlsl, RSX_SHADER_STAGE_VERTEX, RSX_SPIRV_FLAG_CONST_SSBO_INST, &P->vs_inst) == 0);
        if (!ok) {
            if (P->vs_inst) pvkDestroyShaderModule(s_vk.device, P->vs_inst, NULL);
            if (P->vs_ssbo) pvkDestroyShaderModule(s_vk.device, P->vs_ssbo, NULL);
            if (P->fs_ssbo) pvkDestroyShaderModule(s_vk.device, P->fs_ssbo, NULL);
            if (P->vs) pvkDestroyShaderModule(s_vk.device, P->vs, NULL);
            if (P->fs) pvkDestroyShaderModule(s_vk.device, P->fs, NULL);
            memset(P, 0, sizeof *P);
            return 0;
        }
    }
    P->rs = *rs;
    P->nslots = layout->count;
    P->stride = vertex_stride;
    P->rt_count = rt_count;
    P->rt_fmt = cf;
    P->cube_mask = vk_eng_cube_units(ps_hlsl);
    P->depth_or_stencil_write = (u8)((rs->depth_write != 0u) || (rs->stencil_enable && rs->s_write_mask != 0u));
    P->used = 1;
    return slot + 1u;
}

static void vk_eng_pipeline_release(void* user, u32 pipeline)
{
    (void)user;
    if (s_eb_pending || vk_eb_any_inflight()) vk_eng_flush();
    if (!pipeline || pipeline > VK_ENG_MAX_PIPE || !s_epipe[pipeline - 1].used) return;
    vk_eng_pipe* P = &s_epipe[pipeline - 1];
    for (u32 t = 0; t < VK_ENG_TOPOLOGIES; t++) {
        if (P->variant[t]) pvkDestroyPipeline(s_vk.device, P->variant[t], NULL);
        if (P->variant_ssbo[t]) pvkDestroyPipeline(s_vk.device, P->variant_ssbo[t], NULL);
        if (P->variant_inst[t]) pvkDestroyPipeline(s_vk.device, P->variant_inst[t], NULL);
    }
    if (P->vs) pvkDestroyShaderModule(s_vk.device, P->vs, NULL);
    if (P->fs) pvkDestroyShaderModule(s_vk.device, P->fs, NULL);
    if (P->vs_ssbo) pvkDestroyShaderModule(s_vk.device, P->vs_ssbo, NULL);
    if (P->fs_ssbo) pvkDestroyShaderModule(s_vk.device, P->fs_ssbo, NULL);
    if (P->vs_inst) pvkDestroyShaderModule(s_vk.device, P->vs_inst, NULL);
    memset(P, 0, sizeof *P);
    if (s_e2.pipeline == pipeline) s_e2.pipeline = 0;
}

static VkPipeline vk_eng_variant(vk_eng_pipe* P, u32 topo_idx, int ssbo_mode)
{
    static const VkPrimitiveTopology topo[VK_ENG_TOPOLOGIES] = {
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
        VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP };
    VkPipeline* dst_pipe = (ssbo_mode == 2) ? &P->variant_inst[topo_idx]
                         : (ssbo_mode == 1) ? &P->variant_ssbo[topo_idx]
                                            : &P->variant[topo_idx];
    if (*dst_pipe) return *dst_pipe;
    VkShaderModule vs_mod = (ssbo_mode == 2) ? P->vs_inst
                          : (ssbo_mode == 1) ? P->vs_ssbo : P->vs;
    VkShaderModule fs_mod = (ssbo_mode != 0) ? P->fs_ssbo : P->fs;
    if (!vs_mod || !fs_mod) return VK_NULL_HANDLE;
    VkRenderPass rp = vk_eng_render_pass(P->rt_fmt, P->rt_count);
    if (!rp || P->nslots > VK_GUEST_ATTRS) return VK_NULL_HANDLE;
    const VkPipelineLayout layout = (ssbo_mode != 0)
        ? (P->cube_mask ? vk_eng_ssbo_split_layout() : s_g.ssbo_pipe_layout)
        : (P->cube_mask ? vk_eng_split_layout() : s_g.pipe_layout);
    if (!layout) return VK_NULL_HANDLE;
    const rsx_be_render_state* rs = &P->rs;

    VkPipelineShaderStageCreateInfo st[2] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs_mod, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs_mod, .pName = "main" } };
    VkVertexInputBindingDescription vb = { 0, P->stride, VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription va[VK_GUEST_ATTRS];
    for (u32 s = 0; s < P->nslots; s++)
        va[s] = (VkVertexInputAttributeDescription){ s, 0, VK_FORMAT_R32G32B32A32_SFLOAT, s * 16u };
    VkPipelineVertexInputStateCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vb,
        .vertexAttributeDescriptionCount = P->nslots, .pVertexAttributeDescriptions = va };
    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = topo[topo_idx] };
    VkPipelineViewportStateCreateInfo vps = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1 };
    /* CULL_FACE FRONT 0x0404 / BACK 0x0405 / FRONT_AND_BACK 0x0408 (front, as
     * in D3D12 and Metal); FRONT_FACE CCW 0x0901 maps directly: both APIs
     * judge winding on screen, and the Y flip in the vertex stage makes this
     * screen the same one D3D's viewport produces. */
    VkPipelineRasterizationStateCreateInfo ras = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = (rs->depth_clamp && s_vk.depth_clamp_ok) ? VK_TRUE : VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = !rs->cull_enable ? VK_CULL_MODE_NONE
                  : (rs->cull_face == 0x0404u || rs->cull_face == 0x0408u) ? VK_CULL_MODE_FRONT_BIT
                                                                           : VK_CULL_MODE_BACK_BIT,
        .frontFace = (rs->front_face == 0x0901u) ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                 : VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_TRUE,   /* dynamic via vk_eng_set_depth_bias (#702) */
        .lineWidth = 1.0f };
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkStencilOpState front = vk_gcm_stencil_face(rs->s_func, rs->s_fail, rs->s_zfail, rs->s_zpass,
                                                 rs->s_func_mask, rs->s_write_mask);
    VkStencilOpState back = rs->stencil_two_sided
        ? vk_gcm_stencil_face(rs->bs_func, rs->bs_fail, rs->bs_zfail, rs->bs_zpass,
                              rs->s_func_mask, rs->s_write_mask)
        : front;
    VkPipelineDepthStencilStateCreateInfo ds = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = rs->depth_test ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = (rs->depth_test && rs->depth_write) ? VK_TRUE : VK_FALSE,
        .depthCompareOp = vk_gcm_compare(rs->depth_func),
        .stencilTestEnable = rs->stencil_enable ? VK_TRUE : VK_FALSE,
        .front = front, .back = back,
        /* The bounds themselves are dynamic (vk_eng_set_depth_bounds). */
        .depthBoundsTestEnable = (rs->depth_bounds_test && s_vk.depth_bounds_ok) ? VK_TRUE : VK_FALSE,
        .minDepthBounds = 0.0f, .maxDepthBounds = 1.0f };
    if (rs->depth_bounds_test && !s_vk.depth_bounds_ok && !(s_e2.warned & 64u)) {
        VK_LOG("engine: the device has no depthBounds; the depth bounds test is ignored\n");
        s_e2.warned |= 64u;
    }
    /* nv40 COLOR_MASK: B [0:7], G [8:15], R [16:23], A [24:31], any bit set.
     * Target A takes COLOR_MASK and BLEND_ENABLE; targets B..D their own
     * nibble of COLOR_MASK_MRT (A,R,G,B at bits 0..3) and bit of
     * BLEND_ENABLE_MRT, the hardware rule the Metal backend follows. With A's
     * state on every target, a light pass that adds into A with B and C
     * masked off added every light into the G-buffer, and a material pass
     * that writes B and C's alpha with A's alpha masked left them stale.
     * PS3RECOMP_VK_MRT_MASK takes the same values as PS3RECOMP_METAL_MRT_MASK:
     * 1 (default) the hardware rule; 0 target A's state everywhere, the old
     * behaviour; 2 only a target masked off entirely is off, with per-target
     * blend. A device without independentBlend keeps A's state everywhere. */
    VkColorComponentFlags wm = 0;
    if ((rs->color_mask >> 16) & 0xFFu) wm |= VK_COLOR_COMPONENT_R_BIT;
    if ((rs->color_mask >>  8) & 0xFFu) wm |= VK_COLOR_COMPONENT_G_BIT;
    if ((rs->color_mask >>  0) & 0xFFu) wm |= VK_COLOR_COMPONENT_B_BIT;
    if ((rs->color_mask >> 24) & 0xFFu) wm |= VK_COLOR_COMPONENT_A_BIT;
    static int mrt_mode = -1;
    if (mrt_mode < 0) { const char* e = getenv("PS3RECOMP_VK_MRT_MASK"); mrt_mode = e ? atoi(e) : 1; }
    VkPipelineColorBlendAttachmentState cba[RSX_BE_MAX_COLOR_TARGETS];
    /* A device without blending on a float target draws it unblended; that is
     * reported once, never silently. */
    const int blendable = vk_eng_rt_blendable(P->rt_fmt);
    for (u32 r = 0; r < P->rt_count; r++) {
        VkColorComponentFlags twm = wm;
        int tblend = rs->blend_enable != 0;
        if (r > 0 && mrt_mode != 0 && s_vk.indep_blend_ok) {
            const u32 nib = (rs->color_mask_mrt >> (r * 4u)) & 0xFu;
            if (mrt_mode == 1 || nib == 0) {
                twm = 0;
                if (nib & 1u) twm |= VK_COLOR_COMPONENT_A_BIT;
                if (nib & 2u) twm |= VK_COLOR_COMPONENT_R_BIT;
                if (nib & 4u) twm |= VK_COLOR_COMPONENT_G_BIT;
                if (nib & 8u) twm |= VK_COLOR_COMPONENT_B_BIT;
            }
            tblend = (rs->blend_mrt >> r) & 1u;
        }
        if (tblend && !blendable) {
            tblend = 0;
            if (!(s_e2.warned & 256u)) {
                VK_LOG("engine: blending unsupported for colour format %d on this device; drawing unblended\n",
                       (int)P->rt_fmt);
                s_e2.warned |= 256u;
            }
        }
        cba[r] = (VkPipelineColorBlendAttachmentState){
            .blendEnable = tblend ? VK_TRUE : VK_FALSE,
            .srcColorBlendFactor = vk_gcm_blend_factor(rs->sf_rgb),
            .dstColorBlendFactor = vk_gcm_blend_factor(rs->df_rgb),
            .colorBlendOp = vk_gcm_blend_op(rs->eq_rgb),
            .srcAlphaBlendFactor = vk_gcm_blend_factor(rs->sf_a),
            .dstAlphaBlendFactor = vk_gcm_blend_factor(rs->df_a),
            .alphaBlendOp = vk_gcm_blend_op(rs->eq_a),
            .colorWriteMask = twm };
    }
    VkPipelineColorBlendStateCreateInfo cb = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = P->rt_count, .pAttachments = cba };
    VkDynamicState dyn[6] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                              VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_DEPTH_BIAS,
                              VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_DEPTH_BOUNDS };
    VkPipelineDynamicStateCreateInfo dy = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = s_vk.depth_bounds_ok ? 6u : 5u, .pDynamicStates = dyn };
    VkGraphicsPipelineCreateInfo gci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &vps, .pRasterizationState = &ras, .pMultisampleState = &ms,
        .pDepthStencilState = &ds, .pColorBlendState = &cb, .pDynamicState = &dy,
        .layout = layout, .renderPass = rp, .subpass = 0 };
    const double pt0 = g_rsx_sf.enabled ? vk_now_ms() : 0.0;
    if (pvkCreateGraphicsPipelines(s_vk.device, VK_NULL_HANDLE, 1, &gci, NULL,
                                   dst_pipe) != VK_SUCCESS) {
        VK_LOG("engine: vkCreateGraphicsPipelines failed\n");
        *dst_pipe = VK_NULL_HANDLE;
    }
    if (g_rsx_sf.enabled) {
        g_rsx_sf.vk_pipe_creates++;
        g_rsx_sf.vk_pipe_ms += vk_now_ms() - pt0;
    }
    return *dst_pipe;
}

static void vk_eng_bind_pipeline(void* user, u32 pipeline) { (void)user; s_e2.pipeline = pipeline; }

static void vk_eng2_shutdown(void* user)
{
    vk_zconv_shutdown();
    for (u32 i = 1; i <= VK_ENG_MAX_PIPE; i++) vk_eng_pipeline_release(user, i);
    vk_eng_drop_framebuffers();
    for (u32 f = 0; f < VK_ENG_RT_FORMATS; f++)
        for (u32 n = 0; n <= RSX_BE_MAX_COLOR_TARGETS; n++)
            if (s_e2.rp[f][n]) pvkDestroyRenderPass(s_vk.device, s_e2.rp[f][n], NULL);
    vk_destroy_image(&s_e2.scratch_depth);
    vk_destroy_buffer(&s_e2.indices);
    if (s_e2.ssbo_split_pipe_layout) pvkDestroyPipelineLayout(s_vk.device, s_e2.ssbo_split_pipe_layout, NULL);
    if (s_e2.ssbo_split_pool)        pvkDestroyDescriptorPool(s_vk.device, s_e2.ssbo_split_pool, NULL);
    if (s_e2.ssbo_split_tex_layout)  pvkDestroyDescriptorSetLayout(s_vk.device, s_e2.ssbo_split_tex_layout, NULL);
    if (s_e2.split_pipe_layout) pvkDestroyPipelineLayout(s_vk.device, s_e2.split_pipe_layout, NULL);
    if (s_e2.split_pool)        pvkDestroyDescriptorPool(s_vk.device, s_e2.split_pool, NULL);
    if (s_e2.split_layout)      pvkDestroyDescriptorSetLayout(s_vk.device, s_e2.split_layout, NULL);
    free(s_e2.ucpu);
    memset(&s_e2, 0, sizeof s_e2);
    for (u32 i = 0; i < s_esamp_count; i++) pvkDestroySampler(s_vk.device, s_esamp[i].s, NULL);
    s_esamp_count = 0;
}

/* Constants are kept in ordinary memory laid out like the shared UBO (vertex
 * at 0, pixel at ps_off) and copied to the GPU per draw: into the draw's ring
 * slice when batched, into the shared UBO otherwise (every such draw is waited
 * on, so overwriting it is safe). Writing them straight into the UBO meant a
 * batched draw read them back out of it, and that memory is host-coherent,
 * i.e. write-combined and uncached on most drivers. */
static u8* vk_eng_ucpu(void)
{
    if (!s_e2.ucpu && s_g.ubo.size) s_e2.ucpu = (u8*)calloc(1, (size_t)s_g.ubo.size);
    return s_e2.ucpu;
}
static void vk_eng_bind_vs_constants(void* user, const void* data, u32 bytes)
{
    (void)user;
    if (!s_g.on || !data || !vk_eng_ucpu()) { s_e2.vs_bytes = 0; s_e2.vs_src = NULL; return; }
    const u32 cap = (u32)s_g.ps_off;
    if (bytes > cap) { bytes = cap; if (!(s_e2.warned & 2u)) { VK_LOG("engine: VS constants truncated\n"); s_e2.warned |= 2u; } }
    s_e2.vs_src = data;
    if (!vk_eb_active() || !vk_eb_async_use())
        memcpy(s_e2.ucpu, data, bytes);
    s_e2.vs_bytes = bytes;
    s_e2.vs_dirty = 1;
}
static void vk_eng_bind_ps_constants(void* user, const void* data, u32 bytes)
{
    (void)user;
    if (!s_g.on || !data || !vk_eng_ucpu()) { s_e2.ps_bytes = 0; return; }
    const u32 cap = (u32)(s_g.ubo.size - s_g.ps_off);
    if (bytes > cap) { bytes = cap; if (!(s_e2.warned & 4u)) { VK_LOG("engine: PS constants truncated\n"); s_e2.warned |= 4u; } }
    memcpy(s_e2.ucpu + s_g.ps_off, data, bytes);
    s_e2.ps_bytes = bytes;
    s_e2.ps_dirty = 1;
}
/* The image a bound engine handle samples: a texture's own, or the target a
 * view was made from. The hazard check compares images, because a crossbar
 * view is a different handle over the same memory. */
static VkImage vk_eng_handle_image(u32 handle);
static VkSampler vk_eng_sampler(const rsx_be_sampler_desc* d);
static u8 vk_eng_sampler_idx(const rsx_be_sampler_desc* d, VkSampler* out_vk);

static void vk_eng_bind_textures(void* user, const u32* textures,
                                 const rsx_be_sampler_desc* samplers, u32 mask)
{
    (void)user;
    const int m2e4 = vk_m2e4_use();
    int changed = (s_e2.tex_mask != mask);
    const u32 clear = s_e2.tex_mask & ~mask;
    if (clear) {
        u32 cm = clear;
        while (cm) {
            const u32 u = (u32)__builtin_ctz(cm);
            cm &= cm - 1u;
            s_e2.tex[u] = 0;
            s_e2.tex_img[u] = VK_NULL_HANDLE;
            s_e2.samp_vk[u] = VK_NULL_HANDLE;
            s_e2.samp_idx[u] = 0;
        }
        s_e2.tex_rt_mask &= ~clear;
    }
    u32 rt_mask = m2e4 ? s_e2.tex_rt_mask : 0u;
    u32 m = mask;
    while (m) {
        const u32 u = (u32)__builtin_ctz(m);
        m &= m - 1u;
        const u32 hd = textures[u];
        const int hd_chg = (s_e2.tex[u] != hd);
        if (hd_chg) {
            changed = 1;
            s_e2.tex[u] = hd;
        }
        if (!m2e4 || hd_chg) {
            rt_mask &= ~(1u << u);
            if (hd && hd < VK_ENG_MAX_OBJ) {
                const vk_eng_obj* o = &s_eobj[hd];
                if (o->kind == VK_ENG_COLOR) {
                    s_e2.tex_img[u] = o->im.img;
                    rt_mask |= 1u << u;
                } else if (o->kind == VK_ENG_VIEW) {
                    const vk_eng_obj* t = vk_eng_get(o->surface, VK_ENG_COLOR);
                    s_e2.tex_img[u] = t ? t->im.img : VK_NULL_HANDLE;
                    if (t) rt_mask |= 1u << u;
                } else {
                    s_e2.tex_img[u] = (o->kind == VK_ENG_TEXTURE) ? o->im.img : VK_NULL_HANDLE;
                }
            } else {
                s_e2.tex_img[u] = VK_NULL_HANDLE;
            }
        }
        if (samplers) {
            if (!s_e2.samp_vk[u] || memcmp(&s_e2.samp[u], &samplers[u], sizeof(rsx_be_sampler_desc)) != 0) {
                s_e2.samp[u] = samplers[u];
                const u8 nidx = vk_eng_sampler_idx(&samplers[u], &s_e2.samp_vk[u]);
                if (s_e2.samp_idx[u] != nidx) { s_e2.samp_idx[u] = nidx; changed = 1; }
            }
        }
    }
    s_e2.tex_mask = mask;
    s_e2.tex_rt_mask = rt_mask;
    if (rt_mask) vk_eb_inval_fast_rp();
    if (changed) {
        u64 th = (u64)mask * 0x9E3779B97F4A7C15ull;
        u32 hm = mask;
        while (hm) {
            const u32 u = (u32)__builtin_ctz(hm);
            hm &= hm - 1u;
            th ^= ((u64)s_e2.tex[u] << 16) | ((u64)s_e2.samp_idx[u] << 8) | (u64)u;
            th *= 0xBF58476D1CE4E5B9ull;
        }
        s_e2.tex_bind_hash = th;
        s_e2.tex_bind_gen++;
        if (!s_e2.tex_bind_gen) s_e2.tex_bind_gen = 1u;
        s_e2.tex_dirty = 1;
    }
}
static void vk_eng_bind_vertex_textures(void* user, const u32* textures,
                                        const rsx_be_sampler_desc* samplers, u32 mask)
{
    (void)user; (void)textures; (void)samplers;
    if (mask && !(s_e2.warned & 8u)) { VK_LOG("engine: vertex textures not supported yet\n"); s_e2.warned |= 8u; }
}
static void vk_eng_set_viewport(void* user, float x, float y, float w, float h)
{
    (void)user;
    s_e2.vp[0] = x; s_e2.vp[1] = y; s_e2.vp[2] = w; s_e2.vp[3] = h; s_e2.have_vp = 1; s_e2.vp_dirty = 1; s_e2.dyn_dirty_mask |= 1u;
}
static void vk_eng_set_scissor(void* user, u32 x, u32 y, u32 w, u32 h)
{
    (void)user;
    s_e2.sc[0] = x; s_e2.sc[1] = y; s_e2.sc[2] = w; s_e2.sc[3] = h; s_e2.have_sc = 1; s_e2.sc_dirty = 1; s_e2.dyn_dirty_mask |= 2u;
}
static void vk_eng_set_stencil_ref(void* user, u32 ref) { (void)user; s_e2.stencil_ref = ref; s_e2.sref_dirty = 1; s_e2.dyn_dirty_mask |= 4u; }

/* Vulkan wants the bounds inside [0, 1] without VK_EXT_depth_range_unrestricted;
 * a NaN clamps to the open end so a garbage register cannot reject a draw. */
static float vk_depth_bound(float v, float open)
{
    if (!(v == v)) return open;
    return v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
}

static void vk_eng_set_depth_bounds(void* user, float min, float max)
{
    (void)user;
    s_e2.depth_bounds[0] = vk_depth_bound(min, 0.0f);
    s_e2.depth_bounds[1] = vk_depth_bound(max, 1.0f);
    s_e2.db_dirty = 1;
    s_e2.dyn_dirty_mask |= 8u;
}
static void vk_eng_set_depth_bias(void* user, float slope, float units, u32 depth_bits)
{
    (void)user;
    s_e2.depth_bias[0] = (depth_bits == 16u) ? units * 256.0f : units;
    s_e2.depth_bias[1] = slope;
    s_e2.bias_dirty = 1;
    s_e2.dyn_dirty_mask |= 16u;
}
static void vk_eng_set_blend_color(void* user, const float rgba[4])
{
    (void)user;
    memcpy(s_e2.blend_color, rgba, sizeof s_e2.blend_color);
    s_e2.bc_dirty = 1;
    s_e2.dyn_dirty_mask |= 32u;
}

static VkFramebuffer vk_eng_framebuffer(VkRenderPass rp, const u32* rt, u32 n, VkImageView dview,
                                        u32 dkey, u32 w, u32 h)
{
    if (s_e2.last_fb && s_e2.last_fb_rp == rp && s_e2.last_fb_n == n && s_e2.last_fb_dkey == dkey) {
        int same = 1;
        for (u32 i = 0; i < n; i++) if (s_e2.last_fb_rt[i] != rt[i]) { same = 0; break; }
        if (same) return s_e2.last_fb;
    }
    u32 key[RSX_BE_MAX_COLOR_TARGETS + 1] = { 0 };
    for (u32 i = 0; i < n; i++) key[i] = rt[i];
    key[RSX_BE_MAX_COLOR_TARGETS] = dkey;
    for (u32 i = 0; i < s_e2.fb_count; i++) {
        if (!memcmp(s_e2.fb[i].key, key, sizeof key)) {
            s_e2.last_fb = s_e2.fb[i].fb; s_e2.last_fb_rp = rp; s_e2.last_fb_n = n;
            s_e2.last_fb_dkey = dkey;
            for (u32 j = 0; j < n; j++) s_e2.last_fb_rt[j] = rt[j];
            return s_e2.fb[i].fb;
        }
    }
    if (s_e2.fb_count >= VK_ENG_FB_CACHE) vk_eng_drop_framebuffers();
    VkImageView views[RSX_BE_MAX_COLOR_TARGETS + 1];
    for (u32 i = 0; i < n; i++) views[i] = s_eobj[rt[i]].im.view;
    views[n] = dview;
    VkFramebufferCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                    .renderPass = rp, .attachmentCount = n + 1, .pAttachments = views,
                                    .width = w, .height = h, .layers = 1 };
    VkFramebuffer fb = VK_NULL_HANDLE;
    if (pvkCreateFramebuffer(s_vk.device, &fci, NULL, &fb) != VK_SUCCESS) return VK_NULL_HANDLE;
    memcpy(s_e2.fb[s_e2.fb_count].key, key, sizeof key);
    s_e2.fb[s_e2.fb_count++].fb = fb;
    return fb;
}

/* Depth when the draw names no zeta: one scratch image, grown as needed and
 * cleared to the far plane when (re)created. Key 0xFFFFFFFF in the cache. */
static VkImageView vk_eng_scratch_depth(u32 w, u32 h)
{
    if (s_e2.scratch_depth.view && s_e2.scratch_w >= w && s_e2.scratch_h >= h)
        return s_e2.scratch_depth.view;
    vk_eng_drop_framebuffers();
    vk_destroy_image(&s_e2.scratch_depth);
    if (vk_create_image(&s_e2.scratch_depth, s_vk.depth_format, w, h,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        s_vk.depth_aspect) || vk_begin())
        return VK_NULL_HANDLE;
    vk_barrier_image(s_e2.scratch_depth.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_UNDEFINED);
    VkClearDepthStencilValue far_plane = { 1.0f, 0 };
    VkImageSubresourceRange r = { s_vk.depth_aspect, 0, 1, 0, 1 };
    pvkCmdClearDepthStencilImage(s_vk.cmd, s_e2.scratch_depth.img, VK_IMAGE_LAYOUT_GENERAL, &far_plane, 1, &r);
    if (vk_submit_and_wait()) return VK_NULL_HANDLE;
    s_e2.scratch_w = w; s_e2.scratch_h = h;
    return s_e2.scratch_depth.view;
}

static int vk_eng_ensure_indices(VkDeviceSize bytes)
{
    if (s_e2.indices.buf && s_e2.indices.size >= bytes) return 0;
    VkDeviceSize size = s_e2.indices.size ? s_e2.indices.size : 64 * 1024;
    while (size < bytes) size *= 2;
    vk_destroy_buffer(&s_e2.indices);
    if (vk_create_host_buffer(&s_e2.indices, size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT)) return -1;
    return s_e2.indices.coherent ? 0 : -1;
}

/* rsx_be_sampler_desc is the registers decoded the D3D12 way; the wrap codes
 * stay the guest's 1..8, which vk_gcm_wrap already translates. */
static u8 vk_eng_sampler_idx(const rsx_be_sampler_desc* d, VkSampler* out_vk)
{
    /* Guest border colour (A8R8G8B8 -> Vulkan 1.0 core border enum): inFamous'
     * shadow cascades use border mode with 0x00FFFFFF = depth 1.0 = lit. */
    const u32 bcode = (((d->border_color >> 16) & 0xFFu) >= 0x80u) ? 2u
                    : (((d->border_color >> 24) & 0xFFu) >= 0x80u) ? 1u : 0u;
    const VkBorderColor bcol = bcode == 2u ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
                             : bcode == 1u ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK
                                           : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    u32 lo, hi;
    memcpy(&lo, &d->min_lod, 4); memcpy(&hi, &d->max_lod, 4);
    const u64 key = (u64)d->min_linear | ((u64)d->mag_linear << 1) | ((u64)d->mip_linear << 2)
                  | ((u64)d->mip_present << 3) | ((u64)(d->wrap_s & 0xF) << 4)
                  | ((u64)(d->wrap_t & 0xF) << 8) | ((u64)(d->wrap_r & 0xF) << 12)
                  | ((u64)(lo >> 16) << 16) | ((u64)(hi >> 16) << 32)
                  | ((u64)bcode << 48);
    for (u32 i = 0; i < s_esamp_count; i++) {
        if (s_esamp[i].key == key) {
            if (out_vk) *out_vk = s_esamp[i].s;
            return (u8)(i + 1u);
        }
    }
    if (s_esamp_count >= VK_ENG_SAMP_CACHE) {
        if (out_vk) *out_vk = s_vk.sampler;
        return 0u;
    }
    float min_lod = d->min_lod, max_lod = d->mip_present ? d->max_lod : d->min_lod;
    if (max_lod < min_lod) max_lod = min_lod;
    static float s_aniso_req = -1.0f;
    if (__builtin_expect(s_aniso_req < 0.0f, 0)) {
        /* Default 0 (guest/off) so 15 W Steam Deck keeps GPU headroom & 1% lows;
         * opt-in TEX_ANISO=16 (or 8) on BC-250 / desktop GPUs. */
        const char* e = getenv("TEX_ANISO");
        float a = (e && *e) ? (float)atof(e) : 0.0f;
        if (!(a > 1.0f) || !s_vk.aniso_ok) a = 0.0f;
        else if (a > s_vk.max_aniso) a = s_vk.max_aniso;
        s_aniso_req = a;
        if (s_aniso_req > 1.0f)
            VK_LOG("TEX_ANISO=%.0f (device max %.0f, supported=%d)\n",
                   s_aniso_req, s_vk.max_aniso, s_vk.aniso_ok);
    }
    const int use_aniso = (s_aniso_req > 1.0f && d->min_linear && d->mag_linear && d->mip_present);
    VkSamplerCreateInfo sci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = d->mag_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .minFilter = d->min_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .mipmapMode = d->mip_linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = vk_gcm_wrap(d->wrap_s), .addressModeV = vk_gcm_wrap(d->wrap_t),
        .addressModeW = vk_gcm_wrap(d->wrap_r),
        .mipLodBias = d->mip_present ? rsx_upscale_mip_bias() : 0.0f,
        .anisotropyEnable = use_aniso ? VK_TRUE : VK_FALSE,
        .maxAnisotropy = use_aniso ? s_aniso_req : 1.0f,
        .minLod = min_lod, .maxLod = max_lod,
        .borderColor = bcol };
    VkSampler smp;
    if (pvkCreateSampler(s_vk.device, &sci, NULL, &smp) != VK_SUCCESS) {
        if (out_vk) *out_vk = s_vk.sampler;
        return 0u;
    }
    const u32 idx = s_esamp_count++;
    s_esamp[idx].key = key;
    s_esamp[idx].s = smp;
    if (out_vk) *out_vk = smp;
    return (u8)(idx + 1u);
}

static VkSampler vk_eng_sampler(const rsx_be_sampler_desc* d)
{
    VkSampler s = s_vk.sampler;
    vk_eng_sampler_idx(d, &s);
    return s;
}

/* The view a bound engine handle samples through: an uploaded texture, a
 * colour target (bound directly when surface_view was not asked), or a view.
 * A unit gets a view of its own kind only -- a cube for a TextureCube unit, a
 * 2D one otherwise -- and NULL for anything else, which then samples the
 * dummy of the unit's kind. */
static VkImageView vk_eng_sample_view(u32 handle, int want_cube)
{
    if (!handle || handle >= VK_ENG_MAX_OBJ) return VK_NULL_HANDLE;
    const vk_eng_obj* o = &s_eobj[handle];
    if (o->kind != VK_ENG_TEXTURE && o->kind != VK_ENG_COLOR && o->kind != VK_ENG_VIEW)
        return VK_NULL_HANDLE;
    return ((int)o->cube == want_cube) ? o->im.view : VK_NULL_HANDLE;
}

static VkImage vk_eng_handle_image(u32 handle)
{
    if (!handle || handle >= VK_ENG_MAX_OBJ) return VK_NULL_HANDLE;
    const vk_eng_obj* o = &s_eobj[handle];
    if (o->kind == VK_ENG_VIEW) {
        const vk_eng_obj* t = vk_eng_get(o->surface, VK_ENG_COLOR);
        return t ? t->im.img : VK_NULL_HANDLE;
    }
    return (o->kind == VK_ENG_TEXTURE || o->kind == VK_ENG_COLOR) ? o->im.img : VK_NULL_HANDLE;
}

/* sp00nznet/ps3recomp #754: unbound engine texture units sample transparent
 * black (0,0,0,0), matching Metal, for both 2D and cube placeholders. */
static const u8 s_eng_unbound_texel[4] = { 0x00, 0x00, 0x00, 0x00 };
static VkImageView vk_eng_dummy_cube_view(void)
{
    if (!vk_eng_get(s_e2.dummy_cube, VK_ENG_TEXTURE)) {
        s_e2.dummy_cube = vk_eng_texture_create(NULL, RSX_BE_FMT_R8G8B8A8, 1, 1, 1, 6,
                                                0xAAE4u, 0x85u);
        if (!s_e2.dummy_cube) return VK_NULL_HANDLE;
        for (u32 f = 0; f < 6; f++)
            vk_eng_texture_upload(NULL, s_e2.dummy_cube, f, 0, 1, 1, s_eng_unbound_texel, 4, 1);
    }
    return s_eobj[s_e2.dummy_cube].im.view;
}
static VkImageView vk_eng_dummy_2d_view(void)
{
    if (!vk_eng_get(s_e2.dummy_2d, VK_ENG_TEXTURE)) {
        s_e2.dummy_2d = vk_eng_texture_create(NULL, RSX_BE_FMT_R8G8B8A8, 1, 1, 1, 1,
                                              0xAAE4u, 0x85u);
        if (!s_e2.dummy_2d) return VK_NULL_HANDLE;
        vk_eng_texture_upload(NULL, s_e2.dummy_2d, 0, 0, 1, 1, s_eng_unbound_texel, 4, 1);
    }
    return s_eobj[s_e2.dummy_2d].im.view;
}

/* The A/B phase the runtime's driver advances, as in rsx_draw_engine.c: the
 * barriers below read it for their in-run comparison. Weak, so a tool that
 * links this file without the runtime reports phase 0. */
#if defined(__GNUC__) || defined(__clang__)
extern volatile int g_ab_phase __attribute__((weak));
#define ENG_AB_PHASE() (&g_ab_phase ? g_ab_phase : 0)
#else
#define ENG_AB_PHASE() 0
#endif

/* ---------------------------------------------------------------------------
 * Draw batching (VK_DRAW_BATCH=1)
 *
 * Every vk_eng_draw used to submit its own command buffer and wait on a
 * fence, because the vertex, index and uniform buffers and the descriptor set
 * are single objects rewritten per draw. In God of War: Ascension's first level
 * that was 54,759 draws in 20 s at ~168 us each: the FIFO walker spent 46% of
 * its time blocked on the GPU, the RSX ring drained one draw at a time and the
 * SPU geometry jobs waited for ring space. Batched, a draw takes slices of
 * three ring buffers and a descriptor set from a pool, records into the open
 * command buffer, and only s_eb_pending grows. vk_eng_flush() submits, waits
 * once, and resets the rings and the pool; vk_begin, every object destroy, the
 * present and the engine's own submit point call it first, so all other work
 * keeps seeing everything before it complete. Between batched draws a full
 * memory barrier keeps render-to-texture ordering without tracking.
 * VK_DRAW_BATCH_MAX (default 512) caps the draws per submission.
 * -------------------------------------------------------------------------*/
#define VK_EB_SLOTS      2u
#define VK_EB_SLOT_VB    (32u << 20)
#define VK_EB_SLOT_IB    (8u << 20)
#define VK_EB_SLOT_UB    (96u << 20)
#define VK_EB_SLOT_SB    (16u << 20)   /* texture upload staging per slot */
#define VK_EB_SLOT_IND   (1u << 20)    /* indirect draw commands per slot (~52k draws) */
#define VK_EB_IND_MAX    ((u32)(VK_EB_SLOT_IND / sizeof(VkDrawIndexedIndirectCommand)))
#define VK_EB_VB_BYTES   VK_EB_SLOT_VB
#define VK_EB_IB_BYTES   VK_EB_SLOT_IB
#define VK_EB_UB_BYTES   VK_EB_SLOT_UB
#define VK_EB_SB_BYTES   VK_EB_SLOT_SB
#define VK_EB_MAX_SETS     4096u
#define VK_EB_PERSIST_SETS 16384u
#define VK_EB_TOUCHED      64
#define VK_EB_DESC_BUCKETS 4096u
#define VK_EB_DESC_WAYS    4u
#define VK_EB_DESC_SETS    (VK_EB_DESC_BUCKETS * VK_EB_DESC_WAYS)
#define VK_EB_GVB_BYTES  (256u << 20)  /* persistent GPU vertex cache (VRAM/ReBAR) */
#define VK_EB_GIB_BYTES  (64u << 20)   /* persistent GPU index cache (VRAM/ReBAR) */
#define VK_EB_GC_SETS    16384u
#define VK_EB_GC_WAYS    2u
#define VK_EB_GC_ENTS    (VK_EB_GC_SETS * VK_EB_GC_WAYS)

#define VK_EB_MAX_RETIRED 64u
typedef struct {
    VkCommandBuffer  cmd;
    VkFence          fence;
    int              inflight;
    volatile u32     rec_busy;
    vk_image         retired[VK_EB_MAX_RETIRED];
    u32              n_retired;
} vk_eb_slot;

typedef struct {
    u64 id;
    u32 off;
    u32 bytes;
} vk_eb_gcent;

static struct {
    int              on;            /* -1 undecided, 0 off, 1 on */
    u32              max_draws;
    int              max_draws_explicit;
    vk_buffer        vb, ib, ub, sb, ind;
    int              ind_ok;
    u32              max_ind_count;
    VkDeviceSize     vb_head, ib_head, ub_head, sb_head, ub_align, slot_ub;
    VkCommandPool    rec_pool;      /* dedicated command pool for slot[].cmd */
    vk_eb_slot       slot[VK_EB_SLOTS];
    u32              cur;
    int              slot_ready;
    /* Persistent descriptor pool shared across slots/frames (UBO is one shared buffer;
     * slots differ only via dynamic offsets). Reset only when full after GPU idle. */
    VkDescriptorPool desc_pool;
    u32              desc_sets_used;
    /* Dedicated 1-set pool for set-0 persistent constant ring SSBO (never reset with desc_pool). */
    VkDescriptorPool ssbo_const_pool;
    VkDescriptorSet  ssbo_const_set;
    /* Persistent GPU vertex and index cache keyed by rsx_draw_engine monotonic IDs. */
    vk_buffer        gvb, gib;
    VkDeviceSize     gvb_head, gib_head;
    vk_eb_gcent*     gvb_tab;
    vk_eb_gcent*     gib_tab;
    u32              gcache_gen;
    int              gcache_ok;
    unsigned long long draws, flushes;
    VkImage          touched[VK_EB_TOUCHED];
    VkImage          last_touched_depth;
    u32              touched_n;
    int              touched_overflow;
    u32              touch_gen;
    u32              sample_haz_tex_gen;
    u32              sample_haz_touch_gen;
    int              sample_haz_val;
    u32              desc_epoch;
    /* Open render pass coalescing + redundant command state cache. */
    int              in_rp;
    int              fast_rp_ok;
    int              flags_phase;
    u8               f_eb, f_async, f_gcache, f_hazard, f_m2e4, f_m2e4b, f_m2f, f_prof, f_ssbo, f_rec_async, f_ind;
    VkRenderPass     cur_rp;
    VkFramebuffer    cur_fb;
    VkPipeline       cur_pipe;
    int              rp_state_valid;
    VkViewport       cur_vp;
    VkRect2D         cur_sc;
    u32              cur_sref;
    float            cur_db[2];
    float            cur_bias[2];
    float            cur_bc[4];
    VkPipelineLayout cur_playout;
    VkDescriptorSet  cur_set;
    u32              cur_dyn[2];
    u8               cur_ssbo_bound;
    u32              cur_pc_vp, cur_pc_ps;
    VkBuffer         cur_vbuf, cur_ibuf;
    VkDeviceSize     cur_voff, cur_ioff;
    /* Per-batch UBO + descriptor set fast reuse. */
    VkDeviceSize     cur_uvs, cur_ups;
    u32              cur_vsb, cur_psb;
    VkDescriptorSet  last_desc_set;
    u32              last_desc_epoch, last_desc_obj_gen;
    u32              last_desc_cube_mask;
    u32              last_desc_vsb, last_desc_psb;
} s_eb = { .on = -1, .flags_phase = -999999 };

/* VK_DRAW_BATCH_ASYNC=0 restores M1e sync 512-draw batching.
 * VK_DRAW_BATCH_ASYNC_PHASE=<0|1> runs M2a only on that A/B parity. */
static int vk_eb_async_use(void)
{
    static int mode = -1, phase = -1;
    if (mode < 0) {
        const char* e = getenv("VK_DRAW_BATCH_ASYNC");
        const char* p = getenv("VK_DRAW_BATCH_ASYNC_PHASE");
        mode = (e && *e == '0') ? 0 : 1;
        if (p && *p) phase = atoi(p) & 1;
    }
    if (!mode) return 0;
    return phase < 0 || (ENG_AB_PHASE() & 1) == phase;
}

static int vk_eb_gcache_use(void)
{
    static int mode = -1;
    if (mode < 0) {
        const char* e = getenv("VK_GPU_VCACHE");
        mode = (e && *e == '0') ? 0 : 1;
    }
    return mode && s_eb.gcache_ok && vk_eb_async_use();
}

static int vk_eb_any_inflight(void)
{
    if (s_eb.on != 1) return 0;
    for (u32 i = 0; i < VK_EB_SLOTS; i++)
        if (s_eb.slot[i].inflight || __atomic_load_n(&s_eb.slot[i].rec_busy, __ATOMIC_ACQUIRE))
            return 1;
    return 0;
}

static int vk_eb_prepare_slot(void);
static int vk_eb_retire_image(vk_image* im)
{
    if (s_eb.on != 1 || !im || (!im->img && !im->view && !im->mem)) return 0;
    if (!s_eb.slot_ready && vk_eb_prepare_slot()) return 0;
    vk_eb_slot* sl = &s_eb.slot[s_eb.cur];
    if (sl->n_retired >= VK_EB_MAX_RETIRED) return 0;
    sl->retired[sl->n_retired++] = *im;
    memset(im, 0, sizeof *im);
    return 1;
}

/* A draw's descriptor set depends on nothing but its sampler and view tuple:
 * the constants travel as two dynamic uniform offsets. Same tuple, same set,
 * no vkUpdateDescriptorSets. Direct-mapped; a collision simply writes the
 * entry out and allocates another set. */
typedef struct {
    int             split;                       /* the layout the set is from */
    int             ssbo;                        /* 1 = set-1 textures/samplers only (SSBO consts in set 0) */
    u32             vs_bytes, ps_bytes;          /* dynamic UBO descriptor ranges */
    VkImageView     view[RSX_MAX_TEXTURES];
    VkSampler       sampler[RSX_MAX_TEXTURES];
} vk_eb_desc_key;

typedef struct {
    u32             ps_bytes;
    u16             cube_mask;
    u16             tex_mask;
    u32             tex[RSX_MAX_TEXTURES];
    u8              samp[RSX_MAX_TEXTURES];
} vk_eb_fast_key;

typedef struct {
    u64             hash;
    VkDescriptorSet set;
    u32             epoch;
    u32             obj_gen;
    u32             tex_bind_gen;
    u32             ps_bytes;
    u16             cube_mask;
    u16             tex_mask;
} vk_eb_desc_tag;

typedef struct {
    vk_eb_fast_key  fkey;
    u8              live;
} vk_eb_desc;

static vk_eb_desc_tag s_eb_desc_tag[VK_EB_DESC_SETS];
static vk_eb_desc s_eb_desc[VK_EB_DESC_SETS];
static u8 s_eb_desc_rr[VK_EB_DESC_BUCKETS];
static u16 s_eb_desc_live[VK_EB_DESC_SETS];
static u32 s_eb_desc_live_n;

static void vk_eb_inval_fast_rp(void) { s_eb.fast_rp_ok = 0; }

static int vk_m2e4_use(void)
{
    static int cfg = -1, phase = -1;
    if (cfg < 0) {
        const char* e = getenv("M2E4");
        if (e && (e[0] == 'a' || e[0] == 'A')) { cfg = 1; phase = 0; }
        else { cfg = (e && *e == '0') ? 0 : 1; }
        const char* p = getenv("M2E4_PHASE");
        if (p && *p) phase = atoi(p) & 1;
    }
    if (!cfg) return 0;
    return phase < 0 || (ENG_AB_PHASE() & 1) == phase;
}

static int vk_m2e4b_use(void)
{
    static int cfg = -1, phase = -1;
    if (cfg < 0) {
        const char* e = getenv("M2E4B");
        if (e && (e[0] == 'a' || e[0] == 'A')) { cfg = 1; phase = 0; }
        else { cfg = (e && *e == '0') ? 0 : 1; }
        const char* p = getenv("M2E4B_PHASE");
        if (p && *p) phase = atoi(p) & 1;
    }
    if (!cfg) return 0;
    return phase < 0 || (ENG_AB_PHASE() & 1) == phase;
}

static void vk_eb_inval_binds(void)
{
    s_eb.cur_ssbo_bound = 0;
    s_eb.cur_playout = VK_NULL_HANDLE;
    s_eb.cur_set = VK_NULL_HANDLE;
    s_eb.cur_pc_vp = 0xFFFFFFFFu;
    s_eb.cur_pc_ps = 0xFFFFFFFFu;
}

static int vk_const_ssbo_use(void)
{
    int cfg = 0, phase = -1;
    vk_const_ssbo_cfg(&cfg, &phase);
    if (!cfg || !s_g.ssbo_ok) return 0;
    return phase < 0 || (ENG_AB_PHASE() & 1) == phase;
}

static int vk_indirect_batch_use(void)
{
    int cfg = 0, phase = -1;
    vk_indirect_batch_cfg(&cfg, &phase);
    if (!cfg || !s_vk.multi_draw_indirect_ok || !s_eb.ind_ok) return 0;
    return phase < 0 || (ENG_AB_PHASE() & 1) == phase;
}

/* VK_REC_ASYNC=0 records Vulkan commands synchronously on the RSX FIFO thread;
 * default 1 offloads vkCmd* recording and vkQueueSubmit to a dedicated worker
 * thread via a 64B lock-free SPSC ring (m2e-5). VK_REC_ASYNC=a or
 * VK_REC_ASYNC_PHASE=<0|1> runs async recording on that PS3_AB_PHASE parity. */
static int vk_rec_async_use(void)
{
    static int cfg = -1, phase = -1;
    if (cfg < 0) {
        const char* e = getenv("VK_REC_ASYNC");
        if (e && (e[0] == 'a' || e[0] == 'A')) { cfg = 1; phase = 0; }
        else { cfg = (e && *e == '0') ? 0 : 1; }
        const char* p = getenv("VK_REC_ASYNC_PHASE");
        if (p && *p) phase = atoi(p) & 1;
    }
    if (!cfg || !vk_eb_async_use()) return 0;
    return phase < 0 || (ENG_AB_PHASE() & 1) == phase;
}

static int vk_m2f_use(void)
{
    static int cfg = -1, phase = -1;
    if (cfg < 0) {
        const char* e = getenv("ENG_M2F");
        if (e && (e[0] == 'a' || e[0] == 'A')) { cfg = 1; phase = 0; }
        else { cfg = (e && *e == '0') ? 0 : 1; }
        const char* p = getenv("ENG_M2F_PHASE");
        if (p && *p) phase = atoi(p) & 1;
    }
    if (!cfg) return 0;
    return phase < 0 || (ENG_AB_PHASE() & 1) == phase;
}

static int vk_eb_on(void);
static int vk_eb_barrier_hazard(void);
static int vk_m2e4_prof_on(void);
static void vk_rec_drain(void);
static void vk_eb_refresh_flags(int ph)
{
    const u8 new_rec = (u8)(vk_rec_async_use() != 0);
    if (s_eb.flags_phase != -999999 && s_eb.f_rec_async != new_rec) {
        /* Drain async recorder and GPU before switching ownership of rec_pool. */
        if (s_eb_pending || vk_eb_any_inflight()) vk_eng_flush();
        else vk_rec_drain();
    }
    s_eb.f_eb        = (u8)(vk_eb_on() != 0);
    s_eb.f_async     = (u8)(vk_eb_async_use() != 0);
    s_eb.f_gcache    = (u8)(vk_eb_gcache_use() != 0);
    s_eb.f_hazard    = (u8)(vk_eb_barrier_hazard() != 0);
    s_eb.f_m2e4      = (u8)(vk_m2e4_use() != 0);
    s_eb.f_m2e4b     = (u8)(vk_m2e4b_use() != 0);
    s_eb.f_m2f       = (u8)(vk_m2f_use() != 0);
    s_eb.f_prof      = (u8)(vk_m2e4_prof_on() != 0);
    s_eb.f_rec_async = new_rec;
    const u8 new_ssbo = (u8)(vk_const_ssbo_use() != 0);
    const u8 new_ind  = (u8)(new_ssbo && vk_indirect_batch_use() != 0);
    if (s_eb.flags_phase != -999999 && (s_eb.f_ssbo != new_ssbo || s_eb.f_ind != new_ind)) {
        s_eb.cur_pipe = VK_NULL_HANDLE;
        vk_eb_inval_binds();
        s_eb.cur_uvs = (VkDeviceSize)-1;
        s_eb.cur_ups = (VkDeviceSize)-1;
        s_eb.last_desc_set = VK_NULL_HANDLE;
    }
    s_eb.f_ssbo = new_ssbo;
    s_eb.f_ind  = new_ind;
    s_eb.flags_phase = ph;
    g_rsx_vk_gcache_gen = (s_eb.f_eb && s_eb.f_gcache) ? s_eb.gcache_gen : 0u;
}

static int vk_m2e4_prof_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("M2E4_PROF"); on = (e && *e == '1') ? 1 : 0; }
    return on;
}

#if defined(__x86_64__) || defined(_M_X64)
static inline u64 vk_m2e4_rdtsc(void)
{
    u32 lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((u64)hi << 32) | lo;
}
#else
static inline u64 vk_m2e4_rdtsc(void) { return 0; }
#endif

static void vk_eb_desc_invalidate_handle(u32 handle)
{
    if (!handle || handle >= VK_ENG_MAX_OBJ) return;
    u32 tm = s_e2.tex_mask;
    while (tm) {
        const u32 u = (u32)__builtin_ctz(tm);
        tm &= tm - 1u;
        if (s_e2.tex[u] == handle) {
            s_e2.tex[u] = 0;
            s_e2.tex_img[u] = VK_NULL_HANDLE;
            s_e2.tex_rt_mask &= ~(1u << u);
            s_e2.tex_dirty = 1;
        }
    }
    if (!s_obj_in_desc[handle]) return;
    u32 i = 0;
    while (i < s_eb_desc_live_n) {
        const u32 slot = s_eb_desc_live[i];
        vk_eb_desc* const e = &s_eb_desc[slot];
        int hit = 0;
        u32 m = e->fkey.tex_mask;
        while (m) {
            const u32 u = (u32)__builtin_ctz(m);
            m &= m - 1u;
            if (e->fkey.tex[u] == handle) { hit = 1; break; }
        }
        if (hit) {
            s_eb_desc_tag[slot].set = VK_NULL_HANDLE;
            s_eb_desc_tag[slot].epoch = 0;
            e->live = 0;
            s_eb_desc_live[i] = s_eb_desc_live[--s_eb_desc_live_n];
        } else {
            i++;
        }
    }
    s_obj_in_desc[handle] = 0;
    s_eb.last_desc_set = VK_NULL_HANDLE;
}

/* VK_DRAW_BATCH_BARRIER=full keeps one full memory barrier per batched draw.
 * VK_DRAW_BATCH_BARRIER_PHASE=<0|1> runs the hazard path on that A/B parity
 * only, so one run measures both. */
static int vk_eb_barrier_hazard(void)
{
    static int mode = -1, phase = -1;
    if (mode < 0) {
        const char* e = getenv("VK_DRAW_BATCH_BARRIER");
        const char* p = getenv("VK_DRAW_BATCH_BARRIER_PHASE");
        mode = (e && (*e == 'f' || *e == 'F')) ? 0 : 1;   /* 1 = hazard */
        if (p && *p) phase = atoi(p) & 1;
    }
    if (!mode) return 0;
    return phase < 0 || (ENG_AB_PHASE() & 1) == phase;
}

static void vk_eb_touch_clear(void)
{
    s_eb.last_touched_depth = VK_NULL_HANDLE;
    s_eb.fast_rp_ok = 0;
    if (s_eb.touched_n || s_eb.touched_overflow) {
        s_eb.touched_n = 0;
        s_eb.touched_overflow = 0;
        s_eb.touch_gen++;
        if (!s_eb.touch_gen) s_eb.touch_gen = 1u;
    }
}

static int vk_eb_touched(VkImage img)
{
    if (!img) return 0;
    if (s_eb.touched_overflow) return 1;
    for (u32 i = 0; i < s_eb.touched_n; i++)
        if (s_eb.touched[i] == img) return 1;
    return 0;
}

static void vk_eb_touch(VkImage img)
{
    if (!img || vk_eb_touched(img)) return;
    if (s_eb.touched_n < VK_EB_TOUCHED) s_eb.touched[s_eb.touched_n++] = img;
    else s_eb.touched_overflow = 1;      /* a later draw cannot tell: barrier it */
    s_eb.touch_gen++;
    if (!s_eb.touch_gen) s_eb.touch_gen = 1u;
    s_eb.fast_rp_ok = 0;
}

/* Does this draw read or write an image the batch has already written? A
 * sampled target, a blend over one, or a depth test against one all need the
 * barrier; a draw that touches none of them does not. */
static void vk_eb_end_rp(void)
{
    s_eb.fast_rp_ok = 0;
    if (!s_eb.in_rp) return;
    pvkCmdEndRenderPass(s_vk.cmd);
    s_eb.in_rp = 0;
    s_eb.rp_state_valid = 0;
    s_eb.cur_rp = VK_NULL_HANDLE;
    s_eb.cur_fb = VK_NULL_HANDLE;
    s_eb.cur_pipe = VK_NULL_HANDLE;
    s_eb.cur_vbuf = VK_NULL_HANDLE;
    s_eb.cur_ibuf = VK_NULL_HANDLE;
}

static int vk_eb_sample_hazard(void)
{
    u32 m = s_e2.tex_rt_mask;
    if (!m) return 0;
    if (s_eb.sample_haz_tex_gen == s_e2.tex_bind_gen &&
        s_eb.sample_haz_touch_gen == s_eb.touch_gen &&
        s_e2.tex_bind_gen != 0u && s_eb.touch_gen != 0u)
        return s_eb.sample_haz_val;
    int haz = 0;
    while (m) {
        const u32 u = (u32)__builtin_ctz(m);
        m &= m - 1u;
        if (s_e2.tex_img[u] && vk_eb_touched(s_e2.tex_img[u])) { haz = 1; break; }
    }
    s_eb.sample_haz_tex_gen = s_e2.tex_bind_gen;
    s_eb.sample_haz_touch_gen = s_eb.touch_gen;
    s_eb.sample_haz_val = haz;
    return haz;
}

static int vk_eb_hazard(const VkImage* rt, u32 nrt, VkImage depth)
{
    for (u32 i = 0; i < nrt; i++)
        if (vk_eb_touched(rt[i])) return 1;
    if (vk_eb_touched(depth)) return 1;
    return vk_eb_sample_hazard();
}

/* ---------------------------------------------------------------------------
 * Async Vulkan command-recording thread (m2e-5, VK_REC_ASYNC=1)
 *
 * Moves vkCmd* recording, hazard tracking, render pass coalescing, and
 * vkQueueSubmit off the RSX FIFO thread onto a dedicated worker owning
 * s_eb.rec_pool and s_eb.slot[].cmd. 64-byte lock-free SPSC ring with Linux
 * futex batched wake (no busy-spin on 15 W Deck).
 * -------------------------------------------------------------------------*/
#define VK_REC_RING_N    16384u
#define VK_REC_RING_MASK (VK_REC_RING_N - 1u)

enum {
    VK_REC_OP_BEGIN_SLOT = 1,
    VK_REC_OP_SET_TARGETS,
    VK_REC_OP_SET_DYN,
    VK_REC_OP_SET_DYN2,
    VK_REC_OP_SET_TEX_RT,
    VK_REC_OP_SET_OFFS,
    VK_REC_OP_DRAW,
    VK_REC_OP_CLEAR_COLOR,
    VK_REC_OP_CLEAR_DEPTH,
    VK_REC_OP_IMG_INIT,
    VK_REC_OP_UPLOAD,
    VK_REC_OP_COLOR_SNAP,
    VK_REC_OP_DEPTH_SNAP,
    VK_REC_OP_ZCONV,
    VK_REC_OP_UPSCALE_DEPTH,
    VK_REC_OP_UPSCALE_COLOR,
    VK_REC_OP_SUBMIT,
    VK_REC_OP_SYNC,
};

#define REC_DRAW_F_INDEXED (1u << 0)
#define REC_DRAW_F_SSBO    (1u << 1)
#define REC_DRAW_F_DWRITE  (1u << 2)
#define REC_DRAW_F_HAZARD  (1u << 3)
#define REC_DRAW_F_OFFS    (1u << 4)
#define REC_DRAW_F_VGC     (1u << 5)
#define REC_DRAW_F_IGC     (1u << 6)
#define REC_DRAW_F_SPLIT   (1u << 7)

typedef struct __attribute__((aligned(64))) vk_rec_pkt {
    u8  op;
    u8  flags;
    u16 u16_a;
    u16 u16_b;
    u8  u8_a;
    u8  u8_b;
    union {
        struct {
            u32             count;
            u32             first_idx;
            int32_t         vtx_off;
            u32             dyn0;
            u32             dyn1;
            u32             pad;
            VkPipeline      pipe;
            VkDescriptorSet set;
        } draw;
        struct {
            VkRenderPass    rp;
            VkFramebuffer   fb;
            VkImage         depth_img;
            VkImage         rt_img[RSX_BE_MAX_COLOR_TARGETS];
        } targets;
        struct {
            float           vp[4];
            int32_t         sc_x, sc_y;
            u32             sc_w, sc_h;
            float           db[2];
        } dyn;
        struct {
            float           bias[2];
            float           bc[4];
        } dyn2;
        struct {
            VkDeviceSize    voff;
            VkDeviceSize    ioff;
        } offs;
        struct {
            VkImage         imgs[6];
        } tex_rt;
        struct {
            VkImage         img;
            float           rgba[4];
        } clr_col;
        struct {
            VkImage            img;
            VkImageAspectFlags aspect;
            float              depth;
        } clr_dep;
        struct {
            VkImage            img;
            VkImageAspectFlags aspect;
        } img_init;
        struct {
            VkImage         img;
            VkDeviceSize    buf_off;
            u32             mip;
            u32             layer;
            int32_t         ox, oy;
            u32             ew, eh;
        } upload;
        struct {
            VkImage         src;
            VkImage         dst;
        } csnap;
        struct {
            VkImage         zimg;
            VkImage         timg;
            VkBuffer        stbuf;
        } dsnap;
        struct {
            VkFramebuffer   fb;
            VkDescriptorSet set;
            VkImage         cimg;
        } zconv;
        struct {
            VkImage         img;
            u32             fmt;
        } upscale;
        struct {
            u64             seq;
        } sync;
    } u;
} vk_rec_pkt;

_Static_assert(sizeof(vk_rec_pkt) == 64, "vk_rec_pkt must be 64 bytes");

typedef struct {
    VkCommandBuffer  cmd;
    u32              slot_idx;
    int              cmd_open;
    int              first_op_done;
    int              in_rp;
    int              fast_rp_ok;
    int              rp_state_valid;
    /* Current targets from SET_TARGETS */
    VkRenderPass     t_rp;
    VkFramebuffer    t_fb;
    VkImage          t_depth_img;
    VkImage          t_rt_img[RSX_BE_MAX_COLOR_TARGETS];
    u32              t_nrt, t_w, t_h;
    /* Current sampled RT images for hazard check */
    VkImage          tex_rt_imgs[6];
    u8               tex_rt_n, tex_rt_overflow, tex_rt_dirty;
    /* Pending indirect draw run (VK_INDIRECT_BATCH) */
    VkDrawIndexedIndirectCommand* ind_base;
    u32              ind_head;
    u32              ind_run_start;
    u32              ind_run_len;
    /* Desired dynamic state from SET_DYN / SET_DYN2 */
    VkViewport       want_vp;
    VkRect2D         want_sc;
    u32              want_sref;
    float            want_db[2];
    float            want_bias[2];
    float            want_bc[4];
    u8               dyn_dirty;
    u8               dyn2_dirty;
    /* Pending non-zero buffer offsets from SET_OFFS */
    VkDeviceSize     pending_voff, pending_ioff;
    /* Bound state inside current command buffer / render pass */
    VkRenderPass     cur_rp;
    VkFramebuffer    cur_fb;
    VkPipeline       cur_pipe;
    VkViewport       cur_vp;
    VkRect2D         cur_sc;
    u32              cur_sref;
    float            cur_db[2];
    float            cur_bias[2];
    float            cur_bc[4];
    VkPipelineLayout cur_playout;
    VkDescriptorSet  cur_set;
    u32              cur_dyn[2];
    u8               cur_ssbo_bound;
    u32              cur_pc_vp, cur_pc_ps;
    VkBuffer         cur_vbuf, cur_ibuf;
    VkDeviceSize     cur_voff, cur_ioff;
    /* Hazard touched-image set */
    VkImage          touched[VK_EB_TOUCHED];
    VkImage          last_touched_depth;
    u32              touched_n;
    int              touched_overflow;
    /* Consumer-side stats folded on present */
    u64              st_submits, st_barriers, st_rps, st_desc_binds, st_push_consts;
    u64              st_ind_calls, st_ind_draws, st_dir_draws;
    /* Consumer-side stats mirrored for M2E4-VK 300f print */
    u64              prof_ind_calls, prof_ind_draws, prof_dir_draws, prof_push;
} vk_rec_cstate;

static struct {
    vk_rec_pkt*      ring;
    pthread_t        thread;
    int              running;
    int              batch_async;   /* 1 if current s_eb_pending batch was opened on vk_rec */
    volatile u32     quit;
    /* Producer-only shadow caches (FIFO thread) */
    VkRenderPass     p_rp;
    VkFramebuffer    p_fb;
    VkImage          p_depth_img;
    VkImage          p_rt_img[RSX_BE_MAX_COLOR_TARGETS];
    u32              p_nrt, p_w, p_h;
    int              p_targets_valid;
    int              p_dyn_valid;
    int              p_dyn2_valid;
    u32              p_dyn_w, p_dyn_h;
    u32              p_tex_rt_gen;
    u32              p_tex_rt_mask;
    u8               p_tex_rt_n;
    u64              p_sync_seq;
    u32              batch;
    /* Producer-only local cursor + cache line */
    u64              local_head;
    u64              pub_head;
    u64              cached_tail;
    u64              st_wakes;
    u64              st_stalls;
    u64              st_pkts;
    u64              st_wake_tsc;
    /* Shared producer->consumer publication line */
    __attribute__((aligned(64))) volatile u64 head;
    /* Consumer cache line */
    __attribute__((aligned(64))) volatile u64 tail;
    volatile u32     sleeping;      /* futex: 0 = awake, 1 = sleeping */
    volatile u32     notify_seq;    /* futex: bumped on SUBMIT / SYNC */
    volatile u64     c_sync_seq;
    u64              c_busy_tsc;
#ifndef __linux__
    pthread_mutex_t  mu;
    pthread_cond_t   cv_cons;
    pthread_cond_t   cv_prod;
#endif
    vk_rec_cstate    c;
} s_rec;

static inline void vk_cpu_pause(void)
{
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

static inline void vk_futex_wait(volatile u32* addr, u32 val)
{
#ifdef __linux__
    syscall(SYS_futex, (uint32_t*)addr, FUTEX_WAIT_PRIVATE, val, NULL, NULL, 0);
#else
    pthread_mutex_lock(&s_rec.mu);
    if (__atomic_load_n(addr, __ATOMIC_ACQUIRE) == val)
        pthread_cond_wait(addr == &s_rec.sleeping ? &s_rec.cv_cons : &s_rec.cv_prod, &s_rec.mu);
    pthread_mutex_unlock(&s_rec.mu);
#endif
}

static inline void vk_futex_wake(volatile u32* addr, int n)
{
#ifdef __linux__
    syscall(SYS_futex, (uint32_t*)addr, FUTEX_WAKE_PRIVATE, n, NULL, NULL, 0);
#else
    pthread_mutex_lock(&s_rec.mu);
    if (n > 1) pthread_cond_broadcast(addr == &s_rec.sleeping ? &s_rec.cv_cons : &s_rec.cv_prod);
    else       pthread_cond_signal(addr == &s_rec.sleeping ? &s_rec.cv_cons : &s_rec.cv_prod);
    pthread_mutex_unlock(&s_rec.mu);
#endif
}

static void vk_rec_publish_and_wake(void)
{
    s_rec.pub_head = s_rec.local_head;
    __atomic_store_n(&s_rec.head, s_rec.local_head, __ATOMIC_RELEASE);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    if (__builtin_expect(__atomic_load_n(&s_rec.sleeping, __ATOMIC_ACQUIRE) != 0u, 0)) {
        const u64 wt0 = s_eb.f_prof ? vk_m2e4_rdtsc() : 0;
        __atomic_store_n(&s_rec.sleeping, 0u, __ATOMIC_RELEASE);
        vk_futex_wake(&s_rec.sleeping, 1);
        s_rec.st_wakes++;
        if (wt0) s_rec.st_wake_tsc += vk_m2e4_rdtsc() - wt0;
    }
}

static inline void vk_rec_wake(int force)
{
    if (force || (s_rec.local_head - s_rec.pub_head) >= s_rec.batch)
        vk_rec_publish_and_wake();
}

static vk_rec_pkt* vk_rec_alloc_pkt_slow(u64 lh)
{
    s_rec.cached_tail = __atomic_load_n(&s_rec.tail, __ATOMIC_ACQUIRE);
    if (lh - s_rec.cached_tail >= VK_REC_RING_N) {
        s_rec.st_stalls++;
        vk_rec_publish_and_wake();
        u32 spins = 0;
        while (lh - s_rec.cached_tail >= VK_REC_RING_N) {
            if (spins < 128u) { vk_cpu_pause(); spins++; }
            else {
                struct timespec ts = { 0, 50000 }; /* 50 us */
                nanosleep(&ts, NULL);
            }
            s_rec.cached_tail = __atomic_load_n(&s_rec.tail, __ATOMIC_ACQUIRE);
        }
    }
    return &s_rec.ring[lh & VK_REC_RING_MASK];
}

static inline vk_rec_pkt* vk_rec_alloc_pkt(void)
{
    const u64 lh = s_rec.local_head;
    if (__builtin_expect(lh - s_rec.cached_tail >= VK_REC_RING_N, 0))
        return vk_rec_alloc_pkt_slow(lh);
    return &s_rec.ring[lh & VK_REC_RING_MASK];
}

static inline void vk_rec_commit_pkt(int force_wake)
{
    const u64 lh = s_rec.local_head + 1u;
    s_rec.local_head = lh;
    s_rec.st_pkts++;
    if (__builtin_expect(force_wake || (lh - s_rec.pub_head) >= s_rec.batch, 0))
        vk_rec_publish_and_wake();
}

static void vk_rec_wait_slot_free(u32 slot_idx)
{
    vk_eb_slot* sl = &s_eb.slot[slot_idx];
    if (__builtin_expect(__atomic_load_n(&sl->rec_busy, __ATOMIC_ACQUIRE) == 0u, 1)) return;
    vk_rec_wake(1);
    for (u32 i = 0; i < 64u; i++) {
        if (__atomic_load_n(&sl->rec_busy, __ATOMIC_ACQUIRE) == 0u) return;
        vk_cpu_pause();
    }
    while (__atomic_load_n(&sl->rec_busy, __ATOMIC_ACQUIRE) != 0u) {
        const u32 seq = __atomic_load_n(&s_rec.notify_seq, __ATOMIC_ACQUIRE);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        if (__atomic_load_n(&sl->rec_busy, __ATOMIC_ACQUIRE) == 0u) break;
        vk_futex_wait(&s_rec.notify_seq, seq);
    }
}

static void vk_rec_drain(void)
{
    if (!s_rec.running) return;
    const u64 target = ++s_rec.p_sync_seq;
    vk_rec_pkt* p = vk_rec_alloc_pkt();
    p->op = VK_REC_OP_SYNC;
    p->u.sync.seq = target;
    vk_rec_commit_pkt(1);
    for (u32 i = 0; i < 64u; i++) {
        if (__atomic_load_n(&s_rec.c_sync_seq, __ATOMIC_ACQUIRE) >= target) return;
        vk_cpu_pause();
    }
    while (__atomic_load_n(&s_rec.c_sync_seq, __ATOMIC_ACQUIRE) < target) {
        const u32 seq = __atomic_load_n(&s_rec.notify_seq, __ATOMIC_ACQUIRE);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        if (__atomic_load_n(&s_rec.c_sync_seq, __ATOMIC_ACQUIRE) >= target) break;
        vk_futex_wait(&s_rec.notify_seq, seq);
    }
}

static void vk_rec_wait_submitted(void)
{
    if (!s_rec.running) return;
    for (u32 i = 0; i < VK_EB_SLOTS; i++)
        vk_rec_wait_slot_free(i);
    if (s_eng_stats) {
        s_engtime.submits     += s_rec.c.st_submits;
        s_engtime.barriers    += s_rec.c.st_barriers;
        s_engtime.rps         += s_rec.c.st_rps;
        s_engtime.desc_binds  += s_rec.c.st_desc_binds;
        s_engtime.push_consts += s_rec.c.st_push_consts;
        s_engtime.ind_calls   += s_rec.c.st_ind_calls;
        s_engtime.ind_draws   += s_rec.c.st_ind_draws;
        s_engtime.dir_draws   += s_rec.c.st_dir_draws;
        s_rec.c.st_submits = s_rec.c.st_barriers = s_rec.c.st_rps = 0;
        s_rec.c.st_desc_binds = s_rec.c.st_push_consts = 0;
        s_rec.c.st_ind_calls = s_rec.c.st_ind_draws = s_rec.c.st_dir_draws = 0;
    }
}

static int vk_rec_use_batch(void)
{
    if (!s_rec.running) return 0;
    if (s_eb_pending) return s_rec.batch_async;
    const int cur_ph = ENG_AB_PHASE();
    if (__builtin_expect(s_eb.flags_phase != cur_ph, 0)) vk_eb_refresh_flags(cur_ph);
    return s_eb.f_rec_async != 0;
}

static int vk_rec_ensure_open(void)
{
    if (vk_eb_prepare_slot()) return -1;
    vk_eb_slot* sl = &s_eb.slot[s_eb.cur];
    __atomic_store_n(&sl->rec_busy, 1u, __ATOMIC_RELEASE);
    vk_rec_pkt* p = vk_rec_alloc_pkt();
    p->op = VK_REC_OP_BEGIN_SLOT;
    p->u8_a = (u8)s_eb.cur;
    p->u8_b = (u8)(s_vk.present_order != 0);
    s_vk.present_order = 0;
    vk_rec_commit_pkt(0);
    s_rec.batch_async = 1;
    s_rec.p_targets_valid = 0;
    s_rec.p_dyn_valid = 0;
    s_rec.p_dyn2_valid = 0;
    s_rec.p_tex_rt_gen = 0;
    s_rec.p_tex_rt_mask = 0;
    s_rec.p_tex_rt_n = 0;
    return 0;
}

static void vk_rec_emit_clear_color(VkImage img, const float rgba[4])
{
    vk_rec_pkt* p = vk_rec_alloc_pkt();
    p->op = VK_REC_OP_CLEAR_COLOR;
    p->u.clr_col.img = img;
    memcpy(p->u.clr_col.rgba, rgba, 16);
    vk_rec_commit_pkt(0);
}

static void vk_rec_emit_clear_depth(VkImage img, VkImageAspectFlags aspect, float depth_value, u8 stencil)
{
    vk_rec_pkt* p = vk_rec_alloc_pkt();
    p->op = VK_REC_OP_CLEAR_DEPTH;
    p->u8_a = stencil;
    p->u.clr_dep.img = img;
    p->u.clr_dep.aspect = aspect;
    p->u.clr_dep.depth = depth_value;
    vk_rec_commit_pkt(0);
}

static void vk_rec_emit_img_init(VkImage img, VkImageAspectFlags aspect)
{
    vk_rec_pkt* p = vk_rec_alloc_pkt();
    p->op = VK_REC_OP_IMG_INIT;
    p->u.img_init.img = img;
    p->u.img_init.aspect = aspect;
    vk_rec_commit_pkt(0);
}

static void vk_rec_emit_upload(VkImage img, const VkBufferImageCopy* r)
{
    vk_rec_pkt* p = vk_rec_alloc_pkt();
    p->op = VK_REC_OP_UPLOAD;
    p->u.upload.img = img;
    p->u.upload.buf_off = r->bufferOffset;
    p->u.upload.mip = r->imageSubresource.mipLevel;
    p->u.upload.layer = r->imageSubresource.baseArrayLayer;
    p->u.upload.ox = r->imageOffset.x;
    p->u.upload.oy = r->imageOffset.y;
    p->u.upload.ew = r->imageExtent.width;
    p->u.upload.eh = r->imageExtent.height;
    vk_rec_commit_pkt(0);
}

static void vk_rec_emit_color_snap(VkImage src, VkImage dst, u32 w, u32 h)
{
    vk_rec_pkt* p = vk_rec_alloc_pkt();
    p->op = VK_REC_OP_COLOR_SNAP;
    p->u16_a = (u16)w;
    p->u16_b = (u16)h;
    p->u.csnap.src = src;
    p->u.csnap.dst = dst;
    vk_rec_commit_pkt(0);
}

static void vk_rec_emit_depth_snap(VkImage zimg, VkImage timg, VkBuffer stbuf, u32 w, u32 h)
{
    vk_rec_pkt* p = vk_rec_alloc_pkt();
    p->op = VK_REC_OP_DEPTH_SNAP;
    p->u16_a = (u16)w;
    p->u16_b = (u16)h;
    p->u.dsnap.zimg = zimg;
    p->u.dsnap.timg = timg;
    p->u.dsnap.stbuf = stbuf;
    vk_rec_commit_pkt(0);
}

static void vk_rec_emit_zconv(VkFramebuffer fb, VkDescriptorSet set, VkImage cimg, u32 w, u32 h)
{
    vk_rec_pkt* p = vk_rec_alloc_pkt();
    p->op = VK_REC_OP_ZCONV;
    p->u16_a = (u16)w;
    p->u16_b = (u16)h;
    p->u.zconv.fb = fb;
    p->u.zconv.set = set;
    p->u.zconv.cimg = cimg;
    vk_rec_commit_pkt(0);
}

static void vk_rec_emit_upscale_depth(VkImage img, u32 w, u32 h)
{
    vk_rec_pkt* p = vk_rec_alloc_pkt();
    p->op = VK_REC_OP_UPSCALE_DEPTH;
    p->u16_a = (u16)w;
    p->u16_b = (u16)h;
    p->u.upscale.img = img;
    p->u.upscale.fmt = 0u;
    vk_rec_commit_pkt(0);
}

static void vk_rec_emit_upscale_color(VkImage img, VkFormat fmt, u32 w, u32 h)
{
    vk_rec_pkt* p = vk_rec_alloc_pkt();
    p->op = VK_REC_OP_UPSCALE_COLOR;
    p->u16_a = (u16)w;
    p->u16_b = (u16)h;
    p->u.upscale.img = img;
    p->u.upscale.fmt = (u32)fmt;
    vk_rec_commit_pkt(0);
}

/* ---- consumer helpers (run exclusively on vk_rec thread) ---------------- */
static void vk_rec_inval_binds(vk_rec_cstate* c)
{
    c->cur_ssbo_bound = 0;
    c->cur_playout = VK_NULL_HANDLE;
    c->cur_set = VK_NULL_HANDLE;
    c->cur_pc_vp = 0xFFFFFFFFu;
    c->cur_pc_ps = 0xFFFFFFFFu;
}

static void vk_rec_touch_clear(vk_rec_cstate* c)
{
    c->last_touched_depth = VK_NULL_HANDLE;
    c->fast_rp_ok = 0;
    c->touched_n = 0;
    c->touched_overflow = 0;
}

static int vk_rec_touched(const vk_rec_cstate* c, VkImage img)
{
    if (!img) return 0;
    if (c->touched_overflow) return 1;
    for (u32 i = 0; i < c->touched_n; i++)
        if (c->touched[i] == img) return 1;
    return 0;
}

static void vk_rec_touch(vk_rec_cstate* c, VkImage img)
{
    if (!img || vk_rec_touched(c, img)) return;
    if (c->touched_n < VK_EB_TOUCHED) c->touched[c->touched_n++] = img;
    else c->touched_overflow = 1;
    c->fast_rp_ok = 0;
}

static void vk_rec_flush_ind(vk_rec_cstate* c)
{
    const u32 n = c->ind_run_len;
    if (!n) return;
    c->ind_run_len = 0;
    if (n == 1u) {
        const VkDrawIndexedIndirectCommand* e = &c->ind_base[c->ind_run_start];
        pvkCmdDrawIndexed(c->cmd, e->indexCount, 1u, e->firstIndex, e->vertexOffset, e->firstInstance);
        c->ind_head = c->ind_run_start;
        c->st_dir_draws++;
        if (__builtin_expect(s_eb.f_prof, 0)) c->prof_dir_draws++;
    } else {
        const VkDeviceSize off = (VkDeviceSize)c->slot_idx * VK_EB_SLOT_IND +
                                 (VkDeviceSize)c->ind_run_start * sizeof(VkDrawIndexedIndirectCommand);
        pvkCmdDrawIndexedIndirect(c->cmd, s_eb.ind.buf, off, n, (u32)sizeof(VkDrawIndexedIndirectCommand));
        c->st_ind_calls++;
        c->st_ind_draws += n;
        if (__builtin_expect(s_eb.f_prof, 0)) {
            c->prof_ind_calls++;
            c->prof_ind_draws += (u64)n;
        }
    }
}

static void vk_rec_end_rp(vk_rec_cstate* c)
{
    vk_rec_flush_ind(c);
    c->fast_rp_ok = 0;
    if (!c->in_rp) return;
    pvkCmdEndRenderPass(c->cmd);
    c->in_rp = 0;
    c->rp_state_valid = 0;
    c->cur_rp = VK_NULL_HANDLE;
    c->cur_fb = VK_NULL_HANDLE;
    c->cur_pipe = VK_NULL_HANDLE;
    c->cur_vbuf = VK_NULL_HANDLE;
    c->cur_ibuf = VK_NULL_HANDLE;
}

static int vk_rec_sample_hazard(const vk_rec_cstate* c)
{
    if (c->tex_rt_overflow) return 1;
    for (u32 i = 0; i < c->tex_rt_n; i++)
        if (vk_rec_touched(c, c->tex_rt_imgs[i])) return 1;
    return 0;
}

static int vk_rec_hazard(const vk_rec_cstate* c)
{
    for (u32 i = 0; i < c->t_nrt; i++)
        if (vk_rec_touched(c, c->t_rt_img[i])) return 1;
    if (vk_rec_touched(c, c->t_depth_img)) return 1;
    return vk_rec_sample_hazard(c);
}

static void vk_rec_exec_draw(vk_rec_cstate* c, const vk_rec_pkt* p)
{
    if (!c->cmd_open) return;
    const u8 fl = p->flags;
    const int indexed = (fl & REC_DRAW_F_INDEXED) != 0;
    const int ssbo_on = (fl & REC_DRAW_F_SSBO) != 0;
    const int inst_on = (p->u8_a != 0u) && ssbo_on;
    const int dwrite  = (fl & REC_DRAW_F_DWRITE) != 0;
    const int hazard  = (fl & REC_DRAW_F_HAZARD) != 0;
    const int split   = (fl & REC_DRAW_F_SPLIT) != 0;
    VkDeviceSize bind_voff = 0, bind_ioff = 0;
    if (__builtin_expect((fl & REC_DRAW_F_OFFS) != 0, 0)) {
        bind_voff = c->pending_voff;
        bind_ioff = c->pending_ioff;
        c->pending_voff = 0;
        c->pending_ioff = 0;
    }
    const VkPipelineLayout playout = ssbo_on
        ? (split ? s_e2.ssbo_split_pipe_layout : s_g.ssbo_pipe_layout)
        : (split ? s_e2.split_pipe_layout : s_g.pipe_layout);
    const VkPipeline pipe = p->u.draw.pipe;
    const VkDescriptorSet set = p->u.draw.set;
    const VkBuffer vbuf = (fl & REC_DRAW_F_VGC) ? s_eb.gvb.buf : s_eb.vb.buf;
    const u32 vp_f4 = p->u.draw.dyn0 >> 4;
    const u32 ps_f4 = p->u.draw.dyn1 >> 4;

    /* Fast path: coalesce consecutive indexed draws sharing all bound state
     * into one contiguous VkDrawIndexedIndirectCommand array with vp_f4 in
     * firstInstance (zero vkCmd* calls per coalesced draw). */
    if (__builtin_expect(inst_on && indexed && hazard && c->ind_run_len > 0u, 1)) {
        const VkBuffer ibuf = (fl & REC_DRAW_F_IGC) ? s_eb.gib.buf : s_eb.ib.buf;
        if (c->in_rp && c->cur_rp == c->t_rp && c->cur_fb == c->t_fb &&
            !c->tex_rt_dirty && !c->dyn_dirty && !c->dyn2_dirty &&
            (!dwrite || c->t_depth_img == c->last_touched_depth) &&
            c->cur_pipe == pipe && c->cur_playout == playout && c->cur_set == set &&
            c->cur_pc_ps == ps_f4 &&
            c->cur_vbuf == vbuf && c->cur_voff == bind_voff &&
            c->cur_ibuf == ibuf && c->cur_ioff == bind_ioff &&
            c->ind_head < VK_EB_IND_MAX && c->ind_run_len < s_eb.max_ind_count) {
            VkDrawIndexedIndirectCommand* e = &c->ind_base[c->ind_head++];
            e->indexCount    = p->u.draw.count;
            e->instanceCount = 1u;
            e->firstIndex    = p->u.draw.first_idx;
            e->vertexOffset  = p->u.draw.vtx_off;
            e->firstInstance = vp_f4;
            c->ind_run_len++;
            return;
        }
    }
    if (c->ind_run_len) vk_rec_flush_ind(c);

    int reuse_rp = 0;
    if (__builtin_expect(c->fast_rp_ok && c->cur_rp == c->t_rp && c->cur_fb == c->t_fb &&
                         (!dwrite || c->t_depth_img == c->last_touched_depth), 1)) {
        reuse_rp = 1;
    } else {
        if (!c->first_op_done) {
            vk_cmd_after_everything(c->cmd);
            c->st_barriers++;
            vk_rec_touch_clear(c);
            c->first_op_done = 1;
        } else if (hazard && c->in_rp && c->cur_rp == c->t_rp && c->cur_fb == c->t_fb &&
                   !vk_rec_sample_hazard(c)) {
            reuse_rp = 1;
        } else {
            vk_rec_end_rp(c);
            if (!hazard || vk_rec_hazard(c)) {
                vk_cmd_after_everything(c->cmd);
                c->st_barriers++;
                vk_rec_touch_clear(c);
            }
        }
        if (!reuse_rp) {
            for (u32 i = 0; i < c->t_nrt; i++) vk_rec_touch(c, c->t_rt_img[i]);
        }
        if (dwrite && c->t_depth_img != c->last_touched_depth) {
            vk_rec_touch(c, c->t_depth_img);
            c->last_touched_depth = c->t_depth_img;
        }
        if (hazard && !c->tex_rt_n && !c->tex_rt_overflow && reuse_rp)
            c->fast_rp_ok = 1;
    }
    c->tex_rt_dirty = 0;
    if (!reuse_rp) {
        VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                      .renderPass = c->t_rp, .framebuffer = c->t_fb,
                                      .renderArea = { { 0, 0 }, { c->t_w, c->t_h } } };
        pvkCmdBeginRenderPass(c->cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        c->st_rps++;
        if (hazard) {
            c->in_rp = 1;
            c->cur_rp = c->t_rp;
            c->cur_fb = c->t_fb;
            c->cur_pipe = VK_NULL_HANDLE;
            if (!c->tex_rt_n && !c->tex_rt_overflow) c->fast_rp_ok = 1;
        }
    }
    if (!reuse_rp || c->cur_pipe != pipe) {
        pvkCmdBindPipeline(c->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        if (hazard) c->cur_pipe = pipe;
    }
    const int skip_st = hazard && reuse_rp && c->rp_state_valid;
    if (__builtin_expect(!skip_st || c->dyn_dirty, 0)) {
        if (!skip_st || memcmp(&c->cur_vp, &c->want_vp, sizeof c->want_vp) != 0) {
            pvkCmdSetViewport(c->cmd, 0, 1, &c->want_vp);
            c->cur_vp = c->want_vp;
        }
        if (!skip_st || memcmp(&c->cur_sc, &c->want_sc, sizeof c->want_sc) != 0) {
            pvkCmdSetScissor(c->cmd, 0, 1, &c->want_sc);
            c->cur_sc = c->want_sc;
        }
        if (!skip_st || c->cur_sref != c->want_sref) {
            pvkCmdSetStencilReference(c->cmd, VK_STENCIL_FACE_FRONT_AND_BACK, c->want_sref);
            c->cur_sref = c->want_sref;
        }
        if (s_vk.depth_bounds_ok &&
            (!skip_st || c->cur_db[0] != c->want_db[0] || c->cur_db[1] != c->want_db[1])) {
            pvkCmdSetDepthBounds(c->cmd, c->want_db[0], c->want_db[1]);
            c->cur_db[0] = c->want_db[0]; c->cur_db[1] = c->want_db[1];
        }
        c->dyn_dirty = 0;
    }
    if (__builtin_expect(!skip_st || c->dyn2_dirty, 0)) {
        if (!skip_st || c->cur_bias[0] != c->want_bias[0] || c->cur_bias[1] != c->want_bias[1]) {
            pvkCmdSetDepthBias(c->cmd, c->want_bias[0], 0.0f, c->want_bias[1]);
            c->cur_bias[0] = c->want_bias[0]; c->cur_bias[1] = c->want_bias[1];
        }
        if (!skip_st || memcmp(c->cur_bc, c->want_bc, sizeof c->want_bc) != 0) {
            pvkCmdSetBlendConstants(c->cmd, c->want_bc);
            memcpy(c->cur_bc, c->want_bc, sizeof c->want_bc);
        }
        c->dyn2_dirty = 0;
    }
    if (ssbo_on) {
        if (!c->cur_ssbo_bound) {
            VkDescriptorSet cset = s_eb.ssbo_const_set;
            pvkCmdBindDescriptorSets(c->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     playout, 0, 1, &cset, 0, NULL);
            c->cur_ssbo_bound = 1;
            c->st_desc_binds++;
        }
        if (c->cur_playout != playout || c->cur_set != set) {
            pvkCmdBindDescriptorSets(c->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     playout, 1, 1, &set, 0, NULL);
            c->cur_playout = playout; c->cur_set = set;
            c->st_desc_binds++;
        }
        if (inst_on) {
            /* vp_f4 travels in firstInstance (SV_InstanceID); only push ps_f4 on change. */
            c->cur_pc_vp = 0xFFFFFFFFu;
            if (c->cur_pc_ps != ps_f4) {
                pvkCmdPushConstants(c->cmd, playout,
                                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                    4u, 4u, &ps_f4);
                c->cur_pc_ps = ps_f4;
                c->st_push_consts++;
                if (__builtin_expect(s_eb.f_prof, 0)) c->prof_push++;
            }
        } else {
            const int chg_vp = (c->cur_pc_vp != vp_f4);
            const int chg_ps = (c->cur_pc_ps != ps_f4);
            if (chg_vp && chg_ps) {
                const u32 pc[2] = { vp_f4, ps_f4 };
                pvkCmdPushConstants(c->cmd, playout,
                                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                    0u, 8u, pc);
                c->cur_pc_vp = vp_f4; c->cur_pc_ps = ps_f4;
                c->st_push_consts++;
                if (__builtin_expect(s_eb.f_prof, 0)) c->prof_push++;
            } else if (chg_vp) {
                pvkCmdPushConstants(c->cmd, playout,
                                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                    0u, 4u, &vp_f4);
                c->cur_pc_vp = vp_f4;
                c->st_push_consts++;
                if (__builtin_expect(s_eb.f_prof, 0)) c->prof_push++;
            } else if (chg_ps) {
                pvkCmdPushConstants(c->cmd, playout,
                                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                    4u, 4u, &ps_f4);
                c->cur_pc_ps = ps_f4;
                c->st_push_consts++;
                if (__builtin_expect(s_eb.f_prof, 0)) c->prof_push++;
            }
        }
    } else if (!skip_st || c->cur_playout != playout || c->cur_set != set ||
               c->cur_dyn[0] != p->u.draw.dyn0 || c->cur_dyn[1] != p->u.draw.dyn1) {
        const u32 dyn[2] = { p->u.draw.dyn0, p->u.draw.dyn1 };
        pvkCmdBindDescriptorSets(c->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 playout, 0, 1, &set, 2, dyn);
        c->cur_playout = playout; c->cur_set = set;
        c->cur_dyn[0] = dyn[0]; c->cur_dyn[1] = dyn[1];
        c->st_desc_binds++;
    }
    if (!skip_st || c->cur_vbuf != vbuf || c->cur_voff != bind_voff) {
        pvkCmdBindVertexBuffers(c->cmd, 0, 1, &vbuf, &bind_voff);
        c->cur_vbuf = vbuf; c->cur_voff = bind_voff;
    }
    if (indexed) {
        const VkBuffer ibuf = (fl & REC_DRAW_F_IGC) ? s_eb.gib.buf : s_eb.ib.buf;
        if (!skip_st || c->cur_ibuf != ibuf || c->cur_ioff != bind_ioff) {
            pvkCmdBindIndexBuffer(c->cmd, ibuf, bind_ioff, VK_INDEX_TYPE_UINT32);
            c->cur_ibuf = ibuf; c->cur_ioff = bind_ioff;
        }
        if (inst_on && hazard && c->ind_base && c->ind_head < VK_EB_IND_MAX) {
            c->ind_run_start = c->ind_head;
            VkDrawIndexedIndirectCommand* e = &c->ind_base[c->ind_head++];
            e->indexCount    = p->u.draw.count;
            e->instanceCount = 1u;
            e->firstIndex    = p->u.draw.first_idx;
            e->vertexOffset  = p->u.draw.vtx_off;
            e->firstInstance = vp_f4;
            c->ind_run_len   = 1u;
        } else {
            pvkCmdDrawIndexed(c->cmd, p->u.draw.count, 1u, p->u.draw.first_idx, p->u.draw.vtx_off,
                              inst_on ? vp_f4 : 0u);
            c->st_dir_draws++;
            if (__builtin_expect(s_eb.f_prof, 0)) c->prof_dir_draws++;
        }
    } else {
        pvkCmdDraw(c->cmd, p->u.draw.count, 1u, p->u.draw.first_idx, inst_on ? vp_f4 : 0u);
        c->st_dir_draws++;
        if (__builtin_expect(s_eb.f_prof, 0)) c->prof_dir_draws++;
    }
    if (hazard) c->rp_state_valid = 1;
    else pvkCmdEndRenderPass(c->cmd);
}

static void vk_rec_exec_pkt(vk_rec_cstate* c, const vk_rec_pkt* p)
{
    switch (p->op) {
    case VK_REC_OP_DRAW:
        vk_rec_exec_draw(c, p);
        break;
    case VK_REC_OP_SET_TARGETS:
        c->t_rp = p->u.targets.rp;
        c->t_fb = p->u.targets.fb;
        c->t_depth_img = p->u.targets.depth_img;
        c->t_nrt = p->u8_a;
        c->t_w = p->u16_a;
        c->t_h = p->u16_b;
        for (u32 i = 0; i < c->t_nrt; i++) c->t_rt_img[i] = p->u.targets.rt_img[i];
        c->fast_rp_ok = 0;
        break;
    case VK_REC_OP_SET_DYN:
        c->want_vp = (VkViewport){ p->u.dyn.vp[0], p->u.dyn.vp[1], p->u.dyn.vp[2], p->u.dyn.vp[3], 0.0f, 1.0f };
        c->want_sc = (VkRect2D){ { p->u.dyn.sc_x, p->u.dyn.sc_y }, { p->u.dyn.sc_w, p->u.dyn.sc_h } };
        c->want_sref = p->u8_b;
        c->want_db[0] = p->u.dyn.db[0];
        c->want_db[1] = p->u.dyn.db[1];
        c->dyn_dirty = 1;
        break;
    case VK_REC_OP_SET_DYN2:
        c->want_bias[0] = p->u.dyn2.bias[0];
        c->want_bias[1] = p->u.dyn2.bias[1];
        memcpy(c->want_bc, p->u.dyn2.bc, sizeof c->want_bc);
        c->dyn2_dirty = 1;
        break;
    case VK_REC_OP_SET_TEX_RT:
        c->tex_rt_n = p->u8_a;
        c->tex_rt_overflow = p->u8_b;
        c->tex_rt_dirty = 1;
        for (u32 i = 0; i < c->tex_rt_n; i++) c->tex_rt_imgs[i] = p->u.tex_rt.imgs[i];
        if (c->tex_rt_n || c->tex_rt_overflow) c->fast_rp_ok = 0;
        break;
    case VK_REC_OP_SET_OFFS:
        c->pending_voff = p->u.offs.voff;
        c->pending_ioff = p->u.offs.ioff;
        break;
    case VK_REC_OP_BEGIN_SLOT: {
        const u32 slot_idx = p->u8_a;
        vk_eb_slot* sl = &s_eb.slot[slot_idx];
        c->slot_idx = slot_idx;
        c->cmd = sl->cmd;
        c->cmd_open = 0;
        if (pvkResetCommandBuffer(c->cmd, 0) == VK_SUCCESS) {
            VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
            if (pvkBeginCommandBuffer(c->cmd, &bi) == VK_SUCCESS) {
                c->cmd_open = 1;
                if (p->u8_b) vk_cmd_after_everything(c->cmd);
            }
        }
        c->first_op_done = 0;
        c->in_rp = 0;
        c->fast_rp_ok = 0;
        c->rp_state_valid = 0;
        c->cur_rp = VK_NULL_HANDLE;
        c->cur_fb = VK_NULL_HANDLE;
        c->cur_pipe = VK_NULL_HANDLE;
        c->cur_vbuf = VK_NULL_HANDLE;
        c->cur_ibuf = VK_NULL_HANDLE;
        c->tex_rt_n = 0;
        c->tex_rt_overflow = 0;
        c->tex_rt_dirty = 0;
        c->ind_base = s_eb.ind_ok
            ? (VkDrawIndexedIndirectCommand*)((u8*)s_eb.ind.ptr + (VkDeviceSize)slot_idx * VK_EB_SLOT_IND)
            : NULL;
        c->ind_head = 0;
        c->ind_run_start = 0;
        c->ind_run_len = 0;
        c->pending_voff = 0;
        c->pending_ioff = 0;
        vk_rec_inval_binds(c);
        vk_rec_touch_clear(c);
        break;
    }
    case VK_REC_OP_CLEAR_COLOR:
        if (c->cmd_open) {
            vk_rec_end_rp(c);
            VkClearColorValue v;
            memcpy(v.float32, p->u.clr_col.rgba, 16);
            VkImageSubresourceRange r = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vk_barrier_image_on(c->cmd, p->u.clr_col.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
            pvkCmdClearColorImage(c->cmd, p->u.clr_col.img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
            vk_rec_touch(c, p->u.clr_col.img);
            c->first_op_done = 1;
        }
        break;
    case VK_REC_OP_CLEAR_DEPTH:
        if (c->cmd_open) {
            vk_rec_end_rp(c);
            VkClearDepthStencilValue d = { p->u.clr_dep.depth, p->u8_a };
            VkImageSubresourceRange r = { p->u.clr_dep.aspect, 0, 1, 0, 1 };
            vk_barrier_image_on(c->cmd, p->u.clr_dep.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
            pvkCmdClearDepthStencilImage(c->cmd, p->u.clr_dep.img, VK_IMAGE_LAYOUT_GENERAL, &d, 1, &r);
            vk_rec_touch(c, p->u.clr_dep.img);
            c->first_op_done = 1;
        }
        break;
    case VK_REC_OP_IMG_INIT:
        if (c->cmd_open) {
            vk_rec_end_rp(c);
            vk_barrier_image_on(c->cmd, p->u.img_init.img, p->u.img_init.aspect, VK_IMAGE_LAYOUT_UNDEFINED);
            c->first_op_done = 1;
        }
        break;
    case VK_REC_OP_UPLOAD:
        if (c->cmd_open) {
            vk_rec_end_rp(c);
            vk_cmd_after_everything(c->cmd);
            vk_rec_touch_clear(c);
            VkBufferImageCopy reg = {
                .bufferOffset = p->u.upload.buf_off,
                .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, p->u.upload.mip, p->u.upload.layer, 1 },
                .imageOffset = { p->u.upload.ox, p->u.upload.oy, 0 },
                .imageExtent = { p->u.upload.ew, p->u.upload.eh, 1 },
            };
            pvkCmdCopyBufferToImage(c->cmd, s_eb.sb.buf, p->u.upload.img, VK_IMAGE_LAYOUT_GENERAL, 1, &reg);
            vk_barrier_image_on(c->cmd, p->u.upload.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
            c->first_op_done = 1;
        }
        break;
    case VK_REC_OP_COLOR_SNAP:
        if (c->cmd_open) {
            vk_rec_end_rp(c);
            VkImageMemoryBarrier pre[2] = {
                {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = p->u.csnap.src,
                    .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
                },
                {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = p->u.csnap.dst,
                    .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
                },
            };
            pvkCmdPipelineBarrier(c->cmd,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 2, pre);
            VkImageCopy ic = {
                .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                .extent = { p->u16_a, p->u16_b, 1 },
            };
            pvkCmdCopyImage(c->cmd, p->u.csnap.src, VK_IMAGE_LAYOUT_GENERAL,
                            p->u.csnap.dst, VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
            VkImageMemoryBarrier post[2] = {
                {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = p->u.csnap.dst,
                    .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
                },
                {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                    .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = p->u.csnap.src,
                    .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
                },
            };
            pvkCmdPipelineBarrier(c->cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0, 0, NULL, 0, NULL, 2, post);
            c->first_op_done = 1;
        }
        break;
    case VK_REC_OP_DEPTH_SNAP:
        if (c->cmd_open) {
            vk_rec_end_rp(c);
            vk_cmd_after_everything(c->cmd);
            vk_rec_touch_clear(c);
            vk_barrier_image_on(c->cmd, p->u.dsnap.zimg, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
            vk_barrier_image_on(c->cmd, p->u.dsnap.timg, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
            VkBufferImageCopy out = { .imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 },
                                      .imageExtent = { p->u16_a, p->u16_b, 1 } };
            pvkCmdCopyImageToBuffer(c->cmd, p->u.dsnap.zimg, VK_IMAGE_LAYOUT_GENERAL, p->u.dsnap.stbuf, 1, &out);
            VkBufferMemoryBarrier bb = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = p->u.dsnap.stbuf, .offset = 0, .size = VK_WHOLE_SIZE };
            pvkCmdPipelineBarrier(c->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0, 0, NULL, 1, &bb, 0, NULL);
            VkBufferImageCopy in = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                     .imageExtent = { p->u16_a, p->u16_b, 1 } };
            pvkCmdCopyBufferToImage(c->cmd, p->u.dsnap.stbuf, p->u.dsnap.timg, VK_IMAGE_LAYOUT_GENERAL, 1, &in);
            vk_barrier_image_on(c->cmd, p->u.dsnap.timg, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
            c->first_op_done = 1;
        }
        break;
    case VK_REC_OP_ZCONV:
        if (c->cmd_open) {
            vk_rec_end_rp(c);
            vk_cmd_after_everything(c->cmd);
            vk_rec_touch_clear(c);
            VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                          .renderPass = s_zconv.rp, .framebuffer = p->u.zconv.fb,
                                          .renderArea = { { 0, 0 }, { p->u16_a, p->u16_b } } };
            pvkCmdBeginRenderPass(c->cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
            pvkCmdBindPipeline(c->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_zconv.pipe);
            VkViewport vp = { 0.0f, 0.0f, (float)p->u16_a, (float)p->u16_b, 0.0f, 1.0f };
            VkRect2D sc = { { 0, 0 }, { p->u16_a, p->u16_b } };
            pvkCmdSetViewport(c->cmd, 0, 1, &vp);
            pvkCmdSetScissor(c->cmd, 0, 1, &sc);
            VkDescriptorSet zset = p->u.zconv.set;
            pvkCmdBindDescriptorSets(c->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     s_zconv.pipe_layout, 0, 1, &zset, 0, NULL);
            pvkCmdDraw(c->cmd, 3, 1, 0, 0);
            pvkCmdEndRenderPass(c->cmd);
            vk_rec_inval_binds(c);
            vk_rec_touch(c, p->u.zconv.cimg);
            c->first_op_done = 1;
        }
        break;
    case VK_REC_OP_UPSCALE_DEPTH:
        if (c->cmd_open) {
            vk_rec_end_rp(c);
            rsx_vk_fsr_capture_depth(c->cmd, p->u.upscale.img, p->u16_a, p->u16_b);
            c->first_op_done = 1;
        }
        break;
    case VK_REC_OP_UPSCALE_COLOR:
        if (c->cmd_open) {
            vk_rec_end_rp(c);
            rsx_vk_fsr_capture_color(c->cmd, p->u.upscale.img, (VkFormat)p->u.upscale.fmt, p->u16_a, p->u16_b);
            c->first_op_done = 1;
        }
        break;
    case VK_REC_OP_SUBMIT: {
        const u32 slot_idx = p->u8_a;
        vk_eb_slot* sl = &s_eb.slot[slot_idx];
        if (c->cmd_open) {
            vk_rec_end_rp(c);
            if (pvkEndCommandBuffer(sl->cmd) == VK_SUCCESS) {
                VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                    .commandBufferCount = 1, .pCommandBuffers = &sl->cmd };
                pvkResetFences(s_vk.device, 1, &sl->fence);
                if (vk_queue_submit(&si, sl->fence) == VK_SUCCESS) {
                    sl->inflight = 1;
                    c->st_submits++;
                }
            }
            c->cmd_open = 0;
        }
        vk_rec_touch_clear(c);
        __atomic_store_n(&sl->rec_busy, 0u, __ATOMIC_RELEASE);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        __atomic_add_fetch(&s_rec.notify_seq, 1u, __ATOMIC_RELEASE);
        vk_futex_wake(&s_rec.notify_seq, 0x7FFFFFFF);
        break;
    }
    case VK_REC_OP_SYNC:
        __atomic_store_n(&s_rec.c_sync_seq, p->u.sync.seq, __ATOMIC_RELEASE);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        __atomic_add_fetch(&s_rec.notify_seq, 1u, __ATOMIC_RELEASE);
        vk_futex_wake(&s_rec.notify_seq, 0x7FFFFFFF);
        break;
    default:
        break;
    }
}

static void* vk_rec_thread_main(void* arg)
{
    (void)arg;
#ifdef __linux__
    { struct sched_param sp = { 0 }; sched_setscheduler(0, SCHED_BATCH, &sp); }
#endif
    u64 t = 0;
    for (;;) {
        u64 h = __atomic_load_n(&s_rec.head, __ATOMIC_ACQUIRE);
        if (t < h) {
            const u64 ct0 = s_eb.f_prof ? vk_m2e4_rdtsc() : 0;
            while (t < h) {
                vk_rec_exec_pkt(&s_rec.c, &s_rec.ring[t & VK_REC_RING_MASK]);
                t++;
                if ((t & 255u) == 0u) __atomic_store_n(&s_rec.tail, t, __ATOMIC_RELEASE);
            }
            __atomic_store_n(&s_rec.tail, t, __ATOMIC_RELEASE);
            if (ct0) __atomic_add_fetch(&s_rec.c_busy_tsc, vk_m2e4_rdtsc() - ct0, __ATOMIC_RELAXED);
            continue;
        }
        if (__atomic_load_n(&s_rec.quit, __ATOMIC_ACQUIRE)) break;
        int got = 0;
        for (u32 s = 0; s < 32u; s++) {
            vk_cpu_pause();
            if (__atomic_load_n(&s_rec.head, __ATOMIC_ACQUIRE) != t ||
                __atomic_load_n(&s_rec.quit, __ATOMIC_ACQUIRE)) { got = 1; break; }
        }
        if (got) continue;
        __atomic_store_n(&s_rec.sleeping, 1u, __ATOMIC_RELEASE);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        if (__atomic_load_n(&s_rec.head, __ATOMIC_ACQUIRE) != t ||
            __atomic_load_n(&s_rec.quit, __ATOMIC_ACQUIRE)) {
            __atomic_store_n(&s_rec.sleeping, 0u, __ATOMIC_RELAXED);
            continue;
        }
        vk_futex_wait(&s_rec.sleeping, 1u);
        __atomic_store_n(&s_rec.sleeping, 0u, __ATOMIC_RELAXED);
    }
    return NULL;
}

static void vk_rec_start(void)
{
    if (s_rec.running) return;
    const char* e = getenv("VK_REC_ASYNC");
    if (e && *e == '0') return;
    memset(&s_rec, 0, sizeof s_rec);
    u32 batch = 4096u;
    { const char* b = getenv("VK_REC_BATCH");
      if (b && atoi(b) > 0) batch = (u32)atoi(b);
      if (batch < 1u) batch = 1u;
      if (batch > VK_REC_RING_N / 2u) batch = VK_REC_RING_N / 2u; }
    s_rec.batch = batch;
    void* mem = NULL;
    if (posix_memalign(&mem, 64, (size_t)VK_REC_RING_N * sizeof(vk_rec_pkt)) != 0 || !mem) return;
    memset(mem, 0, (size_t)VK_REC_RING_N * sizeof(vk_rec_pkt));
    s_rec.ring = (vk_rec_pkt*)mem;
#ifndef __linux__
    pthread_mutex_init(&s_rec.mu, NULL);
    pthread_cond_init(&s_rec.cv_cons, NULL);
    pthread_cond_init(&s_rec.cv_prod, NULL);
#endif
    if (pthread_create(&s_rec.thread, NULL, vk_rec_thread_main, NULL) != 0) {
#ifndef __linux__
        pthread_cond_destroy(&s_rec.cv_prod);
        pthread_cond_destroy(&s_rec.cv_cons);
        pthread_mutex_destroy(&s_rec.mu);
#endif
        free(s_rec.ring);
        s_rec.ring = NULL;
        return;
    }
    s_rec.running = 1;
    VK_LOG("draw batching: async command recorder thread ON (ring=%u x 64B, batch=%u)\n",
           VK_REC_RING_N, s_rec.batch);
}

static void vk_rec_stop(void)
{
    if (!s_rec.running) return;
    vk_rec_drain();
    __atomic_store_n(&s_rec.quit, 1u, __ATOMIC_RELEASE);
    vk_rec_wake(1);
    pthread_join(s_rec.thread, NULL);
#ifndef __linux__
    pthread_cond_destroy(&s_rec.cv_prod);
    pthread_cond_destroy(&s_rec.cv_cons);
    pthread_mutex_destroy(&s_rec.mu);
#endif
    free(s_rec.ring);
    memset(&s_rec, 0, sizeof s_rec);
}

static int vk_eb_prepare_slot(void)
{
    if (s_eb.slot_ready) return 0;
    vk_rec_wait_slot_free(s_eb.cur);
    vk_eb_slot* sl = &s_eb.slot[s_eb.cur];
    if (sl->inflight) {
        if (s_eng_stats) {
            const double w0 = vk_now_ms();
            VK_CHECK(pvkWaitForFences(s_vk.device, 1, &sl->fence, VK_TRUE, UINT64_MAX), "vkWaitForFences(slot)");
            const double dw = vk_now_ms() - w0;
            s_engtime.gpu_wait_ms += dw;
            if (g_rsx_sf.enabled) g_rsx_sf.vk_flush_wait_ms += dw;
        } else {
            VK_CHECK(pvkWaitForFences(s_vk.device, 1, &sl->fence, VK_TRUE, UINT64_MAX), "vkWaitForFences(slot)");
        }
        sl->inflight = 0;
    }
    for (u32 r = 0; r < sl->n_retired; r++) {
        vk_image* im = &sl->retired[r];
        if (im->view) pvkDestroyImageView(s_vk.device, im->view, NULL);
        if (im->img)  pvkDestroyImage(s_vk.device, im->img, NULL);
        if (im->mem)  pvkFreeMemory(s_vk.device, im->mem, NULL);
    }
    sl->n_retired = 0;
    s_eb.vb_head = s_eb.ib_head = s_eb.ub_head = s_eb.sb_head = 0;
    s_eb.cur_uvs = (VkDeviceSize)-1;
    s_eb.cur_ups = (VkDeviceSize)-1;
    s_eb.slot_ready = 1;
    return 0;
}

static int vk_eb_open_cmd(void)
{
    if (vk_eb_prepare_slot()) return -1;
    vk_eb_slot* sl = &s_eb.slot[s_eb.cur];
    s_vk.cmd = sl->cmd;
    VK_CHECK(pvkResetCommandBuffer(s_vk.cmd, 0), "vkResetCommandBuffer(eb)");
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VK_CHECK(pvkBeginCommandBuffer(s_vk.cmd, &bi), "vkBeginCommandBuffer(eb)");
    if (s_vk.present_order) { vk_cmd_after_everything(s_vk.cmd); s_vk.present_order = 0; }
    s_eb.rp_state_valid = 0;
    vk_eb_inval_binds();
    return 0;
}

static void vk_eb_wait_all_slots(void)
{
    if (s_rec.running) vk_rec_drain();
    for (u32 i = 0; i < VK_EB_SLOTS; i++) {
        vk_rec_wait_slot_free(i);
        vk_eb_slot* sl = &s_eb.slot[i];
        if (!sl->inflight) continue;
        if (s_eng_stats) {
            const double w0 = vk_now_ms();
            pvkWaitForFences(s_vk.device, 1, &sl->fence, VK_TRUE, UINT64_MAX);
            const double dw = vk_now_ms() - w0;
            s_engtime.gpu_wait_ms += dw;
            if (g_rsx_sf.enabled) g_rsx_sf.vk_flush_wait_ms += dw;
        } else {
            pvkWaitForFences(s_vk.device, 1, &sl->fence, VK_TRUE, UINT64_MAX);
        }
        sl->inflight = 0;
    }
    for (u32 i = 0; i < VK_EB_SLOTS; i++) {
        vk_eb_slot* sl = &s_eb.slot[i];
        for (u32 r = 0; r < sl->n_retired; r++) {
            vk_image* im = &sl->retired[r];
            if (im->view) pvkDestroyImageView(s_vk.device, im->view, NULL);
            if (im->img)  pvkDestroyImage(s_vk.device, im->img, NULL);
            if (im->mem)  pvkFreeMemory(s_vk.device, im->mem, NULL);
        }
        sl->n_retired = 0;
    }
    s_eb.vb_head = s_eb.ib_head = s_eb.ub_head = s_eb.sb_head = 0;
    s_eb.cur_uvs = (VkDeviceSize)-1;
    s_eb.cur_ups = (VkDeviceSize)-1;
    s_eb.slot_ready = 1;
    s_vk.cmd = s_vk.sync_cmd;
}

static void vk_eb_flush_slot(int wait_now)
{
    if (s_eb_pending) {
        s_eb_pending = 0;
        if (s_rec.batch_async) {
            s_rec.batch_async = 0;
            vk_rec_pkt* p = vk_rec_alloc_pkt();
            p->op = VK_REC_OP_SUBMIT;
            p->u8_a = (u8)s_eb.cur;
            vk_rec_commit_pkt(1);
            s_eb.slot_ready = 0;
            s_eb.cur = (s_eb.cur + 1u) % VK_EB_SLOTS;
            if ((++s_eb.flushes & 0x3FFFu) == 0)
                VK_LOG("draw batching: %llu draws in %llu submissions\n", s_eb.draws, s_eb.flushes);
        } else {
            vk_eb_end_rp();
            vk_eb_slot* sl = &s_eb.slot[s_eb.cur];
            if (pvkEndCommandBuffer(sl->cmd) == VK_SUCCESS) {
                VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                    .commandBufferCount = 1, .pCommandBuffers = &sl->cmd };
                pvkResetFences(s_vk.device, 1, &sl->fence);
                if (vk_queue_submit(&si, sl->fence) == VK_SUCCESS) {
                    sl->inflight = 1;
                    s_engtime.submits++;
                }
            }
            vk_eb_touch_clear();
            s_eb.slot_ready = 0;
            s_eb.cur = (s_eb.cur + 1u) % VK_EB_SLOTS;
            if ((++s_eb.flushes & 0x3FFFu) == 0)
                VK_LOG("draw batching: %llu draws in %llu submissions\n", s_eb.draws, s_eb.flushes);
        }
    }
    if (wait_now) vk_eb_wait_all_slots();
}

static void vk_eng_flush(void)
{
    if (s_eb.on != 1) return;
    if (g_rsx_sf.enabled) g_rsx_sf.vk_sync_flushes++;
    vk_eb_flush_slot(1);
}

static void vk_eng_flush_async(void)
{
    if (s_eb.on != 1) return;
    vk_eb_flush_slot(!vk_eb_async_use());
}

static int vk_eb_on(void)
{
    if (s_eb.on >= 0) return s_eb.on;
    const char* e = getenv("VK_DRAW_BATCH");
    s_eb.on = 0;
    if (!e || *e != '1' || !s_g.on) return 0;
    const char* m = getenv("VK_DRAW_BATCH_MAX");
    if (m && atoi(m) > 0) {
        s_eb.max_draws = (u32)atoi(m);
        s_eb.max_draws_explicit = 1;
    } else {
        s_eb.max_draws = 512u;
        s_eb.max_draws_explicit = 0;
    }
    VkPhysicalDeviceProperties p;
    pvkGetPhysicalDeviceProperties(s_vk.phys, &p);
    s_eb.ub_align = p.limits.minUniformBufferOffsetAlignment ? p.limits.minUniformBufferOffsetAlignment : 256;
    s_eb.slot_ub = VK_EB_SLOT_UB;
    if (s_g.ssbo_ok) {
        const VkDeviceSize overhead = VK_VPCONST_BYTES + 4096u;
        if (s_g.max_ssbo_range > overhead) {
            const VkDeviceSize max_slot = ((s_g.max_ssbo_range - overhead) / VK_EB_SLOTS) & ~(VkDeviceSize)4095u;
            if (max_slot < s_eb.slot_ub) s_eb.slot_ub = max_slot;
        }
    }
    VkDescriptorPoolSize ps[3] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2 * VK_EB_PERSIST_SETS },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, RSX_MAX_TEXTURES * VK_EB_PERSIST_SETS },
        { VK_DESCRIPTOR_TYPE_SAMPLER, RSX_MAX_TEXTURES * VK_EB_PERSIST_SETS },
    };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = VK_EB_PERSIST_SETS, .poolSizeCount = 3, .pPoolSizes = ps };
    if (vk_create_host_buffer(&s_eb.vb, (VkDeviceSize)VK_EB_SLOTS * VK_EB_SLOT_VB, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) || !s_eb.vb.coherent ||
        vk_create_host_buffer(&s_eb.ib, (VkDeviceSize)VK_EB_SLOTS * VK_EB_SLOT_IB, VK_BUFFER_USAGE_INDEX_BUFFER_BIT) || !s_eb.ib.coherent ||
        vk_create_host_buffer(&s_eb.ub, (VkDeviceSize)VK_EB_SLOTS * s_eb.slot_ub + VK_VPCONST_BYTES + 4096u,
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) || !s_eb.ub.coherent ||
        vk_create_host_buffer(&s_eb.sb, (VkDeviceSize)VK_EB_SLOTS * VK_EB_SLOT_SB, VK_BUFFER_USAGE_TRANSFER_SRC_BIT) || !s_eb.sb.coherent) {
        VK_LOG("draw batching: no host-coherent ring memory, staying off\n");
        return 0;
    }
    if (pvkCreateDescriptorPool(s_vk.device, &pci, NULL, &s_eb.desc_pool) != VK_SUCCESS) {
        VK_LOG("draw batching: failed to create persistent descriptor pool\n");
        return 0;
    }
    if (s_g.ssbo_ok) {
        VkDescriptorPoolSize sps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
        VkDescriptorPoolCreateInfo spci = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &sps
        };
        VkDescriptorSetAllocateInfo sdai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = VK_NULL_HANDLE, .descriptorSetCount = 1,
            .pSetLayouts = &s_g.ssbo_const_layout
        };
        if (pvkCreateDescriptorPool(s_vk.device, &spci, NULL, &s_eb.ssbo_const_pool) == VK_SUCCESS) {
            sdai.descriptorPool = s_eb.ssbo_const_pool;
            if (pvkAllocateDescriptorSets(s_vk.device, &sdai, &s_eb.ssbo_const_set) == VK_SUCCESS) {
                VkDescriptorBufferInfo sbi = { s_eb.ub.buf, 0, s_eb.ub.size };
                VkWriteDescriptorSet sw = {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = s_eb.ssbo_const_set,
                    .dstBinding = RSX_SPIRV_SSBO_BINDING,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &sbi
                };
                pvkUpdateDescriptorSets(s_vk.device, 1, &sw, 0, NULL);
            } else {
                s_eb.ssbo_const_set = VK_NULL_HANDLE;
            }
        }
    }
    s_eb.desc_sets_used = 0;
    VkCommandPoolCreateInfo rpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                     .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                     .queueFamilyIndex = s_vk.queue_family };
    if (pvkCreateCommandPool(s_vk.device, &rpci, NULL, &s_eb.rec_pool) != VK_SUCCESS) {
        VK_LOG("draw batching: failed to create recorder command pool\n");
        return 0;
    }
    VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                        .commandPool = s_eb.rec_pool,
                                        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                        .commandBufferCount = 1 };
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    for (u32 i = 0; i < VK_EB_SLOTS; i++) {
        if (pvkAllocateCommandBuffers(s_vk.device, &cai, &s_eb.slot[i].cmd) != VK_SUCCESS ||
            pvkCreateFence(s_vk.device, &fci, NULL, &s_eb.slot[i].fence) != VK_SUCCESS) {
            VK_LOG("draw batching: failed to allocate async slot %u\n", i);
            return 0;
        }
    }
    s_eb.cur = 0;
    s_eb.slot_ready = 1;
    s_eb.cur_uvs = s_eb.cur_ups = (VkDeviceSize)-1;
    s_eb.gcache_gen = 1u;
    s_eb.touch_gen = 1u;
    s_eb.flags_phase = -999999;
    /* Persistent GPU vertex/index cache in ReBAR VRAM (optional, non-fatal). */
    if (vk_create_rebar_buffer(&s_eb.gvb, VK_EB_GVB_BYTES, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) == 0 &&
        vk_create_rebar_buffer(&s_eb.gib, VK_EB_GIB_BYTES, VK_BUFFER_USAGE_INDEX_BUFFER_BIT) == 0) {
        s_eb.gvb_tab = (vk_eb_gcent*)calloc(VK_EB_GC_ENTS, sizeof(vk_eb_gcent));
        s_eb.gib_tab = (vk_eb_gcent*)calloc(VK_EB_GC_ENTS, sizeof(vk_eb_gcent));
        if (s_eb.gvb_tab && s_eb.gib_tab) {
            s_eb.gcache_ok = 1;
        } else {
            free(s_eb.gvb_tab); free(s_eb.gib_tab);
            s_eb.gvb_tab = s_eb.gib_tab = NULL;
        }
    }
    s_eb.max_ind_count = p.limits.maxDrawIndirectCount ? p.limits.maxDrawIndirectCount : 1u;
    if (s_vk.multi_draw_indirect_ok && s_eb.max_ind_count > 1u) {
        if (vk_create_host_buffer(&s_eb.ind, (VkDeviceSize)VK_EB_SLOTS * VK_EB_SLOT_IND,
                                  VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) == 0 && s_eb.ind.coherent) {
            s_eb.ind_ok = 1;
        } else {
            vk_destroy_buffer(&s_eb.ind);
            s_eb.ind_ok = 0;
        }
    }
    VK_LOG("draw batching: on (%u async slots, gcache=%d, %u MB gvb / %u MB gib, indirect=%d)\n",
           VK_EB_SLOTS, s_eb.gcache_ok,
           (unsigned)(VK_EB_GVB_BYTES >> 20), (unsigned)(VK_EB_GIB_BYTES >> 20), s_eb.ind_ok);
    s_eb.on = 1;
    vk_rec_start();
    return 1;
}

static void vk_eb_release(void)
{
    if (s_eb.on == 1) vk_eng_flush();
    vk_rec_stop();
    for (u32 i = 0; i < VK_EB_SLOTS; i++) {
        if (s_eb.slot[i].fence) pvkDestroyFence(s_vk.device, s_eb.slot[i].fence, NULL);
    }
    if (s_eb.rec_pool) pvkDestroyCommandPool(s_vk.device, s_eb.rec_pool, NULL);
    if (s_eb.ssbo_const_pool) pvkDestroyDescriptorPool(s_vk.device, s_eb.ssbo_const_pool, NULL);
    if (s_eb.desc_pool) pvkDestroyDescriptorPool(s_vk.device, s_eb.desc_pool, NULL);
    vk_destroy_buffer(&s_eb.vb);
    vk_destroy_buffer(&s_eb.ib);
    vk_destroy_buffer(&s_eb.ub);
    vk_destroy_buffer(&s_eb.sb);
    vk_destroy_buffer(&s_eb.ind);
    vk_destroy_buffer(&s_eb.gvb);
    vk_destroy_buffer(&s_eb.gib);
    free(s_eb.gvb_tab);
    free(s_eb.gib_tab);
    memset(&s_eb, 0, sizeof s_eb);
    memset(s_eb_desc_tag, 0, sizeof s_eb_desc_tag);
    memset(s_eb_desc, 0, sizeof s_eb_desc);
    memset(s_eb_desc_rr, 0, sizeof s_eb_desc_rr);
    memset(s_obj_in_desc, 0, sizeof s_obj_in_desc);
    s_eb_desc_live_n = 0;
    g_rsx_vk_gcache_gen = 0u;
    s_eb.flags_phase = -999999;
    s_eb.on = -1;
}

static int vk_eb_active(void) { return s_eb.on == 1; }
static u32 vk_eb_max(void)
{
    if (s_eb.max_draws_explicit) return s_eb.max_draws;
    return vk_eb_async_use() ? 32768u : 512u;
}
static void vk_eb_recorded(void) { if (++s_eb_pending >= vk_eb_max()) vk_eng_flush_async(); }

static VkDeviceSize vk_eb_take(VkDeviceSize* head, VkDeviceSize cap, VkDeviceSize bytes, VkDeviceSize align);

static int vk_eb_gcache_get(vk_buffer* buf, VkDeviceSize* head, VkDeviceSize cap,
                            vk_eb_gcent* tab, u64 id, const void* src, VkDeviceSize bytes,
                            VkDeviceSize align, VkDeviceSize* out_off, u64* hits, u64* misses)
{
    if (!id || !bytes || bytes > (16u << 20)) return 0;
    const u32 set = (u32)((id * 0x9E3779B97F4A7C15ull) >> 18) & (VK_EB_GC_SETS - 1u);
    vk_eb_gcent* base = &tab[set * VK_EB_GC_WAYS];
    for (u32 w = 0; w < VK_EB_GC_WAYS; w++) {
        if (base[w].id == id && base[w].bytes == (u32)bytes) {
            *out_off = base[w].off;
            (*hits)++;
            return 1;
        }
    }
    if (align < 16u) align = 16u;
    VkDeviceSize off = vk_eb_take(head, cap, bytes, align);
    if (off == (VkDeviceSize)-1) {
        /* Cache full: sync GPU, reset bump arena and table. */
        vk_eng_flush();
        *head = 0;
        memset(tab, 0, (size_t)VK_EB_GC_ENTS * sizeof(vk_eb_gcent));
        s_eb.gcache_gen++;
        if (!s_eb.gcache_gen) s_eb.gcache_gen = 1u;
        if (s_eb.f_eb && s_eb.f_gcache) g_rsx_vk_gcache_gen = s_eb.gcache_gen;
        off = vk_eb_take(head, cap, bytes, align);
        if (off == (VkDeviceSize)-1) return 0;
    }
    memcpy((u8*)buf->ptr + off, src, (size_t)bytes);
    u32 way = (base[0].id == 0) ? 0u : (base[1].id == 0 ? 1u : (u32)(id & 1u));
    base[way].id = id;
    base[way].off = (u32)off;
    base[way].bytes = (u32)bytes;
    *out_off = off;
    (*misses)++;
    return 1;
}

static int vk_eb_upload(VkImage img, VkBufferImageCopy* region, const void* src, size_t tight,
                        u32 rows, u32 row_bytes)
{
    if (!vk_eb_active()) return -2;
    const VkDeviceSize bytes = (VkDeviceSize)tight * rows;
    if (!bytes || bytes > VK_EB_SB_BYTES) return -2;
    if (vk_eb_prepare_slot()) return -1;
    VkDeviceSize off = vk_eb_take(&s_eb.sb_head, VK_EB_SB_BYTES, bytes, 16u);
    if (off == (VkDeviceSize)-1) {
        vk_eng_flush_async();
        if (vk_eb_prepare_slot()) return -1;
        off = vk_eb_take(&s_eb.sb_head, VK_EB_SB_BYTES, bytes, 16u);
        if (off == (VkDeviceSize)-1) return -2;
    }
    const VkDeviceSize abs_off = (VkDeviceSize)s_eb.cur * VK_EB_SLOT_SB + off;
    for (u32 y = 0; y < rows; y++)
        memcpy((u8*)s_eb.sb.ptr + abs_off + (size_t)y * tight, (const u8*)src + (size_t)y * row_bytes, tight);
    region->bufferOffset = abs_off;
    if (vk_rec_use_batch()) {
        if (!s_eb_pending && vk_rec_ensure_open()) return -1;
        vk_rec_emit_upload(img, region);
        vk_eb_recorded();
        return 0;
    }
    if (!s_eb_pending && vk_eb_open_cmd()) return -1;
    vk_eb_end_rp();
    vk_cmd_after_everything(s_vk.cmd);
    vk_eb_touch_clear();
    pvkCmdCopyBufferToImage(s_vk.cmd, s_eb.sb.buf, img, VK_IMAGE_LAYOUT_GENERAL, 1, region);
    vk_barrier_image(img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    vk_eb_recorded();
    return 0;
}

static VkDeviceSize vk_eb_take(VkDeviceSize* head, VkDeviceSize cap, VkDeviceSize bytes, VkDeviceSize align)
{
    const VkDeviceSize at = ((align & (align - 1u)) == 0u)
        ? ((*head + align - 1u) & ~(align - 1u))
        : (((*head + align - 1u) / align) * align);
    if (at + bytes > cap) return (VkDeviceSize)-1;
    *head = at + bytes;
    return at;
}

/* The views and samplers a pipeline's set names: what a batched draw's
 * descriptor cache is keyed on. The constants are not in here; they move
 * through the two dynamic uniform offsets. */
static int vk_eb_key_build(const vk_eng_pipe* P, vk_eb_desc_key* k)
{
    static vk_eb_desc_key s_def_key;
    static VkImageView s_def_view;
    static VkSampler s_def_samp;
    const VkImageView dummy_2d = vk_eng_dummy_2d_view();
    if (!dummy_2d) return -1;
    if (s_def_view != dummy_2d || s_def_samp != s_vk.sampler) {
        memset(&s_def_key, 0, sizeof s_def_key);
        s_def_key.vs_bytes = VK_VPCONST_BYTES;
        for (u32 u = 0; u < RSX_MAX_TEXTURES; u++) {
            s_def_key.view[u] = dummy_2d;
            s_def_key.sampler[u] = s_vk.sampler;
        }
        s_def_view = dummy_2d;
        s_def_samp = s_vk.sampler;
    }
    const int split = P->cube_mask != 0;
    const VkImageView dummy_cube = split ? vk_eng_dummy_cube_view() : VK_NULL_HANDLE;
    if (split && !dummy_cube) return -1;         /* may flush: before any slice */
    *k = s_def_key;
    k->split = split;
    k->ps_bytes = s_e2.ps_bytes ? s_e2.ps_bytes : 16u;
    u32 m = s_e2.tex_mask | P->cube_mask;
    while (m) {
        const u32 u = (u32)__builtin_ctz(m);
        m &= m - 1u;
        const int cube = (P->cube_mask >> u) & 1u;
        VkImageView v = ((s_e2.tex_mask >> u) & 1u) ? vk_eng_sample_view(s_e2.tex[u], cube)
                                                     : VK_NULL_HANDLE;
        if (v) {
            k->view[u] = v;
            k->sampler[u] = s_e2.samp_vk[u] ? s_e2.samp_vk[u] : vk_eng_sampler(&s_e2.samp[u]);
        } else if (cube) {
            k->view[u] = dummy_cube;
        }
    }
    return 0;
}

static VkDescriptorSet vk_eng_write_descriptors_to(const vk_eb_desc_key* k, VkDescriptorSet set,
                                                   VkBuffer ubo)
{
    if (!set) return VK_NULL_HANDLE;
    VkDescriptorBufferInfo bi[2] = {
        { ubo, 0, k->vs_bytes },
        { ubo, 0, k->ps_bytes },
    };
    VkDescriptorImageInfo ii[RSX_MAX_TEXTURES], si[RSX_MAX_TEXTURES];
    for (u32 u = 0; u < RSX_MAX_TEXTURES; u++) {
        ii[u] = (VkDescriptorImageInfo){ VK_NULL_HANDLE, k->view[u], VK_IMAGE_LAYOUT_GENERAL };
        si[u] = (VkDescriptorImageInfo){ k->sampler[u], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
    }
    VkWriteDescriptorSet w[3 + RSX_MAX_TEXTURES];
    u32 n = 0;
    if (!k->ssbo) {
        w[n++] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
              .dstBinding = RSX_SPIRV_VPCONST_BINDING, .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &bi[0] };
        w[n++] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
              .dstBinding = RSX_SPIRV_PSCONST_BINDING, .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &bi[1] };
    }
    w[n++] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
          .dstBinding = RSX_SPIRV_SAMPLER_BINDING, .descriptorCount = RSX_MAX_TEXTURES,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .pImageInfo = si };
    if (k->split) {
        for (u32 u = 0; u < RSX_MAX_TEXTURES; u++)
            w[n++] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                  .dstSet = set, .dstBinding = RSX_SPIRV_TEXTURE_BINDING + u, .descriptorCount = 1,
                  .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &ii[u] };
    } else {
        w[n++] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
              .dstBinding = RSX_SPIRV_TEXTURE_BINDING, .descriptorCount = RSX_MAX_TEXTURES,
              .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = ii };
    }
    pvkUpdateDescriptorSets(s_vk.device, n, w, 0, NULL);
    return set;
}

/* Fill the set the bound pipeline uses for the unbatched path: the shared one
 * (sixteen textures in one binding), or for a pipeline with cube units the
 * split one (one binding per unit). Returns the set, or VK_NULL_HANDLE. */
static VkDescriptorSet vk_eng_write_descriptors(const vk_eng_pipe* P, int ssbo_on)
{
    vk_eb_desc_key k;
    if (vk_eb_key_build(P, &k)) return VK_NULL_HANDLE;
    k.ssbo = ssbo_on;
    VkDescriptorSet dst = ssbo_on
        ? (P->cube_mask ? s_e2.ssbo_split_tex_set : s_g.ssbo_tex_set)
        : (P->cube_mask ? s_e2.split_set : s_g.set);
    return vk_eng_write_descriptors_to(&k, dst, s_g.ubo.buf);
}

/* A batched draw's set. The tuple is the whole key and the comparison is a
 * full memcmp, so a hash collision cannot bind another draw's textures.
 * In SSBO mode psb_key == 0 (ps_bytes is not in set 1; UBO sets always have
 * ps_bytes >= 16, so both modes coexist in the cache during in-run A/B). */
static VkDescriptorSet vk_eb_descriptors(const vk_eng_pipe* P, int async_on, int ssbo_on)
{
    const u32 vsb = VK_VPCONST_BYTES;
    const u32 psb = ssbo_on ? 0u : (s_e2.ps_bytes ? s_e2.ps_bytes : 16u);
    if (async_on && !s_e2.tex_dirty && s_eb.last_desc_set &&
        s_eb.last_desc_epoch == s_eb.desc_epoch &&
        s_eb.last_desc_obj_gen == s_obj_gen &&
        s_eb.last_desc_cube_mask == P->cube_mask &&
        s_eb.last_desc_vsb == vsb && s_eb.last_desc_psb == psb) {
        s_engtime.desc_hits++;
        return s_eb.last_desc_set;
    }
    u64 h = s_e2.tex_bind_hash ^ ((u64)psb << 32) ^ ((u64)P->cube_mask << 16);
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 33;
    const u32 bkt = (u32)(h & (VK_EB_DESC_BUCKETS - 1u));
    const u32 base_i = bkt * VK_EB_DESC_WAYS;
    vk_eb_desc_tag* const tbase = &s_eb_desc_tag[base_i];
    vk_eb_desc* const base = &s_eb_desc[base_i];
    int victim_way = -1;
    const u32 cur_epoch = s_eb.desc_epoch;
    const u32 cur_ogen = s_obj_gen;
    const u16 cur_cmask = (u16)P->cube_mask;
    const u16 cur_tmask = (u16)s_e2.tex_mask;
    const u32 cur_tbgen = s_e2.tex_bind_gen;
    for (u32 way = 0; way < VK_EB_DESC_WAYS; way++) {
        vk_eb_desc_tag* const tg = &tbase[way];
        if (tg->set && tg->epoch == cur_epoch && tg->obj_gen == cur_ogen) {
            if (tg->hash == h && tg->ps_bytes == psb &&
                tg->cube_mask == cur_cmask && tg->tex_mask == cur_tmask) {
                int match = (tg->tex_bind_gen == cur_tbgen && cur_tbgen != 0u);
                if (!match) {
                    vk_eb_desc* const e = &base[way];
                    match = 1;
                    u32 m = cur_tmask;
                    while (m) {
                        const u32 u = (u32)__builtin_ctz(m);
                        m &= m - 1u;
                        if (e->fkey.tex[u] != s_e2.tex[u] || e->fkey.samp[u] != s_e2.samp_idx[u]) {
                            match = 0; break;
                        }
                    }
                    if (match) tg->tex_bind_gen = cur_tbgen;
                }
                if (match) {
                    s_engtime.desc_hits++;
                    s_e2.tex_dirty = 0;
                    s_eb.last_desc_set = tg->set;
                    s_eb.last_desc_epoch = cur_epoch;
                    s_eb.last_desc_obj_gen = cur_ogen;
                    s_eb.last_desc_cube_mask = P->cube_mask;
                    s_eb.last_desc_vsb = vsb;
                    s_eb.last_desc_psb = psb;
                    return tg->set;
                }
            }
        } else if (victim_way < 0) {
            victim_way = (int)way;
        }
    }
    const u32 dst_way = (victim_way >= 0) ? (u32)victim_way
                                          : (s_eb_desc_rr[bkt]++ & (VK_EB_DESC_WAYS - 1u));
    vk_eb_desc_tag* const tg = &tbase[dst_way];
    vk_eb_desc* const e = &base[dst_way];
    vk_eb_desc_key k;
    if (vk_eb_key_build(P, &k)) return VK_NULL_HANDLE;
    k.ssbo = ssbo_on;
    if (s_eb.desc_sets_used >= VK_EB_PERSIST_SETS) return VK_NULL_HANDLE;
    s_engtime.desc_misses++;
    VkDescriptorSetLayout lay = ssbo_on
        ? (k.split ? s_e2.ssbo_split_tex_layout : s_g.ssbo_tex_layout)
        : (k.split ? s_e2.split_layout : s_g.set_layout);
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = s_eb.desc_pool, .descriptorSetCount = 1,
                                        .pSetLayouts = &lay };
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (pvkAllocateDescriptorSets(s_vk.device, &dai, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
    s_eb.desc_sets_used++;
    if (!vk_eng_write_descriptors_to(&k, set, s_eb.ub.buf)) return VK_NULL_HANDLE;
    tg->hash = h; tg->epoch = cur_epoch; tg->obj_gen = cur_ogen;
    tg->tex_bind_gen = cur_tbgen;
    tg->ps_bytes = psb; tg->cube_mask = cur_cmask; tg->tex_mask = cur_tmask;
    tg->set = set;
    e->fkey.ps_bytes = psb;
    e->fkey.cube_mask = cur_cmask;
    e->fkey.tex_mask = cur_tmask;
    { u32 m = cur_tmask;
      while (m) {
          const u32 u = (u32)__builtin_ctz(m);
          m &= m - 1u;
          const u32 thd = s_e2.tex[u];
          e->fkey.tex[u] = thd;
          if (thd < VK_ENG_MAX_OBJ) s_obj_in_desc[thd] = 1;
          e->fkey.samp[u] = s_e2.samp_idx[u];
      } }
    if (!e->live && s_eb_desc_live_n < VK_EB_DESC_SETS) {
        s_eb_desc_live[s_eb_desc_live_n++] = (u16)(base_i + dst_way);
        e->live = 1;
    }
    s_e2.tex_dirty = 0;
    s_eb.last_desc_set = set;
    s_eb.last_desc_epoch = cur_epoch;
    s_eb.last_desc_obj_gen = cur_ogen;
    s_eb.last_desc_cube_mask = P->cube_mask;
    s_eb.last_desc_vsb = vsb;
    s_eb.last_desc_psb = psb;
    return set;
}

static void vk_eng_draw(void* user, rsx_topology topology, const void* vertices, u32 vertex_count,
                        u32 stride, const u32* indices, u32 index_count)
{
    (void)user;
    if (!s_e2.pipeline || !vertices || !vertex_count || !stride) return;
    if (topology < RSX_TOPOLOGY_POINTS || topology > RSX_TOPOLOGY_TRIANGLE_STRIP) return;
    { const int cur_ph = ENG_AB_PHASE();
      if (__builtin_expect(s_eb.flags_phase != cur_ph, 0)) vk_eb_refresh_flags(cur_ph); }
    const int vk_prof = s_eb.f_prof;
    const int m2e4 = s_eb.f_m2e4;
    const int m2e4b = s_eb.f_m2e4b;
    const int m2f = s_eb.f_m2f;
    const u64 vp_t0 = vk_prof ? vk_m2e4_rdtsc() : 0;
    static u64 s_vp_gc, s_vp_desc, s_vp_ubo, s_vp_haz, s_vp_bind, s_vp_draw, s_vp_tot, s_vp_t0;
    static u64 s_vp_npipe, s_vp_ndesc, s_vp_npush, s_vp_nvb, s_vp_nib, s_vp_ndraw;
    static u32 s_vp_fr0;
    static double s_vp_ms0;
    const int ssbo_on = (int)s_eb.f_ssbo && (!s_eb.f_eb || s_eb.ssbo_const_set != VK_NULL_HANDLE);
    const int inst_on = ssbo_on && (int)s_eb.f_ind;
    const int ssbo_mode = inst_on ? 2 : (ssbo_on ? 1 : 0);
    vk_eng_pipe* P = &s_epipe[s_e2.pipeline - 1];
    if (__builtin_expect(!P->used || stride != P->stride, 0)) return;

    /* Ultra-fast path (ENG_M2F): 98%+ of stationary draws are indexed gcache hits
     * with cached RT/FB, cached pipeline, cached descriptor set (or fast desc hit),
     * and open async slot. */
    if (__builtin_expect(m2f && !vk_prof && s_eb.f_eb && s_eb.f_async && s_eb.f_gcache &&
                         ssbo_on && s_eb.slot_ready && s_eb_pending && s_rec.batch_async &&
                         indices && index_count && s_e2.rt_cache_ok &&
                         s_e2.rt_cache_n >= 1u && s_e2.rt_cache_n == s_eng_bound.nrt &&
                         s_e2.rt_cache_n == P->rt_count && s_e2.rt_cache_fmt == P->rt_fmt &&
                         ((s_e2.tex_rt_mask == 0u && s_rec.p_tex_rt_n == 0u) ||
                          (s_rec.p_tex_rt_gen == s_e2.tex_bind_gen && s_rec.p_tex_rt_mask == s_e2.tex_rt_mask)), 1)) {
        const u32 topo_idx = (u32)topology - 1u;
        const VkPipeline pipe = (ssbo_mode == 2) ? P->variant_inst[topo_idx]
                              : (ssbo_mode == 1) ? P->variant_ssbo[topo_idx]
                                                 : P->variant[topo_idx];
        const rsx_eng_gcache_memo* gcm_f = g_rsx_eng_cur_gcache;
        const u32 gc_gen = s_eb.gcache_gen;
        if (__builtin_expect(pipe && gcm_f && gcm_f->v_gen == gc_gen && gcm_f->i_gen == gc_gen &&
                             s_rec.p_targets_valid && s_rec.p_fb == s_e2.rt_cache_fb &&
                             s_rec.p_rp == s_e2.rt_cache_rp &&
                             s_rec.p_dyn_valid && s_rec.p_dyn2_valid && !s_e2.dyn_dirty_mask &&
                             s_rec.p_dyn_w == s_e2.rt_cache_w && s_rec.p_dyn_h == s_e2.rt_cache_h, 1)) {
            VkDescriptorSet set_f = VK_NULL_HANDLE;
            if (__builtin_expect(!s_e2.tex_dirty && s_eb.last_desc_set &&
                                 s_eb.last_desc_epoch == s_eb.desc_epoch &&
                                 s_eb.last_desc_obj_gen == s_obj_gen &&
                                 s_eb.last_desc_cube_mask == P->cube_mask &&
                                 s_eb.last_desc_psb == 0u, 1)) {
                set_f = s_eb.last_desc_set;
            } else if (s_e2.dummy_2d != 0u && (!P->cube_mask || s_e2.dummy_cube != 0u)) {
                set_f = vk_eb_descriptors(P, 1, 1);
            }
            if (__builtin_expect(set_f != VK_NULL_HANDLE, 1)) {
            const u32 vsb_f = s_e2.vs_bytes ? s_e2.vs_bytes : 16u;
            const u32 psb_f = s_e2.ps_bytes ? s_e2.ps_bytes : 16u;
            const int reuse_vs_f = !s_e2.vs_dirty && s_eb.cur_uvs != (VkDeviceSize)-1 && s_eb.cur_vsb == vsb_f;
            const int reuse_ps_f = !s_e2.ps_dirty && s_eb.cur_ups != (VkDeviceSize)-1 && s_eb.cur_psb == psb_f;
            VkDeviceSize uh_f = s_eb.ub_head;
            VkDeviceSize ruvs_f = 0, rups_f = 0;
            if (!reuse_vs_f) { ruvs_f = (uh_f + 15u) & ~(VkDeviceSize)15u; uh_f = ruvs_f + vsb_f; }
            if (!reuse_ps_f) { rups_f = (uh_f + 15u) & ~(VkDeviceSize)15u; uh_f = rups_f + psb_f; }
            if (__builtin_expect(uh_f <= s_eb.slot_ub, 1)) {
                s_eb.ub_head = uh_f;
                const VkDeviceSize ub_base_f = (VkDeviceSize)s_eb.cur * s_eb.slot_ub;
                VkDeviceSize abs_uvs_f, abs_ups_f;
                if (reuse_vs_f) {
                    abs_uvs_f = s_eb.cur_uvs;
                } else {
                    abs_uvs_f = ub_base_f + ruvs_f;
                    const void* vs_ptr_f = s_e2.vs_src ? s_e2.vs_src : (const void*)vk_eng_ucpu();
                    memcpy((u8*)s_eb.ub.ptr + abs_uvs_f, vs_ptr_f, vsb_f);
                    s_eb.cur_uvs = abs_uvs_f; s_eb.cur_vsb = vsb_f; s_e2.vs_dirty = 0;
                }
                if (reuse_ps_f) {
                    abs_ups_f = s_eb.cur_ups;
                } else {
                    abs_ups_f = ub_base_f + rups_f;
                    memcpy((u8*)s_eb.ub.ptr + abs_ups_f, vk_eng_ucpu() + s_g.ps_off, psb_f);
                    s_eb.cur_ups = abs_ups_f; s_eb.cur_psb = psb_f; s_e2.ps_dirty = 0;
                }
                if (__builtin_expect(s_eng_stats, 0)) {
                    s_engtime.gvb_hits++;
                    s_engtime.gib_hits++;
                    s_engtime.desc_hits++;
                    if (reuse_vs_f) s_engtime.ubo_reuses++; else s_engtime.ubo_bytes += vsb_f;
                    if (reuse_ps_f) s_engtime.ubo_reuses++; else s_engtime.ubo_bytes += psb_f;
                }
                vk_rec_pkt* dp = vk_rec_alloc_pkt();
                dp->op = VK_REC_OP_DRAW;
                dp->flags = (u8)(REC_DRAW_F_INDEXED | REC_DRAW_F_SSBO | REC_DRAW_F_VGC | REC_DRAW_F_IGC |
                                 (P->depth_or_stencil_write ? REC_DRAW_F_DWRITE : 0u) |
                                 (s_eb.f_hazard ? REC_DRAW_F_HAZARD : 0u) |
                                 (P->cube_mask ? REC_DRAW_F_SPLIT : 0u));
                dp->u8_a = (u8)inst_on;
                dp->u.draw.count = index_count;
                dp->u.draw.first_idx = gcm_f->i_first;
                dp->u.draw.vtx_off = gcm_f->v_base;
                dp->u.draw.dyn0 = (u32)abs_uvs_f;
                dp->u.draw.dyn1 = (u32)abs_ups_f;
                dp->u.draw.pipe = pipe;
                dp->u.draw.set = set_f;
                vk_rec_commit_pkt(0);
                s_eb.draws++;
                if (__builtin_expect(++s_eb_pending >= vk_eb_max(), 0)) vk_eng_flush_async();
                return;
            }
            }
        }
    }

    /* Targets: every member of a set is target A's size (engine contract). */
    const u32 n = s_eng_bound.nrt;
    if (n < 1 || n != P->rt_count) {
        if (!(s_e2.warned & 16u)) { VK_LOG("engine: draw with %u colour targets for a %u-target pipeline skipped\n", n, P->rt_count); s_e2.warned |= 16u; }
        return;
    }
    VkImage rt_img[RSX_BE_MAX_COLOR_TARGETS];
    u32 w, h;
    VkImage depth_img;
    VkRenderPass rp;
    VkFramebuffer fb;
    if (s_e2.rt_cache_ok && s_e2.rt_cache_n == n && s_e2.rt_cache_fmt == P->rt_fmt) {
        for (u32 i = 0; i < n; i++) rt_img[i] = s_e2.rt_cache_img[i];
        w = s_e2.rt_cache_w; h = s_e2.rt_cache_h;
        depth_img = s_e2.rt_cache_depth_img;
        rp = s_e2.rt_cache_rp;
        fb = s_e2.rt_cache_fb;
    } else {
        vk_eng_obj* rt0 = vk_eng_get(s_eng_bound.rt[0], VK_ENG_COLOR);
        if (!rt0) return;
        rt_img[0] = rt0->im.img;
        for (u32 i = 1; i < n; i++) {
            vk_eng_obj* o = vk_eng_get(s_eng_bound.rt[i], VK_ENG_COLOR);
            if (!o) return;
            rt_img[i] = o->im.img;
        }
        if (rt0->fmt != P->rt_fmt) {
            if (!(s_e2.warned & 32u)) { VK_LOG("engine: draw into colour format %d with a pipeline built for %d skipped\n", (int)rt0->fmt, (int)P->rt_fmt); s_e2.warned |= 32u; }
            return;
        }
        w = rt0->w; h = rt0->h;
        vk_eng_obj* zo = vk_eng_get(s_eng_bound.depth, VK_ENG_DEPTH);
        VkImageView dview = (zo && zo->w >= w && zo->h >= h) ? zo->im.view : vk_eng_scratch_depth(w, h);
        if (!dview) return;
        const u32 dkey = (zo && dview == zo->im.view) ? s_eng_bound.depth : 0xFFFFFFFFu;
        depth_img = (zo && dview == zo->im.view) ? zo->im.img : s_e2.scratch_depth.img;
        rp = vk_eng_render_pass(P->rt_fmt, n);
        fb = vk_eng_framebuffer(rp, s_eng_bound.rt, n, dview, dkey, w, h);
        if (!rp || !fb) return;
        for (u32 i = 0; i < n; i++) s_e2.rt_cache_img[i] = rt_img[i];
        s_e2.rt_cache_n = n; s_e2.rt_cache_w = w; s_e2.rt_cache_h = h;
        s_e2.rt_cache_fmt = rt0->fmt; s_e2.rt_cache_dkey = dkey;
        s_e2.rt_cache_depth_img = depth_img; s_e2.rt_cache_dview = dview;
        s_e2.rt_cache_rp = rp; s_e2.rt_cache_fb = fb;
        s_e2.rt_cache_ok = 1;
    }
    VkPipeline pipe = vk_eng_variant(P, (u32)topology - 1u, ssbo_mode);
    if (!pipe) return;

    const VkDeviceSize vbytes = (VkDeviceSize)vertex_count * stride;
    const VkDeviceSize ibytes = (indices && index_count) ? (VkDeviceSize)index_count * 4u : 0;
    const VkDeviceSize vb_align = stride >= 16u ? (VkDeviceSize)stride : 16u;
    VkBuffer vbuf = s_vk.vertices.buf, ibuf = s_e2.indices.buf;
    VkDeviceSize voff = 0, ioff = 0;
    VkDescriptorSet set = VK_NULL_HANDLE;
    u32 dyn[2] = { 0u, 0u };
    const u32 vsb = s_e2.vs_bytes ? s_e2.vs_bytes : 16u, psb = s_e2.ps_bytes ? s_e2.ps_bytes : 16u;
    const int async_on = s_eb.f_async;
    int batched = s_eb.f_eb && vbytes <= VK_EB_VB_BYTES && ibytes <= VK_EB_IB_BYTES;
    if (batched && (!vk_eng_dummy_2d_view() || (P->cube_mask && !vk_eng_dummy_cube_view()))) return;   /* may flush: before any slice */
    int vb_from_gc = 0, ib_from_gc = 0;
    rsx_eng_gcache_memo* gcm = NULL;
    int gcm_vbase_ok = 0, gcm_ifirst_ok = 0;
    int32_t gcm_vbase = 0;
    u32 gcm_ifirst = 0;
    if (batched && s_eb.f_gcache) {
        gcm = m2e4b ? g_rsx_eng_cur_gcache : rsx_draw_engine_current_mesh_gcache();
        if (gcm && gcm->v_gen == s_eb.gcache_gen) {
            vbuf = s_eb.gvb.buf;
            voff = gcm->v_off;
            gcm_vbase = gcm->v_base;
            gcm_vbase_ok = 1;
            vb_from_gc = 1;
            s_engtime.gvb_hits++;
        }
        if (ibytes && gcm && gcm->i_gen == s_eb.gcache_gen) {
            ibuf = s_eb.gib.buf;
            ioff = gcm->i_off;
            gcm_ifirst = gcm->i_first;
            gcm_ifirst_ok = 1;
            ib_from_gc = 1;
            s_engtime.gib_hits++;
        }
        if (__builtin_expect(!vb_from_gc || (ibytes && !ib_from_gc), 0)) {
            if (gcm) rsx_draw_engine_resolve_cur_ptrs(&vertices, &indices);
            u64 vid = 0, iid = 0;
            rsx_draw_engine_current_mesh_ids(&vid, &iid);
            if (!vb_from_gc && vid && vk_eb_gcache_get(&s_eb.gvb, &s_eb.gvb_head, VK_EB_GVB_BYTES,
                                        s_eb.gvb_tab, vid, vertices, vbytes, vb_align,
                                        &voff, &s_engtime.gvb_hits, &s_engtime.gvb_misses)) {
                vbuf = s_eb.gvb.buf;
                vb_from_gc = 1;
                if (gcm && (voff % stride) == 0 && (voff / stride) <= 0x7FFFFFFFull) {
                    gcm->v_off = (u32)voff;
                    gcm->v_base = (int32_t)(voff / stride);
                    gcm->v_gen = s_eb.gcache_gen;
                    gcm_vbase = gcm->v_base;
                    gcm_vbase_ok = 1;
                }
            }
            if (ibytes && !ib_from_gc && iid && vk_eb_gcache_get(&s_eb.gib, &s_eb.gib_head, VK_EB_GIB_BYTES,
                                                  s_eb.gib_tab, iid, indices, ibytes, 16u,
                                                  &ioff, &s_engtime.gib_hits, &s_engtime.gib_misses)) {
                ibuf = s_eb.gib.buf;
                ib_from_gc = 1;
                if (gcm && (ioff & 3u) == 0 && (ioff >> 2) <= 0xFFFFFFFFull) {
                    gcm->i_off = (u32)ioff;
                    gcm->i_first = (u32)(ioff >> 2);
                    gcm->i_gen = s_eb.gcache_gen;
                    gcm_ifirst = gcm->i_first;
                    gcm_ifirst_ok = 1;
                }
            }
        }
    }
    const u64 vp_t1 = vk_prof ? vk_m2e4_rdtsc() : 0;
    if (batched) {
        for (int tries = 0; tries < 3; tries++) {
            if (vk_eb_prepare_slot()) { batched = 0; break; }
            const VkDeviceSize vb_base = (VkDeviceSize)s_eb.cur * VK_EB_SLOT_VB;
            const VkDeviceSize ib_base = (VkDeviceSize)s_eb.cur * VK_EB_SLOT_IB;
            const VkDeviceSize ub_base = (VkDeviceSize)s_eb.cur * s_eb.slot_ub;
            const VkDeviceSize u_align = ssbo_on ? 16u : s_eb.ub_align;
            VkDeviceSize vh = s_eb.vb_head, ih = s_eb.ib_head, uh = s_eb.ub_head;
            VkDeviceSize rvoff = vb_from_gc ? 0 : vk_eb_take(&vh, VK_EB_VB_BYTES, vbytes, vb_align);
            VkDeviceSize rioff = (ibytes && !ib_from_gc) ? vk_eb_take(&ih, VK_EB_IB_BYTES, ibytes, 16u) : 0;
            const int reuse_vs = async_on && !s_e2.vs_dirty && s_eb.cur_uvs != (VkDeviceSize)-1 && s_eb.cur_vsb == vsb;
            const int reuse_ps = async_on && !s_e2.ps_dirty && s_eb.cur_ups != (VkDeviceSize)-1 && s_eb.cur_psb == psb;
            const VkDeviceSize ruvs = reuse_vs ? 0 : vk_eb_take(&uh, s_eb.slot_ub, vsb, u_align);
            const VkDeviceSize rups = (ruvs == (VkDeviceSize)-1) ? ruvs
                                    : (reuse_ps ? 0 : vk_eb_take(&uh, s_eb.slot_ub, psb, u_align));
            if (rvoff != (VkDeviceSize)-1 && rioff != (VkDeviceSize)-1 && rups != (VkDeviceSize)-1) {
                const u64 vp_td0 = vk_prof ? vk_m2e4_rdtsc() : 0;
                set = vk_eb_descriptors(P, async_on, ssbo_on);
                if (__builtin_expect(vk_prof, 0)) s_vp_desc += vk_m2e4_rdtsc() - vp_td0;
                if (!set && s_eb.desc_sets_used >= VK_EB_PERSIST_SETS) {
                    /* Persistent descriptor pool full: sync all slots and reset cleanly. */
                    vk_eng_flush();
                    pvkResetDescriptorPool(s_vk.device, s_eb.desc_pool, 0);
                    s_eb.desc_sets_used = 0;
                    s_eb.desc_epoch++;
                    s_eb.last_desc_set = VK_NULL_HANDLE;
                    for (u32 li = 0; li < s_eb_desc_live_n; li++)
                        s_eb_desc[s_eb_desc_live[li]].live = 0;
                    s_eb_desc_live_n = 0;
                    memset(s_obj_in_desc, 0, sizeof s_obj_in_desc);
                    continue;
                }
                if (set) {
                    s_eb.vb_head = vh; s_eb.ib_head = ih; s_eb.ub_head = uh;
                    if (!vb_from_gc) { vbuf = s_eb.vb.buf; voff = vb_base + rvoff; }
                    if (ibytes && !ib_from_gc) { ibuf = s_eb.ib.buf; ioff = ib_base + rioff; }
                    const u64 vp_tu0 = vk_prof ? vk_m2e4_rdtsc() : 0;
                    VkDeviceSize abs_uvs, abs_ups;
                    if (reuse_vs) {
                        abs_uvs = s_eb.cur_uvs;
                        s_engtime.ubo_reuses++;
                    } else {
                        abs_uvs = ub_base + ruvs;
                        const void* vs_ptr = (async_on && s_e2.vs_src) ? s_e2.vs_src : (const void*)vk_eng_ucpu();
                        memcpy((u8*)s_eb.ub.ptr + abs_uvs, vs_ptr, vsb);
                        s_engtime.ubo_bytes += vsb;
                        s_eb.cur_uvs = abs_uvs; s_eb.cur_vsb = vsb; s_e2.vs_dirty = 0;
                    }
                    if (reuse_ps) {
                        abs_ups = s_eb.cur_ups;
                        s_engtime.ubo_reuses++;
                    } else {
                        abs_ups = ub_base + rups;
                        memcpy((u8*)s_eb.ub.ptr + abs_ups, vk_eng_ucpu() + s_g.ps_off, psb);
                        s_engtime.ubo_bytes += psb;
                        s_eb.cur_ups = abs_ups; s_eb.cur_psb = psb; s_e2.ps_dirty = 0;
                    }
                    if (__builtin_expect(vk_prof, 0)) s_vp_ubo += vk_m2e4_rdtsc() - vp_tu0;
                    dyn[0] = (u32)abs_uvs; dyn[1] = (u32)abs_ups;
                    break;
                }
            }
            if (tries || !s_eb_pending) { batched = 0; break; }
            vk_eng_flush_async();
        }
        if (batched && !set) batched = 0;
    }
    const u64 vp_t2 = vk_prof ? vk_m2e4_rdtsc() : 0;
    const int hazard = batched ? (int)s_eb.f_hazard : 0;
    const int rec_async = batched && s_rec.running && (s_eb_pending ? s_rec.batch_async : (s_eb.f_rec_async != 0));
    if (rec_async) {
        static u64 s_vp_ecp, s_vp_est, s_vp_epk;
        if (__builtin_expect(!vb_from_gc, 0)) {
            if (gcm) rsx_draw_engine_resolve_cur_ptrs(&vertices, NULL);
            memcpy((u8*)s_eb.vb.ptr + voff, vertices, (size_t)vbytes);
        }
        if (__builtin_expect(ibytes && !ib_from_gc, 0)) {
            if (gcm) rsx_draw_engine_resolve_cur_ptrs(NULL, &indices);
            memcpy((u8*)s_eb.ib.ptr + ioff, indices, (size_t)ibytes);
        }
        const u64 vp_e1 = vk_prof ? vk_m2e4_rdtsc() : 0;
        if (__builtin_expect(!s_eb_pending, 0)) {
            if (vk_rec_ensure_open()) return;
        }
        if (__builtin_expect(!s_rec.p_targets_valid || s_rec.p_fb != fb || s_rec.p_rp != rp, 0)) {
            vk_rec_pkt* tp = vk_rec_alloc_pkt();
            tp->op = VK_REC_OP_SET_TARGETS;
            tp->u8_a = (u8)n;
            tp->u16_a = (u16)w;
            tp->u16_b = (u16)h;
            tp->u.targets.rp = rp;
            tp->u.targets.fb = fb;
            tp->u.targets.depth_img = depth_img;
            for (u32 i = 0; i < n; i++) tp->u.targets.rt_img[i] = rt_img[i];
            vk_rec_commit_pkt(0);
            s_rec.p_rp = rp; s_rec.p_fb = fb; s_rec.p_depth_img = depth_img;
            s_rec.p_nrt = n; s_rec.p_w = w; s_rec.p_h = h;
            for (u32 i = 0; i < n; i++) s_rec.p_rt_img[i] = rt_img[i];
            s_rec.p_targets_valid = 1;
        }
        if (__builtin_expect(!s_rec.p_dyn_valid || (s_e2.dyn_dirty_mask & 15u) ||
                             s_rec.p_dyn_w != w || s_rec.p_dyn_h != h, 0)) {
            vk_rec_pkt* dp = vk_rec_alloc_pkt();
            dp->op = VK_REC_OP_SET_DYN;
            dp->u.dyn.vp[0] = 0.0f; dp->u.dyn.vp[1] = 0.0f;
            dp->u.dyn.vp[2] = (float)w; dp->u.dyn.vp[3] = (float)h;
            if (s_e2.have_vp && s_e2.vp[2] > 0.0f && s_e2.vp[3] > 0.0f) {
                dp->u.dyn.vp[0] = s_e2.vp[0]; dp->u.dyn.vp[1] = s_e2.vp[1];
                dp->u.dyn.vp[2] = s_e2.vp[2]; dp->u.dyn.vp[3] = s_e2.vp[3];
            }
            if (s_e2.have_sc) {
                u32 x0 = s_e2.sc[0] < w ? s_e2.sc[0] : w, y0 = s_e2.sc[1] < h ? s_e2.sc[1] : h;
                u32 x1 = s_e2.sc[0] + s_e2.sc[2], y1 = s_e2.sc[1] + s_e2.sc[3];
                if (x1 > w) x1 = w;
                if (y1 > h) y1 = h;
                dp->u.dyn.sc_x = (int32_t)x0; dp->u.dyn.sc_y = (int32_t)y0;
                dp->u.dyn.sc_w = x1 > x0 ? x1 - x0 : 0; dp->u.dyn.sc_h = y1 > y0 ? y1 - y0 : 0;
            } else {
                dp->u.dyn.sc_x = 0; dp->u.dyn.sc_y = 0; dp->u.dyn.sc_w = w; dp->u.dyn.sc_h = h;
            }
            dp->u8_b = (u8)(s_e2.stencil_ref & 0xFFu);
            dp->u.dyn.db[0] = s_e2.depth_bounds[0];
            dp->u.dyn.db[1] = s_e2.depth_bounds[1];
            vk_rec_commit_pkt(0);
            s_e2.vp_dirty = s_e2.sc_dirty = s_e2.sref_dirty = s_e2.db_dirty = 0;
            s_e2.dyn_dirty_mask &= (u8)~15u;
            s_rec.p_dyn_w = w; s_rec.p_dyn_h = h; s_rec.p_dyn_valid = 1;
        }
        if (__builtin_expect(!s_rec.p_dyn2_valid || (s_e2.dyn_dirty_mask & 48u), 0)) {
            vk_rec_pkt* d2p = vk_rec_alloc_pkt();
            d2p->op = VK_REC_OP_SET_DYN2;
            d2p->u.dyn2.bias[0] = s_e2.depth_bias[0];
            d2p->u.dyn2.bias[1] = s_e2.depth_bias[1];
            memcpy(d2p->u.dyn2.bc, s_e2.blend_color, sizeof d2p->u.dyn2.bc);
            vk_rec_commit_pkt(0);
            s_e2.bias_dirty = s_e2.bc_dirty = 0;
            s_e2.dyn_dirty_mask &= (u8)~48u;
            s_rec.p_dyn2_valid = 1;
        }
        if (__builtin_expect((s_e2.tex_rt_mask != 0u || s_rec.p_tex_rt_n != 0u) &&
                             (s_rec.p_tex_rt_gen != s_e2.tex_bind_gen || s_rec.p_tex_rt_mask != s_e2.tex_rt_mask), 0)) {
            vk_rec_pkt* rp_tex = vk_rec_alloc_pkt();
            rp_tex->op = VK_REC_OP_SET_TEX_RT;
            u8 cnt = 0, ov = 0;
            u32 m = s_e2.tex_rt_mask;
            while (m) {
                const u32 u = (u32)__builtin_ctz(m);
                m &= m - 1u;
                VkImage ti = s_e2.tex_img[u];
                if (!ti) continue;
                int dup = 0;
                for (u8 k = 0; k < cnt; k++) if (rp_tex->u.tex_rt.imgs[k] == ti) { dup = 1; break; }
                if (dup) continue;
                if (cnt < 6u) rp_tex->u.tex_rt.imgs[cnt++] = ti;
                else { ov = 1; break; }
            }
            rp_tex->u8_a = cnt;
            rp_tex->u8_b = ov;
            vk_rec_commit_pkt(0);
            s_rec.p_tex_rt_n = cnt | ov;
            s_rec.p_tex_rt_gen = s_e2.tex_bind_gen;
            s_rec.p_tex_rt_mask = s_e2.tex_rt_mask;
        }
        VkDeviceSize bind_voff = voff;
        int32_t vtx_offset = 0;
        if (gcm_vbase_ok) {
            bind_voff = 0;
            vtx_offset = gcm_vbase;
        } else {
            static const u64 kRec16[17] = {
                0ull, 0ull,
                0x80000000ull, 0x55555556ull, 0x40000000ull, 0x33333334ull,
                0x2AAAAAABull, 0x24924925ull, 0x20000000ull, 0x1C71C71Dull,
                0x1999999Aull, 0x1745D175ull, 0x15555556ull, 0x13B13B14ull,
                0x12492493ull, 0x11111112ull, 0x10000000ull
            };
            const u32 d = stride >> 4;
            if ((stride & 15u) == 0u && d >= 1u && d <= 16u && (voff & 15u) == 0u && voff < (1ull << 32)) {
                const u32 v16 = (u32)(voff >> 4);
                const u32 q = (d == 1u) ? v16 : (u32)(((u64)v16 * kRec16[d]) >> 32);
                if ((VkDeviceSize)q * stride == voff) { bind_voff = 0; vtx_offset = (int32_t)q; }
            } else if ((voff % stride) == 0 && (voff / stride) <= 0x7FFFFFFFull) {
                bind_voff = 0; vtx_offset = (int32_t)(voff / stride);
            }
        }
        VkDeviceSize bind_ioff = 0;
        u32 first_idx = (u32)vtx_offset;
        if (ibytes) {
            bind_ioff = ioff;
            first_idx = 0;
            if (gcm_ifirst_ok) { bind_ioff = 0; first_idx = gcm_ifirst; }
            else if ((ioff & 3u) == 0 && (ioff >> 2) <= 0xFFFFFFFFull) { bind_ioff = 0; first_idx = (u32)(ioff >> 2); }
        }
        u8 fl = (u8)((ibytes ? REC_DRAW_F_INDEXED : 0u) |
                     (ssbo_on ? REC_DRAW_F_SSBO : 0u) |
                     (P->depth_or_stencil_write ? REC_DRAW_F_DWRITE : 0u) |
                     (hazard ? REC_DRAW_F_HAZARD : 0u) |
                     (vb_from_gc ? REC_DRAW_F_VGC : 0u) |
                     (ib_from_gc ? REC_DRAW_F_IGC : 0u) |
                     (P->cube_mask ? REC_DRAW_F_SPLIT : 0u));
        if (__builtin_expect(bind_voff != 0 || bind_ioff != 0, 0)) {
            vk_rec_pkt* op = vk_rec_alloc_pkt();
            op->op = VK_REC_OP_SET_OFFS;
            op->u.offs.voff = bind_voff;
            op->u.offs.ioff = bind_ioff;
            vk_rec_commit_pkt(0);
            fl |= REC_DRAW_F_OFFS;
        }
        const u64 vp_e2 = vk_prof ? vk_m2e4_rdtsc() : 0;
        vk_rec_pkt* dp = vk_rec_alloc_pkt();
        dp->op = VK_REC_OP_DRAW;
        dp->flags = fl;
        dp->u8_a = (u8)inst_on;
        dp->u.draw.count = ibytes ? index_count : vertex_count;
        dp->u.draw.first_idx = first_idx;
        dp->u.draw.vtx_off = vtx_offset;
        dp->u.draw.dyn0 = dyn[0];
        dp->u.draw.dyn1 = dyn[1];
        dp->u.draw.pipe = pipe;
        dp->u.draw.set = set;
        vk_rec_commit_pkt(0);
        if (__builtin_expect(vk_prof, 0)) {
            const u64 vp_t5 = vk_m2e4_rdtsc();
            s_vp_gc += vp_t1 - vp_t0;
            s_vp_ecp += vp_e1 - vp_t2;
            s_vp_est += vp_e2 - vp_e1;
            s_vp_epk += vp_t5 - vp_e2;
            s_vp_draw += vp_t5 - vp_t2;
            s_vp_tot += vp_t5 - vp_t0;
            s_vp_ndraw++;
            const u32 cur_fr = rsx_draw_engine_frames();
            if (!s_vp_t0) {
                s_vp_t0 = vp_t5; s_vp_fr0 = cur_fr; s_vp_ms0 = vk_now_ms();
            } else if (cur_fr - s_vp_fr0 >= 300u) {
                const double now_ms = vk_now_ms();
                const double dt_ms = now_ms - s_vp_ms0;
                const u64 dt_tsc = vp_t5 - s_vp_t0;
                const double df = (double)(cur_fr - s_vp_fr0);
                const double c = (dt_tsc && dt_ms > 0.0) ? (dt_ms / (double)dt_tsc) / df : 0.0;
                const u64 c_tsc = __atomic_exchange_n(&s_rec.c_busy_tsc, 0ull, __ATOMIC_RELAXED);
                const u64 ic = __atomic_exchange_n(&s_rec.c.prof_ind_calls, 0ull, __ATOMIC_RELAXED);
                const u64 id = __atomic_exchange_n(&s_rec.c.prof_ind_draws, 0ull, __ATOMIC_RELAXED);
                const u64 dd = __atomic_exchange_n(&s_rec.c.prof_dir_draws, 0ull, __ATOMIC_RELAXED);
                const u64 np = __atomic_exchange_n(&s_rec.c.prof_push, 0ull, __ATOMIC_RELAXED);
                fprintf(stderr, "[M2E4-VK] 300f fr=%u (m2e4=%d m2e4b=%d ssbo=%d ind=%d rec=1): tot=%.2f ms/f | gc=%.2f desc=%.2f ubo=%.2f enq=%.2f (cp=%.2f st=%.2f pk=%.2f wk=%.2f) | rec_thr=%.2f | ind=%.0f(%.0f) dir=%.0f push=%.0f wk=%.1f/f stl=%.1f/f draws/f=%.0f\n",
                        cur_fr, m2e4, m2e4b, ssbo_on, inst_on,
                        (double)s_vp_tot * c, (double)s_vp_gc * c, (double)s_vp_desc * c,
                        (double)s_vp_ubo * c, (double)s_vp_draw * c,
                        (double)s_vp_ecp * c, (double)s_vp_est * c, (double)s_vp_epk * c,
                        (double)s_rec.st_wake_tsc * c, (double)c_tsc * c,
                        (double)ic / df, (double)id / df, (double)dd / df, (double)np / df,
                        (double)s_rec.st_wakes / df, (double)s_rec.st_stalls / df,
                        (double)s_vp_ndraw / df);
                s_vp_gc = s_vp_desc = s_vp_ubo = s_vp_haz = s_vp_bind = s_vp_draw = s_vp_tot = 0;
                s_vp_ecp = s_vp_est = s_vp_epk = 0;
                s_rec.st_wakes = s_rec.st_stalls = s_rec.st_pkts = s_rec.st_wake_tsc = 0;
                s_vp_npipe = s_vp_ndesc = s_vp_npush = s_vp_nvb = s_vp_nib = s_vp_ndraw = 0;
                s_vp_t0 = vp_t5; s_vp_fr0 = cur_fr; s_vp_ms0 = now_ms;
            }
        }
        s_eb.draws++;
        if (++s_eb_pending >= vk_eb_max()) vk_eng_flush_async();
        return;
    }
    int reuse_rp = 0;
    if (batched) {
        if (__builtin_expect(!vb_from_gc, 0)) {
            if (gcm) rsx_draw_engine_resolve_cur_ptrs(&vertices, NULL);
            memcpy((u8*)s_eb.vb.ptr + voff, vertices, (size_t)vbytes);
        }
        if (__builtin_expect(ibytes && !ib_from_gc, 0)) {
            if (gcm) rsx_draw_engine_resolve_cur_ptrs(NULL, &indices);
            memcpy((u8*)s_eb.ib.ptr + ioff, indices, (size_t)ibytes);
        }
        if (__builtin_expect(m2e4b && s_eb.fast_rp_ok && s_eb.cur_rp == rp && s_eb.cur_fb == fb &&
                             (!P->depth_or_stencil_write || depth_img == s_eb.last_touched_depth), 1)) {
            reuse_rp = 1;
        } else {
            if (!s_eb_pending) {
                if (vk_eb_open_cmd()) return;
                vk_cmd_after_everything(s_vk.cmd);
                s_engtime.barriers++;
                vk_eb_touch_clear();
            } else if (hazard && s_eb.in_rp && s_eb.cur_rp == rp && s_eb.cur_fb == fb &&
                       !vk_eb_sample_hazard()) {
                reuse_rp = 1;
            } else {
                vk_eb_end_rp();
                if (!hazard || vk_eb_hazard(rt_img, n, depth_img)) {
                    vk_cmd_after_everything(s_vk.cmd);
                    s_engtime.barriers++;
                    vk_eb_touch_clear();
                }
            }
            if (!reuse_rp) {
                for (u32 i = 0; i < n; i++) vk_eb_touch(rt_img[i]);
            }
            if (P->depth_or_stencil_write) {
                if (!m2e4 || depth_img != s_eb.last_touched_depth) {
                    vk_eb_touch(depth_img);
                    s_eb.last_touched_depth = depth_img;
                }
            }
            if (m2e4b && hazard && !s_e2.tex_rt_mask && (reuse_rp || !s_eb.in_rp)) {
                /* fast_rp_ok will be armed once the RP is active below */
                if (reuse_rp) s_eb.fast_rp_ok = 1;
            }
        }
    } else {
        if (gcm) rsx_draw_engine_resolve_cur_ptrs(&vertices, &indices);
        if (s_eb_pending) vk_eng_flush();
        if (vk_ensure_vertex_buffer(vbytes)) return;
        memcpy(s_vk.vertices.ptr, vertices, (size_t)vbytes);
        if (ibytes) {
            if (vk_eng_ensure_indices(ibytes)) return;
            memcpy(s_e2.indices.ptr, indices, (size_t)ibytes);
        }
        vbuf = s_vk.vertices.buf; ibuf = s_e2.indices.buf; voff = ioff = 0;
        if (!vk_eng_ucpu()) return;
        memcpy(s_g.ubo.ptr, s_e2.vs_src ? s_e2.vs_src : (const void*)s_e2.ucpu, vsb);
        memcpy((u8*)s_g.ubo.ptr + s_g.ps_off, s_e2.ucpu + s_g.ps_off, psb);
        set = vk_eng_write_descriptors(P, ssbo_on);
        if (!set) return;
        dyn[1] = (u32)s_g.ps_off;
        if (vk_begin()) return;
        /* Everything submitted before is complete (each submission is waited
         * on); this orders memory for the surfaces this draw touches. */
        for (u32 i = 0; i < n; i++)
            vk_barrier_image(rt_img[i], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    }
    if (!reuse_rp) {
        VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                      .renderPass = rp, .framebuffer = fb,
                                      .renderArea = { { 0, 0 }, { w, h } } };
        pvkCmdBeginRenderPass(s_vk.cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        s_engtime.rps++;
        if (batched && hazard) {
            s_eb.in_rp = 1;
            s_eb.cur_rp = rp;
            s_eb.cur_fb = fb;
            s_eb.cur_pipe = VK_NULL_HANDLE;
            if (m2e4b && !s_e2.tex_rt_mask) s_eb.fast_rp_ok = 1;
        }
    }
    const u64 vp_t3 = vk_prof ? vk_m2e4_rdtsc() : 0;
    if (!reuse_rp || s_eb.cur_pipe != pipe) {
        pvkCmdBindPipeline(s_vk.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        if (batched && hazard) s_eb.cur_pipe = pipe;
        if (__builtin_expect(vk_prof, 0)) s_vp_npipe++;
    }
    const VkPipelineLayout playout = ssbo_on
        ? (P->cube_mask ? s_e2.ssbo_split_pipe_layout : s_g.ssbo_pipe_layout)
        : (P->cube_mask ? s_e2.split_pipe_layout : s_g.pipe_layout);
    const int skip_st = batched && hazard && async_on && reuse_rp && s_eb.rp_state_valid;
    if (__builtin_expect(!skip_st || s_e2.dyn_dirty_mask, 0)) {
        if (!skip_st || s_e2.vp_dirty) {
            VkViewport vp = { 0.0f, 0.0f, (float)w, (float)h, 0.0f, 1.0f };
            if (s_e2.have_vp && s_e2.vp[2] > 0.0f && s_e2.vp[3] > 0.0f) {
                vp.x = s_e2.vp[0]; vp.y = s_e2.vp[1]; vp.width = s_e2.vp[2]; vp.height = s_e2.vp[3];
            }
            if (!skip_st || memcmp(&s_eb.cur_vp, &vp, sizeof vp) != 0) {
                pvkCmdSetViewport(s_vk.cmd, 0, 1, &vp);
                s_eb.cur_vp = vp;
            }
            s_e2.vp_dirty = 0;
        }
        if (!skip_st || s_e2.sc_dirty) {
            VkRect2D sc = { { 0, 0 }, { w, h } };
            if (s_e2.have_sc) {                   /* clamped: Vulkan rejects negative or over-size */
                u32 x0 = s_e2.sc[0] < w ? s_e2.sc[0] : w, y0 = s_e2.sc[1] < h ? s_e2.sc[1] : h;
                u32 x1 = s_e2.sc[0] + s_e2.sc[2], y1 = s_e2.sc[1] + s_e2.sc[3];
                if (x1 > w) x1 = w;
                if (y1 > h) y1 = h;
                sc.offset.x = (int32_t)x0; sc.offset.y = (int32_t)y0;
                sc.extent.width = x1 > x0 ? x1 - x0 : 0; sc.extent.height = y1 > y0 ? y1 - y0 : 0;
            }
            if (!skip_st || memcmp(&s_eb.cur_sc, &sc, sizeof sc) != 0) {
                pvkCmdSetScissor(s_vk.cmd, 0, 1, &sc);
                s_eb.cur_sc = sc;
            }
            s_e2.sc_dirty = 0;
        }
        if (!skip_st || s_e2.sref_dirty) {
            const u32 sref = s_e2.stencil_ref & 0xFFu;
            if (!skip_st || s_eb.cur_sref != sref) {
                pvkCmdSetStencilReference(s_vk.cmd, VK_STENCIL_FACE_FRONT_AND_BACK, sref);
                s_eb.cur_sref = sref;
            }
            s_e2.sref_dirty = 0;
        }
        if (s_vk.depth_bounds_ok && (!skip_st || s_e2.db_dirty)) {
            if (!skip_st || s_eb.cur_db[0] != s_e2.depth_bounds[0] || s_eb.cur_db[1] != s_e2.depth_bounds[1]) {
                pvkCmdSetDepthBounds(s_vk.cmd, s_e2.depth_bounds[0], s_e2.depth_bounds[1]);
                s_eb.cur_db[0] = s_e2.depth_bounds[0]; s_eb.cur_db[1] = s_e2.depth_bounds[1];
            }
            s_e2.db_dirty = 0;
        }
        if (!skip_st || s_e2.bias_dirty) {
            if (!skip_st || s_eb.cur_bias[0] != s_e2.depth_bias[0] || s_eb.cur_bias[1] != s_e2.depth_bias[1]) {
                pvkCmdSetDepthBias(s_vk.cmd, s_e2.depth_bias[0], 0.0f, s_e2.depth_bias[1]);
                s_eb.cur_bias[0] = s_e2.depth_bias[0]; s_eb.cur_bias[1] = s_e2.depth_bias[1];
            }
            s_e2.bias_dirty = 0;
        }
        if (!skip_st || s_e2.bc_dirty) {
            if (!skip_st || memcmp(s_eb.cur_bc, s_e2.blend_color, sizeof s_e2.blend_color) != 0) {
                pvkCmdSetBlendConstants(s_vk.cmd, s_e2.blend_color);
                memcpy(s_eb.cur_bc, s_e2.blend_color, sizeof s_e2.blend_color);
            }
            s_e2.bc_dirty = 0;
        }
        s_e2.dyn_dirty_mask = 0;
    }
    if (ssbo_on) {
        if (!batched || !s_eb.cur_ssbo_bound) {
            VkDescriptorSet cset = batched ? s_eb.ssbo_const_set : s_g.ssbo_const_set;
            pvkCmdBindDescriptorSets(s_vk.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     playout, 0, 1, &cset, 0, NULL);
            if (batched) s_eb.cur_ssbo_bound = 1;
            if (__builtin_expect(vk_prof, 0)) s_vp_ndesc++;
            if (__builtin_expect(s_eng_stats, 0)) s_engtime.desc_binds++;
        }
        if (!batched || s_eb.cur_playout != playout || s_eb.cur_set != set) {
            pvkCmdBindDescriptorSets(s_vk.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     playout, 1, 1, &set, 0, NULL);
            if (batched) { s_eb.cur_playout = playout; s_eb.cur_set = set; }
            if (__builtin_expect(vk_prof, 0)) s_vp_ndesc++;
            if (__builtin_expect(s_eng_stats, 0)) s_engtime.desc_binds++;
        }
        const u32 vp_f4 = dyn[0] >> 4;
        const u32 ps_f4 = dyn[1] >> 4;
        if (inst_on) {
            if (batched) s_eb.cur_pc_vp = 0xFFFFFFFFu;
            if (!batched || s_eb.cur_pc_ps != ps_f4) {
                pvkCmdPushConstants(s_vk.cmd, playout,
                                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                    4u, 4u, &ps_f4);
                if (batched) s_eb.cur_pc_ps = ps_f4;
                if (__builtin_expect(vk_prof, 0)) s_vp_npush++;
                if (__builtin_expect(s_eng_stats, 0)) s_engtime.push_consts++;
            }
        } else {
            const int chg_vp = (!batched || s_eb.cur_pc_vp != vp_f4);
            const int chg_ps = (!batched || s_eb.cur_pc_ps != ps_f4);
            if (chg_vp && chg_ps) {
                const u32 pc[2] = { vp_f4, ps_f4 };
                pvkCmdPushConstants(s_vk.cmd, playout,
                                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                    0u, 8u, pc);
                if (batched) { s_eb.cur_pc_vp = vp_f4; s_eb.cur_pc_ps = ps_f4; }
                if (__builtin_expect(vk_prof, 0)) s_vp_npush++;
                if (__builtin_expect(s_eng_stats, 0)) s_engtime.push_consts++;
            } else if (chg_vp) {
                pvkCmdPushConstants(s_vk.cmd, playout,
                                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                    0u, 4u, &vp_f4);
                if (batched) s_eb.cur_pc_vp = vp_f4;
                if (__builtin_expect(vk_prof, 0)) s_vp_npush++;
                if (__builtin_expect(s_eng_stats, 0)) s_engtime.push_consts++;
            } else if (chg_ps) {
                pvkCmdPushConstants(s_vk.cmd, playout,
                                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                    4u, 4u, &ps_f4);
                if (batched) s_eb.cur_pc_ps = ps_f4;
                if (__builtin_expect(vk_prof, 0)) s_vp_npush++;
                if (__builtin_expect(s_eng_stats, 0)) s_engtime.push_consts++;
            }
        }
    } else if (!skip_st || s_eb.cur_playout != playout || s_eb.cur_set != set ||
        s_eb.cur_dyn[0] != dyn[0] || s_eb.cur_dyn[1] != dyn[1]) {
        pvkCmdBindDescriptorSets(s_vk.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 playout, 0, 1, &set, 2, dyn);
        s_eb.cur_playout = playout; s_eb.cur_set = set;
        s_eb.cur_dyn[0] = dyn[0]; s_eb.cur_dyn[1] = dyn[1];
        if (__builtin_expect(vk_prof, 0)) s_vp_ndesc++;
        if (__builtin_expect(s_eng_stats, 0)) s_engtime.desc_binds++;
    }
    VkDeviceSize bind_voff = voff;
    int32_t vtx_offset = 0;
    if (batched && async_on) {
        if (gcm_vbase_ok) {
            bind_voff = 0;
            vtx_offset = gcm_vbase;
        } else {
            /* Fast exact division for stride = 16 * d (d in 1..16, all RSX layouts). */
            static const u64 kRec16[17] = {
                0ull, 0ull,
                0x80000000ull, 0x55555556ull, 0x40000000ull, 0x33333334ull,
                0x2AAAAAABull, 0x24924925ull, 0x20000000ull, 0x1C71C71Dull,
                0x1999999Aull, 0x1745D175ull, 0x15555556ull, 0x13B13B14ull,
                0x12492493ull, 0x11111112ull, 0x10000000ull
            };
            const u32 d = stride >> 4;
            if ((stride & 15u) == 0u && d >= 1u && d <= 16u && (voff & 15u) == 0u && voff < (1ull << 32)) {
                const u32 v16 = (u32)(voff >> 4);
                const u32 q = (d == 1u) ? v16 : (u32)(((u64)v16 * kRec16[d]) >> 32);
                if ((VkDeviceSize)q * stride == voff) {
                    bind_voff = 0;
                    vtx_offset = (int32_t)q;
                }
            } else if ((voff % stride) == 0 && (voff / stride) <= 0x7FFFFFFFull) {
                bind_voff = 0;
                vtx_offset = (int32_t)(voff / stride);
            }
        }
    }
    if (!skip_st || s_eb.cur_vbuf != vbuf || s_eb.cur_voff != bind_voff) {
        pvkCmdBindVertexBuffers(s_vk.cmd, 0, 1, &vbuf, &bind_voff);
        s_eb.cur_vbuf = vbuf; s_eb.cur_voff = bind_voff;
        if (__builtin_expect(vk_prof, 0)) s_vp_nvb++;
    }
    const u64 vp_t4 = vk_prof ? vk_m2e4_rdtsc() : 0;
    if (ibytes) {
        VkDeviceSize bind_ioff = ioff;
        u32 first_idx = 0;
        if (batched && async_on) {
            if (gcm_ifirst_ok) {
                bind_ioff = 0;
                first_idx = gcm_ifirst;
            } else if ((ioff & 3u) == 0 && (ioff >> 2) <= 0xFFFFFFFFull) {
                bind_ioff = 0;
                first_idx = (u32)(ioff >> 2);
            }
        }
        if (!skip_st || s_eb.cur_ibuf != ibuf || s_eb.cur_ioff != bind_ioff) {
            pvkCmdBindIndexBuffer(s_vk.cmd, ibuf, bind_ioff, VK_INDEX_TYPE_UINT32);
            s_eb.cur_ibuf = ibuf; s_eb.cur_ioff = bind_ioff;
            if (__builtin_expect(vk_prof, 0)) s_vp_nib++;
        }
        pvkCmdDrawIndexed(s_vk.cmd, index_count, 1, first_idx, vtx_offset, inst_on ? (dyn[0] >> 4) : 0u);
    } else {
        pvkCmdDraw(s_vk.cmd, vertex_count, 1, (u32)vtx_offset, inst_on ? (dyn[0] >> 4) : 0u);
    }
    if (__builtin_expect(vk_prof, 0)) {
        const u64 vp_t5 = vk_m2e4_rdtsc();
        s_vp_gc += vp_t1 - vp_t0;
        s_vp_haz += vp_t3 - vp_t2;
        s_vp_bind += vp_t4 - vp_t3;
        s_vp_draw += vp_t5 - vp_t4;
        s_vp_tot += vp_t5 - vp_t0;
        s_vp_ndraw++;
        const u32 cur_fr = rsx_draw_engine_frames();
        if (!s_vp_t0) {
            s_vp_t0 = vp_t5; s_vp_fr0 = cur_fr; s_vp_ms0 = vk_now_ms();
        } else if (cur_fr - s_vp_fr0 >= 300u) {
            const double now_ms = vk_now_ms();
            const double dt_ms = now_ms - s_vp_ms0;
            const u64 dt_tsc = vp_t5 - s_vp_t0;
            const double df = (double)(cur_fr - s_vp_fr0);
            const double c = (dt_tsc && dt_ms > 0.0) ? (dt_ms / (double)dt_tsc) / df : 0.0;
            fprintf(stderr, "[M2E4-VK] 300f fr=%u (m2e4=%d m2e4b=%d ssbo=%d rec=0): tot=%.2f ms/f | gc=%.2f desc=%.2f ubo=%.2f haz=%.2f cmdbind=%.2f cmddraw=%.2f | calls/f: pipe=%.0f desc=%.0f push=%.0f vb=%.0f ib=%.0f draw=%.0f\n",
                    cur_fr, m2e4, m2e4b, ssbo_on,
                    (double)s_vp_tot * c, (double)s_vp_gc * c, (double)s_vp_desc * c,
                    (double)s_vp_ubo * c, (double)s_vp_haz * c, (double)s_vp_bind * c,
                    (double)s_vp_draw * c,
                    (double)s_vp_npipe / df, (double)s_vp_ndesc / df,
                    (double)s_vp_npush / df,
                    (double)s_vp_nvb / df, (double)s_vp_nib / df, (double)s_vp_ndraw / df);
            s_vp_gc = s_vp_desc = s_vp_ubo = s_vp_haz = s_vp_bind = s_vp_draw = s_vp_tot = 0;
            s_vp_npipe = s_vp_ndesc = s_vp_npush = s_vp_nvb = s_vp_nib = s_vp_ndraw = 0;
            s_vp_t0 = vp_t5; s_vp_fr0 = cur_fr; s_vp_ms0 = now_ms;
        }
    }
    if (batched && hazard) s_eb.rp_state_valid = 1;
    if (!batched || !hazard) pvkCmdEndRenderPass(s_vk.cmd);
    if (!batched) { vk_submit_and_wait(); return; }
    s_eb.draws++;
    if (++s_eb_pending >= vk_eb_max()) vk_eng_flush_async();
}

static void vk_eng_upscale_capture_depth(void* user, u32 depth)
{
    (void)user;
    if (!rsx_vk_fsr_active()) return;
    vk_eng_obj* z = vk_eng_get(depth, VK_ENG_DEPTH);
    if (!z) return;
    const int batched = vk_eb_active();
    if (batched && vk_rec_use_batch()) {
        if (!s_eb_pending && vk_rec_ensure_open()) return;
        vk_rec_emit_upscale_depth(z->im.img, z->w, z->h);
        vk_eb_recorded();
        return;
    }
    if (batched ? (!s_eb_pending && vk_eb_open_cmd()) : vk_begin()) return;
    if (batched) vk_eb_end_rp();
    rsx_vk_fsr_capture_depth(s_vk.cmd, z->im.img, z->w, z->h);
    if (batched) vk_eb_recorded();
    else vk_submit_and_wait();
}

static void vk_eng_upscale_capture_color(void* user, u32 surface)
{
    (void)user;
    if (!rsx_vk_fsr_active()) return;
    vk_eng_obj* c = vk_eng_get(surface, VK_ENG_COLOR);
    if (!c) return;
    const int batched = vk_eb_active();
    if (batched && vk_rec_use_batch()) {
        if (!s_eb_pending && vk_rec_ensure_open()) return;
        vk_rec_emit_upscale_color(c->im.img, c->fmt, c->w, c->h);
        vk_eb_recorded();
        return;
    }
    if (batched ? (!s_eb_pending && vk_eb_open_cmd()) : vk_begin()) return;
    if (batched) vk_eb_end_rp();
    rsx_vk_fsr_capture_color(s_vk.cmd, c->im.img, c->fmt, c->w, c->h);
    if (batched) vk_eb_recorded();
    else vk_submit_and_wait();
}

static const rsx_draw_backend s_vk_engine_backend = {
    .user                 = NULL,
    .init                 = vk_eng_init,
    .shutdown             = vk_eng_shutdown,
    .submit_and_wait      = vk_eng_submit_and_wait,
    .texture_create       = vk_eng_texture_create,
    .texture_upload       = vk_eng_texture_upload,
    .texture_release      = vk_eng_texture_release,
    .color_target_create  = vk_eng_color_target_create,
    .color_target_release = vk_eng_color_target_release,
    .surface_view         = vk_eng_surface_view,
    .color_snapshot       = vk_eng_color_snapshot,
    .depth_target_create  = vk_eng_depth_target_create,
    .depth_target_release = vk_eng_depth_target_release,
    .depth_snapshot       = vk_eng_depth_snapshot,
    .depth_color_snapshot = vk_eng_depth_color_snapshot,
    .pipeline_create      = vk_eng_pipeline_create,
    .pipeline_release     = vk_eng_pipeline_release,
    .bind_targets         = vk_eng_bind_targets,
    .bind_pipeline        = vk_eng_bind_pipeline,
    .bind_vs_constants    = vk_eng_bind_vs_constants,
    .bind_ps_constants    = vk_eng_bind_ps_constants,
    .bind_textures        = vk_eng_bind_textures,
    .bind_vertex_textures = vk_eng_bind_vertex_textures,
    .set_viewport         = vk_eng_set_viewport,
    .set_scissor          = vk_eng_set_scissor,
    .set_stencil_ref      = vk_eng_set_stencil_ref,
    .draw                 = vk_eng_draw,
    .clear_color          = vk_eng_clear_color,
    .clear_depth_stencil  = vk_eng_clear_depth_stencil,
    .present              = vk_eng_present,
    .readback             = vk_eng_readback,
    .set_depth_bounds     = vk_eng_set_depth_bounds,
    .set_depth_bias       = vk_eng_set_depth_bias,
    .set_blend_color      = vk_eng_set_blend_color,
    .upscale_capture_depth = vk_eng_upscale_capture_depth,
    .upscale_capture_color = vk_eng_upscale_capture_color,
};

static rsx_backend s_vulkan_backend = {
    .userdata           = &s_vk,
    .set_render_target  = vk_cb_track_state,
    .set_vertex_attribs = vk_cb_track_state,
    .set_depth_stencil  = vk_cb_track_state,
    .clear              = vk_cb_clear,
    .draw_arrays        = vk_cb_draw_arrays,
    .draw_indexed       = vk_cb_draw_indexed,
    .bind_texture       = vk_cb_bind_texture,
    .present            = vk_cb_present,
};

/* ---------------------------------------------------------------------------
 * Public entry points
 * -------------------------------------------------------------------------*/
static int vk_init_all(u32 width, u32 height, const char* title)
{
    pthread_mutex_init(&s_vk.queue_mu, NULL);
    pthread_mutex_init(&s_vk.present_mu, NULL);
    s_vk.guest_w = width  ? width  : 1280;
    s_vk.guest_h = height ? height : 720;
    {
        const char* e = getenv("RES_SCALE");
        float sc = (e && *e) ? (float)atof(e) : 1.0f;
        if (!(sc >= 1.0f)) sc = 1.0f;
        if (sc > 3.0f) sc = 3.0f;
        u32 out_w = (sc <= 1.0f) ? s_vk.guest_w : (((u32)((float)s_vk.guest_w * sc + 0.5f) + 1u) & ~1u);
        u32 out_h = (sc <= 1.0f) ? s_vk.guest_h : (((u32)((float)s_vk.guest_h * sc + 0.5f) + 1u) & ~1u);
        u32 ren_w = out_w, ren_h = out_h;
        const rsx_upscale_cfg* ucfg = rsx_upscale_get_cfg();
        if (ucfg->mode) {
            const float ratio = rsx_upscale_render_ratio();
            if (ratio < 0.999f) {
                ren_w = ((u32)((float)out_w * ratio + 0.5f) + 1u) & ~1u;
                ren_h = ((u32)((float)out_h * ratio + 0.5f) + 1u) & ~1u;
                const u32 min_w = ucfg->min_720p ? s_vk.guest_w : 320u;
                const u32 min_h = ucfg->min_720p ? s_vk.guest_h : 180u;
                if (ren_w < min_w) ren_w = min_w;
                if (ren_h < min_h) ren_h = min_h;
                if (ren_w > out_w) ren_w = out_w;
                if (ren_h > out_h) ren_h = out_h;
            }
        }
        s_vk.width = out_w;
        s_vk.height = out_h;
        s_vk.render_w = ren_w;
        s_vk.render_h = ren_h;
        if (sc > 1.0f || ren_w != out_w || ren_h != out_h)
            VK_LOG("resolution: guest %ux%u, render %ux%u, output %ux%u (RES_SCALE %.2f)\n",
                   s_vk.guest_w, s_vk.guest_h, s_vk.render_w, s_vk.render_h, s_vk.width, s_vk.height, sc);
    }
    s_eng_stats = (getenv("PS3RECOMP_ENG_STATS") || rsx_slowframe_init()) ? 1 : 0;
    s_vk.dump_path = getenv("PS3RECOMP_VK_DUMP");
    /* Writing a full frame to disk on every present costs more than the
     * present itself and slows a title enough to change what it does; a
     * live title wants one now and then, not all of them. */
    { const char* e = getenv("PS3RECOMP_VK_DUMP_EVERY");
      s_vk.dump_every = (e && atoi(e) > 0) ? (u32)atoi(e) : 1; }
    /* PS3RECOMP_VK_DUMP_FRAME=<n>[,<n>...]: dump exactly these presented frames.
     * A scene capture wants named frames (the menu, a gameplay frame) at the
     * same numbers on every run, not every Nth frame of the whole boot: the
     * thinning knob cannot name one, and a dump costs a readback plus a write. */
    { const char* e = getenv("PS3RECOMP_VK_DUMP_FRAME");
      while (e && *e && s_vk.dump_n_frames < 8u) {
          char* end;
          const long v = strtol(e, &end, 10);
          if (end == e) { e++; continue; }
          if (v > 0) s_vk.dump_frames[s_vk.dump_n_frames++] = (u32)v;
          e = end;
      } }

    vk_window_open(title);           /* no-op unless PS3RECOMP_VK_WINDOW is set */
    if (vk_load_library()) return -1;

    u32 inst_api = VK_API_VERSION_1_0;
    {
        PFN_vkEnumerateInstanceVersion fn_ver =
            (PFN_vkEnumerateInstanceVersion)pvkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
        u32 loader_ver = VK_API_VERSION_1_0;
        if (fn_ver && fn_ver(&loader_ver) == VK_SUCCESS && loader_ver >= VK_API_VERSION_1_2)
            inst_api = VK_API_VERSION_1_2;
    }
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                              .pApplicationName = "ps3recomp", .pEngineName = "ps3recomp RSX",
                              .apiVersion = inst_api };
    VkInstanceCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
    if (s_vk.windowed) {
        ci.enabledExtensionCount   = s_vk.inst_ext_count;
        ci.ppEnabledExtensionNames = s_vk.inst_exts;
    }
    VkResult ir = pvkCreateInstance(&ci, NULL, &s_vk.instance);
    if (ir != VK_SUCCESS && s_vk.windowed) {
        vk_window_drop("vkCreateInstance with the surface extensions failed");
        ci.enabledExtensionCount = 0;
        ci.ppEnabledExtensionNames = NULL;
        ir = pvkCreateInstance(&ci, NULL, &s_vk.instance);
    }
    if (ir != VK_SUCCESS) { VK_LOG("vkCreateInstance failed (VkResult %d)\n", (int)ir); return -1; }
    if (vk_load_instance_functions()) return -1;
    if (s_vk.windowed) {
        if (vk_load_wsi_instance_functions()) {
            vk_window_drop("the driver has no surface functions");
        } else if (vk_create_surface()) {
            char why[256];
            snprintf(why, sizeof why, "SDL_Vulkan_CreateSurface failed: %s", vk_surface_error());
            vk_window_drop(why);
        }
    }
    if (vk_pick_device()) {
        if (!s_vk.windowed) return -1;
        vk_window_drop("no device can present to the window");
        if (vk_pick_device()) return -1;
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                    .queueFamilyIndex = s_vk.queue_family,
                                    .queueCount = 1, .pQueuePriorities = &prio };
    const char* dev_exts[4];
    u32 dev_ext_count = 0;
    if (s_vk.windowed) dev_exts[dev_ext_count++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    /* DXT (BC1-3) is how PS3 titles ship most textures; Vulkan gates it
     * behind a feature, enabled only when the device reports it. */
    VkPhysicalDeviceFeatures have, want;
    memset(&want, 0, sizeof want);
    pvkGetPhysicalDeviceFeatures(s_vk.phys, &have);
    want.textureCompressionBC = have.textureCompressionBC;
    want.samplerAnisotropy = have.samplerAnisotropy;
    s_vk.aniso_ok = have.samplerAnisotropy ? 1 : 0;
    /* User clip planes (rsx_vp_apply_user_clip) write ClipDistance. */
    want.shaderClipDistance = have.shaderClipDistance;
    s_vk.bc_ok = have.textureCompressionBC ? 1 : 0;
    /* The depth bounds test: a deferred renderer bounds its light pass to
     * the geometry and its sky to the far plane with it. Optional in
     * Vulkan; without it the test stays off, as before. */
    want.depthBounds = have.depthBounds;
    s_vk.depth_bounds_ok = have.depthBounds ? 1 : 0;
    /* SET_ZMIN_MAX_CONTROL's depth clamp (rs->depth_clamp). Optional too;
     * without it such draws are clipped, as before. */
    want.depthClamp = have.depthClamp;
    s_vk.depth_clamp_ok = have.depthClamp ? 1 : 0;
    /* Per-target colour mask and blend enable (an MRT set's targets B..D
     * have their own); without it every target takes target A's. */
    want.independentBlend = have.independentBlend;
    s_vk.indep_blend_ok = have.independentBlend ? 1 : 0;
    want.multiDrawIndirect = have.multiDrawIndirect;
    want.drawIndirectFirstInstance = have.drawIndirectFirstInstance;
    s_vk.multi_draw_indirect_ok = (have.multiDrawIndirect && have.drawIndirectFirstInstance) ? 1 : 0;
    /* Required by FSR 3.1.5 / 3.1.6 compute shaders (formatless storage ops). */
    want.shaderStorageImageWriteWithoutFormat = have.shaderStorageImageWriteWithoutFormat;
    want.shaderStorageImageReadWithoutFormat  = have.shaderStorageImageReadWithoutFormat;
    void* dci_pnext = NULL;
#ifdef PS3RECOMP_VK_FFX
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR csd_want = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR,
    };
    if (vk_has_device_extension(s_vk.phys, VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME)) {
        PFN_vkGetPhysicalDeviceFeatures2 fn_feat2 =
            (PFN_vkGetPhysicalDeviceFeatures2)pvkGetInstanceProcAddr(s_vk.instance, "vkGetPhysicalDeviceFeatures2");
        if (fn_feat2) {
            VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR csd_have = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR,
            };
            VkPhysicalDeviceFeatures2 f2 = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                .pNext = &csd_have,
            };
            fn_feat2(s_vk.phys, &f2);
            if (csd_have.computeDerivativeGroupLinear) {
                csd_want.computeDerivativeGroupLinear = VK_TRUE;
                csd_want.computeDerivativeGroupQuads  = csd_have.computeDerivativeGroupQuads;
                dci_pnext = &csd_want;
                dev_exts[dev_ext_count++] = VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME;
            }
        }
    }
#endif
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                               .pNext = dci_pnext,
                               .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
                               .enabledExtensionCount = dev_ext_count,
                               .ppEnabledExtensionNames = dev_ext_count ? dev_exts : NULL,
                               .pEnabledFeatures = &want };
    VK_CHECK(pvkCreateDevice(s_vk.phys, &dci, NULL, &s_vk.device), "vkCreateDevice");
    if (vk_load_device_functions()) return -1;
    if (s_vk.windowed && vk_load_wsi_device_functions())
        vk_window_drop("the driver has no swapchain functions");
    pvkGetDeviceQueue(s_vk.device, s_vk.queue_family, 0, &s_vk.queue);

    VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                    .queueFamilyIndex = s_vk.queue_family };
    VK_CHECK(pvkCreateCommandPool(s_vk.device, &pci, NULL, &s_vk.pool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                        .commandPool = s_vk.pool,
                                        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                        .commandBufferCount = 1 };
    VK_CHECK(pvkAllocateCommandBuffers(s_vk.device, &cai, &s_vk.cmd), "vkAllocateCommandBuffers");
    s_vk.sync_cmd = s_vk.cmd;
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VK_CHECK(pvkCreateFence(s_vk.device, &fci, NULL, &s_vk.fence), "vkCreateFence");
    /* The present's own (see vk_state). */
    VK_CHECK(pvkCreateCommandPool(s_vk.device, &pci, NULL, &s_vk.present_pool), "vkCreateCommandPool");
    cai.commandPool = s_vk.present_pool;
    VK_CHECK(pvkAllocateCommandBuffers(s_vk.device, &cai, &s_vk.present_cmd), "vkAllocateCommandBuffers");
    VK_CHECK(pvkCreateFence(s_vk.device, &fci, NULL, &s_vk.present_fence), "vkCreateFence");
    VK_CHECK(pvkCreateFence(s_vk.device, &fci, NULL, &s_vk.window_fence), "vkCreateFence");

    if (vk_pick_depth_format()) return -1;
    {
        rsx_vk_fsr_device fdev = {
            .phys = s_vk.phys,
            .device = s_vk.device,
            .memprops = s_vk.memprops,
            .depth_format = s_vk.depth_format,
            .depth_aspect = s_vk.depth_aspect,
            .render_w = s_vk.render_w,
            .render_h = s_vk.render_h,
            .out_w = s_vk.width,
            .out_h = s_vk.height,
            .get_device_proc_addr = pvkGetDeviceProcAddr,
        };
        if (rsx_vk_fsr_init(&fdev) != 0 ||
            (rsx_upscale_get_cfg()->mode && !rsx_vk_fsr_active())) {
            if (s_vk.render_w != s_vk.width || s_vk.render_h != s_vk.height) {
                s_vk.render_w = s_vk.width;
                s_vk.render_h = s_vk.height;
                VK_LOG("FSR unavailable: rendering at output size %ux%u\n", s_vk.render_w, s_vk.render_h);
            }
        }
    }
    if (vk_create_image(&s_vk.color, VK_FORMAT_R8G8B8A8_UNORM, s_vk.width, s_vk.height,
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT)) return -1;
    if (vk_create_image(&s_vk.depth, s_vk.depth_format, s_vk.width, s_vk.height,
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, s_vk.depth_aspect)) return -1;
    if (vk_create_host_buffer(&s_vk.readback, (VkDeviceSize)s_vk.width * s_vk.height * 4u,
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return -1;
    if (vk_create_render_pass()) return -1;
    if (vk_create_pipeline_objects()) return -1;
    if (vk_guest_init()) {
        VK_LOG("guest programs: setup failed, fixed-function path only\n");
        vk_guest_shutdown();
    }

    /* Both targets UNDEFINED -> GENERAL once. Colour starts opaque black so a
     * present before the first clear reads defined memory; depth starts at
     * the FAR plane, not zero -- zero is the near plane, and a guest that
     * enables LESS before its first depth clear would draw nothing (the
     * null backend makes the same choice). */
    if (vk_begin()) return -1;
    vk_barrier_image(s_vk.color.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
    vk_barrier_image(s_vk.depth.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_UNDEFINED);
    VkClearColorValue black = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } };
    VkClearDepthStencilValue far_plane = { 1.0f, 0 };
    VkImageSubresourceRange cr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VkImageSubresourceRange dr = { s_vk.depth_aspect, 0, 1, 0, 1 };
    pvkCmdClearColorImage(s_vk.cmd, s_vk.color.img, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &cr);
    pvkCmdClearDepthStencilImage(s_vk.cmd, s_vk.depth.img, VK_IMAGE_LAYOUT_GENERAL, &far_plane, 1, &dr);
    if (vk_submit_and_wait()) return -1;

    if (s_vk.windowed) {
        VkSemaphoreCreateInfo semi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        if (pvkCreateSemaphore(s_vk.device, &semi, NULL, &s_vk.sem_acquire) != VK_SUCCESS ||
            pvkCreateSemaphore(s_vk.device, &semi, NULL, &s_vk.sem_acquire_real) != VK_SUCCESS ||
            vk_create_swapchain())
            vk_window_drop("the swapchain could not be created");
    }
    return 0;
}

int rsx_vulkan_backend_init(u32 width, u32 height, const char* title)
{
    memset(&s_vk, 0, sizeof s_vk);
    if (vk_init_all(width, height, title)) {
        rsx_vulkan_backend_shutdown();
        return -1;
    }
    /* One path at a time: the FIFO walker feeds both the rsx_state vtable
     * and the register-file engine, so registering both would record every
     * draw twice (the rule the Metal backend's init follows).
     *
     * With guest programs on, the engine is the default -- as it became
     * Metal's once every host mode passed through it. It builds every
     * pipeline from the guest's programs, so without them it has nothing to
     * draw with, and the vtable's fixed-function path stays the default.
     * Either way PS3RECOMP_RSX_ENGINE=dispatch|vtable overrides. */
    rsx_draw_engine_set_backend(&s_vk_engine_backend);
    rsx_draw_engine_set_default(s_g.on ? 1 : 0);
    rsx_draw_engine_set_render_size(s_vk.render_w, s_vk.render_h);
    if (rsx_draw_engine_enabled() && rsx_draw_engine_init(s_vk.guest_w, s_vk.guest_h) == 0)
        s_vk.eng_active = 1;
    else
        rsx_set_backend(&s_vulkan_backend);
    VK_LOG("%s path\n", s_vk.eng_active ? "register-file draw engine" : "vtable");
    VK_LOG("offscreen target %ux%u ready, depth format %d (%s)\n",
           s_vk.width, s_vk.height, (int)s_vk.depth_format, title ? title : "");
    return 0;
}

void rsx_vulkan_backend_shutdown(void)
{
    rsx_vulkan_backend_present_finish();   /* a host that began one and quit */
    if (rsx_get_backend() == &s_vulkan_backend) rsx_set_backend(NULL);
    if (s_vk.eng_active) { rsx_draw_engine_shutdown(); s_vk.eng_active = 0; }
    rsx_draw_engine_set_backend(NULL);
    if (s_vk.windowed && s_vk.swapchain && s_vk.presented && s_vk.hold_seconds > 0.0) {
        VK_LOG("holding the last frame on screen for %.1f s "
               "(Esc or closing the window ends it early)\n", s_vk.hold_seconds);
#if VK_WINDOW_SDL
        const Uint64 end = SDL_GetTicks() + (Uint64)(s_vk.hold_seconds * 1000.0);
        while (s_vk.windowed && SDL_GetTicks() < end) {
            if (rsx_vulkan_backend_pump_messages() != 0) break;
            vk_present_window();         /* FIFO: paced by the display */
        }
#endif
    }
    if (s_vk.windowed || s_vk.window) vk_window_drop(NULL);
    if (s_vk.device) {
        vk_rec_stop();
        vk_device_idle();
        rsx_vk_fsr_shutdown();
        vk_guest_shutdown();
        for (int i = 0; i < VK_PIPELINE_SLOTS; i++)
            if (s_vk.pipelines[i]) pvkDestroyPipeline(s_vk.device, s_vk.pipelines[i], NULL);
        if (s_vk.pipe_layout) pvkDestroyPipelineLayout(s_vk.device, s_vk.pipe_layout, NULL);
        if (s_vk.desc_pool)   pvkDestroyDescriptorPool(s_vk.device, s_vk.desc_pool, NULL);
        if (s_vk.set_layout)  pvkDestroyDescriptorSetLayout(s_vk.device, s_vk.set_layout, NULL);
        if (s_vk.sampler)     pvkDestroySampler(s_vk.device, s_vk.sampler, NULL);
        for (u32 i = 0; i < s_vk.samp_count; i++)
            pvkDestroySampler(s_vk.device, s_vk.samp_cache[i].s, NULL);
        if (s_vk.vs)          pvkDestroyShaderModule(s_vk.device, s_vk.vs, NULL);
        if (s_vk.fs)          pvkDestroyShaderModule(s_vk.device, s_vk.fs, NULL);
        if (s_vk.framebuffer) pvkDestroyFramebuffer(s_vk.device, s_vk.framebuffer, NULL);
        if (s_vk.render_pass) pvkDestroyRenderPass(s_vk.device, s_vk.render_pass, NULL);
        for (u32 u = 0; u < RSX_MAX_TEXTURES; u++) vk_destroy_image(&s_vk.tex[u]);
        vk_destroy_image(&s_vk.dummy);
        vk_destroy_image(&s_vk.depth);
        vk_destroy_image(&s_vk.color);
        vk_destroy_buffer(&s_vk.vertices);
        vk_destroy_buffer(&s_vk.readback);
        vk_destroy_buffer(&s_vk.fg_readback);
        if (s_vk.fence) pvkDestroyFence(s_vk.device, s_vk.fence, NULL);
        if (s_vk.pool)  pvkDestroyCommandPool(s_vk.device, s_vk.pool, NULL);
        if (s_vk.present_fence) pvkDestroyFence(s_vk.device, s_vk.present_fence, NULL);
        if (s_vk.window_fence)  pvkDestroyFence(s_vk.device, s_vk.window_fence, NULL);
        if (s_vk.present_pool)  pvkDestroyCommandPool(s_vk.device, s_vk.present_pool, NULL);
        pvkDestroyDevice(s_vk.device, NULL);
    }
    if (s_vk.instance && pvkDestroyInstance) pvkDestroyInstance(s_vk.instance, NULL);
    if (s_vk.lib) dlclose(s_vk.lib);
    free(s_vk.cpu_verts);
    pthread_mutex_destroy(&s_vk.queue_mu);
    pthread_mutex_destroy(&s_vk.present_mu);
    memset(&s_vk, 0, sizeof s_vk);
}

#ifdef PS3RECOMP_SDL3_NATIVE
extern void pad_sdl3_native_notify_hotplug(void);   /* libs/input/pad_sdl3_native.c */
#endif

int rsx_vulkan_backend_pump_messages(void)
{
    if (!s_vk.windowed) return 0;
#if !VK_WINDOW_SDL
    return 0;
#else
    int quit = 0;
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_EVENT_QUIT) quit = 1;
        else if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_ESCAPE) quit = 1;
        else if (e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat && s_vk.window &&
                 (e.key.key == SDLK_F11 ||
                  (e.key.key == SDLK_RETURN && (e.key.mod & SDL_KMOD_ALT)))) {
            /* Borderless fullscreen on the window's display, and back. */
            const int full = (SDL_GetWindowFlags(s_vk.window) & SDL_WINDOW_FULLSCREEN) != 0;
            if (SDL_SetWindowFullscreen(s_vk.window, !full)) {
                if (full) SDL_ShowCursor(); else SDL_HideCursor();
            }
            s_vk_resized = 1;
        } else if (e.type == SDL_EVENT_GAMEPAD_ADDED || e.type == SDL_EVENT_GAMEPAD_REMOVED) {
            /* This pump consumes the window's events; tell cellPad so it rescans now (Deck waking from sleep, a
             * Bluetooth pad reconnecting) instead of on its 500 ms timer. */
            pad_sdl3_native_notify_hotplug();
        } else if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED ||
                   e.type == SDL_EVENT_WINDOW_RESTORED || e.type == SDL_EVENT_WINDOW_MAXIMIZED ||
                   e.type == SDL_EVENT_WINDOW_EXPOSED || e.type == SDL_EVENT_WINDOW_FOCUS_GAINED ||
                   e.type == SDL_EVENT_WINDOW_ENTER_FULLSCREEN || e.type == SDL_EVENT_WINDOW_LEAVE_FULLSCREEN ||
                   e.type == SDL_EVENT_WINDOW_DISPLAY_CHANGED) {
            /* also what a Deck resume or a compositor handing the window back looks like */
            s_vk_resized = 1;
        }
    }
    return quit ? -1 : 0;
#endif /* VK_WINDOW_SDL */
}

void rsx_vulkan_backend_present(void)
{
    if (!s_vk.device) return;
    if (s_vk.eng_active && vk_eb_async_use()) {
        rsx_draw_engine_present();
        return;
    }
    rsx_vulkan_backend_present_begin();
    rsx_vulkan_backend_present_finish();
}

void rsx_vulkan_backend_present_begin(void)
{
    if (!s_vk.device) return;
    s_vk.present_split = 1;
    if (s_vk.eng_active) rsx_draw_engine_present();
    else                 vk_cb_present(&s_vk, 0);
    s_vk.present_split = 0;
}

void rsx_vulkan_backend_present_finish(void)
{
    if (!s_vk.device) return;
    pthread_mutex_lock(&s_vk.present_mu);
    vk_present_complete_locked();
    pthread_mutex_unlock(&s_vk.present_mu);
}

u32 rsx_vulkan_backend_presented_center(void)
{
    return s_vk.presented ? (0xFF000000u | s_vk.last_center) : 0u;
}

u32 rsx_vulkan_backend_debug_color(void) { return s_vk.clear_argb; }

u32 rsx_vulkan_backend_readback_center(void)
{
    if (s_vk.eng_active) return rsx_draw_engine_readback_center();
    return s_vk.presented ? (0xFF000000u | s_vk.last_center) : 0u;
}

u32 rsx_vulkan_backend_guest_draws(void)
{
    return s_vk.eng_active ? rsx_draw_engine_guest_draws() : s_g.last_draws;
}

int rsx_vulkan_backend_depth_bounds(void)
{
    return s_vk.eng_active && s_vk.depth_bounds_ok;
}

#endif /* !_WIN32 */
