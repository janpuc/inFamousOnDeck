/* Native Linux Vulkan temporal upscaling & FSR 3.1 compute post-process
 * module for rsx_vulkan_backend.c.
 *
 * Uses the API-agnostic camera/jitter/draw-tracking state from
 * rsx_upscale_common.h and runs SPIR-V compute passes on the backend's own
 * VkDevice + command buffers:
 *   - depth capture (pre-HUD scene depth snapshot)
 *   - pre-HUD color capture (right after the composite draw)
 *   - MV + R32F depth generation (camera reprojection + depth-validated fallback)
 *   - temporal upscaling (Job 1 passthrough blit -> Job 2 FSR 3.1.5)
 *   - HUD merge (per-pixel / 2x2 bilinear diff blend of HUD over upscaled output)
 *   - debug overlays (FSR_DEBUG / DLSS_DEBUG = mv|depth|warp|nowarp|zcheck|hud)
 */
#ifndef PS3RECOMP_RSX_VK_FSR_H
#define PS3RECOMP_RSX_VK_FSR_H

#ifndef _WIN32

#include <stdint.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rsx_vk_fsr_device {
    VkPhysicalDevice phys;
    VkDevice device;
    VkPhysicalDeviceMemoryProperties memprops;
    VkFormat depth_format;
    VkImageAspectFlags depth_aspect;
    uint32_t render_w, render_h;
    uint32_t out_w, out_h;
    PFN_vkGetDeviceProcAddr get_device_proc_addr;
} rsx_vk_fsr_device;

int  rsx_vk_fsr_init(const rsx_vk_fsr_device* dev);
void rsx_vk_fsr_shutdown(void);
int  rsx_vk_fsr_active(void);

/* Capture scene depth (depth_img in VK_IMAGE_LAYOUT_GENERAL) on the active
 * draw command buffer. */
void rsx_vk_fsr_capture_depth(VkCommandBuffer cb, VkImage depth_img,
                              uint32_t w, uint32_t h);

/* Capture pre-HUD scene color (color_img in VK_IMAGE_LAYOUT_GENERAL) on the
 * active draw command buffer right after the composite draw. */
void rsx_vk_fsr_capture_color(VkCommandBuffer cb, VkImage color_img,
                              VkFormat fmt, uint32_t w, uint32_t h);

/* End-of-frame post-process on the present command buffer.
 * finished_img = final display surface (render size, GENERAL).
 * out_color_img = s_vk.color.img (output size, GENERAL).
 * Returns 1 if out_color_img was written, 0 if caller should run its normal
 * blit (e.g. non-gameplay frames when FSR is inactive or skipped). */
int  rsx_vk_fsr_frame_end(VkCommandBuffer cb,
                          VkImage finished_img, VkFormat finished_fmt,
                          uint32_t finished_w, uint32_t finished_h,
                          VkImage out_color_img,
                          uint32_t out_w, uint32_t out_h);

/* Retire transient FSR descriptor/UBO allocations after the present submission
 * fence has signaled. Safe to call unconditionally (no-op when inactive). */
void rsx_vk_fsr_retire_frame(void);

/* Returns 1 if FSR 3.1.6 Frame Generation (FSR_FG=1) is enabled and supported. */
int  rsx_vk_fsr_fg_active(void);

/* Returns 1 if an interpolated frame was recorded on the most recent
 * rsx_vk_fsr_frame_end() call. Sets *out_img to the HUD-composited generated
 * RGBA8 image (output size, VK_IMAGE_LAYOUT_GENERAL) and *out_reset to 1 if
 * the dispatch was a history reset (presenter should display the real frame). */
int  rsx_vk_fsr_fg_frame_ready(VkImage* out_img, int* out_reset);

#ifdef __cplusplus
}
#endif
#endif /* !_WIN32 */
#endif /* PS3RECOMP_RSX_VK_FSR_H */
