/* Native Linux Vulkan temporal upscaling & FSR 3.1 compute post-process
 * module for rsx_vulkan_backend.c. See rsx_vk_fsr.h.
 */
#ifndef _WIN32

#define VK_NO_PROTOTYPES
#include "rsx_vk_fsr.h"
#include "rsx_upscale_common.h"

#include "vulkan/rsx_vk_upscale_mv_comp.spv.h"
#include "vulkan/rsx_vk_upscale_merge_comp.spv.h"
#include "vulkan/rsx_vk_upscale_blit_comp.spv.h"
#include "vulkan/rsx_vk_upscale_dbg_comp.spv.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef PS3RECOMP_VK_FFX
#include "ffx_vk_portable.h"
#include "ffx_vk_fsr3_3_1_5_bridge.h"
#endif

#define FSR_VK_DEV_FNS(X) \
    X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements) X(vkBindImageMemory) \
    X(vkCreateImageView) X(vkDestroyImageView) \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory) X(vkUnmapMemory) \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) \
    X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) \
    X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) \
    X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) \
    X(vkCreateSampler) X(vkDestroySampler) \
    X(vkCmdPipelineBarrier) X(vkCmdCopyImage) X(vkCmdBlitImage) \
    X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) \
    X(vkCmdPushConstants) X(vkCmdDispatch) \
    X(vkCmdClearColorImage)

#define FSR_VK_DECL(name) static PFN_##name f##name;
FSR_VK_DEV_FNS(FSR_VK_DECL)
#undef FSR_VK_DECL

typedef struct fsr_vk_img {
    VkImage        img;
    VkDeviceMemory mem;
    VkImageView    view;
    VkFormat       fmt;
    uint32_t       w, h;
    int            inited;
} fsr_vk_img;

static struct {
    rsx_vk_fsr_device dev;
    int ready;
    int failed;
    int pipes_ready;
    int res_ready;

    VkSampler             sampler;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout      playout;
    VkDescriptorPool      dpool;
    VkDescriptorSet       dset;
    VkDescriptorSet       dset_fg;
    VkPipeline            pipe_mv;
    VkPipeline            pipe_merge;
    VkPipeline            pipe_blit;
    VkPipeline            pipe_dbg;

    VkBuffer       ubo;
    VkDeviceMemory ubo_mem;
    void*          ubo_ptr;

    fsr_vk_img zcopy;    /* scene depth snapshot (depth_format, DEPTH aspect SRV) */
    fsr_vk_img in;       /* pre-HUD scene color (render size, RGBA8) */
    fsr_vk_img hud;      /* finished frame with HUD (render size, RGBA8) */
    fsr_vk_img out;      /* upscaled output (output size, RGBA8) */
    fsr_vk_img fg_out;   /* interpolated frame output (output size, RGBA8) */
    fsr_vk_img mv;       /* motion vectors (render size, RG16F) */
    fsr_vk_img dz;       /* R32F scene depth (render size) */
    fsr_vk_img pdz;      /* previous frame R32F depth (render size) */
    fsr_vk_img prev;     /* previous frame pre-HUD color (render size, warp/nowarp) */

    int zsnap;
    int csnap;
    int hist_ok;
    int desc_dirty;
    uint64_t last_ns;

#ifdef PS3RECOMP_VK_FFX
    FfxVkFsr3_3_1_5UpscalerContext* ffx_ctx;
    fsr_vk_img ffx_out;        /* RGBA16F output matching FSR 3.1.5 rw_upscaled_output SPIR-V format */
    fsr_vk_img ffx_dil_depth;
    fsr_vk_img ffx_dil_mv;
    fsr_vk_img ffx_rec_prev_z;
    FfxVkFsr3_3_1_5Resource imp_in;
    FfxVkFsr3_3_1_5Resource imp_dz;
    FfxVkFsr3_3_1_5Resource imp_mv;
    FfxVkFsr3_3_1_5Resource imp_out;
    FfxVkFsr3_3_1_5Resource imp_dil_depth;
    FfxVkFsr3_3_1_5Resource imp_dil_mv;
    FfxVkFsr3_3_1_5Resource imp_rec_prev_z;
    int ffx_ready;
    uint64_t ffx_frame_id;
    uint64_t ffx_pending_retire_id;
    uint64_t ffx_retired_id;

    FfxVkFsr3_3_1_6FrameGenerationContext* fg_ctx;
    int fg_enabled;            /* FSR_FG=1 and GPU supports FG prerequisites */
    int fg_ready;              /* fg_ctx created successfully */
    int fg_failed;
    int fg_frame_dispatched;   /* current frame recorded FG output into fg_out */
    int fg_last_reset;         /* current frame FG dispatch had reset=VK_TRUE */
    uint64_t fg_frame_id;
    uint64_t fg_pending_retire_id;
    uint64_t fg_retired_id;
    uint64_t fg_generated_total;
    float fg_cam_pos[3];
#endif
} s_fsr;

static int fsr_find_mem(uint32_t bits, VkMemoryPropertyFlags want, uint32_t* out)
{
    const VkMemoryPropertyFlags forbidden =
        VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD |
        VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
    for (uint32_t i = 0; i < s_fsr.dev.memprops.memoryTypeCount; i++) {
        const VkMemoryPropertyFlags f = s_fsr.dev.memprops.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (f & want) == want && (f & forbidden) == 0u) {
            *out = i;
            return 0;
        }
    }
    return -1;
}

static void fsr_destroy_img(fsr_vk_img* im)
{
    if (!s_fsr.dev.device) return;
    if (im->view) fvkDestroyImageView(s_fsr.dev.device, im->view, NULL);
    if (im->img)  fvkDestroyImage(s_fsr.dev.device, im->img, NULL);
    if (im->mem)  fvkFreeMemory(s_fsr.dev.device, im->mem, NULL);
    memset(im, 0, sizeof *im);
}

static int fsr_create_img(fsr_vk_img* im, VkFormat fmt, uint32_t w, uint32_t h,
                          VkImageUsageFlags usage, VkImageAspectFlags view_aspect)
{
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = fmt,
        .extent = { w, h, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    if (fvkCreateImage(s_fsr.dev.device, &ici, NULL, &im->img) != VK_SUCCESS)
        return -1;
    VkMemoryRequirements req;
    fvkGetImageMemoryRequirements(s_fsr.dev.device, im->img, &req);
    uint32_t mtype = 0;
    if (fsr_find_mem(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &mtype) != 0 &&
        fsr_find_mem(req.memoryTypeBits, 0, &mtype) != 0) {
        fsr_destroy_img(im);
        return -1;
    }
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = mtype,
    };
    if (fvkAllocateMemory(s_fsr.dev.device, &mai, NULL, &im->mem) != VK_SUCCESS ||
        fvkBindImageMemory(s_fsr.dev.device, im->img, im->mem, 0) != VK_SUCCESS) {
        fsr_destroy_img(im);
        return -1;
    }
    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = im->img,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = fmt,
        .subresourceRange = { view_aspect, 0, 1, 0, 1 },
    };
    if (fvkCreateImageView(s_fsr.dev.device, &vci, NULL, &im->view) != VK_SUCCESS) {
        fsr_destroy_img(im);
        return -1;
    }
    im->fmt = fmt;
    im->w = w;
    im->h = h;
    im->inited = 0;
    return 0;
}

static void fsr_img_barrier(VkCommandBuffer cb, VkImage img, VkImageAspectFlags aspect,
                            VkImageLayout old_lay, VkImageLayout new_lay,
                            VkPipelineStageFlags src_stage, VkAccessFlags src_access,
                            VkPipelineStageFlags dst_stage, VkAccessFlags dst_access)
{
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_lay,
        .newLayout = new_lay,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img,
        .subresourceRange = { aspect, 0, 1, 0, 1 },
    };
    fvkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &b);
}

static void fsr_ensure_general(VkCommandBuffer cb, fsr_vk_img* im, VkImageAspectFlags aspect)
{
    if (im->inited) return;
    fsr_img_barrier(cb, im->img, aspect,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
    im->inited = 1;
}

static VkPipeline fsr_make_compute_pipe(const uint32_t* words, size_t byte_size)
{
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = byte_size,
        .pCode = words,
    };
    VkShaderModule sm = VK_NULL_HANDLE;
    if (fvkCreateShaderModule(s_fsr.dev.device, &smci, NULL, &sm) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sm,
            .pName = "main",
        },
        .layout = s_fsr.playout,
    };
    VkPipeline pipe = VK_NULL_HANDLE;
    VkResult r = fvkCreateComputePipelines(s_fsr.dev.device, VK_NULL_HANDLE, 1, &pci, NULL, &pipe);
    fvkDestroyShaderModule(s_fsr.dev.device, sm, NULL);
    return (r == VK_SUCCESS) ? pipe : VK_NULL_HANDLE;
}

static int fsr_build_pipelines(void)
{
    if (s_fsr.pipes_ready) return 1;
    if (s_fsr.failed) return 0;

    VkSamplerCreateInfo sci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxLod = 0.0f,
    };
    if (fvkCreateSampler(s_fsr.dev.device, &sci, NULL, &s_fsr.sampler) != VK_SUCCESS)
        goto fail;

    /* Shared layout for all 4 compute shaders:
     *   0: UBO (K, 176 bytes)
     *   1: combined image sampler zt (scene depth snapshot)
     *   2: storage image mv (rg16f)
     *   3: storage image dz (r32f)
     *   4: storage image o  (rgba8)
     *   5: combined image sampler sc (pre-HUD scene)
     *   6: combined image sampler hu (finished frame with HUD)
     *   7: combined image sampler pr (prev frame pre-HUD)
     *   8: combined image sampler pz (prev frame depth) */
    VkDescriptorSetLayoutBinding bindings[9] = {
        { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 9,
        .pBindings = bindings,
    };
    if (fvkCreateDescriptorSetLayout(s_fsr.dev.device, &dslci, NULL, &s_fsr.dsl) != VK_SUCCESS)
        goto fail;

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 4,
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &s_fsr.dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr,
    };
    if (fvkCreatePipelineLayout(s_fsr.dev.device, &plci, NULL, &s_fsr.playout) != VK_SUCCESS)
        goto fail;

    VkDescriptorPoolSize sizes[3] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         2 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          6 },
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2,
        .poolSizeCount = 3,
        .pPoolSizes = sizes,
    };
    if (fvkCreateDescriptorPool(s_fsr.dev.device, &dpci, NULL, &s_fsr.dpool) != VK_SUCCESS)
        goto fail;

    VkDescriptorSetLayout dsls[2] = { s_fsr.dsl, s_fsr.dsl };
    VkDescriptorSet dsets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = s_fsr.dpool,
        .descriptorSetCount = 2,
        .pSetLayouts = dsls,
    };
    if (fvkAllocateDescriptorSets(s_fsr.dev.device, &dsai, dsets) != VK_SUCCESS)
        goto fail;
    s_fsr.dset = dsets[0];
    s_fsr.dset_fg = dsets[1];

    /* 256-byte host-coherent UBO for K (44 floats = 176 bytes). */
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = 256,
        .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (fvkCreateBuffer(s_fsr.dev.device, &bci, NULL, &s_fsr.ubo) != VK_SUCCESS)
        goto fail;
    VkMemoryRequirements breq;
    fvkGetBufferMemoryRequirements(s_fsr.dev.device, s_fsr.ubo, &breq);
    uint32_t bmtype = 0;
    const VkMemoryPropertyFlags coh = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (fsr_find_mem(breq.memoryTypeBits, coh, &bmtype) != 0)
        goto fail;
    VkMemoryAllocateInfo bmai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = breq.size,
        .memoryTypeIndex = bmtype,
    };
    if (fvkAllocateMemory(s_fsr.dev.device, &bmai, NULL, &s_fsr.ubo_mem) != VK_SUCCESS ||
        fvkBindBufferMemory(s_fsr.dev.device, s_fsr.ubo, s_fsr.ubo_mem, 0) != VK_SUCCESS ||
        fvkMapMemory(s_fsr.dev.device, s_fsr.ubo_mem, 0, 256, 0, &s_fsr.ubo_ptr) != VK_SUCCESS)
        goto fail;

    s_fsr.pipe_mv    = fsr_make_compute_pipe(rsx_vk_upscale_mv_comp_spv,    sizeof rsx_vk_upscale_mv_comp_spv);
    s_fsr.pipe_merge = fsr_make_compute_pipe(rsx_vk_upscale_merge_comp_spv, sizeof rsx_vk_upscale_merge_comp_spv);
    s_fsr.pipe_blit  = fsr_make_compute_pipe(rsx_vk_upscale_blit_comp_spv,  sizeof rsx_vk_upscale_blit_comp_spv);
    s_fsr.pipe_dbg   = fsr_make_compute_pipe(rsx_vk_upscale_dbg_comp_spv,   sizeof rsx_vk_upscale_dbg_comp_spv);
    if (!s_fsr.pipe_mv || !s_fsr.pipe_merge || !s_fsr.pipe_blit || !s_fsr.pipe_dbg)
        goto fail;

    s_fsr.pipes_ready = 1;
    s_fsr.desc_dirty = 1;
    return 1;

fail:
    fprintf(stderr, "[FSR] Vulkan compute pipeline build failed -- off for this run\n");
    s_fsr.failed = 1;
    rsx_upscale_set_active(0);
    return 0;
}

#ifdef PS3RECOMP_VK_FFX
static void fsr_ffx_destroy(void)
{
    if (s_fsr.fg_ctx) {
        if (s_fsr.fg_pending_retire_id > s_fsr.fg_retired_id) {
            ffxVkFsr3_3_1_6FrameGenerationContextRetireFrame(s_fsr.fg_ctx, s_fsr.fg_pending_retire_id);
            s_fsr.fg_retired_id = s_fsr.fg_pending_retire_id;
        }
        ffxVkFsr3_3_1_6FrameGenerationContextDestroy(s_fsr.fg_ctx);
        s_fsr.fg_ctx = NULL;
    }
    s_fsr.fg_ready = 0;
    s_fsr.fg_frame_dispatched = 0;

    if (s_fsr.ffx_ctx) {
        FfxVkFsr3_3_1_5Bridge* br = ffxVkFsr3_3_1_5UpscalerContextGetBridge(s_fsr.ffx_ctx);
        if (br) {
            if (s_fsr.imp_in.resource)         ffxVkFsr3_3_1_5BridgeReleaseImportedImage(br, s_fsr.imp_in);
            if (s_fsr.imp_dz.resource)         ffxVkFsr3_3_1_5BridgeReleaseImportedImage(br, s_fsr.imp_dz);
            if (s_fsr.imp_mv.resource)         ffxVkFsr3_3_1_5BridgeReleaseImportedImage(br, s_fsr.imp_mv);
            if (s_fsr.imp_out.resource)        ffxVkFsr3_3_1_5BridgeReleaseImportedImage(br, s_fsr.imp_out);
            if (s_fsr.imp_dil_depth.resource)  ffxVkFsr3_3_1_5BridgeReleaseImportedImage(br, s_fsr.imp_dil_depth);
            if (s_fsr.imp_dil_mv.resource)     ffxVkFsr3_3_1_5BridgeReleaseImportedImage(br, s_fsr.imp_dil_mv);
            if (s_fsr.imp_rec_prev_z.resource) ffxVkFsr3_3_1_5BridgeReleaseImportedImage(br, s_fsr.imp_rec_prev_z);
        }
        memset(&s_fsr.imp_in, 0, sizeof s_fsr.imp_in);
        memset(&s_fsr.imp_dz, 0, sizeof s_fsr.imp_dz);
        memset(&s_fsr.imp_mv, 0, sizeof s_fsr.imp_mv);
        memset(&s_fsr.imp_out, 0, sizeof s_fsr.imp_out);
        memset(&s_fsr.imp_dil_depth, 0, sizeof s_fsr.imp_dil_depth);
        memset(&s_fsr.imp_dil_mv, 0, sizeof s_fsr.imp_dil_mv);
        memset(&s_fsr.imp_rec_prev_z, 0, sizeof s_fsr.imp_rec_prev_z);
        ffxVkFsr3_3_1_5UpscalerContextDestroy(s_fsr.ffx_ctx);
        s_fsr.ffx_ctx = NULL;
    }
    fsr_destroy_img(&s_fsr.ffx_out);
    fsr_destroy_img(&s_fsr.ffx_dil_depth);
    fsr_destroy_img(&s_fsr.ffx_dil_mv);
    fsr_destroy_img(&s_fsr.ffx_rec_prev_z);
    s_fsr.ffx_ready = 0;
}

static FfxVkFsr3_3_1_5Resource fsr_ffx_import(FfxVkFsr3_3_1_5Bridge* br,
                                              const fsr_vk_img* im,
                                              VkImageUsageFlags usage,
                                              uint32_t state)
{
    FfxVkFsr3_3_1_5ImportedImageDescription d;
    memset(&d, 0, sizeof d);
    d.image = im->img;
    d.format = im->fmt;
    d.width = im->w;
    d.height = im->h;
    d.mipCount = 1u;
    d.arrayLayers = 1u;
    d.layout = VK_IMAGE_LAYOUT_GENERAL;
    d.state = state;
    d.usage = usage;
    return ffxVkFsr3_3_1_5BridgeImportImage(br, &d);
}

static void fsr_ffx_init_context(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh,
                                 VkImageUsageFlags color_ro, VkImageUsageFlags stor_rw)
{
    fsr_ffx_destroy();
    FfxVkFsr3_3_1_5UpscalerCreateInfo ci;
    memset(&ci, 0, sizeof ci);
    ci.physicalDevice = s_fsr.dev.phys;
    ci.device = s_fsr.dev.device;
    ci.maxRenderWidth = w;
    ci.maxRenderHeight = h;
    ci.maxUpscaleWidth = ow;
    ci.maxUpscaleHeight = oh;
    /* FireBurn embedded SPIR-V modules are compiled with BASE = LANCZOS|HDR|LOW_RES_MV. */
    ci.hdrColorInput = VK_TRUE;
    ci.autoExposure = VK_FALSE;

    const FfxVkFsr3_3_1_5Result cr = ffxVkFsr3_3_1_5UpscalerContextCreate(&ci, &s_fsr.ffx_ctx);
    if (cr != FFX_VK_FSR3_3_1_5_OK || !s_fsr.ffx_ctx) {
        fprintf(stderr, "[FSR] FSR 3.1.5 context creation failed (%d) -- using bilinear fallback\n", (int)cr);
        fsr_ffx_destroy();
        return;
    }

    FfxVkFsr3_3_1_5SharedResourceDescriptions sd;
    memset(&sd, 0, sizeof sd);
    const FfxVkFsr3_3_1_5Result sr =
        ffxVkFsr3_3_1_5UpscalerContextGetSharedResourceDescriptions(s_fsr.ffx_ctx, &sd);
    if (sr != FFX_VK_FSR3_3_1_5_OK ||
        sd.dilatedDepth.format == VK_FORMAT_UNDEFINED ||
        sd.dilatedMotionVectors.format == VK_FORMAT_UNDEFINED ||
        sd.reconstructedPrevNearestDepth.format == VK_FORMAT_UNDEFINED) {
        fprintf(stderr, "[FSR] FSR 3.1.5 shared resource descriptions failed (%d) -- using bilinear fallback\n", (int)sr);
        fsr_ffx_destroy();
        return;
    }

    if (fsr_create_img(&s_fsr.ffx_out, VK_FORMAT_R16G16B16A16_SFLOAT,
                       ow, oh, stor_rw, VK_IMAGE_ASPECT_COLOR_BIT) != 0 ||
        fsr_create_img(&s_fsr.ffx_dil_depth, sd.dilatedDepth.format,
                       sd.dilatedDepth.width, sd.dilatedDepth.height,
                       stor_rw, VK_IMAGE_ASPECT_COLOR_BIT) != 0 ||
        fsr_create_img(&s_fsr.ffx_dil_mv, sd.dilatedMotionVectors.format,
                       sd.dilatedMotionVectors.width, sd.dilatedMotionVectors.height,
                       stor_rw, VK_IMAGE_ASPECT_COLOR_BIT) != 0 ||
        fsr_create_img(&s_fsr.ffx_rec_prev_z, sd.reconstructedPrevNearestDepth.format,
                       sd.reconstructedPrevNearestDepth.width, sd.reconstructedPrevNearestDepth.height,
                       stor_rw, VK_IMAGE_ASPECT_COLOR_BIT) != 0) {
        fprintf(stderr, "[FSR] FSR 3.1.5 shared image allocation failed -- using bilinear fallback\n");
        fsr_ffx_destroy();
        return;
    }

    FfxVkFsr3_3_1_5Bridge* br = ffxVkFsr3_3_1_5UpscalerContextGetBridge(s_fsr.ffx_ctx);
    s_fsr.imp_in         = fsr_ffx_import(br, &s_fsr.in, color_ro, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
    s_fsr.imp_dz         = fsr_ffx_import(br, &s_fsr.dz, stor_rw,  FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
    s_fsr.imp_mv         = fsr_ffx_import(br, &s_fsr.mv, stor_rw,  FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
    s_fsr.imp_out        = fsr_ffx_import(br, &s_fsr.ffx_out, stor_rw, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);
    s_fsr.imp_dil_depth  = fsr_ffx_import(br, &s_fsr.ffx_dil_depth, stor_rw, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);
    s_fsr.imp_dil_mv     = fsr_ffx_import(br, &s_fsr.ffx_dil_mv, stor_rw, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);
    s_fsr.imp_rec_prev_z = fsr_ffx_import(br, &s_fsr.ffx_rec_prev_z, stor_rw, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!s_fsr.imp_in.resource || !s_fsr.imp_dz.resource || !s_fsr.imp_mv.resource ||
        !s_fsr.imp_out.resource || !s_fsr.imp_dil_depth.resource ||
        !s_fsr.imp_dil_mv.resource || !s_fsr.imp_rec_prev_z.resource) {
        fprintf(stderr, "[FSR] FSR 3.1.5 image import failed -- using bilinear fallback\n");
        fsr_ffx_destroy();
        return;
    }

    FfxVkFsr3_3_1_5MemoryUsage mem = { 0, 0 };
    (void)ffxVkFsr3_3_1_5UpscalerContextGetMemoryUsage(s_fsr.ffx_ctx, &mem);
    s_fsr.ffx_ready = 1;
    fprintf(stderr, "[FSR] FSR 3.1.5 Vulkan upscaler created: %ux%u -> %ux%u (VRAM %.1f MB, sharpening %.2f)\n",
            w, h, ow, oh, (double)mem.totalUsageInBytes / (1024.0 * 1024.0),
            rsx_upscale_get_cfg()->fsr_sharp);

    if (s_fsr.fg_enabled && !s_fsr.fg_failed && s_fsr.fg_out.img) {
        FfxVkFsr3_3_1_6FrameGenerationCreateInfo fgci;
        memset(&fgci, 0, sizeof fgci);
        fgci.physicalDevice = s_fsr.dev.phys;
        fgci.device = s_fsr.dev.device;
        fgci.maxRenderWidth = w;
        fgci.maxRenderHeight = h;
        fgci.displayWidth = ow;
        fgci.displayHeight = oh;
        fgci.colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
        const FfxVkFsr3_3_1_6FrameGenerationResult fgr =
            ffxVkFsr3_3_1_6FrameGenerationContextCreate(&fgci, &s_fsr.fg_ctx);
        if (fgr == FFX_VK_FSR3_3_1_6_FRAMEGEN_OK && s_fsr.fg_ctx) {
            FfxVkFsr3_3_1_6FrameGenerationMemoryUsage fgmem = { 0, 0 };
            (void)ffxVkFsr3_3_1_6FrameGenerationContextGetMemoryUsage(s_fsr.fg_ctx, &fgmem);
            s_fsr.fg_ready = 1;
            fprintf(stderr, "[FSR] FSR 3.1.6 Optical Flow + Frame Generation created: render %ux%u, display %ux%u (VRAM %.1f MB)\n",
                    w, h, ow, oh, (double)fgmem.totalUsageInBytes / (1024.0 * 1024.0));
        } else {
            fprintf(stderr, "[FSR] FSR 3.1.6 Frame Generation creation failed (%d) -- FG off\n", (int)fgr);
            s_fsr.fg_failed = 1;
            s_fsr.fg_enabled = 0;
            s_fsr.fg_ctx = NULL;
        }
    }
}
#endif

static int fsr_ensure_resources(void)
{
    const uint32_t w = s_fsr.dev.render_w, h = s_fsr.dev.render_h;
    const uint32_t ow = s_fsr.dev.out_w, oh = s_fsr.dev.out_h;
    if (!w || !h || !ow || !oh) return 0;
    if (s_fsr.res_ready && s_fsr.in.w == w && s_fsr.in.h == h &&
        s_fsr.out.w == ow && s_fsr.out.h == oh)
        return 1;
    if (!fsr_build_pipelines()) return 0;

#ifdef PS3RECOMP_VK_FFX
    fsr_ffx_destroy();
#endif
    fsr_destroy_img(&s_fsr.in);
    fsr_destroy_img(&s_fsr.hud);
    fsr_destroy_img(&s_fsr.out);
    fsr_destroy_img(&s_fsr.fg_out);
    fsr_destroy_img(&s_fsr.mv);
    fsr_destroy_img(&s_fsr.dz);
    fsr_destroy_img(&s_fsr.pdz);
    fsr_destroy_img(&s_fsr.prev);
    s_fsr.csnap = 0;

    const VkImageUsageFlags color_ro = VK_IMAGE_USAGE_SAMPLED_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    const VkImageUsageFlags stor_rw  = VK_IMAGE_USAGE_STORAGE_BIT |
                                       VK_IMAGE_USAGE_SAMPLED_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    if (fsr_create_img(&s_fsr.in,  VK_FORMAT_R8G8B8A8_UNORM, w, h, color_ro, VK_IMAGE_ASPECT_COLOR_BIT) != 0 ||
        fsr_create_img(&s_fsr.hud, VK_FORMAT_R8G8B8A8_UNORM, w, h, color_ro, VK_IMAGE_ASPECT_COLOR_BIT) != 0 ||
        fsr_create_img(&s_fsr.out, VK_FORMAT_R8G8B8A8_UNORM, ow, oh, stor_rw, VK_IMAGE_ASPECT_COLOR_BIT) != 0 ||
        fsr_create_img(&s_fsr.mv,  VK_FORMAT_R16G16_SFLOAT,  w, h, stor_rw,  VK_IMAGE_ASPECT_COLOR_BIT) != 0 ||
        fsr_create_img(&s_fsr.dz,  VK_FORMAT_R32_SFLOAT,     w, h, stor_rw,  VK_IMAGE_ASPECT_COLOR_BIT) != 0 ||
        fsr_create_img(&s_fsr.pdz, VK_FORMAT_R32_SFLOAT,     w, h, color_ro, VK_IMAGE_ASPECT_COLOR_BIT) != 0 ||
        fsr_create_img(&s_fsr.prev, VK_FORMAT_R8G8B8A8_UNORM, w, h, color_ro, VK_IMAGE_ASPECT_COLOR_BIT) != 0) {
        fprintf(stderr, "[FSR] Vulkan resource allocation failed (%ux%u -> %ux%u) -- off\n", w, h, ow, oh);
        s_fsr.failed = 1;
        rsx_upscale_set_active(0);
        return 0;
    }
#ifdef PS3RECOMP_VK_FFX
    if (s_fsr.fg_enabled && !s_fsr.fg_failed) {
        if (fsr_create_img(&s_fsr.fg_out, VK_FORMAT_R8G8B8A8_UNORM, ow, oh,
                           stor_rw, VK_IMAGE_ASPECT_COLOR_BIT) != 0) {
            fprintf(stderr, "[FSR] FG output image allocation failed (%ux%u) -- FG off\n", ow, oh);
            s_fsr.fg_failed = 1;
            s_fsr.fg_enabled = 0;
        }
    }
#endif
    s_fsr.res_ready = 1;
    s_fsr.desc_dirty = 1;
    fprintf(stderr, "[FSR] Vulkan buffers ready: render %ux%u, output %ux%u\n", w, h, ow, oh);
#ifdef PS3RECOMP_VK_FFX
    fsr_ffx_init_context(w, h, ow, oh, color_ro, stor_rw);
#endif
    return 1;
}

static void fsr_update_descriptors(void)
{
    if (!s_fsr.desc_dirty || !s_fsr.res_ready || !s_fsr.zcopy.view) return;
    VkDescriptorBufferInfo bi = { .buffer = s_fsr.ubo, .offset = 0, .range = 256 };
    VkDescriptorImageInfo ii_zt   = { s_fsr.sampler, s_fsr.zcopy.view, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo ii_mv   = { VK_NULL_HANDLE, s_fsr.mv.view,   VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo ii_dz   = { VK_NULL_HANDLE, s_fsr.dz.view,   VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo ii_out  = { VK_NULL_HANDLE, s_fsr.out.view,  VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo ii_sc   = { s_fsr.sampler, s_fsr.in.view,    VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo ii_hu   = { s_fsr.sampler, s_fsr.hud.view,   VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo ii_pr   = { s_fsr.sampler, s_fsr.prev.view,  VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo ii_pz   = { s_fsr.sampler, s_fsr.pdz.view,   VK_IMAGE_LAYOUT_GENERAL };

    VkWriteDescriptorSet w[9] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, s_fsr.dset, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         NULL, &bi, NULL },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, s_fsr.dset, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii_zt,  NULL, NULL },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, s_fsr.dset, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &ii_mv,  NULL, NULL },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, s_fsr.dset, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &ii_dz,  NULL, NULL },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, s_fsr.dset, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &ii_out, NULL, NULL },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, s_fsr.dset, 5, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii_sc,  NULL, NULL },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, s_fsr.dset, 6, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii_hu,  NULL, NULL },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, s_fsr.dset, 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii_pr,  NULL, NULL },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, s_fsr.dset, 8, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii_pz,  NULL, NULL },
    };
    fvkUpdateDescriptorSets(s_fsr.dev.device, 9, w, 0, NULL);
    if (s_fsr.dset_fg && s_fsr.fg_out.view) {
        VkDescriptorImageInfo ii_fg = { VK_NULL_HANDLE, s_fsr.fg_out.view, VK_IMAGE_LAYOUT_GENERAL };
        for (int i = 0; i < 9; i++) w[i].dstSet = s_fsr.dset_fg;
        w[4].pImageInfo = &ii_fg;
        fvkUpdateDescriptorSets(s_fsr.dev.device, 9, w, 0, NULL);
    }
    s_fsr.desc_dirty = 0;
}

int rsx_vk_fsr_init(const rsx_vk_fsr_device* dev)
{
    memset(&s_fsr, 0, sizeof s_fsr);
    if (!dev || !dev->device || !dev->get_device_proc_addr) return 0;
    s_fsr.dev = *dev;
    rsx_upscale_set_sizes(dev->render_w, dev->render_h, dev->out_w, dev->out_h);
    const rsx_upscale_cfg* cfg = rsx_upscale_get_cfg();
    if (!cfg->mode) {
        rsx_upscale_set_active(0);
        return 0;
    }

#define LOAD_F(name) \
    if (!(f##name = (PFN_##name)dev->get_device_proc_addr(dev->device, #name))) { \
        fprintf(stderr, "[FSR] missing Vulkan device function " #name " -- off\n"); \
        s_fsr.failed = 1; rsx_upscale_set_active(0); return -1; \
    }
    FSR_VK_DEV_FNS(LOAD_F)
#undef LOAD_F

    s_fsr.ready = 1;
    rsx_upscale_set_active(1);
#ifdef PS3RECOMP_VK_FFX
    if (dev->phys) {
        FfxVkPortableDeviceCapabilities caps;
        memset(&caps, 0, sizeof caps);
        caps.structSize = sizeof caps;
        if (ffxVkPortableQueryDeviceCapabilities(dev->phys, &caps) == FFX_VK_PORTABLE_OK) {
            fprintf(stderr, "[FSR] GPU capabilities: Vulkan %u.%u.%u, subgroup %u, "
                            "FSR3 compute=%d, FG=%d\n",
                    VK_API_VERSION_MAJOR(caps.apiVersion),
                    VK_API_VERSION_MINOR(caps.apiVersion),
                    VK_API_VERSION_PATCH(caps.apiVersion),
                    caps.subgroupSize,
                    caps.fsr3ComputePrerequisites ? 1 : 0,
                    caps.fsr3FrameGenerationPrerequisites ? 1 : 0);
            if (cfg->fg) {
                if (caps.fsr3FrameGenerationPrerequisites) {
                    s_fsr.fg_enabled = 1;
                } else {
                    fprintf(stderr, "[FSR] FSR_FG=1 requested but GPU lacks FG prerequisites -- FG off\n");
                }
            }
        }
    }
#endif
    const int mode = rsx_upscale_is_upscaling() ? (cfg->up ? cfg->up : 1) : 0;
    fprintf(stderr, "[FSR] Vulkan tracking ready: %s, render %ux%u -> output %ux%u, "
                    "jitter %s (%u phases), motion vectors %s, mip bias %+.2f, debug %d, fg %s\n",
            rsx_upscale_mode_name(mode, 1),
            dev->render_w, dev->render_h, dev->out_w, dev->out_h,
            cfg->jitter ? "on" : "off", rsx_upscale_jitter_phases(),
            cfg->mv ? "on" : "off", rsx_upscale_mip_bias(), cfg->debug_mv,
            rsx_vk_fsr_fg_active() ? "on" : "off");
    return 0;
}

void rsx_vk_fsr_shutdown(void)
{
    if (s_fsr.dev.device) {
#ifdef PS3RECOMP_VK_FFX
        fsr_ffx_destroy();
#endif
        fsr_destroy_img(&s_fsr.zcopy);
        fsr_destroy_img(&s_fsr.in);
        fsr_destroy_img(&s_fsr.hud);
        fsr_destroy_img(&s_fsr.out);
        fsr_destroy_img(&s_fsr.fg_out);
        fsr_destroy_img(&s_fsr.mv);
        fsr_destroy_img(&s_fsr.dz);
        fsr_destroy_img(&s_fsr.pdz);
        fsr_destroy_img(&s_fsr.prev);
        if (s_fsr.ubo_ptr) fvkUnmapMemory(s_fsr.dev.device, s_fsr.ubo_mem);
        if (s_fsr.ubo)     fvkDestroyBuffer(s_fsr.dev.device, s_fsr.ubo, NULL);
        if (s_fsr.ubo_mem) fvkFreeMemory(s_fsr.dev.device, s_fsr.ubo_mem, NULL);
        if (s_fsr.pipe_mv)    fvkDestroyPipeline(s_fsr.dev.device, s_fsr.pipe_mv, NULL);
        if (s_fsr.pipe_merge) fvkDestroyPipeline(s_fsr.dev.device, s_fsr.pipe_merge, NULL);
        if (s_fsr.pipe_blit)  fvkDestroyPipeline(s_fsr.dev.device, s_fsr.pipe_blit, NULL);
        if (s_fsr.pipe_dbg)   fvkDestroyPipeline(s_fsr.dev.device, s_fsr.pipe_dbg, NULL);
        if (s_fsr.dpool)      fvkDestroyDescriptorPool(s_fsr.dev.device, s_fsr.dpool, NULL);
        if (s_fsr.playout)    fvkDestroyPipelineLayout(s_fsr.dev.device, s_fsr.playout, NULL);
        if (s_fsr.dsl)        fvkDestroyDescriptorSetLayout(s_fsr.dev.device, s_fsr.dsl, NULL);
        if (s_fsr.sampler)    fvkDestroySampler(s_fsr.dev.device, s_fsr.sampler, NULL);
    }
    rsx_upscale_set_active(0);
    memset(&s_fsr, 0, sizeof s_fsr);
}

int rsx_vk_fsr_active(void) { return s_fsr.ready && !s_fsr.failed && rsx_upscale_is_active(); }

void rsx_vk_fsr_capture_depth(VkCommandBuffer cb, VkImage depth_img, uint32_t w, uint32_t h)
{
    if (!rsx_vk_fsr_active() || !cb || !depth_img) return;
    if (w != s_fsr.dev.render_w || h != s_fsr.dev.render_h) return;
    if (!fsr_ensure_resources()) return;
    if (!s_fsr.zcopy.img || s_fsr.zcopy.w != w || s_fsr.zcopy.h != h) {
        fsr_destroy_img(&s_fsr.zcopy);
        if (fsr_create_img(&s_fsr.zcopy, s_fsr.dev.depth_format, w, h,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           VK_IMAGE_ASPECT_DEPTH_BIT) != 0) {
            fprintf(stderr, "[FSR] depth snapshot buffer failed (%ux%u) -- off\n", w, h);
            s_fsr.failed = 1;
            rsx_upscale_set_active(0);
            return;
        }
        s_fsr.desc_dirty = 1;
    }
    fsr_ensure_general(cb, &s_fsr.zcopy, s_fsr.dev.depth_aspect);
    fsr_img_barrier(cb, depth_img, s_fsr.dev.depth_aspect,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    fsr_img_barrier(cb, s_fsr.zcopy.img, s_fsr.dev.depth_aspect,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageCopy region = {
        .srcSubresource = { s_fsr.dev.depth_aspect, 0, 0, 1 },
        .dstSubresource = { s_fsr.dev.depth_aspect, 0, 0, 1 },
        .extent = { w, h, 1 },
    };
    fvkCmdCopyImage(cb, depth_img, VK_IMAGE_LAYOUT_GENERAL,
                    s_fsr.zcopy.img, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    fsr_img_barrier(cb, depth_img, s_fsr.dev.depth_aspect,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    fsr_img_barrier(cb, s_fsr.zcopy.img, s_fsr.dev.depth_aspect,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    s_fsr.zsnap = 1;
}

void rsx_vk_fsr_capture_color(VkCommandBuffer cb, VkImage color_img,
                              VkFormat fmt, uint32_t w, uint32_t h)
{
    if (!rsx_vk_fsr_active() || !cb || !color_img) return;
    if (fmt != VK_FORMAT_R8G8B8A8_UNORM ||
        w != s_fsr.dev.render_w || h != s_fsr.dev.render_h)
        return;
    if (!fsr_ensure_resources()) return;
    fsr_ensure_general(cb, &s_fsr.in, VK_IMAGE_ASPECT_COLOR_BIT);
    fsr_img_barrier(cb, color_img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    fsr_img_barrier(cb, s_fsr.in.img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageCopy region = {
        .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .extent = { w, h, 1 },
    };
    fvkCmdCopyImage(cb, color_img, VK_IMAGE_LAYOUT_GENERAL,
                    s_fsr.in.img, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    fsr_img_barrier(cb, color_img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    fsr_img_barrier(cb, s_fsr.in.img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    s_fsr.csnap = 1;
}

int rsx_vk_fsr_frame_end(VkCommandBuffer cb,
                         VkImage finished_img, VkFormat finished_fmt,
                         uint32_t finished_w, uint32_t finished_h,
                         VkImage out_color_img,
                         uint32_t out_w, uint32_t out_h)
{
#ifdef PS3RECOMP_VK_FFX
    s_fsr.fg_frame_dispatched = 0;
    s_fsr.fg_last_reset = 1;
#endif
    if (!rsx_vk_fsr_active() || !cb || !finished_img || !out_color_img) return 0;
    const uint32_t w = s_fsr.dev.render_w, h = s_fsr.dev.render_h;
    const uint32_t ow = s_fsr.dev.out_w, oh = s_fsr.dev.out_h;
    if (out_w != ow || out_h != oh) return 0;
    if (finished_fmt != VK_FORMAT_R8G8B8A8_UNORM || finished_w != w || finished_h != h) {
        s_fsr.hist_ok = 0;
        s_fsr.zsnap = s_fsr.csnap = 0;
        rsx_upscale_reset_history();
        return 0;
    }

    const rsx_upscale_batch* ex = rsx_upscale_exec();
    const rsx_upscale_cfg* cfg = rsx_upscale_get_cfg();

    if (ex->qualifying < 50 || !s_fsr.zsnap || !s_fsr.zcopy.img) {
        s_fsr.hist_ok = 0;
        s_fsr.zsnap = s_fsr.csnap = 0;
        rsx_upscale_reset_history();
        return 0;
    }
    s_fsr.zsnap = 0;

    if (!fsr_ensure_resources()) return 0;

    /* Transition any first-use internal images to GENERAL once. */
    fsr_ensure_general(cb, &s_fsr.in,   VK_IMAGE_ASPECT_COLOR_BIT);
    fsr_ensure_general(cb, &s_fsr.hud,  VK_IMAGE_ASPECT_COLOR_BIT);
    fsr_ensure_general(cb, &s_fsr.out,  VK_IMAGE_ASPECT_COLOR_BIT);
    fsr_ensure_general(cb, &s_fsr.mv,   VK_IMAGE_ASPECT_COLOR_BIT);
    fsr_ensure_general(cb, &s_fsr.dz,   VK_IMAGE_ASPECT_COLOR_BIT);
    if (!s_fsr.pdz.inited) {
        fsr_ensure_general(cb, &s_fsr.pdz, VK_IMAGE_ASPECT_COLOR_BIT);
        VkClearColorValue far_z = { .float32 = { 1.0f, 1.0f, 1.0f, 1.0f } };
        VkImageSubresourceRange rng = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        fvkCmdClearColorImage(cb, s_fsr.pdz.img, VK_IMAGE_LAYOUT_GENERAL, &far_z, 1, &rng);
    }
    if (!s_fsr.prev.inited) {
        fsr_ensure_general(cb, &s_fsr.prev, VK_IMAGE_ASPECT_COLOR_BIT);
        VkClearColorValue blk = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } };
        VkImageSubresourceRange rng = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        fvkCmdClearColorImage(cb, s_fsr.prev.img, VK_IMAGE_LAYOUT_GENERAL, &blk, 1, &rng);
    }
    fsr_update_descriptors();

    /* 1. Copy finished display frame into s_fsr.hud (and s_fsr.in if no pre-HUD snapshot). */
    const int merge = s_fsr.csnap;
    s_fsr.csnap = 0;
    fsr_img_barrier(cb, finished_img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    fsr_img_barrier(cb, s_fsr.hud.img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageCopy cpy = {
        .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .extent = { w, h, 1 },
    };
    fvkCmdCopyImage(cb, finished_img, VK_IMAGE_LAYOUT_GENERAL,
                    s_fsr.hud.img, VK_IMAGE_LAYOUT_GENERAL, 1, &cpy);
    if (!merge) {
        fsr_img_barrier(cb, s_fsr.in.img, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        fvkCmdCopyImage(cb, finished_img, VK_IMAGE_LAYOUT_GENERAL,
                        s_fsr.in.img, VK_IMAGE_LAYOUT_GENERAL, 1, &cpy);
        static int n_no_hud = 0;
        if (n_no_hud++ < 3)
            fprintf(stderr, "[FSR] no pre-HUD snapshot this frame -- HUD included\n");
    }

    /* 2. Pack MV / camera reprojection constants and dispatch MV + R32F depth pass. */
    float k[44];
    int cut = 0;
    const int mv_on = rsx_upscale_pack_mv_consts(k, s_fsr.hist_ok, &cut);
    memcpy(s_fsr.ubo_ptr, k, sizeof k);
    (void)mv_on;

    if (ex->cam_valid) {
        static int cam_logged = 0;
        if (!cam_logged++) {
            float nearp = 10.0f, farp = 100000.0f, fov = 1.0f;
            rsx_upscale_cam_planes(ex->cam, &nearp, &farp, &fov);
            fprintf(stderr, "[FSR] camera: near %.3f, far %.1f, vertical fov %.1f deg\n",
                    nearp, farp, fov * 57.29578f);
        }
    }

    fsr_img_barrier(cb, s_fsr.zcopy.img, s_fsr.dev.depth_aspect,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    fsr_img_barrier(cb, s_fsr.pdz.img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    fsr_img_barrier(cb, s_fsr.in.img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    fsr_img_barrier(cb, s_fsr.hud.img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    fsr_img_barrier(cb, s_fsr.mv.img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    fsr_img_barrier(cb, s_fsr.dz.img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);

    fvkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s_fsr.playout,
                             0, 1, &s_fsr.dset, 0, NULL);
    fvkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s_fsr.pipe_mv);
    fvkCmdDispatch(cb, (w + 7u) / 8u, (h + 7u) / 8u, 1u);

    fsr_img_barrier(cb, s_fsr.mv.img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    fsr_img_barrier(cb, s_fsr.dz.img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);

    /* 3. Upscaler pass: AMD FidelityFX FSR 3.1.5 temporal upscaling / Native AA
     *    (falls back to bilinear blit only if FSR 3.1.5 was not compiled/created). */
    float dt_ms = 16.6667f;
    {
        struct timespec ts;
        if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
            const uint64_t now_ns = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
            if (s_fsr.last_ns && now_ns > s_fsr.last_ns)
                dt_ms = (float)((double)(now_ns - s_fsr.last_ns) * 1e-6);
            s_fsr.last_ns = now_ns;
        }
        if (!isfinite(dt_ms) || dt_ms < 1.0f) dt_ms = 1.0f;
        if (dt_ms > 1000.0f) dt_ms = 1000.0f;
    }

    float nearp = 10.0f, farp = 100000.0f, fov = 1.0f;
    if (ex->cam_valid) rsx_upscale_cam_planes(ex->cam, &nearp, &farp, &fov);
    if (!isfinite(nearp) || nearp < 0.001f) nearp = 10.0f;
    if (!isfinite(farp) || farp <= nearp + 1.0f) farp = nearp + 100000.0f;
    if (!isfinite(fov) || fov < 0.05f || fov > 3.0f) fov = 1.0f;

    const VkBool32 hist_reset = (s_fsr.hist_ok && !cut) ? VK_FALSE : VK_TRUE;

    int fsr_dispatched = 0;
#ifdef PS3RECOMP_VK_FFX
    if (s_fsr.ffx_ready && s_fsr.ffx_ctx) {
        fsr_ensure_general(cb, &s_fsr.ffx_out,        VK_IMAGE_ASPECT_COLOR_BIT);
        fsr_ensure_general(cb, &s_fsr.ffx_dil_depth,  VK_IMAGE_ASPECT_COLOR_BIT);
        fsr_ensure_general(cb, &s_fsr.ffx_dil_mv,     VK_IMAGE_ASPECT_COLOR_BIT);
        fsr_ensure_general(cb, &s_fsr.ffx_rec_prev_z, VK_IMAGE_ASPECT_COLOR_BIT);

        fsr_img_barrier(cb, s_fsr.ffx_out.img, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        FfxVkFsr3_3_1_5UpscalerDispatchInfo di;
        memset(&di, 0, sizeof di);
        di.commandBuffer = cb;
        di.color = s_fsr.imp_in;
        di.depth = s_fsr.imp_dz;
        di.motionVectors = s_fsr.imp_mv;
        di.dilatedDepth = s_fsr.imp_dil_depth;
        di.dilatedMotionVectors = s_fsr.imp_dil_mv;
        di.reconstructedPrevNearestDepth = s_fsr.imp_rec_prev_z;
        di.output = s_fsr.imp_out;
        di.jitterOffsetX = cfg->jsign * ex->jx;
        di.jitterOffsetY = cfg->jsign * ex->jy;
        di.motionVectorScaleX = 1.0f;
        di.motionVectorScaleY = 1.0f;
        di.renderWidth = w;
        di.renderHeight = h;
        di.upscaleWidth = ow;
        di.upscaleHeight = oh;
        di.frameTimeMilliseconds = dt_ms;
        di.preExposure = 1.0f;
        di.enableSharpening = cfg->fsr_sharp > 0.0f ? VK_TRUE : VK_FALSE;
        di.sharpness = cfg->fsr_sharp;
        di.reset = hist_reset;
        di.cameraNear = nearp;
        di.cameraFar = farp;
        di.cameraVerticalFovRadians = fov;
        di.viewSpaceToMeters = 0.01f;   /* inFamous world units are centimetres */
        di.flags = 0u;
        di.frameId = ++s_fsr.ffx_frame_id;

        const FfxVkFsr3_3_1_5Result dr =
            ffxVkFsr3_3_1_5UpscalerContextRecordDispatch(s_fsr.ffx_ctx, &di);
        s_fsr.ffx_pending_retire_id = di.frameId;
        if (dr == FFX_VK_FSR3_3_1_5_OK) {
            fsr_dispatched = 1;
            /* Convert RGBA16F FSR output -> RGBA8 s_fsr.out (1:1 nearest blit). */
            fsr_img_barrier(cb, s_fsr.ffx_out.img, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            fsr_img_barrier(cb, s_fsr.out.img, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkImageBlit bl = {
                .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                .srcOffsets = { { 0, 0, 0 }, { (int32_t)ow, (int32_t)oh, 1 } },
                .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                .dstOffsets = { { 0, 0, 0 }, { (int32_t)ow, (int32_t)oh, 1 } },
            };
            fvkCmdBlitImage(cb, s_fsr.ffx_out.img, VK_IMAGE_LAYOUT_GENERAL,
                            s_fsr.out.img, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_NEAREST);
        } else {
            static int n_err = 0;
            if (n_err++ < 5)
                fprintf(stderr, "[FSR] FSR 3.1.5 dispatch failed (%d) -- falling back to blit\n", (int)dr);
            fvkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     s_fsr.playout, 0, 1, &s_fsr.dset, 0, NULL);
        }
    }
#endif
    if (!fsr_dispatched) {
        fsr_img_barrier(cb, s_fsr.out.img, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        const uint32_t use_sc = 1u;
        fvkCmdPushConstants(cb, s_fsr.playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &use_sc);
        fvkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s_fsr.pipe_blit);
        fvkCmdDispatch(cb, (ow + 7u) / 8u, (oh + 7u) / 8u, 1u);
    }

    /* 3a. FSR 3.1.6 analytical Optical Flow + Frame Interpolation on HUDless scene.
     *     HUD is merged afterward onto BOTH s_fsr.out and s_fsr.fg_out so UI stays crisp. */
#ifdef PS3RECOMP_VK_FFX
    if (s_fsr.fg_ready && s_fsr.fg_ctx && s_fsr.fg_out.img && !cfg->debug_mv) {
        fsr_ensure_general(cb, &s_fsr.fg_out, VK_IMAGE_ASPECT_COLOR_BIT);

        fsr_img_barrier(cb, s_fsr.out.img, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        fsr_img_barrier(cb, s_fsr.dz.img, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        fsr_img_barrier(cb, s_fsr.mv.img, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        fsr_img_barrier(cb, s_fsr.fg_out.img, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        if (hist_reset) {
            s_fsr.fg_cam_pos[0] = s_fsr.fg_cam_pos[1] = s_fsr.fg_cam_pos[2] = 0.0f;
        } else if (ex->delta_valid) {
            for (int i = 0; i < 3; i++) {
                const float np = s_fsr.fg_cam_pos[i] + ex->delta[i];
                s_fsr.fg_cam_pos[i] = isfinite(np) ? np : 0.0f;
            }
        }

        const VkImageUsageFlags stor_rw = VK_IMAGE_USAGE_STORAGE_BIT |
                                          VK_IMAGE_USAGE_SAMPLED_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        FfxVkFsr3_3_1_6FrameGenerationPrepareInfo fpi;
        memset(&fpi, 0, sizeof fpi);
        fpi.commandBuffer = cb;
        fpi.color.image = s_fsr.out.img;
        fpi.color.format = VK_FORMAT_R8G8B8A8_UNORM;
        fpi.color.width = ow;
        fpi.color.height = oh;
        fpi.color.layout = VK_IMAGE_LAYOUT_GENERAL;
        fpi.color.usage = stor_rw;
        fpi.depth.image = s_fsr.dz.img;
        fpi.depth.format = VK_FORMAT_R32_SFLOAT;
        fpi.depth.width = w;
        fpi.depth.height = h;
        fpi.depth.layout = VK_IMAGE_LAYOUT_GENERAL;
        fpi.depth.usage = stor_rw;
        fpi.motionVectors.image = s_fsr.mv.img;
        fpi.motionVectors.format = VK_FORMAT_R16G16_SFLOAT;
        fpi.motionVectors.width = w;
        fpi.motionVectors.height = h;
        fpi.motionVectors.layout = VK_IMAGE_LAYOUT_GENERAL;
        fpi.motionVectors.usage = stor_rw;
        fpi.renderWidth = w;
        fpi.renderHeight = h;
        fpi.jitterOffsetX = cfg->jsign * ex->jx;
        fpi.jitterOffsetY = cfg->jsign * ex->jy;
        fpi.motionVectorScaleX = 1.0f;
        fpi.motionVectorScaleY = 1.0f;
        fpi.frameTimeMilliseconds = dt_ms;
        fpi.minLuminance = 0.0f;
        fpi.maxLuminance = 1.0f;
        fpi.transferFunction = FFX_VK_FSR3_3_1_6_FRAMEGEN_TRANSFER_SRGB;
        fpi.cameraNear = nearp;
        fpi.cameraFar = farp;
        fpi.viewSpaceToMeters = 0.01f;
        fpi.cameraVerticalFovRadians = fov;
        fpi.cameraRight[0] = 1.0f;
        fpi.cameraUp[1] = 1.0f;
        fpi.cameraForward[2] = 1.0f;
        if (ex->cam_valid) {
            const float* m = ex->cam;
            const float* rows[3] = { m, m + 4, m + 12 };
            float* outs[3] = { fpi.cameraRight, fpi.cameraUp, fpi.cameraForward };
            for (int k_ax = 0; k_ax < 3; k_ax++) {
                const float l = sqrtf(rows[k_ax][0] * rows[k_ax][0] +
                                      rows[k_ax][1] * rows[k_ax][1] +
                                      rows[k_ax][2] * rows[k_ax][2]);
                if (isfinite(l) && l > 1e-9f) {
                    for (int i = 0; i < 3; i++) outs[k_ax][i] = rows[k_ax][i] / l;
                }
            }
            for (int i = 0; i < 3; i++) fpi.cameraPosition[i] = s_fsr.fg_cam_pos[i];
        }
        const uint64_t fg_fid = ++s_fsr.fg_frame_id;
        fpi.frameId = fg_fid;
        fpi.reset = hist_reset;

        const FfxVkFsr3_3_1_6FrameGenerationResult pr =
            ffxVkFsr3_3_1_6FrameGenerationContextRecordPrepare(s_fsr.fg_ctx, &fpi);
        /* Always retain pending retire ID once Prepare has been attempted. */
        s_fsr.fg_pending_retire_id = fg_fid;
        if (pr == FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
            FfxVkFsr3_3_1_6FrameGenerationDispatchInfo fdi;
            memset(&fdi, 0, sizeof fdi);
            fdi.commandBuffer = cb;
            fdi.color = fpi.color;
            fdi.output.image = s_fsr.fg_out.img;
            fdi.output.format = VK_FORMAT_R8G8B8A8_UNORM;
            fdi.output.width = ow;
            fdi.output.height = oh;
            fdi.output.layout = VK_IMAGE_LAYOUT_GENERAL;
            fdi.output.usage = stor_rw;
            fdi.displayWidth = ow;
            fdi.displayHeight = oh;
            fdi.frameTimeMilliseconds = dt_ms;
            fdi.cameraNear = nearp;
            fdi.cameraFar = farp;
            fdi.viewSpaceToMeters = 0.01f;
            fdi.cameraVerticalFovRadians = fov;
            fdi.minLuminance = 0.0f;
            fdi.maxLuminance = 1.0f;
            fdi.transferFunction = FFX_VK_FSR3_3_1_6_FRAMEGEN_TRANSFER_SRGB;
            fdi.frameId = fg_fid;
            fdi.reset = hist_reset;

            const FfxVkFsr3_3_1_6FrameGenerationResult fdr =
                ffxVkFsr3_3_1_6FrameGenerationContextRecordDispatch(s_fsr.fg_ctx, &fdi);
            if (fdr == FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
                s_fsr.fg_frame_dispatched = 1;
                s_fsr.fg_last_reset = (hist_reset == VK_TRUE) ? 1 : 0;
                if (!s_fsr.fg_last_reset) s_fsr.fg_generated_total++;
                static int fg_first_logged = 0;
                if (!fg_first_logged++) {
                    fprintf(stderr, "[FSR] first FG frame (FSR 3.1.6 Optical Flow + Frame Interpolation, "
                                    "%ux%u, HUD-post-FG %s)\n",
                            ow, oh, merge ? "yes" : "no");
                }
            } else {
                static int n_fg_derr = 0;
                if (n_fg_derr++ < 5)
                    fprintf(stderr, "[FSR] FSR 3.1.6 FG dispatch failed (%d)\n", (int)fdr);
            }
        } else {
            static int n_fg_perr = 0;
            if (n_fg_perr++ < 5)
                fprintf(stderr, "[FSR] FSR 3.1.6 FG prepare failed (%d)\n", (int)pr);
        }
    }
#endif

    /* Re-bind our descriptor set if FSR 3.1.5 or 3.1.6 altered compute pipeline state. */
    if ((fsr_dispatched
#ifdef PS3RECOMP_VK_FFX
         || s_fsr.fg_ready
#endif
        ) && (merge || cfg->debug_mv)) {
        fvkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 s_fsr.playout, 0, 1, &s_fsr.dset, 0, NULL);
    }

    /* 3b. Merge HUD back over upscaled output (and over generated frame when FG ran). */
    if (merge) {
        fsr_img_barrier(cb, s_fsr.out.img, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        fvkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s_fsr.pipe_merge);
        fvkCmdDispatch(cb, (ow + 7u) / 8u, (oh + 7u) / 8u, 1u);
#ifdef PS3RECOMP_VK_FFX
        if (s_fsr.fg_frame_dispatched && s_fsr.dset_fg) {
            fsr_img_barrier(cb, s_fsr.fg_out.img, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            fvkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     s_fsr.playout, 0, 1, &s_fsr.dset_fg, 0, NULL);
            fvkCmdDispatch(cb, (ow + 7u) / 8u, (oh + 7u) / 8u, 1u);
            if (cfg->debug_mv) {
                fvkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                         s_fsr.playout, 0, 1, &s_fsr.dset, 0, NULL);
            }
        }
#endif
    }

    /* 3c. Optional debug overlay (mv / depth / warp / nowarp / zcheck / hud). */
    if (cfg->debug_mv) {
        fsr_img_barrier(cb, s_fsr.out.img, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        fsr_img_barrier(cb, s_fsr.prev.img, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        fvkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s_fsr.pipe_dbg);
        fvkCmdDispatch(cb, (ow + 7u) / 8u, (oh + 7u) / 8u, 1u);
        /* Save current pre-HUD scene as next frame's previous frame (for warp/nowarp). */
        if (cfg->debug_mv == 3 || cfg->debug_mv == 4) {
            fsr_img_barrier(cb, s_fsr.in.img, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            fsr_img_barrier(cb, s_fsr.prev.img, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            fvkCmdCopyImage(cb, s_fsr.in.img, VK_IMAGE_LAYOUT_GENERAL,
                            s_fsr.prev.img, VK_IMAGE_LAYOUT_GENERAL, 1, &cpy);
        }
    }

    /* 4. Copy this frame's depth -> previous depth, and output -> s_vk.color.img. */
    fsr_img_barrier(cb, s_fsr.dz.img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    fsr_img_barrier(cb, s_fsr.pdz.img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    fvkCmdCopyImage(cb, s_fsr.dz.img, VK_IMAGE_LAYOUT_GENERAL,
                    s_fsr.pdz.img, VK_IMAGE_LAYOUT_GENERAL, 1, &cpy);

    fsr_img_barrier(cb, s_fsr.out.img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    fsr_img_barrier(cb, out_color_img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageCopy out_cpy = {
        .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .extent = { ow, oh, 1 },
    };
    fvkCmdCopyImage(cb, s_fsr.out.img, VK_IMAGE_LAYOUT_GENERAL,
                    out_color_img, VK_IMAGE_LAYOUT_GENERAL, 1, &out_cpy);
#ifdef PS3RECOMP_VK_FFX
    if (s_fsr.fg_frame_dispatched && s_fsr.fg_out.img) {
        fsr_img_barrier(cb, s_fsr.fg_out.img, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    }
#endif

    rsx_upscale_commit_history();
    s_fsr.hist_ok = 1;
    {
        static int first_logged = 0;
        if (!first_logged++) {
            const int mode = rsx_upscale_is_upscaling() ? (cfg->up ? cfg->up : 1) : 0;
            fprintf(stderr, "[FSR] first %s frame (%s, %u jittered draws, depth 0x%X, pre-HUD %s, %ux%u -> %ux%u)\n",
                    rsx_upscale_mode_name(mode, 1),
                    fsr_dispatched ? "FSR 3.1.5" : "bilinear fallback",
                    ex->qualifying, ex->zeta,
                    merge ? "ok" : "missing", w, h, ow, oh);
        }
    }
    return 1;
}

void rsx_vk_fsr_retire_frame(void)
{
#ifdef PS3RECOMP_VK_FFX
    if (s_fsr.ffx_ctx && s_fsr.ffx_pending_retire_id > s_fsr.ffx_retired_id) {
        ffxVkFsr3_3_1_5UpscalerContextRetireFrame(s_fsr.ffx_ctx, s_fsr.ffx_pending_retire_id);
        s_fsr.ffx_retired_id = s_fsr.ffx_pending_retire_id;
    }
    if (s_fsr.fg_ctx && s_fsr.fg_pending_retire_id > s_fsr.fg_retired_id) {
        ffxVkFsr3_3_1_6FrameGenerationContextRetireFrame(s_fsr.fg_ctx, s_fsr.fg_pending_retire_id);
        s_fsr.fg_retired_id = s_fsr.fg_pending_retire_id;
    }
#endif
}

int rsx_vk_fsr_fg_active(void)
{
#ifdef PS3RECOMP_VK_FFX
    return rsx_vk_fsr_active() && s_fsr.fg_enabled && !s_fsr.fg_failed;
#else
    return 0;
#endif
}

int rsx_vk_fsr_fg_frame_ready(VkImage* out_img, int* out_reset)
{
#ifdef PS3RECOMP_VK_FFX
    if (!s_fsr.fg_frame_dispatched || !s_fsr.fg_out.img) return 0;
    if (out_img)   *out_img   = s_fsr.fg_out.img;
    if (out_reset) *out_reset = s_fsr.fg_last_reset;
    return 1;
#else
    (void)out_img; (void)out_reset;
    return 0;
#endif
}

#endif /* !_WIN32 */
