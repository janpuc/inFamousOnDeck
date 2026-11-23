# Vendored FSR-Vulkan (`ffx_vk`)

Minimal MIT-licensed Vulkan FSR 3.1.5 (temporal upscaling) + FSR 3.1.6 (analytical
Optical Flow + Frame Interpolation) + WSI presenter subset vendored from
[FireBurn/FSR-Vulkan](https://github.com/FireBurn/FSR-Vulkan) commit `d38393c`
(upstream AMD FidelityFX SDK 2.3.0 MIT sources + embedded Vulkan SPIR-V shaders).

See `LICENSE.txt` for full MIT license terms.

## Local Patches Applied
1. `src/ffx_vk_fsr3_3_1_5_bridge.cpp`:
   - `find_memory_type`: exclude `VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD` and
     `VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD` (forbidden without
     `deviceCoherentMemory` feature), and prefer `HOST_COHERENT_BIT` when
     `HOST_VISIBLE_BIT` is requested.
   - `vkFlushMappedMemoryRanges`: use `VK_WHOLE_SIZE` instead of unaligned
     `size` / `initData.size` to satisfy `VUID-VkMappedMemoryRange-size-01390`.
   - Set `srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED` and
     `dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED` on image/buffer barriers.
2. `src/ffx_vk_fsr3_3_1_6_framegen.cpp`:
   - `find_memory_type`: exclude `VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD` and
     `VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD`.
   - `initialize_shared_image_layouts`: set `srcQueueFamilyIndex` and
     `dstQueueFamilyIndex` to `VK_QUEUE_FAMILY_IGNORED`.
   - `ffxVkFsr3_3_1_6FrameGenerationContextRecordPrepare`: call
     `ffxVkFsr3_3_1_5BridgeBeginFrame(context->bridge, info->frameId)` before
     dispatching Optical Flow / FI Prepare so per-frame transient UBOs and
     descriptor sets are tagged with `frameId`.
   - `ffxVkFsr3_3_1_6FrameGenerationContextRetireFrame`: call
     `ffxVkFsr3_3_1_5BridgeRetireFrame(context->bridge, completedFrameId)` so
     per-frame UBOs and descriptor pools are freed instead of leaking every frame.
3. `src/ffx_vk_fsr3_3_1_5_shaderblobs_stub.cpp`:
   - Added C-linkage stubs for `fsr3UpscalerGetPermutationBlobByIndex` and
     `fsr3UpscalerIsWave64` (matching upstream `tests/fsr3_3_1_5_bridge_test.cpp`)
     so `ffx_fsr3upscaler.cpp` links cleanly when SPIR-V modules are resolved via
     `ffx_vk_fsr3_3_1_5_embedded_spirv.h`.
