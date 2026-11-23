/*
 * Stub shader-blob accessors for the AMD FidelityFX SDK 2.3 FSR 3.1.5 host
 * scheduler. Pipeline creation in ffx_vk_fsr3_3_1_5_bridge.cpp resolves its
 * SPIR-V words directly from ffx_vk_fsr3_3_1_5_embedded_spirv.h by pass name
 * and permutation, so these legacy DX12 blob accessors only return an empty
 * blob and Wave32 preference.
 *
 * SPDX-License-Identifier: MIT
 */

#include "ffx_fsr3upscaler.h"

#include <cstdint>
#include <cstring>

extern "C" FfxErrorCode fsr3UpscalerGetPermutationBlobByIndex(
    FfxFsr3UpscalerPass, uint32_t, FfxShaderBlob* outBlob)
{
    if (!outBlob)
        return static_cast<FfxErrorCode>(FFX_ERROR_INVALID_POINTER);
    std::memset(outBlob, 0, sizeof(*outBlob));
    return FFX_OK;
}

extern "C" FfxErrorCode fsr3UpscalerIsWave64(uint32_t, bool& isWave64)
{
    isWave64 = false;
    return FFX_OK;
}
