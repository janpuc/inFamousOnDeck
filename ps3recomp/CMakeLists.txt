cmake_minimum_required(VERSION 3.20)
# Keep in step with the changelog in README.md. This sat at 0.3.0 until August
# 2026 while the project shipped 0.6.x through 0.8.0, so every build printed
# "ps3recomp runtime v0.3.0" and was believed. v0.9.0 is the first TAGGED one.
project(ps3recomp_runtime
    VERSION 0.11.0
    LANGUAGES C CXX
    DESCRIPTION "ps3recomp - Static recompilation runtime for PS3 games"
)

# ---------------------------------------------------------------------------
# Standards
# ---------------------------------------------------------------------------
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(PS3RECOMP_BUILD_TESTS "Build unit tests" OFF)

# ---------------------------------------------------------------------------
# Collect source files
# ---------------------------------------------------------------------------

# HLE library sources
file(GLOB_RECURSE HLE_SOURCES
    "libs/system/*.c"
    "libs/filesystem/*.c"
    "libs/input/*.c"
    "libs/audio/*.c"
    "libs/video/*.c"
    "libs/network/*.c"
    "libs/spurs/*.c"
    "libs/sync/*.c"
    "libs/codec/*.c"
    "libs/font/*.c"
    "libs/misc/*.c"
    "libs/hardware/*.c"
    "runtime/diag/*.c"      # yz_frontier_trace, used by the live draw engine
)
# Exclude test harnesses (e.g. libs/video/tests/*): they define their own main().
list(FILTER HLE_SOURCES EXCLUDE REGEX "/tests/")

# The Vulkan RSX backend is opt-in while it grows. It loads libvulkan at run
# time, so building needs only the Vulkan headers -- no target-architecture
# libvulkan, which keeps cross-compiling simple.
option(PS3RECOMP_RSX_VULKAN "Build the Vulkan RSX backend (Linux)" OFF)
list(FILTER HLE_SOURCES EXCLUDE REGEX "rsx_vulkan_backend\\.c$")
list(FILTER HLE_SOURCES EXCLUDE REGEX "rsx_vk_fsr\\.c$")
if(PS3RECOMP_RSX_VULKAN)
    if(WIN32 OR APPLE)
        message(FATAL_ERROR "PS3RECOMP_RSX_VULKAN is Linux-only for now")
    endif()
    find_path(PS3RECOMP_VULKAN_INCLUDE_DIR vulkan/vulkan.h)
    if(NOT PS3RECOMP_VULKAN_INCLUDE_DIR)
        message(FATAL_ERROR "Vulkan headers not found. Debian/Ubuntu: "
            "libvulkan-dev, Fedora: vulkan-headers -- or pass "
            "-DPS3RECOMP_VULKAN_INCLUDE_DIR=<dir containing vulkan/vulkan.h>")
    endif()
    list(APPEND HLE_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/video/rsx_vulkan_backend.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/video/rsx_vk_fsr.c"
    )
    # The windowed build presents through an SDL3 window, and off Windows SDL3
    # also carries cellPad and cellAudio. One SDL in the process: SDL2 and SDL3
    # export the same symbols with different ABIs, so mixing them silently
    # breaks whichever module resolves to the other library. Without SDL3 the
    # backend still builds and renders offscreen; PS3RECOMP_VK_WINDOW is then
    # unsupported and says so.
    find_package(SDL3 CONFIG QUIET)
    if(SDL3_FOUND)
        set(_ps3recomp_sdl3_native ON)
        message(STATUS "  SDL3:         found (Vulkan window, cellPad, cellAudio)")
    else()
        message(STATUS "  SDL3:         not found; the Vulkan window is disabled "
                       "(install libsdl3-dev for a window and SDL3 gamepads)")
    endif()
    # HLSL -> SPIR-V for guest programs: a stub unless glslang is found below.
    list(APPEND HLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/libs/video/rsx_shader_spirv.cpp")
endif()

# The Metal backend is Objective-C and Apple-only, so it is added explicitly
# rather than globbed. It needs no platform #ifdef of its own as a result.
if(APPLE)
    enable_language(OBJC)
    list(APPEND HLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/libs/video/rsx_metal_backend.m")
endif()
# API-agnostic temporal upscaling tracking shared by D3D12 (rsx_dlss.cpp) and
# Vulkan (rsx_draw_engine.c / rsx_vk_fsr.c). Pure C++/C with no GPU API headers.
list(APPEND HLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/libs/video/rsx_upscale_common.cpp")
# The HLSL -> MSL translator is C++ (it drives glslang's C++ API) and compiles
# to a stub wherever the shader tools are absent, so it is part of the
# library everywhere. See the "Guest shaders on Metal" section below.
list(APPEND HLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/libs/video/rsx_shader_msl.cpp")
# Optional SDL3 gamepads for cellPad (libs/input/pad_sdl3.c: DualSense, DualShock 4, Steam Controller). Built only
# when the SDL3 headers are present next to the checkout (sdl3/include); SDL3.dll is loaded at run time.
set(PS3RECOMP_SDL3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sdl3" CACHE PATH "SDL3 folder for PlayStation / Steam controllers (optional)")
set(_ps3recomp_sdl3 OFF)
if(WIN32 AND EXISTS "${PS3RECOMP_SDL3_DIR}/include/SDL3/SDL_gamepad.h")
    set(_ps3recomp_sdl3 ON)
    list(APPEND HLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/libs/input/pad_sdl3.c")
endif()
# Optional temporal upscaling for the D3D12 backend (libs/video/rsx_dlss.cpp): NVIDIA DLSS (DLSS=1 at run time) when
# the DLSS SDK is next to the checkout (dlss-sdk/include + lib/Windows_x86_64/x64/nvsdk_ngx_d.lib), AMD FSR 3.1 / 4
# (FSR=1) when the AMD FSR SDK is (fsr-sdk/Kits/FidelityFX: headers; its signed DLLs are loaded at run time).
set(PS3RECOMP_DLSS_SDK "${CMAKE_CURRENT_SOURCE_DIR}/../dlss-sdk" CACHE PATH "NVIDIA DLSS SDK folder (optional)")
set(PS3RECOMP_FSR_SDK "${CMAKE_CURRENT_SOURCE_DIR}/../fsr-sdk" CACHE PATH "AMD FSR SDK folder (optional)")
set(_ps3recomp_ngx OFF)
set(_ps3recomp_ffx OFF)
if(WIN32 AND EXISTS "${PS3RECOMP_DLSS_SDK}/include/nvsdk_ngx.h"
         AND EXISTS "${PS3RECOMP_DLSS_SDK}/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib")
    set(_ps3recomp_ngx ON)
endif()
if(WIN32 AND EXISTS "${PS3RECOMP_FSR_SDK}/Kits/FidelityFX/upscalers/include/ffx_upscale.h")
    set(_ps3recomp_ffx ON)
endif()
set(_ps3recomp_dlss OFF)
if(_ps3recomp_ngx OR _ps3recomp_ffx)
    set(_ps3recomp_dlss ON)
    list(APPEND HLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/libs/video/rsx_dlss.cpp")
endif()
# cellAtrac decodes with FFmpeg's ATRAC3plus decoder as extracted by PPSSPP
# (LGPL-2.1, kept separate in third_party/at3_standalone -- see its README).
file(GLOB AT3_SOURCES "third_party/at3_standalone/*.cpp")
set_source_files_properties(${AT3_SOURCES} PROPERTIES COMPILE_OPTIONS "-w")  # upstream code
list(APPEND HLE_SOURCES ${AT3_SOURCES})

# Runtime core (syscalls, PPU, SPU, memory)
file(GLOB_RECURSE RUNTIME_SOURCES
    "runtime/*.c"
    "runtime/*.cpp"
)
# Exclude:
#  - test harnesses (runtime/*/tests/*): they have their own main() and #include
#    per-game generated headers (spu_recomp.h) that a plain library build lacks.
#  - the PPU boot scaffold (runtime/ppu/, runtime/host/): these are compiled
#    per-game against the lifter-generated ppu_recomp.h (and the game's RSX/D3D12
#    backend), not as part of the game-agnostic runtime library. See
#    docs/PPU_RECOMP.md.
#  - spu_dispatch_mt.c: the musttail SPU indirect-branch fast path. It uses
#    __attribute__((musttail)), which MSVC cannot parse, and it must be compiled
#    *inside the game build* by clang so the tail call is guaranteed -- a plain
#    call leaks one resolver frame per guest loop iteration. A port adds it to
#    its own target; see templates/project/CMakeLists.txt.
list(FILTER RUNTIME_SOURCES EXCLUDE REGEX "/tests/|/runtime/ppu/|/runtime/host/|/spu_dispatch_mt\.c$")

# ---------------------------------------------------------------------------
# Static library
# ---------------------------------------------------------------------------
add_library(ps3recomp_runtime STATIC
    ${HLE_SOURCES}
    ${RUNTIME_SOURCES}
)

target_include_directories(ps3recomp_runtime PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
if(_ps3recomp_sdl3_native)
    target_link_libraries(ps3recomp_runtime PRIVATE SDL3::SDL3)
    target_compile_definitions(ps3recomp_runtime PRIVATE PS3RECOMP_SDL3_NATIVE=1)
endif()

if(_ps3recomp_dlss)
    target_compile_definitions(ps3recomp_runtime PRIVATE PS3RECOMP_DLSS=1)
endif()
if(_ps3recomp_ngx)
    target_include_directories(ps3recomp_runtime PRIVATE "${PS3RECOMP_DLSS_SDK}/include")
    target_compile_definitions(ps3recomp_runtime PRIVATE PS3RECOMP_NGX=1)
    target_link_libraries(ps3recomp_runtime PUBLIC "${PS3RECOMP_DLSS_SDK}/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib")
    message(STATUS "  DLSS:         ON (${PS3RECOMP_DLSS_SDK})")
else()
    message(STATUS "  DLSS:         off (no DLSS SDK in ${PS3RECOMP_DLSS_SDK})")
endif()
if(_ps3recomp_ffx)
    target_include_directories(ps3recomp_runtime PRIVATE "${PS3RECOMP_FSR_SDK}/Kits/FidelityFX")
    target_compile_definitions(ps3recomp_runtime PRIVATE PS3RECOMP_FFX=1)
    message(STATUS "  FSR:          ON (${PS3RECOMP_FSR_SDK})")
else()
    message(STATUS "  FSR:          off (no AMD FSR SDK in ${PS3RECOMP_FSR_SDK})")
endif()

if(_ps3recomp_sdl3)
    target_include_directories(ps3recomp_runtime PRIVATE "${PS3RECOMP_SDL3_DIR}/include")
    target_compile_definitions(ps3recomp_runtime PRIVATE PS3RECOMP_SDL3=1)
    message(STATUS "  SDL3 pads:    ON (${PS3RECOMP_SDL3_DIR})")
else()
    message(STATUS "  SDL3 pads:    off (no SDL3 in ${PS3RECOMP_SDL3_DIR})")
endif()
# Player builds pass -DPORT_REQUIRE_SDL3=ON so a missing sdl3/ folder stops the build instead of
# silently dropping DualSense / DualShock 4 / Steam Controller support (XInput only). Dev builds: OFF.
option(PORT_REQUIRE_SDL3 "Fail the configure when SDL3 (PlayStation / Steam controllers) is missing" OFF)
if(WIN32 AND PORT_REQUIRE_SDL3 AND NOT _ps3recomp_sdl3)
    message(FATAL_ERROR "SDL3 not found in ${PS3RECOMP_SDL3_DIR} (needs include/SDL3/SDL_gamepad.h): "
        "PlayStation and Steam controllers would be disabled. Restore the sdl3/ folder or pass -DPORT_REQUIRE_SDL3=OFF.")
endif()

# HLE modules include their own headers
target_include_directories(ps3recomp_runtime PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/system"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/filesystem"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/input"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/audio"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/video"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/network"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/spurs"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/sync"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/codec"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/font"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/misc"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/spu"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Platform-specific link libraries
if(WIN32)
    target_link_libraries(ps3recomp_runtime PRIVATE
        ws2_32      # Winsock for network modules
        xinput      # XInput for gamepad
        ole32       # COM for WASAPI audio
        bcrypt      # BCryptGenRandom for SSL RNG
    )
else()
    if(_ps3recomp_sdl3_native)
        # SDL3 already provides cellPad and cellAudio here; adding SDL2 would
        # put two libs with the same symbol names into one process.
        message(STATUS "  SDL2:         not used (SDL3 supplies input and audio)")
    else()
    # cellPad and cellAudio select their SDL2 backends unconditionally off
    # Windows (see PAD_BACKEND_SDL2 / AUDIO_BACKEND_SDL), so SDL2 is required
    # here -- docs/BUILDING.md already lists it as the Linux/macOS dependency.
    find_package(SDL2 QUIET)
    if(SDL2_FOUND)
        if(TARGET SDL2::SDL2)
            target_link_libraries(ps3recomp_runtime PRIVATE SDL2::SDL2)
        else()
            target_include_directories(ps3recomp_runtime PRIVATE ${SDL2_INCLUDE_DIRS})
            target_link_libraries(ps3recomp_runtime PRIVATE ${SDL2_LIBRARIES})
        endif()
        message(STATUS "  SDL2:         found (input + audio backends enabled)")
    else()
        message(FATAL_ERROR
            "SDL2 not found. It provides the cellPad and cellAudio backends on "
            "non-Windows hosts.\n"
            "  macOS:  brew install sdl2\n"
            "  Debian: apt install libsdl2-dev\n"
            "If SDL2 is installed in a non-standard prefix, pass "
            "-DCMAKE_PREFIX_PATH=<prefix>.")
    endif()
    endif()

    find_package(Threads REQUIRED)
    target_link_libraries(ps3recomp_runtime PRIVATE Threads::Threads)
endif()

if(APPLE)
    target_link_libraries(ps3recomp_runtime PUBLIC
        "-framework Metal"
        "-framework QuartzCore"
        "-framework Foundation"
        "-framework AppKit"
    )
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/video/rsx_metal_backend.m"
        PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    message(STATUS "  Metal:        RSX backend enabled (rsx_metal_backend.m)")
endif()

# ---------------------------------------------------------------------------
# Guest shaders on Metal: HLSL -> SPIR-V -> MSL
# ---------------------------------------------------------------------------
# The RSX decompilers emit HLSL for the D3D12 backend. Metal cannot compile
# that, so libs/video/rsx_shader_msl.cpp lowers it with glslang (HLSL ->
# SPIR-V) and spirv-cross (SPIR-V -> MSL). Both are optional: without them the
# module compiles to a stub, rsx_hlsl_to_msl_available() says so, and the
# Metal backend stays on its fixed-function path. The translation has no GPU
# in it, so it is built and tested on Linux as well -- that is where the
# decompilers' HLSL meets glslang's front end on every push.
#
#   macOS:  brew install glslang spirv-cross
#   Debian: apt install glslang-dev glslang-tools spirv-tools \
#                       libspirv-cross-c-shared-dev
#           (glslang-tools only because the packaged CMake config declares
#           the standalone binary as an imported target and fails to load
#           without it)
set(PS3RECOMP_MSL_TRANSLATION OFF)
if(UNIX)
    find_package(glslang CONFIG QUIET)

    # spirv-cross's C API. Debian ships it as a shared library and Homebrew as
    # a static one, under different pkg-config names -- and pkg-config itself
    # is not a given on a Mac, so fall back to a plain header + library search.
    set(SPVC_OK OFF)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(SPVC QUIET spirv-cross-c-shared)
        if(NOT SPVC_FOUND)
            pkg_check_modules(SPVC QUIET spirv-cross-c)
        endif()
        if(SPVC_FOUND)
            set(SPVC_INC     ${SPVC_INCLUDE_DIRS})
            set(SPVC_LIBS    ${SPVC_LINK_LIBRARIES})
            set(SPVC_LIBDIRS ${SPVC_LIBRARY_DIRS})
            set(SPVC_OK ON)
        endif()
    endif()
    if(NOT SPVC_OK)
        find_path(SPVC_INC_DIR spirv_cross_c.h PATH_SUFFIXES spirv_cross)
        find_library(SPVC_LIB NAMES spirv-cross-c-shared spirv-cross-c)
        if(SPVC_INC_DIR AND SPVC_LIB)
            set(SPVC_INC  ${SPVC_INC_DIR})
            set(SPVC_LIBS ${SPVC_LIB})
            get_filename_component(SPVC_LIBDIRS "${SPVC_LIB}" DIRECTORY)
            set(SPVC_OK ON)
        endif()
    endif()

    if(glslang_FOUND AND SPVC_OK
       AND TARGET glslang::glslang AND TARGET glslang::SPIRV)
        add_library(ps3recomp_msl_translation INTERFACE)
        target_compile_definitions(ps3recomp_msl_translation INTERFACE
            PS3RECOMP_HAVE_MSL_TRANSLATION=1)
        target_include_directories(ps3recomp_msl_translation INTERFACE ${SPVC_INC})
        target_link_libraries(ps3recomp_msl_translation INTERFACE
            glslang::glslang glslang::SPIRV ${SPVC_LIBS})
        if(TARGET glslang::glslang-default-resource-limits)
            target_link_libraries(ps3recomp_msl_translation INTERFACE
                glslang::glslang-default-resource-limits)
        endif()
        # A static spirv-cross C API (Homebrew's) needs the backends it was
        # built from, and neither pkg-config nor the library name says so:
        # pick up the static siblings next to it. The shared one carries
        # its own.
        if(SPVC_LIBS MATCHES "\\.a$")
            foreach(part msl glsl hlsl cpp reflect util core)
                find_library(SPVC_PART_${part} NAMES spirv-cross-${part}
                    HINTS ${SPVC_LIBDIRS} NO_DEFAULT_PATH)
                if(SPVC_PART_${part})
                    target_link_libraries(ps3recomp_msl_translation INTERFACE
                        ${SPVC_PART_${part}})
                endif()
            endforeach()
        endif()
        target_link_libraries(ps3recomp_runtime PRIVATE ps3recomp_msl_translation)
        set(PS3RECOMP_MSL_TRANSLATION ON)
        message(STATUS "  Shaders:      guest programs translated to MSL (glslang ${glslang_VERSION}, ${SPVC_LIBS})")
    else()
        message(STATUS "  Shaders:      glslang/spirv-cross not found; Metal backend stays fixed-function")
    endif()
endif()

if(PS3RECOMP_RSX_VULKAN)
    target_include_directories(ps3recomp_runtime PRIVATE "${PS3RECOMP_VULKAN_INCLUDE_DIR}")
    target_compile_definitions(ps3recomp_runtime PUBLIC PS3RECOMP_RSX_VULKAN)
    if(UNIX AND NOT APPLE)
        target_link_libraries(ps3recomp_runtime PRIVATE ${CMAKE_DL_LIBS})
    endif()
    message(STATUS "  Vulkan:       RSX backend enabled (rsx_vulkan_backend.c, headers in ${PS3RECOMP_VULKAN_INCLUDE_DIR})")
else()
    message(STATUS "  Vulkan:       off (PS3RECOMP_RSX_VULKAN=ON enables the Linux backend)")
endif()

# Guest shaders on Vulkan: HLSL -> SPIR-V. glslang only -- Vulkan consumes
# SPIR-V, so spirv-cross is not needed. Without glslang
# libs/video/rsx_shader_spirv.cpp is a stub and the Vulkan backend stays on
# its fixed-function path.
if(PS3RECOMP_RSX_VULKAN)
    find_package(glslang CONFIG QUIET)
    if(glslang_FOUND AND TARGET glslang::glslang AND TARGET glslang::SPIRV)
        set_source_files_properties(
            "${CMAKE_CURRENT_SOURCE_DIR}/libs/video/rsx_shader_spirv.cpp"
            PROPERTIES COMPILE_DEFINITIONS PS3RECOMP_HAVE_SPIRV_TRANSLATION=1)
        target_link_libraries(ps3recomp_runtime PRIVATE glslang::glslang glslang::SPIRV)
        if(TARGET glslang::glslang-default-resource-limits)
            target_link_libraries(ps3recomp_runtime PRIVATE
                glslang::glslang-default-resource-limits)
        endif()
        set(PS3RECOMP_SPIRV_TRANSLATION ON)
        message(STATUS "  Vulkan:       guest programs available (glslang ${glslang_VERSION}); "
                       "opt in with PS3RECOMP_VK_GUEST_PROGRAMS=1")
    else()
        message(STATUS "  Vulkan:       glslang not found; guest programs unavailable")
    endif()

    # Vendored MIT Vulkan FSR 3.1.5 (temporal upscaling) + 3.1.6 (analytical
    # Optical Flow + Frame Interpolation) + WSI presenter (FireBurn/FSR-Vulkan).
    set(_ffx_vk_dir "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ffx_vk")
    if(EXISTS "${_ffx_vk_dir}/include/ffx_vk_fsr3_3_1_5_bridge.h")
        find_package(Vulkan QUIET)
        if(Vulkan_FOUND)
            set(_ffx_vk_link Vulkan::Vulkan)
        else()
            find_library(_ffx_vk_lib NAMES vulkan vulkan-1)
            if(_ffx_vk_lib)
                set(_ffx_vk_link "${_ffx_vk_lib}")
            endif()
        endif()
        if(_ffx_vk_link)
            set(_ffx_230 "${_ffx_vk_dir}/upstream/ffx-2.3.0/Kits/FidelityFX")
            set(_ffx_vk_private_syms
                ffxAssertReport=ffxVk315AssertReport
                ffxAssertSetPrintingCallback=ffxVk315AssertSetPrintingCallback
                ffxFsr3UpscalerContextCreate=ffxVk315Fsr3UpscalerContextCreate
                ffxFsr3UpscalerContextDestroy=ffxVk315Fsr3UpscalerContextDestroy
                ffxFsr3UpscalerContextDispatch=ffxVk315Fsr3UpscalerContextDispatch
                ffxFsr3UpscalerContextGenerateReactiveMask=ffxVk315Fsr3UpscalerContextGenerateReactiveMask
                ffxFsr3UpscalerContextGetGpuMemoryUsage=ffxVk315Fsr3UpscalerContextGetGpuMemoryUsage
                ffxFsr3UpscalerGetGpuMemoryUsage=ffxVk315Fsr3UpscalerGetGpuMemoryUsage
                ffxFsr3UpscalerGetJitterOffset=ffxVk315Fsr3UpscalerGetJitterOffset
                ffxFsr3UpscalerGetJitterPhaseCount=ffxVk315Fsr3UpscalerGetJitterPhaseCount
                ffxFsr3UpscalerGetRenderResolutionFromQualityMode=ffxVk315Fsr3UpscalerGetRenderResolutionFromQualityMode
                ffxFsr3UpscalerGetSharedResourceDescriptions=ffxVk315Fsr3UpscalerGetSharedResourceDescriptions
                ffxFsr3UpscalerGetUpscaleRatioFromQualityMode=ffxVk315Fsr3UpscalerGetUpscaleRatioFromQualityMode
                ffxFsr3UpscalerResourceIsNull=ffxVk315Fsr3UpscalerResourceIsNull
                ffxFsr3UpscalerSetConstant=ffxVk315Fsr3UpscalerSetConstant
                ffxFsr3UpscalerSetGlobalDebugMessage=ffxVk315Fsr3UpscalerSetGlobalDebugMessage
                frameInterpolationGetPermutationBlobByIndex=ffxVk316FrameInterpolationGetPermutationBlobByIndex
                frameInterpolationIsWave64=ffxVk316FrameInterpolationIsWave64
                opticalflowGetPermutationBlobByIndex=ffxVk316OpticalflowGetPermutationBlobByIndex
                opticalflowIsWave64=ffxVk316OpticalflowIsWave64
                ffxFrameInterpolationContextCreate=ffxVk316FrameInterpolationContextCreate
                ffxFrameInterpolationContextDestroy=ffxVk316FrameInterpolationContextDestroy
                ffxFrameInterpolationContextEnqueueRefreshPipelineRequest=ffxVk316FrameInterpolationContextEnqueueRefreshPipelineRequest
                ffxFrameInterpolationContextGetGpuMemoryUsage=ffxVk316FrameInterpolationContextGetGpuMemoryUsage
                ffxFrameInterpolationDispatch=ffxVk316FrameInterpolationDispatch
                ffxFrameInterpolationGetGpuMemoryUsage=ffxVk316FrameInterpolationGetGpuMemoryUsage
                ffxFrameInterpolationGetSharedResourceDescriptions=ffxVk316FrameInterpolationGetSharedResourceDescriptions
                ffxFrameInterpolationPrepare=ffxVk316FrameInterpolationPrepare
                ffxFrameInterpolationResourceIsNull=ffxVk316FrameInterpolationResourceIsNull
                ffxFrameInterpolationSetGlobalDebugMessage=ffxVk316FrameInterpolationSetGlobalDebugMessage
                ffxOpticalflowContextCreate=ffxVk316OpticalflowContextCreate
                ffxOpticalflowContextDestroy=ffxVk316OpticalflowContextDestroy
                ffxOpticalflowContextDispatch=ffxVk316OpticalflowContextDispatch
                ffxOpticalflowContextGetGpuMemoryUsage=ffxVk316OpticalflowContextGetGpuMemoryUsage
                ffxOpticalflowGetGpuMemoryUsage=ffxVk316OpticalflowGetGpuMemoryUsage
                ffxOpticalflowGetSharedResourceDescriptions=ffxVk316OpticalflowGetSharedResourceDescriptions
                ffxOpticalflowResourceIsNull=ffxVk316OpticalflowResourceIsNull
                ffxSharedContextGetGpuMemoryUsage=ffxVk316SharedContextGetGpuMemoryUsage
                ffxPrintMessage=ffxVk315PrintMessage
                ffxSafeReleaseCopyResource=ffxVk315SafeReleaseCopyResource
                ffxSafeReleasePipeline=ffxVk315SafeReleasePipeline
                ffxSafeReleaseResource=ffxVk315SafeReleaseResource
                ffxSetPrintMessageCallback=ffxVk315SetPrintMessageCallback
            )
            add_library(ps3recomp_ffx_vk STATIC
                "${_ffx_vk_dir}/src/ffx_vk_portable.c"
                "${_ffx_vk_dir}/src/ffx_vk_fsr3_3_1_5_reflection.c"
                "${_ffx_vk_dir}/src/ffx_vk_fsr3_3_1_5_catalog.c"
                "${_ffx_vk_dir}/src/ffx_vk_fsr3_3_1_5_pipeline.c"
                "${_ffx_vk_dir}/src/ffx_vk_fsr3_3_1_5_bundle.c"
                "${_ffx_vk_dir}/src/ffx_vk_fsr3_3_1_5_descriptor.c"
                "${_ffx_vk_dir}/src/ffx_vk_framegeneration_presenter.c"
                "${_ffx_vk_dir}/src/ffx_vk_fsr3_3_1_5_bridge.cpp"
                "${_ffx_vk_dir}/src/ffx_vk_fsr3_3_1_5_upscaler.cpp"
                "${_ffx_vk_dir}/src/ffx_vk_fsr3_3_1_5_shaderblobs_stub.cpp"
                "${_ffx_vk_dir}/src/ffx_vk_fsr3_3_1_6_framegen.cpp"
                "${_ffx_vk_dir}/src/ffx_vk_fsr3_3_1_6_framegen_shaderblobs.cpp"
                "${_ffx_230}/api/internal/ffx_assert.cpp"
                "${_ffx_230}/api/internal/ffx_message.cpp"
                "${_ffx_230}/api/internal/ffx_object_management.cpp"
                "${_ffx_230}/upscalers/fsr3/internal/ffx_fsr3upscaler.cpp"
                "${_ffx_230}/framegeneration/fsr3/internal/ffx_frameinterpolation.cpp"
                "${_ffx_230}/framegeneration/fsr3/internal/ffx_opticalflow.cpp"
            )
            set_target_properties(ps3recomp_ffx_vk PROPERTIES
                CXX_STANDARD 20
                CXX_STANDARD_REQUIRED YES
                CXX_EXTENSIONS NO
            )
            target_include_directories(ps3recomp_ffx_vk
                PUBLIC
                    "${_ffx_vk_dir}/include"
                PRIVATE
                    "${PS3RECOMP_VULKAN_INCLUDE_DIR}"
                    "${_ffx_vk_dir}/generated/ffx-2.3.0/vk/fsr3upscaler-q2-v2"
                    "${_ffx_vk_dir}/generated/ffx-2.3.0/vk/fsr3framegeneration-q2-v1"
                    "${_ffx_230}/api/include"
                    "${_ffx_230}/api/internal"
                    "${_ffx_230}/upscalers/include"
                    "${_ffx_230}/upscalers/fsr3/include"
                    "${_ffx_230}/upscalers/fsr3/internal"
                    "${_ffx_230}/framegeneration/fsr3/include"
                    "${_ffx_230}/framegeneration/fsr3/internal"
            )
            target_compile_definitions(ps3recomp_ffx_vk PRIVATE
                FFX_VK_PORTABLE_NO_WATERMARK=1
                ${_ffx_vk_private_syms}
            )
            target_compile_options(ps3recomp_ffx_vk PRIVATE
                -w
                $<$<CONFIG:Release,RelWithDebInfo>:-O2>
                $<$<COMPILE_LANGUAGE:CXX>:-include>
                $<$<COMPILE_LANGUAGE:CXX>:${_ffx_vk_dir}/src/ffx_1_1_4_portability.hpp>
            )
            target_link_libraries(ps3recomp_ffx_vk PUBLIC ${_ffx_vk_link} PRIVATE m)
            target_link_libraries(ps3recomp_runtime PRIVATE ps3recomp_ffx_vk)
            target_compile_definitions(ps3recomp_runtime PRIVATE PS3RECOMP_VK_FFX=1)
            message(STATUS "  Vulkan FSR:   ON (vendored FSR 3.1.5 upscaler + 3.1.6 FI/OF)")
        else()
            message(STATUS "  Vulkan FSR:   off (libvulkan not found)")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(ps3recomp_runtime PRIVATE
        -Wall -Wextra -Wno-unused-parameter
        -fno-strict-aliasing
        # No spontaneous fused multiply-add. MSVC never contracts, and x86-64
        # without -mfma has nothing to contract into, so every validated
        # result came from separate multiplies and adds; arm64 has FMA in
        # the base ISA and Clang fuses by default there. A PPC fmadd is
        # spelt as one where the lifter means one.
        -ffp-contract=off
        # Explicit, because the per-config defaults cannot be trusted: this
        # tree's caches carry CMAKE_C_FLAGS_RELEASE="" and CMAKE_CXX_FLAGS_RELEASE="",
        # and clang-cl with no /O flag compiles at /Od -- a "Release" runtime
        # was unoptimised without a word. Duplicating a default -O2 is harmless.
        $<$<CONFIG:Release,RelWithDebInfo>:-O3>
        # SSE4.1 (any x86-64 since ~2008): pshufb/blendv for spu_shufb, the
        # hottest SPU op in lifted audio code.
        -msse4.1
    )
    option(PS3RECOMP_X86_AVX2 "Enable x86-64-v3 / znver2 tuning for runtime" ON)
    if(PS3RECOMP_X86_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        target_compile_options(ps3recomp_runtime PRIVATE
            -march=x86-64-v3 -mtune=znver2
        )
    endif()
elseif(MSVC)
    # Real cl.exe -- clang-cl reports its id as Clang and is handled above.
    #
    # This does not currently build. The runtime and the lifted output use GCC/
    # Clang builtins that cl has no equivalent for, and the failure lands as a
    # wall of errors in whichever file the build happens to reach first rather
    # than as anything that names the cause. Say it here instead.
    message(WARNING
        "Building with cl.exe (MSVC). This is not currently supported.\n"
        "  The runtime uses __atomic_* builtins, __int128 and __builtin_bswap*, "
        "and the lifted output uses __int128 for the PPC 64x64 multiplies. "
        "cl has none of them.\n"
        "  Use clang-cl instead -- it consumes the MSVC headers and libraries, "
        "so nothing else about a Visual Studio build changes:\n"
        "    cmake -S . -B build -G Ninja "
        "-DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl\n"
        "  See docs/BUILDING.md. Porting the intrinsics to cl is a welcome "
        "contribution; nothing else stands in the way.")
    target_compile_options(ps3recomp_runtime PRIVATE
        /W3 /wd4100 /wd4244 /wd4267 /wd4996
        /Zc:__cplusplus
        /experimental:c11atomics   # Enable C11 <stdatomic.h> support
    )
    target_compile_definitions(ps3recomp_runtime PRIVATE
        _CRT_SECURE_NO_WARNINGS
    )
endif()

if(NOT MSVC AND NOT WIN32)
    # dladdr/Dl_info (runtime/platform/win32_backtrace.h) are GNU extensions
    # that glibc hides unless _GNU_SOURCE is set. g++ defines it implicitly,
    # so the C++ boot scaffold never noticed; a C file including the same
    # shim does not get it for free.
    target_compile_definitions(ps3recomp_runtime PRIVATE _GNU_SOURCE)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(PS3RECOMP_BUILD_TESTS)
    include(CTest)
    # Pure engine checks use a recording backend and synthetic guest memory,
    # so these exercise draw state even on hosts without a Vulkan GPU.
    add_executable(test_rsx_draw_engine
        libs/video/tests/test_rsx_draw_engine.c
        libs/video/rsx_draw_engine.c
        libs/video/rsx_dispatch.c
        libs/video/rsx_texture_layout.c
        libs/video/rsx_vertex_compact.c
        libs/video/rsx_vertex_fetch.c
        libs/video/rsx_vp_decompiler.c
        libs/video/rsx_fp_decompiler.c
        libs/video/rsx_upscale_common.cpp)
    set_target_properties(test_rsx_draw_engine PROPERTIES LINKER_LANGUAGE CXX)
    target_include_directories(test_rsx_draw_engine PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/video"
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
    if(NOT WIN32)
        target_link_libraries(test_rsx_draw_engine PRIVATE m)
    endif()
    add_test(NAME rsx_draw_engine COMMAND test_rsx_draw_engine)
    set_tests_properties(rsx_draw_engine PROPERTIES LABELS "runtime;video" TIMEOUT 60)
    # sync_stress is written directly against the Win32 threading API
    # (WaitForMultipleObjects, joinable HANDLEs, Interlocked*, CRITICAL_SECTION).
    # Off Windows all of that comes from runtime/platform/win32_compat.h, so
    # the suite runs everywhere -- and is the arm64 memory-model regression
    # net for the lv2 sync primitives.
    add_subdirectory(tests/sync_stress)
    # cellFiber switches contexts with Windows fibers on Windows and ucontext
    # everywhere else. Apple deprecated the ucontext routines and gates both
    # them and the shape of ucontext_t on _XOPEN_SOURCE, so whether that path
    # works on arm64 is a question to answer rather than assume.
    add_subdirectory(tests/fiber_switch)
endif()

# ---------------------------------------------------------------------------
# POSIX host harness
# ---------------------------------------------------------------------------
# Drives the cellGcm -> RSX -> backend bridge with no lifted game, so a backend
# can be brought up and regression-tested without a game binary or a recompiler
# run. Builds on any POSIX host: it picks Metal on Apple and the null backend's
# headless software path everywhere else, so Linux -- which has no real backend
# yet -- can still run the whole guest command stream end to end in CI.
if(UNIX)
    add_executable(ps3recomp_host
        runtime/host/host_posix.c
        runtime/host/host_stubs.c
    )
    target_link_libraries(ps3recomp_host PRIVATE ps3recomp_runtime)
    if(NOT WIN32)
        # glibc keeps the math functions in a separate library; MSVC does not.
        target_link_libraries(ps3recomp_host PRIVATE m)
    endif()
    target_include_directories(ps3recomp_host PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/video"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    # The runtime carries C++ (and, with translation on, spirv-cross), so the
    # C++ driver must do the final link or the standard library goes missing.
    set_target_properties(ps3recomp_host PROPERTIES LINKER_LANGUAGE CXX)

    # Guest shader translation, decompilers -> HLSL -> MSL, with no GPU: the
    # regression net for the Metal backend's shader path on every platform.
    if(PS3RECOMP_MSL_TRANSLATION)
        add_executable(test_shader_msl
            libs/video/tests/test_shader_msl.c
            libs/video/rsx_shader_msl.cpp
            libs/video/rsx_vp_decompiler.c
            libs/video/rsx_fp_decompiler.c
        )
        target_include_directories(test_shader_msl PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/libs/video"
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
        )
        target_link_libraries(test_shader_msl PRIVATE ps3recomp_msl_translation)
    endif()
endif()

# ---------------------------------------------------------------------------
# PPU boot smoke title
# ---------------------------------------------------------------------------
# The one target that LINKS AND RUNS the PPU boot scaffold. runtime/ppu/ is the
# game-agnostic boot path every port is built on, and nothing here has ever
# built it: it #includes the lifter's per-game ppu_recomp.h, so
# tools/check_ppu_scaffold.py compiles it against a stub header and stops
# there. Compiling is not running, and on POSIX the loader, the NID bridge, the
# lv2 syscalls, guest threads, the frame clock and the RSX backend had never
# once run together.
#
# runtime/ppu/tests/smoke/ supplies what was missing: a hand-written program in
# the lifter's ABI, and tools/make_smoke_elf.py writes both the PPU ELF it is
# loaded from and the ppu_recomp.h it compiles against -- the header from
# ppu_lifter.py's own HEADER_PREAMBLE, so the synthetic title cannot drift away
# from the ABI a real one gets.
#
# The HLE NID table is generated exactly as a port generates it, so the smoke
# title's firmware imports resolve through the same bridge a game's do.
if(UNIX)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(NOT Python3_Interpreter_FOUND)
        message(STATUS "  Boot smoke:   no python3; ps3recomp_boot_smoke not built")
    else()
        set(SMOKE_GEN "${CMAKE_CURRENT_BINARY_DIR}/smoke")

        add_custom_command(
            OUTPUT "${SMOKE_GEN}/smoke.elf" "${SMOKE_GEN}/ppu_recomp.h"
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tools/make_smoke_elf.py"
                    --out    "${SMOKE_GEN}/smoke.elf"
                    --header "${SMOKE_GEN}/ppu_recomp.h"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tools/make_smoke_elf.py"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tools/ppu_lifter.py"
            COMMENT "Generating the smoke title's image and ppu_recomp.h")

        add_custom_command(
            OUTPUT "${SMOKE_GEN}/ppu_hle_nids.cpp"
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_hle_nids.py"
                    --out "${SMOKE_GEN}/ppu_hle_nids.cpp" --all
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_hle_nids.py"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tools/nid_database.py"
            COMMENT "Generating the HLE NID -> handler table")

        add_executable(ps3recomp_boot_smoke
            runtime/ppu/tests/boot_main.cpp
            runtime/ppu/tests/smoke/smoke_recomp.c
            runtime/ppu/ppu_loader.cpp
            runtime/ppu/ppu_hle.cpp
            runtime/ppu/ppu_fs.cpp
            runtime/ppu/ppu_sysprx.cpp
            "${SMOKE_GEN}/ppu_hle_nids.cpp"
            "${SMOKE_GEN}/ppu_recomp.h"
        )
        target_include_directories(ps3recomp_boot_smoke PRIVATE
            "${SMOKE_GEN}"                                  # the generated ppu_recomp.h
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/runtime/ppu"
            "${CMAKE_CURRENT_SOURCE_DIR}/runtime/spu"
            "${CMAKE_CURRENT_SOURCE_DIR}/runtime/memory"
            "${CMAKE_CURRENT_SOURCE_DIR}/runtime/syscalls"
            "${CMAKE_CURRENT_SOURCE_DIR}/runtime/platform"
            "${CMAKE_CURRENT_SOURCE_DIR}/libs/video"
        )
        target_link_libraries(ps3recomp_boot_smoke PRIVATE ps3recomp_runtime)
        set_target_properties(ps3recomp_boot_smoke PROPERTIES LINKER_LANGUAGE CXX)

        # Where the runnable image lands, so CI and a developer both say
        #   ./build/ps3recomp_boot_smoke build/smoke/smoke.elf
        message(STATUS "  Boot smoke:   ps3recomp_boot_smoke + smoke/smoke.elf")

        # The same lift again, arranged the way a port's own tree arranges one:
        # the generated ppu_recomp.h and the lifted source in one directory,
        # which is what templates/project wants for -DRECOMP_DIR. The header is
        # generated into the build tree and the source lives in the repository,
        # so a directory holding both has to be made rather than pointed at.
        #
        # This is what lets CI build the project template against a real lift
        # and run it -- the check that a port started from templates/project
        # still boots, on a platform where it can be run.
        set(SMOKE_LIFT "${SMOKE_GEN}/lift")
        set(SMOKE_LIFT_SRC
            "${CMAKE_CURRENT_SOURCE_DIR}/runtime/ppu/tests/smoke/smoke_recomp.c")
        add_custom_command(
            OUTPUT "${SMOKE_LIFT}/ppu_recomp.h" "${SMOKE_LIFT}/smoke_recomp.c"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${SMOKE_LIFT}"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${SMOKE_GEN}/ppu_recomp.h" "${SMOKE_LIFT}/ppu_recomp.h"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${SMOKE_LIFT_SRC}" "${SMOKE_LIFT}/smoke_recomp.c"
            DEPENDS "${SMOKE_GEN}/ppu_recomp.h" "${SMOKE_LIFT_SRC}"
            COMMENT "Staging the smoke title's lift as a RECOMP_DIR")
        add_custom_target(ps3recomp_smoke_lift ALL
            DEPENDS "${SMOKE_LIFT}/ppu_recomp.h" "${SMOKE_LIFT}/smoke_recomp.c")
        message(STATUS "  Smoke lift:   smoke/lift (a RECOMP_DIR for templates/project)")
    endif()
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS ps3recomp_runtime
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ps3emu
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "ps3recomp runtime v${PROJECT_VERSION}")
message(STATUS "  C compiler:   ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "")
