/* Temporal upscaling for the D3D12 backend: NVIDIA DLSS (DLAA + upscaling, DLSS=1) or AMD FSR 3.1 / FSR 4 (FSR=1,
 * through the AMD FSR API: FSR 4 on Radeon RX 7000 / 9000, FSR 3.1 on every other GPU). Both get the same inputs from
 * here; see rsx_dlss.h for what the backend hands over and the env switches.
 *
 * Per presented gameplay frame:
 *   1. copy the finished back buffer into colour_in (DLSS needs a shader-readable input),
 *   2. compute pass: scene depth -> R32F copy for DLSS + per-pixel motion vectors (camera reprojection),
 *   3. DLSS evaluate (DLAA: input size == output size) -> out,
 *   4. copy out back into the back buffer.
 * The port's overlay is drawn after this, so the FPS counter and menus never go through DLSS. */
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef PS3RECOMP_NGX
#include "nvsdk_ngx.h"
#include "nvsdk_ngx_defs.h"
#include "nvsdk_ngx_params.h"
#include "nvsdk_ngx_helpers_d3d.h"
#endif
#ifdef PS3RECOMP_FFX
#include "api/include/ffx_api.h"
#include "api/include/dx12/ffx_api_dx12.h"
#include "upscalers/include/ffx_upscale.h"
#include "framegeneration/include/ffx_framegeneration.h"
#include "framegeneration/include/dx12/ffx_api_framegeneration_dx12.h"
#include <dxgi1_6.h>
#endif
#include "rsx_dlss.h"

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "d3dcompiler.lib")

namespace {

inline const rsx_upscale_cfg& cfg() { return *rsx_upscale_get_cfg(); }
inline const char* name_of(int m) { return rsx_upscale_mode_name(m, cfg().backend); }
#define TAG (rsx_upscale_tag())

/* ---------------------------------------------------------------- NGX state */
ID3D12Device* s_dev = nullptr;
int s_ngx_ok = 0, s_ffx_ok = 0, s_failed = 0;
#ifdef PS3RECOMP_NGX
NVSDK_NGX_Parameter* s_params = nullptr;
NVSDK_NGX_Handle* s_feature = nullptr;
#endif
#ifdef PS3RECOMP_FFX
/* AMD FSR API (amd_fidelityfx_loader_dx12.dll, loaded at run time; it loads amd_fidelityfx_upscaler_dx12.dll) */
PfnFfxCreateContext p_ffxCreate = nullptr;
PfnFfxDestroyContext p_ffxDestroy = nullptr;
PfnFfxQuery p_ffxQuery = nullptr;
PfnFfxDispatch p_ffxDispatch = nullptr;
PfnFfxConfigure p_ffxConfigure = nullptr;
ffxContext s_ffx = nullptr;
/* frame generation: AMD's swap chain proxy (wraps the port's swap chain) + the generation context */
ffxContext s_fg_sc = nullptr, s_fg = nullptr;
IDXGISwapChain4* s_fg_swap = nullptr;
ID3D12Resource* s_hudless = nullptr;        /* this frame's upscaled image before the HUD went on (output size) */
D3D12_RESOURCE_STATES s_hudless_st = D3D12_RESOURCE_STATE_COPY_DEST;
int s_hudless_ok = 0;                       /* s_hudless holds this frame's HUD-less image */
uint64_t s_fg_frame = 0;
float s_cam_pos[3] = {0, 0, 0};             /* the camera's position, summed from its per-frame moves */
const char* s_fg_provider = "?";

/* amd_fidelityfx_loader_dx12.dll next to the exe, else in ..\\..\\fsr-sdk\\Kits\\FidelityFX\\signedbin (FSR_DLL_DIR overrides). That
 * folder also goes on the DLL search path: the loader loads the effect DLLs by name. */
int ffx_load()
{
    static int done = 0, ok = 0;
    if (done) return ok;
    done = 1;
    wchar_t exe_dir[MAX_PATH], dir[MAX_PATH], path[MAX_PATH + 64];
    GetModuleFileNameW(nullptr, exe_dir, MAX_PATH);
    if (wchar_t* sl = wcsrchr(exe_dir, L'\\')) *sl = 0;
    const char* fd = getenv("FSR_DLL_DIR");
    if (fd && *fd) MultiByteToWideChar(CP_ACP, 0, fd, -1, dir, MAX_PATH);
    else _snwprintf(dir, MAX_PATH, L"%ls\\..\\..\\fsr-sdk\\Kits\\FidelityFX\\signedbin", exe_dir);
    HMODULE h = nullptr;
    _snwprintf(path, MAX_PATH + 64, L"%ls\\amd_fidelityfx_loader_dx12.dll", exe_dir);
    if (GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES) h = LoadLibraryW(path);
    if (!h) {
        SetDllDirectoryW(dir);
        _snwprintf(path, MAX_PATH + 64, L"%ls\\amd_fidelityfx_loader_dx12.dll", dir);
        h = LoadLibraryW(path);
    }
    if (h) {
        p_ffxCreate = (PfnFfxCreateContext)(void*)GetProcAddress(h, "ffxCreateContext");
        p_ffxDestroy = (PfnFfxDestroyContext)(void*)GetProcAddress(h, "ffxDestroyContext");
        p_ffxQuery = (PfnFfxQuery)(void*)GetProcAddress(h, "ffxQuery");
        p_ffxDispatch = (PfnFfxDispatch)(void*)GetProcAddress(h, "ffxDispatch");
        p_ffxConfigure = (PfnFfxConfigure)(void*)GetProcAddress(h, "ffxConfigure");
    }
    ok = h && p_ffxCreate && p_ffxDestroy && p_ffxQuery && p_ffxDispatch && p_ffxConfigure;
    if (!ok) fprintf(stderr, "[FSR] amd_fidelityfx_loader_dx12.dll not found or incomplete (next to the exe or in %ls)%c", dir, 10);
    return ok;
}
#endif
uint32_t s_feat_w = 0, s_feat_h = 0;
uint32_t s_render_w = 0, s_render_h = 0;   /* the size the game renders at */
uint32_t s_out_w = 0, s_out_h = 0;         /* the size DLSS writes (the backbuffer); larger when upscaling */
inline bool upscaling() { return s_out_w != s_render_w || s_out_h != s_render_h; }

void fail(const char* what, long long code)
{
    fprintf(stderr, "%s %s failed (0x%llX) -- %s off for this run%c", TAG, what, (unsigned long long)code, cfg().backend ? "FSR" : "DLSS", 10);
    s_failed = 1;
    rsx_upscale_set_active(0);
}

/* ---------------------------------------------------------------- GPU resources */
/* s_in = DLSS's colour input: the scene right after the composite (no HUD), or failing that the finished frame.
 * s_hud = the finished frame (with the HUD), used to put the HUD back over DLSS's output unchanged. */
ID3D12Resource *s_in = nullptr, *s_out = nullptr, *s_mv = nullptr, *s_dz = nullptr, *s_hud = nullptr;
D3D12_RESOURCE_STATES s_in_st, s_out_st, s_mv_st, s_dz_st, s_hud_st;
ID3D12Resource* s_pdz = nullptr;           /* last frame's s_dz: checks the camera's motion vectors */
ID3D12Resource* s_prev = nullptr;          /* DLSS_DEBUG=warp/nowarp only: last frame's s_in */
D3D12_RESOURCE_STATES s_prev_st, s_pdz_st;
ID3D12Resource* s_zcopy = nullptr;       /* the scene depth, copied right after the last scene draw */
D3D12_RESOURCE_STATES s_zcopy_st;
uint32_t s_zc_w = 0, s_zc_h = 0;
int s_zsnap = 0;                          /* scene depth taken since the last DLAA frame */
int s_csnap = 0;                          /* pre-HUD scene colour taken (into s_in) since the last DLAA frame */
uint32_t s_res_w = 0, s_res_h = 0;
ID3D12DescriptorHeap* s_heap = nullptr;   /* 0 depth SRV, 1-3 UAV mv/dz/out, 4-7 SRV scene/hud/prev/prev depth */
UINT s_heap_inc = 0;
ID3D12RootSignature* s_rs = nullptr;
ID3D12PipelineState *s_pso_mv = nullptr, *s_pso_dbg = nullptr, *s_pso_merge = nullptr, *s_pso_blit = nullptr;

void release(ID3D12Resource*& r) { if (r) { r->Release(); r = nullptr; } }

ID3D12Resource* make_tex(DXGI_FORMAT fmt, uint32_t w, uint32_t h, D3D12_RESOURCE_FLAGS fl, D3D12_RESOURCE_STATES st)
{
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; rd.Width = w; rd.Height = h;
    rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.Format = fmt; rd.SampleDesc.Count = 1; rd.Flags = fl;
    ID3D12Resource* r = nullptr;
    if (FAILED(s_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
    return r;
}

/* Shared by all passes. Constants: ic = inverse of this frame's camera, pv = last frame's camera x the camera move;
 * dims = render size, this frame's jitter (px); opt.x = 0 no vectors, 1 camera only, 2 camera + fallback;
 * opt.zw = last frame's jitter; outd.xy = output size (= dims.xy unless upscaling). pz = last frame's depth. */
#define DLSS_HLSL_COMMON \
    "cbuffer K : register(b0) { float4 ic0, ic1, ic2, ic3; float4 pv0, pv1, pv2, pv3; float4 dims; float4 opt; float4 outd; };\n" \
    "Texture2D<float> pz : register(t4);\n" \
    /* relative distance error of a predicted last-frame depth zq against last frame's depth at pixel position pos, \
       best of the 2x2 texels around it (1 - z is proportional to 1 / distance here): .x its size (-1 = off-screen  \
       last frame), .y signed: > 0 = last frame had something nearer there (this point was hidden behind it) */ \
    "float2 zerr(float2 pos, float zq) {\n" \
    "  if (pos.x < 0 || pos.y < 0 || pos.x >= dims.x || pos.y >= dims.y) return float2(-1, 0);\n" \
    "  int2 i = (int2)floor(pos - 0.5); float2 b = float2(1e9, 0);\n" \
    "  [unroll] for (int k = 0; k < 4; k++) {\n" \
    "    float zp = pz.Load(int3(clamp(i + int2(k & 1, k >> 1), int2(0, 0), int2(dims.xy) - 1), 0));\n" \
    "    float s = (zq - zp) / max(1.0 - zp, 1e-7);\n" \
    "    if (abs(s) < b.x) b = float2(abs(s), s);\n" \
    "  }\n" \
    "  return b;\n" \
    "}\n" \
    /* camera reprojection of the pixel id (depth z): .xy motion (px, last frame - this frame, unjittered), .z the  \
       depth it had last frame, .w 0 if it was behind the camera. The sample at a pixel centre was rasterized with  \
       the jitter added, so its unjittered position is centre - jitter. */ \
    "float4 reproj(uint2 id, float z) {\n" \
    "  float2 pc = (float2(id) + 0.5) - dims.zw;\n" \
    "  float4 p = float4(pc.x / dims.x * 2.0 - 1.0, 1.0 - pc.y / dims.y * 2.0, z, 1.0);\n" \
    "  float4 w = float4(dot(ic0, p), dot(ic1, p), dot(ic2, p), dot(ic3, p));\n" \
    "  float4 q = float4(dot(pv0, w), dot(pv1, w), dot(pv2, w), dot(pv3, w));\n" \
    "  if (q.w <= 1e-6) return float4(0, 0, z, 0);\n" \
    "  float2 n = q.xy / q.w;\n" \
    "  return float4(float2((n.x * 0.5 + 0.5) * dims.x, (0.5 - n.y * 0.5) * dims.y) - pc, q.z / q.w, 1);\n" \
    "}\n" \
    /* the vector to use: the camera's, unless this point would have been in front of what last frame shows there  \
       (so it moved) while 'moved with the camera' (zero motion, same depth) fits -- the player character the camera \
       follows. Where last frame had something nearer in front, the point is newly visible and the camera's vector \
       is right. .z: 0 camera (fits), 1 zero motion, 2 camera (nothing fits: animated/moving), 3 camera (newly visible) */ \
    "float3 pick(uint2 id, float z) {\n" \
    "  float4 r = reproj(id, z);\n" \
    "  if (r.w == 0) return float3(0, 0, 2);\n" \
    "  if (z >= 1.0 || opt.x < 1.5) return float3(r.xy, 0);\n" \
    "  float2 pc = (float2(id) + 0.5) - dims.zw;\n" \
    "  float2 ec = zerr(pc + r.xy + opt.zw, r.z);\n" \
    "  if (ec.x < 0.02) return float3(r.xy, 0);\n" \
    "  if (ec.y > 0.0) return float3(r.xy, 3);\n" \
    "  float2 e0 = zerr(pc + opt.zw, z);\n" \
    "  if (e0.x >= 0.0 && e0.x < 0.005 && e0.x * 4.0 < ec.x) return float3(0, 0, 1);\n" \
    "  return float3(r.xy, 2);\n" \
    "}\n"

const char* kCS =
    DLSS_HLSL_COMMON
    "Texture2D<float> zt : register(t0);\n"
    "RWTexture2D<float2> mv : register(u0);\n"
    "RWTexture2D<float> dz : register(u1);\n"
    "[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID) {\n"
    "  if (id.x >= (uint)dims.x || id.y >= (uint)dims.y) return;\n"
    "  float z = zt.Load(int3(id.xy, 0));\n"
    "  dz[id.xy] = z;\n"
    "  mv[id.xy] = opt.x > 0.5 ? pick(id.xy, z).xy : float2(0, 0);\n"
    "}\n";

/* DLSS_DEBUG=mv: motion vectors as colour (red = x, green = y, 8 px = full); DLSS_DEBUG=depth: the depth DLSS
 * gets (bright = near); DLSS_DEBUG=warp: |this frame - last frame fetched along the motion vectors| x4 (black where
 * the vectors are right), nowarp: the same without the vectors, for comparison; zcheck: which vector each pixel got
 * (green = the camera's, consistent with last frame's depth; cyan = zero motion, an object moving with the camera;
 * red = nothing fits, e.g. an animated character; orange = newly visible; blue = sky). Written over DLSS's output,
 * which is the output size: each output pixel shows the render pixel under it. */
const char* kCSDbg =
    DLSS_HLSL_COMMON
    "RWTexture2D<float2> mv : register(u0);\n"
    "RWTexture2D<float> dz : register(u1);\n"
    "RWTexture2D<unorm float4> o : register(u2);\n"
    "Texture2D<float4> sc : register(t1);\n"
    "Texture2D<float4> pr : register(t3);\n"
    "float4 ld(int2 i) { return pr.Load(int3(clamp(i, int2(0, 0), int2(dims.xy) - 1), 0)); }\n"
    "[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID) {\n"
    "  if (id.x >= (uint)outd.x || id.y >= (uint)outd.y) return;\n"
    "  const uint2 r = min(uint2(float2(id.xy) * dims.xy / outd.xy), uint2(dims.xy) - 1);\n"
    "  if (opt.y > 4.5) {\n"
    "    float z = dz[r];\n"
    "    if (z >= 1.0) { o[id.xy] = float4(0, 0, 0.4, 1); return; }\n"
    "    const float3 cols[4] = { float3(0, 1, 0), float3(0, 0.8, 1), float3(1, 0, 0), float3(1, 0.6, 0) };\n"
    "    float3 c = opt.x > 0.5 ? cols[(int)pick(r, z).z] : float3(1, 0, 1);\n"
    "    float lum = dot(sc.Load(int3(r, 0)).rgb, float3(0.3, 0.5, 0.2));\n"
    "    o[id.xy] = float4(c * (0.35 + 0.65 * lum), 1);\n"
    "    return;\n"
    "  }\n"
    "  if (opt.y > 2.5) {\n"
    /* this pixel's sample, unjittered, moved by its vector, re-jittered with last frame's jitter (opt.zw) */
    "    float2 m = opt.y > 3.5 ? float2(0, 0) : mv[r];\n"
    "    float2 q = float2(r) + m - dims.zw + opt.zw;\n"
    "    int2 i = (int2)floor(q); float2 f = q - floor(q);\n"
    "    float4 p = lerp(lerp(ld(i), ld(i + int2(1, 0)), f.x), lerp(ld(i + int2(0, 1)), ld(i + int2(1, 1)), f.x), f.y);\n"
    "    o[id.xy] = float4(saturate(abs(sc.Load(int3(r, 0)).rgb - p.rgb) * 4.0), 1);\n"
    "    return;\n"
    "  }\n"
    "  if (opt.y > 1.5) { float z = dz[r]; float v = saturate((1.0 - z) * 200.0); o[id.xy] = float4(v, v, z >= 1.0 ? 1 : v, 1); return; }\n"
    "  float2 m = mv[r];\n"
    "  o[id.xy] = float4(saturate(0.5 + m.x / 16.0), saturate(0.5 + m.y / 16.0), saturate(length(m) / 32.0), 1);\n"
    "}\n";

/* Puts the HUD back: wherever the finished frame differs from the pre-HUD scene, the HUD (or a dialog) drew there,
 * so that pixel is taken from the finished frame instead of DLSS's output. Both are render size and DLSS's output is
 * the output size: each output pixel blends the 2x2 render pixels under it, so when upscaling the HUD is enlarged
 * smoothly and its edges fade into the DLSS image. At the same size this is exactly a per-pixel replace. */
const char* kCSMerge =
    DLSS_HLSL_COMMON
    "Texture2D<float4> sc : register(t1);\n"
    "Texture2D<float4> hu : register(t2);\n"
    "RWTexture2D<unorm float4> o : register(u2);\n"
    "[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID) {\n"
    "  if (id.x >= (uint)outd.x || id.y >= (uint)outd.y) return;\n"
    "  float2 p = (float2(id.xy) + 0.5) * dims.xy / outd.xy - 0.5;\n"
    "  int2 i = (int2)floor(p); float2 f = p - floor(p);\n"
    "  float a = 0.0; float3 c = 0.0;\n"
    "  [unroll] for (int k = 0; k < 4; k++) {\n"
    "    int2 t = clamp(i + int2(k & 1, k >> 1), int2(0, 0), int2(dims.xy) - 1);\n"
    "    float w = ((k & 1) ? f.x : 1.0 - f.x) * ((k >> 1) ? f.y : 1.0 - f.y);\n"
    "    float4 s = sc.Load(int3(t, 0)), h = hu.Load(int3(t, 0));\n"
    "    a += any(abs(s.rgb - h.rgb) > 0.5 / 255.0) ? w : 0.0;\n"
    "    c += w * h.rgb;\n"
    "  }\n"
    "  if (a > 0.0) o[id.xy] = float4(a >= 1.0 ? c : lerp(o[id.xy].rgb, c, a), 1);\n"
    "}\n";

/* Upscaling, on frames DLSS does not run (menus, loading screens, videos): the finished frame (render size) enlarged
 * to the output size, bilinear. */
const char* kCSBlit =
    DLSS_HLSL_COMMON
    "Texture2D<float4> hu : register(t2);\n"
    "RWTexture2D<unorm float4> o : register(u2);\n"
    "[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID) {\n"
    "  if (id.x >= (uint)outd.x || id.y >= (uint)outd.y) return;\n"
    "  float2 p = (float2(id.xy) + 0.5) * dims.xy / outd.xy - 0.5;\n"
    "  int2 i = (int2)floor(p); float2 f = p - floor(p);\n"
    "  float3 c = 0.0;\n"
    "  [unroll] for (int k = 0; k < 4; k++) {\n"
    "    int2 t = clamp(i + int2(k & 1, k >> 1), int2(0, 0), int2(dims.xy) - 1);\n"
    "    c += ((k & 1) ? f.x : 1.0 - f.x) * ((k >> 1) ? f.y : 1.0 - f.y) * hu.Load(int3(t, 0)).rgb;\n"
    "  }\n"
    "  o[id.xy] = float4(c, 1);\n"
    "}\n";

int build_pipeline()
{
    if (s_pso_mv) return 1;
    /* root: [0] 44 constants b0, [1] SRV t0, [2] UAVs u0..u2, [3] SRVs t1..t4 */
    D3D12_DESCRIPTOR_RANGE r0 = {}; r0.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; r0.NumDescriptors = 1;
    D3D12_DESCRIPTOR_RANGE r1 = {}; r1.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; r1.NumDescriptors = 3;
    D3D12_DESCRIPTOR_RANGE r2 = {}; r2.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; r2.NumDescriptors = 4; r2.BaseShaderRegister = 1;
    D3D12_ROOT_PARAMETER p[4] = {};
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; p[0].Constants.Num32BitValues = 44;
    p[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[1].DescriptorTable.NumDescriptorRanges = 1;
    p[1].DescriptorTable.pDescriptorRanges = &r0; p[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[2].DescriptorTable.NumDescriptorRanges = 1;
    p[2].DescriptorTable.pDescriptorRanges = &r1; p[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[3].DescriptorTable.NumDescriptorRanges = 1;
    p[3].DescriptorTable.pDescriptorRanges = &r2; p[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd = {}; rsd.NumParameters = 4; rsd.pParameters = p;
    ID3DBlob *sig = nullptr, *err = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err))) {
        if (err) { fprintf(stderr, "[DLSS] root signature: %s%c", (const char*)err->GetBufferPointer(), 10); err->Release(); }
        return 0;
    }
    HRESULT hr = s_dev->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&s_rs));
    sig->Release();
    if (FAILED(hr)) return 0;
    const char* srcs[4] = { kCS, kCSDbg, kCSMerge, kCSBlit };
    const char* names[4] = { "dlss_mv", "dlss_dbg", "dlss_merge", "dlss_blit" };
    ID3D12PipelineState** outs[4] = { &s_pso_mv, &s_pso_dbg, &s_pso_merge, &s_pso_blit };
    for (int k = 0; k < 4; k++) {
        ID3DBlob *cs = nullptr; err = nullptr;
        if (FAILED(D3DCompile(srcs[k], strlen(srcs[k]), names[k], nullptr, nullptr, "main", "cs_5_0",
                              D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &cs, &err))) {
            if (err) { fprintf(stderr, "[DLSS] shader: %s%c", (const char*)err->GetBufferPointer(), 10); err->Release(); }
            return 0;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = s_rs; pd.CS.pShaderBytecode = cs->GetBufferPointer(); pd.CS.BytecodeLength = cs->GetBufferSize();
        hr = s_dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(outs[k]));
        cs->Release();
        if (FAILED(hr)) return 0;
    }
    D3D12_DESCRIPTOR_HEAP_DESC hd = {}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 8;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(s_dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&s_heap)))) return 0;
    s_heap_inc = s_dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return 1;
}

/* Buffers at the render size (DLSS's inputs, the HUD snapshots, last frame's depth) and DLSS's output at the
 * output size (the backbuffer). */
uint32_t s_res_ow = 0, s_res_oh = 0;
int ensure_resources()
{
    const uint32_t w = s_render_w, h = s_render_h, ow = s_out_w, oh = s_out_h;
    if (!w || !h || !ow || !oh) return 0;
    if (s_in && s_res_w == w && s_res_h == h && s_res_ow == ow && s_res_oh == oh) return 1;
    release(s_in); release(s_out); release(s_mv); release(s_dz); release(s_hud); release(s_prev); release(s_pdz);
    s_csnap = 0;
    if (cfg().debug_mv == 3 || cfg().debug_mv == 4) {
        s_prev = make_tex(DXGI_FORMAT_R8G8B8A8_UNORM, w, h, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        s_prev_st = D3D12_RESOURCE_STATE_COPY_DEST;
    }
    s_pdz = make_tex(DXGI_FORMAT_R32_FLOAT, w, h, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    s_pdz_st = D3D12_RESOURCE_STATE_COPY_DEST;
    s_in  = make_tex(DXGI_FORMAT_R8G8B8A8_UNORM, w, h, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    s_hud = make_tex(DXGI_FORMAT_R8G8B8A8_UNORM, w, h, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    s_out = make_tex(DXGI_FORMAT_R8G8B8A8_UNORM, ow, oh, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s_mv  = make_tex(DXGI_FORMAT_R16G16_FLOAT, w, h, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s_dz  = make_tex(DXGI_FORMAT_R32_FLOAT, w, h, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!s_in || !s_out || !s_mv || !s_dz || !s_hud || !s_pdz) return 0;
    s_in_st = s_hud_st = D3D12_RESOURCE_STATE_COPY_DEST; s_out_st = s_mv_st = s_dz_st = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    s_res_w = w; s_res_h = h; s_res_ow = ow; s_res_oh = oh;
    {   /* slots 4, 5: the pre-HUD scene and the finished frame, read by the merge */
        D3D12_CPU_DESCRIPTOR_HANDLE c = s_heap->GetCPUDescriptorHandleForHeapStart();
        D3D12_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_R8G8B8A8_UNORM; sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sv.Texture2D.MipLevels = 1;
        c.ptr += 4 * (SIZE_T)s_heap_inc; s_dev->CreateShaderResourceView(s_in, &sv, c);
        c.ptr += s_heap_inc;             s_dev->CreateShaderResourceView(s_hud, &sv, c);
        c.ptr += s_heap_inc;             s_dev->CreateShaderResourceView(s_prev, &sv, c);   /* null views when off */
        sv.Format = DXGI_FORMAT_R32_FLOAT;
        c.ptr += s_heap_inc;             s_dev->CreateShaderResourceView(s_pdz, &sv, c);
    }
    /* fixed UAV descriptors: slots 1..3 = mv, dz, out (slot 0 = the scene depth SRV, rewritten per frame) */
    D3D12_CPU_DESCRIPTOR_HANDLE c = s_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_UNORDERED_ACCESS_VIEW_DESC u = {}; u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    u.Format = DXGI_FORMAT_R16G16_FLOAT; c.ptr += s_heap_inc; s_dev->CreateUnorderedAccessView(s_mv, nullptr, &u, c);
    u.Format = DXGI_FORMAT_R32_FLOAT;    c.ptr += s_heap_inc; s_dev->CreateUnorderedAccessView(s_dz, nullptr, &u, c);
    u.Format = DXGI_FORMAT_R8G8B8A8_UNORM; c.ptr += s_heap_inc; s_dev->CreateUnorderedAccessView(s_out, nullptr, &u, c);
    fprintf(stderr, "[DLSS] buffers ready: render %ux%u, output %ux%u%c", w, h, ow, oh, 10);
    return 1;
}

void barrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* r, D3D12_RESOURCE_STATES& cur, D3D12_RESOURCE_STATES to)
{
    if (cur == to) return;
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = cur; b.Transition.StateAfter = to;
    cl->ResourceBarrier(1, &b);
    cur = to;
}
void uav_barrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* r)
{
    D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; b.UAV.pResource = r;
    cl->ResourceBarrier(1, &b);
}

/* The upscaler: render size in, output size out. Quality mode = the user's DLSS_MODE / FSR_MODE when upscaling, else
 * DLAA / Native AA (also when upscaling was asked for but the render size ended up equal to the output, e.g. clamped
 * at 720p). */
int s_feat_mode = 0;
#ifdef PS3RECOMP_FFX
void ffx_message(uint32_t type, const wchar_t* m)
{
    static int n = 0;
    if (n++ < 40) fprintf(stderr, "[FSR] %s: %ls%c", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", m, 10);
}
const char* s_ffx_provider = "?";
int ffx_create()
{
    const int mode = upscaling() ? (cfg().up ? cfg().up : 1) : 0;
    if (s_ffx && s_feat_w == s_render_w && s_feat_h == s_render_h && s_feat_mode == mode) return 1;
    if (s_ffx) { p_ffxDestroy(&s_ffx, nullptr); s_ffx = nullptr; }
    ffxCreateBackendDX12Desc be = {};
    be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12; be.device = s_dev;
    ffxCreateContextDescUpscaleVersion ver = {};
    ver.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION; ver.version = FFX_UPSCALER_VERSION;
    ver.header.pNext = &be.header;
    ffxCreateContextDescUpscale cd = {};
    cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE; cd.header.pNext = &ver.header;
    /* the game's colour is the finished, tone-mapped 8-bit image; depth is standard (0 near, 1 far) */
    cd.flags = FFX_UPSCALE_ENABLE_AUTO_EXPOSURE | FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE;
    cd.maxRenderSize.width = s_render_w; cd.maxRenderSize.height = s_render_h;
    cd.maxUpscaleSize.width = s_out_w; cd.maxUpscaleSize.height = s_out_h;
    cd.fpMessage = ffx_message;
    /* FSR_VERSION: a specific provider (by name, e.g. "3.1"), otherwise the loader's best one for this GPU */
    ffxOverrideVersion ov = {};
    {
        uint64_t cnt = 0;
        ffxQueryDescGetVersions q = {};
        q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS; q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        q.device = s_dev; q.outputCount = &cnt;
        if (p_ffxQuery(nullptr, &q.header) == FFX_API_RETURN_OK && cnt > 0 && cnt < 32) {
            uint64_t ids[32] = {0}; const char* names[32] = {0};
            q.versionIds = ids; q.versionNames = names;
            if (p_ffxQuery(nullptr, &q.header) == FFX_API_RETURN_OK) {
                static int logged = 0;
                for (uint64_t i = 0; i < cnt && !logged; i++) fprintf(stderr, "[FSR] provider %llu: %s%c", (unsigned long long)i, names[i] ? names[i] : "?", 10);
                logged = 1;
                if (cfg().fsr_ver[0])
                    for (uint64_t i = 0; i < cnt; i++)
                        if (names[i] && strstr(names[i], cfg().fsr_ver)) {
                            ov.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION; ov.versionId = ids[i];
                            ov.header.pNext = cd.header.pNext; cd.header.pNext = &ov.header;
                            fprintf(stderr, "[FSR] FSR_VERSION=%s: using %s%c", cfg().fsr_ver, names[i], 10);
                            break;
                        }
            }
        }
    }
    const ffxReturnCode_t r = p_ffxCreate(&s_ffx, &cd.header, nullptr);
    if (r != FFX_API_RETURN_OK || !s_ffx) { fail("FSR context creation", r); s_ffx = nullptr; return 0; }
    ffxQueryGetProviderVersion pv = {};
    pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    if (p_ffxQuery(&s_ffx, &pv.header) == FFX_API_RETURN_OK && pv.versionName) s_ffx_provider = pv.versionName;
    s_feat_w = s_render_w; s_feat_h = s_render_h; s_feat_mode = mode;
    fprintf(stderr, "[FSR] %s %ux%u -> %ux%u created (provider %s, sharpening %.2f)%c", rsx_upscale_mode_name(mode, 1), s_render_w, s_render_h,
            s_out_w, s_out_h, s_ffx_provider, cfg().fsr_sharp, 10);
    return 1;
}
#endif
#ifdef PS3RECOMP_NGX
int ngx_create(ID3D12GraphicsCommandList* cl)
{
    const int mode = upscaling() ? (cfg().up ? cfg().up : 1) : 0;
    if (s_feature && s_feat_w == s_render_w && s_feat_h == s_render_h && s_feat_mode == mode) return 1;
    if (s_feature) { NVSDK_NGX_D3D12_ReleaseFeature(s_feature); s_feature = nullptr; }
    static const NVSDK_NGX_PerfQuality_Value pq[5] = { NVSDK_NGX_PerfQuality_Value_DLAA, NVSDK_NGX_PerfQuality_Value_MaxQuality,
        NVSDK_NGX_PerfQuality_Value_Balanced, NVSDK_NGX_PerfQuality_Value_MaxPerf, NVSDK_NGX_PerfQuality_Value_UltraPerformance };
    NVSDK_NGX_DLSS_Create_Params cp = {};
    cp.Feature.InWidth = s_render_w; cp.Feature.InHeight = s_render_h;
    cp.Feature.InTargetWidth = s_out_w; cp.Feature.InTargetHeight = s_out_h;
    cp.Feature.InPerfQualityValue = pq[mode];
    cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    if (cfg().preset) {   /* DLSS_PRESET: the same model for whichever mode is in use */
        NVSDK_NGX_Parameter_SetUI(s_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, (unsigned)cfg().preset);
        NVSDK_NGX_Parameter_SetUI(s_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, (unsigned)cfg().preset);
        NVSDK_NGX_Parameter_SetUI(s_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, (unsigned)cfg().preset);
        NVSDK_NGX_Parameter_SetUI(s_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, (unsigned)cfg().preset);
        NVSDK_NGX_Parameter_SetUI(s_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, (unsigned)cfg().preset);
    }
    NVSDK_NGX_Result r = NGX_D3D12_CREATE_DLSS_EXT(cl, 1, 1, &s_feature, s_params, &cp);
    if (NVSDK_NGX_FAILED(r)) { fail("DLSS feature creation", r); s_feature = nullptr; return 0; }
    s_feat_w = s_render_w; s_feat_h = s_render_h; s_feat_mode = mode;
    fprintf(stderr, "[DLSS] %s %ux%u -> %ux%u created (preset %s)%c", rsx_upscale_mode_name(mode, 0), s_render_w, s_render_h,
            s_out_w, s_out_h, cfg().preset ? "set" : "default", 10);
    return 1;
}
#endif
int create_feature(ID3D12GraphicsCommandList* cl)
{
#ifdef PS3RECOMP_FFX
    if (cfg().backend) return ffx_create();
#endif
#ifdef PS3RECOMP_NGX
    if (!cfg().backend) return ngx_create(cl);
#endif
    (void)cl;
    return 0;
}

} // namespace

/* ======================================================================= C API */
extern "C" {

float rsx_dlss_render_ratio(void) { return rsx_upscale_render_ratio(); }

void rsx_dlss_device_init(ID3D12Device* dev, uint32_t render_w, uint32_t render_h, uint32_t out_w, uint32_t out_h)
{
    const rsx_upscale_cfg& g = cfg();
    s_render_w = render_w; s_render_h = render_h;
    s_out_w = out_w ? out_w : render_w; s_out_h = out_h ? out_h : render_h;
    rsx_upscale_set_sizes(s_render_w, s_render_h, s_out_w, s_out_h);
    if (g.probe) fprintf(stderr, "[DLSS] probe armed: frame after %d gameplay frames%c", g.probe, 10);
    if (!g.mode || !dev || s_ngx_ok || s_ffx_ok || s_failed) return;
    s_dev = dev;
    if (g.backend) {
#ifdef PS3RECOMP_FFX
        if (!ffx_load()) { fprintf(stderr, "[FSR] -- off%c", 10); s_failed = 1; return; }
        s_ffx_ok = 1;
        rsx_upscale_set_active(1);
        fprintf(stderr, "[FSR] ready: %s, render %ux%u -> output %ux%u, jitter %s (%u phases), motion vectors %s, mip bias %+.2f%c",
                rsx_upscale_mode_name(upscaling() ? (g.up ? g.up : 1) : 0, 1), s_render_w, s_render_h, s_out_w, s_out_h,
                g.jitter ? "on" : "off", rsx_upscale_jitter_phases(), g.mv ? "on" : "off", rsx_upscale_mip_bias(), 10);
#else
        fprintf(stderr, "[FSR] FSR=1, but this build has no AMD FSR (no fsr-sdk folder at build time) -- off%c", 10);
        s_failed = 1;
#endif
        return;
    }
#ifndef PS3RECOMP_NGX
    fprintf(stderr, "[DLSS] DLSS=1, but this build has no NVIDIA DLSS (no dlss-sdk folder at build time) -- off%c", 10);
    s_failed = 1;
    return;
#else
    /* NGX logs + scratch next to the game (cwd = infamous-recomp); the DLSS model DLL is looked up next to the
     * exe and in ..\..\dlss-sdk\lib\Windows_x86_64\rel (DLSS_DLL_DIR overrides). */
    static wchar_t data_dir[MAX_PATH], dll_dir[MAX_PATH], exe_dir[MAX_PATH];
    GetCurrentDirectoryW(MAX_PATH, data_dir);
    GetModuleFileNameW(nullptr, exe_dir, MAX_PATH);
    if (wchar_t* s = wcsrchr(exe_dir, L'\\')) *s = 0;
    const char* dd = getenv("DLSS_DLL_DIR");
    if (dd && *dd) MultiByteToWideChar(CP_ACP, 0, dd, -1, dll_dir, MAX_PATH);
    else _snwprintf(dll_dir, MAX_PATH, L"%ls\\..\\..\\dlss-sdk\\lib\\Windows_x86_64\\rel", exe_dir);
    static const wchar_t* paths[2]; paths[0] = dll_dir; paths[1] = exe_dir;
    NVSDK_NGX_FeatureCommonInfo fi = {};
    fi.PathListInfo.Path = paths; fi.PathListInfo.Length = 2;
    fi.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;
    NVSDK_NGX_Result r = NVSDK_NGX_D3D12_Init_with_ProjectID("7c3a1e52-2f4b-4d6e-9a8c-1b5d0f2e6a93",
        NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", data_dir, dev, &fi, NVSDK_NGX_Version_API);
    if (NVSDK_NGX_FAILED(r)) { fail("NGX init", r); return; }
    r = NVSDK_NGX_D3D12_GetCapabilityParameters(&s_params);
    if (NVSDK_NGX_FAILED(r) || !s_params) { fail("NGX capability query", r); return; }
    int avail = 0, need_drv = 0; unsigned drv_maj = 0, drv_min = 0;
    NVSDK_NGX_Parameter_GetI(s_params, NVSDK_NGX_Parameter_SuperSampling_Available, &avail);
    NVSDK_NGX_Parameter_GetI(s_params, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &need_drv);
    NVSDK_NGX_Parameter_GetUI(s_params, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &drv_maj);
    NVSDK_NGX_Parameter_GetUI(s_params, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &drv_min);
    if (!avail) {
        fprintf(stderr, "[DLSS] DLSS is not available on this system%s (min driver %u.%u) -- off%c",
                need_drv ? ": the graphics driver is too old" : "", drv_maj, drv_min, 10);
        s_failed = 1; return;
    }
    s_ngx_ok = 1;
    rsx_upscale_set_active(1);
    fprintf(stderr, "[DLSS] NGX ready: %s, render %ux%u -> output %ux%u, jitter %s (%u phases), motion vectors %s, mip bias %+.2f%c",
            rsx_upscale_mode_name(upscaling() ? (g.up ? g.up : 1) : 0, 0), s_render_w, s_render_h, s_out_w, s_out_h,
            g.jitter ? "on" : "off", rsx_upscale_jitter_phases(), g.mv ? "on" : "off", rsx_upscale_mip_bias(), 10);
#endif
}

int rsx_dlss_active(void) { return (s_ngx_ok || s_ffx_ok) && !s_failed; }
int rsx_dlss_upscaling(void) { return rsx_dlss_active() && upscaling(); }
int rsx_dlss_probe_on(void) { return rsx_upscale_probe_on(); }
float rsx_dlss_mip_bias(void) { return rsx_dlss_active() ? rsx_upscale_mip_bias() : 0.0f; }

int rsx_dlss_draw_qualifies(int depth_test, uint32_t zeta_off, uint32_t zw, uint32_t zh, uint32_t rt_off,
                            uint32_t rt_w, uint32_t rt_h, uint32_t vp_w, uint32_t vp_h, uint32_t guest_w, uint32_t guest_h)
{
    return rsx_upscale_draw_qualifies(depth_test, zeta_off, zw, zh, rt_off, rt_w, rt_h, vp_w, vp_h, guest_w, guest_h);
}

void rsx_dlss_jitter_ndc(float* dx, float* dy) { rsx_upscale_jitter_ndc(dx, dy); }

void rsx_dlss_note_draw(const rsx_dlss_draw_info* di, int qualifies) { (void)rsx_upscale_note_draw(di, qualifies); }

int rsx_dlss_hooks_on(void) { return rsx_upscale_hooks_on(); }

void rsx_dlss_handoff(int present) { rsx_upscale_handoff(present); }

uint32_t rsx_dlss_exec_zeta(void) { return rsx_upscale_exec()->zeta; }
uint32_t rsx_dlss_exec_last_scene_draw(void) { return rsx_dlss_active() ? rsx_upscale_exec()->last_q : 0xFFFFFFFFu; }
uint32_t rsx_dlss_exec_composite_draw(void) { return rsx_dlss_active() ? rsx_upscale_exec()->composite : 0xFFFFFFFFu; }

/* The render-size colour snapshots must match the render size DLSS was set up for. */
static int capture_ok(ID3D12Resource* src, const char* what)
{
    const D3D12_RESOURCE_DESC d = src->GetDesc();
    if (d.Format != DXGI_FORMAT_R8G8B8A8_UNORM) {
        static int n = 0; if (n++ < 2) fprintf(stderr, "[DLSS] %s format %d is not RGBA8 -- not used%c", what, (int)d.Format, 10);
        return 0;
    }
    if ((uint32_t)d.Width != s_render_w || d.Height != s_render_h) {
        static int n = 0; if (n++ < 2) fprintf(stderr, "[DLSS] %s is %ux%u, not the render size %ux%u -- not used%c", what,
                                               (uint32_t)d.Width, d.Height, s_render_w, s_render_h, 10);
        return 0;
    }
    return build_pipeline() && ensure_resources();
}

void rsx_dlss_capture_color(ID3D12GraphicsCommandList* cl, ID3D12Resource* src, uint32_t* state)
{
    if (!rsx_dlss_active() || !cl || !src || !state) return;
    if (!capture_ok(src, "display buffer")) return;
    D3D12_RESOURCE_STATES st = (D3D12_RESOURCE_STATES)*state;
    const D3D12_RESOURCE_STATES orig = st;
    barrier(cl, src, st, D3D12_RESOURCE_STATE_COPY_SOURCE);
    barrier(cl, s_in, s_in_st, D3D12_RESOURCE_STATE_COPY_DEST);
    cl->CopyResource(s_in, src);
    barrier(cl, src, st, orig);
    *state = (uint32_t)st;
    s_csnap = 1;
}

static int s_fsnap = 0;   /* upscaling: the finished frame (render size, with the HUD) taken into s_hud for this present */
void rsx_dlss_capture_finished(ID3D12GraphicsCommandList* cl, ID3D12Resource* src)
{
    if (!rsx_dlss_upscaling() || !cl || !src) return;
    if (!capture_ok(src, "finished frame")) return;
    barrier(cl, s_hud, s_hud_st, D3D12_RESOURCE_STATE_COPY_DEST);
    cl->CopyResource(s_hud, src);   /* src: COPY_SOURCE (the caller's) */
    s_fsnap = 1;
}

void rsx_dlss_capture_depth(ID3D12GraphicsCommandList* cl, ID3D12Resource* z, uint32_t* zstate)
{
    if (!rsx_dlss_active() || !cl || !z || !zstate) return;
    const D3D12_RESOURCE_DESC d = z->GetDesc();
    if (!s_zcopy || s_zc_w != (uint32_t)d.Width || s_zc_h != d.Height) {
        release(s_zcopy);
        s_zcopy = make_tex(DXGI_FORMAT_R24G8_TYPELESS, (uint32_t)d.Width, d.Height, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                           D3D12_RESOURCE_STATE_DEPTH_WRITE);
        if (!s_zcopy) { fail("depth snapshot buffer", 0); return; }
        s_zcopy_st = D3D12_RESOURCE_STATE_DEPTH_WRITE; s_zc_w = (uint32_t)d.Width; s_zc_h = d.Height;
    }
    D3D12_RESOURCE_STATES st = (D3D12_RESOURCE_STATES)*zstate;
    const D3D12_RESOURCE_STATES orig = st;
    barrier(cl, z, st, D3D12_RESOURCE_STATE_COPY_SOURCE);
    barrier(cl, s_zcopy, s_zcopy_st, D3D12_RESOURCE_STATE_COPY_DEST);
    cl->CopyResource(s_zcopy, z);
    barrier(cl, z, st, orig);
    *zstate = (uint32_t)st;
    s_zsnap = 1;
}

} // extern "C"

namespace {
int s_hist_ok = 0;   /* DLSS history belongs to an unbroken run of gameplay frames */
int s_last_reset = 1; float s_last_dt = 16.6f;   /* this frame's upscaler reset flag and frame time, for frame generation */

void bind_compute(ID3D12GraphicsCommandList* cl, const float* k)
{
    ID3D12DescriptorHeap* heaps[] = { s_heap };
    cl->SetDescriptorHeaps(1, heaps);
    cl->SetComputeRootSignature(s_rs);
    cl->SetComputeRoot32BitConstants(0, 44, k, 0);
    D3D12_GPU_DESCRIPTOR_HANDLE g0 = s_heap->GetGPUDescriptorHandleForHeapStart(), g1 = g0, g4 = g0;
    g1.ptr += s_heap_inc; g4.ptr += 4 * (UINT64)s_heap_inc;
    cl->SetComputeRootDescriptorTable(1, g0);
    cl->SetComputeRootDescriptorTable(2, g1);
    cl->SetComputeRootDescriptorTable(3, g4);
}

void out_to_backbuffer(ID3D12GraphicsCommandList* cl, ID3D12Resource* bb)
{
    D3D12_RESOURCE_STATES bb_st = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier(cl, s_out, s_out_st, D3D12_RESOURCE_STATE_COPY_SOURCE);
    barrier(cl, bb, bb_st, D3D12_RESOURCE_STATE_COPY_DEST);
    cl->CopyResource(bb, s_out);
    barrier(cl, bb, bb_st, D3D12_RESOURCE_STATE_RENDER_TARGET);
}

/* Upscaling without DLSS this frame: the finished frame enlarged into the backbuffer. */
void upscale_fallback(ID3D12GraphicsCommandList* cl, ID3D12Resource* bb)
{
    if (!s_fsnap || !build_pipeline() || !ensure_resources()) return;
    float k[44] = {0};
    k[32] = (float)s_render_w; k[33] = (float)s_render_h; k[40] = (float)s_out_w; k[41] = (float)s_out_h;
    barrier(cl, s_hud, s_hud_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    barrier(cl, s_out, s_out_st, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    bind_compute(cl, k);
    cl->SetPipelineState(s_pso_blit);
    cl->Dispatch((s_out_w + 7) / 8, (s_out_h + 7) / 8, 1);
    uav_barrier(cl, s_out);
    out_to_backbuffer(cl, bb);
}

/* One DLSS frame into the backbuffer. Returns 0 when DLSS did not run (nothing written to the backbuffer). */
int dlss_frame(ID3D12GraphicsCommandList* cl, ID3D12Resource* bb)
{
    const rsx_upscale_batch& ex = *rsx_upscale_exec();
    const rsx_upscale_cfg& g = cfg();
    const bool up = upscaling();
    const uint32_t w = s_render_w, h = s_render_h, ow = s_out_w, oh = s_out_h;
    if (ex.qualifying < 50) { s_hist_ok = 0; s_zsnap = 0; rsx_upscale_reset_history(); return 0; }   /* menus, loading screens, video */
    if (!s_zsnap || !s_zcopy) {
        static int n = 0; if (n++ < 3) fprintf(stderr, "[DLSS] no scene depth snapshot this frame -- skipped%c", 10);
        s_hist_ok = 0; return 0;
    }
    if (up && !s_fsnap) { s_hist_ok = 0; return 0; }   /* no finished frame to merge the HUD from */
    s_zsnap = 0;
    ID3D12Resource* const zs = s_zcopy;
    if (!build_pipeline() || !ensure_resources()) { fail("DLSS resources", 0); return 0; }
    if (!create_feature(cl)) return 0;

    /* 1. colour: the pre-HUD scene (taken right after the composite) + the finished frame for the HUD merge; without
     *    that snapshot, the finished frame itself (the HUD then goes through DLSS too). DLAA takes the finished frame
     *    from the backbuffer; upscaling took it from the display buffer (rsx_dlss_capture_finished). */
    const int merge = s_csnap;
    s_csnap = 0;
    if (up) {
        if (!merge) {
            barrier(cl, s_hud, s_hud_st, D3D12_RESOURCE_STATE_COPY_SOURCE);
            barrier(cl, s_in, s_in_st, D3D12_RESOURCE_STATE_COPY_DEST);
            cl->CopyResource(s_in, s_hud);
            static int n = 0; if (n++ < 3) fprintf(stderr, "[DLSS] no pre-HUD snapshot this frame -- HUD included%c", 10);
        }
        barrier(cl, s_hud, s_hud_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    } else {
        D3D12_RESOURCE_STATES bb_st = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier(cl, bb, bb_st, D3D12_RESOURCE_STATE_COPY_SOURCE);
        if (merge) {
            barrier(cl, s_hud, s_hud_st, D3D12_RESOURCE_STATE_COPY_DEST);
            cl->CopyResource(s_hud, bb);
            barrier(cl, s_hud, s_hud_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        } else {
            barrier(cl, s_in, s_in_st, D3D12_RESOURCE_STATE_COPY_DEST);
            cl->CopyResource(s_in, bb);
            static int n = 0; if (n++ < 3) fprintf(stderr, "[DLSS] no pre-HUD snapshot this frame -- HUD included%c", 10);
        }
        barrier(cl, bb, bb_st, D3D12_RESOURCE_STATE_RENDER_TARGET);
    }
    barrier(cl, s_in, s_in_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    /* 2. depth copy + motion vectors (render size) */
    barrier(cl, zs, s_zcopy_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    barrier(cl, s_mv, s_mv_st, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    barrier(cl, s_dz, s_dz_st, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    {   /* the previous frame's command list may still read this descriptor: rewrite it only for a new surface */
        static ID3D12Resource* s_srv_for = nullptr;
        if (s_srv_for != zs) {
            D3D12_SHADER_RESOURCE_VIEW_DESC sv = {};
            sv.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS; sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sv.Texture2D.MipLevels = 1;
            s_dev->CreateShaderResourceView(zs, &sv, s_heap->GetCPUDescriptorHandleForHeapStart());
            s_srv_for = zs;
        }
    }
    float k[44] = {0};
    int cut = 0;
    (void)rsx_upscale_pack_mv_consts(k, s_hist_ok, &cut);
    barrier(cl, s_pdz, s_pdz_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);   /* last frame's depth */
    bind_compute(cl, k);
    cl->SetPipelineState(s_pso_mv);
    cl->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
    uav_barrier(cl, s_mv); uav_barrier(cl, s_dz);
    barrier(cl, s_mv, s_mv_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    barrier(cl, s_dz, s_dz_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    /* 3. the upscaler (DLSS or FSR): render size in, output size out. Both take motion vectors in render pixels,
     *    pointing from this frame to the last one, and the jitter in pixels as applied to the projection. */
    barrier(cl, s_out, s_out_st, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const int reset = (s_hist_ok && !cut) ? 0 : 1;   /* new run of gameplay frames, or a camera cut */
    float dt_ms;
    {   static LARGE_INTEGER f, t0; LARGE_INTEGER t; QueryPerformanceCounter(&t);
        if (!f.QuadPart) QueryPerformanceFrequency(&f);
        dt_ms = t0.QuadPart ? (float)((double)(t.QuadPart - t0.QuadPart) * 1000.0 / (double)f.QuadPart) : 16.6f;
        t0 = t; }
#ifdef PS3RECOMP_FFX
    if (g.backend) {
        float nearp = 10.0f, farp = 100000.0f, fov = 1.0f;
        if (ex.cam_valid) rsx_upscale_cam_planes(ex.cam, &nearp, &farp, &fov);
        { static int n = 0; if (n++ == 0) fprintf(stderr, "[FSR] camera: near %.3f, far %.1f, vertical fov %.1f deg%c", nearp, farp, fov * 57.29578f, 10); }
        ffxDispatchDescUpscale d = {};
        d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        d.commandList = cl;
        d.color = ffxApiGetResourceDX12(s_in, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.depth = ffxApiGetResourceDX12(s_dz, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.motionVectors = ffxApiGetResourceDX12(s_mv, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.output = ffxApiGetResourceDX12(s_out, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        d.jitterOffset.x = g.jsign * ex.jx; d.jitterOffset.y = g.jsign * ex.jy;
        d.motionVectorScale.x = 1.0f; d.motionVectorScale.y = 1.0f;
        d.renderSize.width = w; d.renderSize.height = h;
        d.upscaleSize.width = ow; d.upscaleSize.height = oh;
        d.enableSharpening = g.fsr_sharp > 0.0f; d.sharpness = g.fsr_sharp;
        d.frameTimeDelta = dt_ms; d.preExposure = 1.0f; d.reset = reset != 0;
        d.cameraNear = nearp; d.cameraFar = farp; d.cameraFovAngleVertical = fov;
        d.viewSpaceToMetersFactor = 0.01f;   /* the game works in centimetres */
        d.flags = FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB;
        const ffxReturnCode_t r = p_ffxDispatch(&s_ffx, &d.header);
        if (r != FFX_API_RETURN_OK) {
            static int n = 0; if (n++ < 5) fprintf(stderr, "[FSR] dispatch failed (%u)%c", (unsigned)r, 10);
            s_hist_ok = 0;
            return 0;
        }
    }
#endif
#ifdef PS3RECOMP_NGX
    if (!g.backend) {
        NVSDK_NGX_D3D12_DLSS_Eval_Params ep = {};
        ep.Feature.pInColor = s_in; ep.Feature.pInOutput = s_out;
        ep.pInDepth = s_dz; ep.pInMotionVectors = s_mv;
        ep.InJitterOffsetX = g.jsign * ex.jx; ep.InJitterOffsetY = g.jsign * ex.jy;
        ep.InRenderSubrectDimensions.Width = w; ep.InRenderSubrectDimensions.Height = h;
        ep.InReset = reset;
        ep.InMVScaleX = 1.0f; ep.InMVScaleY = 1.0f;
        ep.InFrameTimeDeltaInMsec = dt_ms;
        NVSDK_NGX_Result r = NGX_D3D12_EVALUATE_DLSS_EXT(cl, s_feature, s_params, &ep);
        if (NVSDK_NGX_FAILED(r)) {
            static int n = 0; if (n++ < 5) fprintf(stderr, "[DLSS] evaluate failed (0x%X)%c", (unsigned)r, 10);
            s_hist_ok = 0;
            return 0;
        }
    }
#endif
#ifdef PS3RECOMP_FFX
    s_hudless_ok = 0;
    if (s_fg && merge) {   /* frame generation: the upscaled image before the HUD, so it can keep the HUD undistorted */
        if (!s_hudless) {
            s_hudless = make_tex(DXGI_FORMAT_R8G8B8A8_UNORM, ow, oh, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            s_hudless_st = D3D12_RESOURCE_STATE_COPY_DEST;
        }
        if (s_hudless) {
            barrier(cl, s_out, s_out_st, D3D12_RESOURCE_STATE_COPY_SOURCE);
            barrier(cl, s_hudless, s_hudless_st, D3D12_RESOURCE_STATE_COPY_DEST);
            cl->CopyResource(s_hudless, s_out);
            barrier(cl, s_hudless, s_hudless_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            barrier(cl, s_out, s_out_st, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            s_hudless_ok = 1;
        }
    }
#endif
    if (merge || g.debug_mv) bind_compute(cl, k);   /* the upscaler left its own heaps and state on the list */
    if (merge) {   /* 3b. the HUD back on top (output size; enlarged smoothly when upscaling) */
        cl->SetPipelineState(s_pso_merge);
        uav_barrier(cl, s_out);
        cl->Dispatch((ow + 7) / 8, (oh + 7) / 8, 1);
        uav_barrier(cl, s_out);
    }
    if (g.debug_mv) {   /* motion vectors / depth / reprojection error over the image */
        cl->SetPipelineState(s_pso_dbg);
        barrier(cl, s_mv, s_mv_st, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        barrier(cl, s_dz, s_dz_st, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (s_prev) barrier(cl, s_prev, s_prev_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        uav_barrier(cl, s_out);
        cl->Dispatch((ow + 7) / 8, (oh + 7) / 8, 1);
        uav_barrier(cl, s_out);
        if (s_prev) {   /* this frame's input becomes the next one's "last frame" */
            barrier(cl, s_prev, s_prev_st, D3D12_RESOURCE_STATE_COPY_DEST);
            barrier(cl, s_in, s_in_st, D3D12_RESOURCE_STATE_COPY_SOURCE);
            cl->CopyResource(s_prev, s_in);
        }
    }
    /* this frame's depth and jitter become the next one's "last frame" */
    barrier(cl, s_pdz, s_pdz_st, D3D12_RESOURCE_STATE_COPY_DEST);
    barrier(cl, s_dz, s_dz_st, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cl->CopyResource(s_pdz, s_dz);
    s_last_reset = reset; s_last_dt = dt_ms;

    /* 4. out -> back buffer */
    out_to_backbuffer(cl, bb);

    rsx_upscale_commit_history();
    s_hist_ok = 1;
    { static int n = 0; if (n++ == 0) fprintf(stderr, "%s first %s frame (%u jittered draws, depth 0x%X, %ux%u -> %ux%u)%c",
                                              TAG, name_of(s_feat_mode), ex.qualifying, ex.zeta, w, h, ow, oh, 10); }
    return 1;
}
} // namespace

#ifdef PS3RECOMP_FFX
namespace {
ffxReturnCode_t fg_dispatch(ffxDispatchDescFrameGeneration* params, void* ctx)
{
    return p_ffxDispatch((ffxContext*)ctx, &params->header);
}

/* Once per presented frame, before Present: frame generation runs for gameplay frames the upscaler handled (it has
 * their depth and motion vectors), and is off for menus, loading screens and videos. */
void fg_frame(ID3D12GraphicsCommandList* cl, int ran)
{
    if (!s_fg || !s_fg_swap) return;
    s_fg_frame++;
    const int on = ran && s_dz && s_mv;
    ffxConfigureDescFrameGeneration c = {};
    c.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    c.swapChain = s_fg_swap;
    c.frameGenerationCallback = fg_dispatch; c.frameGenerationCallbackUserContext = &s_fg;
    c.frameGenerationEnabled = on != 0;
    c.allowAsyncWorkloads = false;
    if (on && s_hudless_ok) c.HUDLessColor = ffxApiGetResourceDX12(s_hudless, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    c.generationRect.left = 0; c.generationRect.top = 0; c.generationRect.width = (int32_t)s_out_w; c.generationRect.height = (int32_t)s_out_h;
    c.frameID = s_fg_frame;
    ffxReturnCode_t r = p_ffxConfigure(&s_fg, &c.header);
    if (r != FFX_API_RETURN_OK) { static int n = 0; if (n++ < 5) fprintf(stderr, "[FSR] frame generation configure failed (%u)%c", (unsigned)r, 10); return; }
    if (!on) return;

    /* the generation inputs: this frame's depth and motion vectors (render size), jitter and camera */
    const rsx_upscale_batch& ex = *rsx_upscale_exec();
    const rsx_upscale_cfg& g = cfg();
    barrier(cl, s_dz, s_dz_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    barrier(cl, s_mv, s_mv_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    float nearp = 10.0f, farp = 100000.0f, fov = 1.0f;
    if (ex.cam_valid) rsx_upscale_cam_planes(ex.cam, &nearp, &farp, &fov);
    if (ex.delta_valid) for (int i = 0; i < 3; i++) s_cam_pos[i] += ex.delta[i];
    ffxDispatchDescFrameGenerationPrepareV2 d = {};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2;
    d.frameID = s_fg_frame; d.flags = 0; d.commandList = cl;
    d.renderSize.width = s_render_w; d.renderSize.height = s_render_h;
    d.jitterOffset.x = g.jsign * ex.jx; d.jitterOffset.y = g.jsign * ex.jy;
    d.motionVectorScale.x = 1.0f; d.motionVectorScale.y = 1.0f;
    d.frameTimeDelta = s_last_dt; d.reset = s_last_reset != 0;
    d.cameraNear = nearp; d.cameraFar = farp; d.cameraFovAngleVertical = fov; d.viewSpaceToMetersFactor = 0.01f;
    d.depth = ffxApiGetResourceDX12(s_dz, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.motionVectors = ffxApiGetResourceDX12(s_mv, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    if (ex.cam_valid) {   /* camera axes from the camera matrix rows (right, up, forward), position from its moves */
        const float* m = ex.cam;
        const float* rows[3] = { m, m + 4, m + 12 };
        float* outs[3] = { d.cameraRight, d.cameraUp, d.cameraForward };
        for (int k = 0; k < 3; k++) {
            const float l = sqrtf(rows[k][0] * rows[k][0] + rows[k][1] * rows[k][1] + rows[k][2] * rows[k][2]);
            for (int i = 0; i < 3; i++) outs[k][i] = l > 1e-9f ? rows[k][i] / l : 0.0f;
        }
        for (int i = 0; i < 3; i++) d.cameraPosition[i] = s_cam_pos[i];
    }
    r = p_ffxDispatch(&s_fg, &d.header);
    if (r != FFX_API_RETURN_OK) { static int n = 0; if (n++ < 5) fprintf(stderr, "[FSR] frame generation prepare failed (%u)%c", (unsigned)r, 10); }
    { static int n = 0; if (n++ == 0) fprintf(stderr, "[FSR] frame generation running (%s, HUD-less image %s)%c", s_fg_provider, s_hudless_ok ? "yes" : "no", 10); }
}
} // namespace
#endif

/* FSR_FG=1: right after the port created its swap chain (before it takes the back buffers), replace it with AMD's
 * frame generation swap chain and create the generation context. *swap3 is the port's IDXGISwapChain3 and must be
 * its only reference: AMD's proxy releases it and makes a new swap chain on the same window. Returns 1 when the
 * swap chain was replaced (*swap3 is then the proxy). */
extern "C" int rsx_dlss_wrap_swapchain(void** swap3, ID3D12CommandQueue* queue)
{
    const rsx_upscale_cfg& g = cfg();
#ifdef PS3RECOMP_FFX
    if (!g.fg) return 0;
    if (!rsx_dlss_active()) { fprintf(stderr, "[FSR] frame generation needs DLSS or FSR on (their depth and motion vectors) -- off%c", 10); return 0; }
    if (!swap3 || !*swap3 || !queue || !ffx_load()) return 0;
    IDXGISwapChain3* sc3 = (IDXGISwapChain3*)*swap3;
    IDXGISwapChain4* sc4 = nullptr;
    if (FAILED(sc3->QueryInterface(IID_PPV_ARGS(&sc4))) || !sc4) return 0;
    sc3->Release();   /* sc4 now holds the only reference */
    *swap3 = nullptr;
    ffxCreateContextDescFrameGenerationSwapChainVersionDX12 v = {};
    v.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_VERSION_DX12;
    v.version = FFX_FRAMEGENERATION_SWAPCHAIN_DX12_VERSION;
    ffxCreateContextDescFrameGenerationSwapChainWrapDX12 w = {};
    w.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_WRAP_DX12; w.header.pNext = &v.header;
    w.swapchain = &sc4; w.gameQueue = queue;
    ffxReturnCode_t r = p_ffxCreate(&s_fg_sc, &w.header, nullptr);
    int wrapped = r == FFX_API_RETURN_OK && sc4;
    if (!wrapped) fprintf(stderr, "[FSR] frame generation swap chain failed (%u) -- frame generation off%c", (unsigned)r, 10);
    if (!sc4) return -1;   /* no swap chain left at all */
    IDXGISwapChain3* out = nullptr;
    if (FAILED(sc4->QueryInterface(IID_PPV_ARGS(&out))) || !out) { sc4->Release(); return -1; }
    *swap3 = out;
    if (!wrapped) { sc4->Release(); s_fg_sc = nullptr; return 0; }
    s_fg_swap = sc4;   /* kept for ffxConfigure (one reference) */

    ffxCreateBackendDX12Desc be = {};
    be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12; be.device = s_dev;
    ffxCreateContextDescFrameGenerationVersion fv = {};
    fv.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION; fv.version = FFX_FRAMEGENERATION_VERSION;
    fv.header.pNext = &be.header;
    ffxCreateContextDescFrameGenerationHudless hl = {};
    hl.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS; hl.hudlessBackBufferFormat = FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
    hl.header.pNext = &fv.header;
    ffxCreateContextDescFrameGeneration fd = {};
    fd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION; fd.header.pNext = &hl.header;
    fd.flags = 0;
    fd.displaySize.width = s_out_w; fd.displaySize.height = s_out_h;
    fd.maxRenderSize.width = s_render_w; fd.maxRenderSize.height = s_render_h;
    fd.backBufferFormat = FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
    r = p_ffxCreate(&s_fg, &fd.header, nullptr);
    if (r != FFX_API_RETURN_OK || !s_fg) {
        fprintf(stderr, "[FSR] frame generation context failed (%u) -- the swap chain stays, generation off%c", (unsigned)r, 10);
        s_fg = nullptr;
        return 1;
    }
    ffxQueryGetProviderVersion pv = {};
    pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    if (p_ffxQuery(&s_fg, &pv.header) == FFX_API_RETURN_OK && pv.versionName) s_fg_provider = pv.versionName;
    fprintf(stderr, "[FSR] frame generation ready: provider %s, %ux%u output, render up to %ux%u%c", s_fg_provider, s_out_w, s_out_h,
            s_render_w, s_render_h, 10);
    return 1;
#else
    (void)swap3; (void)queue;
    if (g.fg) fprintf(stderr, "[FSR] FSR_FG=1, but this build has no AMD FSR (no fsr-sdk folder at build time) -- off%c", 10);
    return 0;
#endif
}

extern "C" void rsx_dlss_frame_end(ID3D12GraphicsCommandList* cl, ID3D12Resource* bb, uint32_t w, uint32_t h,
                                   ID3D12Resource* zeta, uint32_t zeta_state)
{
    if (!rsx_dlss_active() || !cl || !bb || !w || !h) return;
    (void)zeta; (void)zeta_state;   /* the frame-end depth is the HUD's; the snapshot holds the scene */
    if (w != s_out_w || h != s_out_h) {
        static int n = 0; if (n++ < 3) fprintf(stderr, "[DLSS] backbuffer %ux%u is not the output size %ux%u -- skipped%c", w, h, s_out_w, s_out_h, 10);
        return;
    }
    const int ran = dlss_frame(cl, bb);
    if (!ran && upscaling()) upscale_fallback(cl, bb);   /* the backbuffer always gets the frame */
    s_fsnap = 0;
#ifdef PS3RECOMP_FFX
    fg_frame(cl, ran);
#endif
}
