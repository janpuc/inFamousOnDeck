/* API-agnostic temporal upscaling & frame-generation tracking.
 * Extracted from rsx_dlss.cpp so D3D12 (DLSS/FSR) and Vulkan (FSR 3.1.5/3.1.6)
 * share the exact same camera-voting, Halton(2,3) jitter, near/far/FOV
 * extraction, draw-qualification, and composite-draw detection logic.
 */
#include "rsx_upscale_common.h"

#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifdef _WIN32
#define UPSCALE_STRICMP  _stricmp
#define UPSCALE_STRNICMP _strnicmp
#else
#include <strings.h>
#define UPSCALE_STRICMP  strcasecmp
#define UPSCALE_STRNICMP strncasecmp
#endif

namespace {

const float kDlssRatio[5] = { 1.0f, 0.6667f, 0.58f, 0.5f, 0.3333f };
const char* const kDlssName[5] = { "DLAA", "Quality", "Balanced", "Performance", "Ultra Performance" };
const float kFsrRatio[5] = { 1.0f, 1.0f / 1.5f, 1.0f / 1.7f, 0.5f, 1.0f / 3.0f };
const char* const kFsrName[5] = { "Native AA", "Quality", "Balanced", "Performance", "Ultra Performance" };

rsx_upscale_cfg g_cfg = {};
int g_cfg_loaded = 0;
int g_active = 0;

uint32_t s_render_w = 0, s_render_h = 0;
uint32_t s_out_w = 0, s_out_h = 0;
uint32_t s_jphases = 16;

uint32_t s_rec_jidx = 0;
rsx_upscale_batch s_rec = {};
rsx_upscale_batch s_exec = {};
std::map<uint32_t, uint32_t> s_zeta_votes;

float s_prev_cam[16] = {0};
int   s_prev_cam_valid = 0;
float s_prev_jx = 0.0f, s_prev_jy = 0.0f;

/* Camera & static-object tracking (record side):
 * inFamous renders camera-relative:
 *   r   = c256 * v.x + c257 * v.y + c258 * v.z + c259        (object -> cam-rel world)
 *   pos = c263 * r.x + c264 * r.y + c265 * r.z + c266 * r.w  (proj x cam rotation)
 * Camera matrix = majority c263..c266 across qualifying scene draws.
 * Camera translation delta = consensus t_prev - t_cur across static objects
 * matched by orientation (c256..c258) + vs_hash + fp_addr + vcount. */
struct CamVote { uint32_t n; float m[16]; float zs, zo; };
std::map<uint64_t, CamVote> s_cam_votes;
struct ObjT { float t[3]; uint32_t n; };
std::map<uint64_t, ObjT> s_obj_cur, s_obj_prev;
int s_obj_prev_valid = 0;

/* Optional single-frame vertex-constant probe dump */
struct ProbeDraw {
    rsx_upscale_draw_info di;
    int q;
    std::vector<std::pair<int, std::vector<float>>> c;
};
std::vector<ProbeDraw> s_pd;
std::map<int, std::string> s_pvs;
uint32_t s_pframe = 0xFFFFFFFFu;
uint32_t s_pq = 0;
int s_pstate = 0;
int s_pskip = 0;

inline const char* env_pair(const char* a, const char* b)
{
    const char* v = getenv(a);
    if (v && *v) return v;
    v = getenv(b);
    return (v && *v) ? v : nullptr;
}

int parse_quality_mode(const char* e)
{
    if (!e || !*e) return 0;
    if (!UPSCALE_STRICMP(e, "quality")) return 1;
    if (!UPSCALE_STRICMP(e, "balanced")) return 2;
    if (!UPSCALE_STRICMP(e, "performance")) return 3;
    if (!UPSCALE_STRICMP(e, "ultra") || !UPSCALE_STRICMP(e, "ultraperformance")) return 4;
    return 0;
}

int parse_debug_mode(const char* e)
{
    if (!e || !*e) return 0;
    if (!UPSCALE_STRICMP(e, "mv")) return 1;
    if (!UPSCALE_STRICMP(e, "depth")) return 2;
    if (!UPSCALE_STRICMP(e, "warp")) return 3;
    if (!UPSCALE_STRICMP(e, "nowarp")) return 4;
    if (!UPSCALE_STRICMP(e, "zcheck")) return 5;
    if (!UPSCALE_STRICMP(e, "hud")) return 6;
    return 0;
}

void cfg_load()
{
    if (g_cfg_loaded) return;
    g_cfg_loaded = 1;

    memset(&g_cfg, 0, sizeof g_cfg);
    g_cfg.jitter = 1;
    g_cfg.mv = 1;
    g_cfg.mvfix = 1;
    g_cfg.mipbias = -1.0f;
    g_cfg.jsign = 1.0f;
#ifdef _WIN32
    g_cfg.min_720p = 1;
#else
    g_cfg.min_720p = 0;
#endif

    const char* d_env = getenv("DLSS");
    const char* f_env = getenv("FSR");
    const char* u_env = getenv("UPSCALER");
    const int has_explicit = (d_env && *d_env) || (f_env && *f_env);

    /* UPSCALER=<off|dlaa|dlss_*|fsr_*> from settings.ini; explicit DLSS/FSR wins. */
    if (!has_explicit && u_env && *u_env && UPSCALE_STRICMP(u_env, "off") != 0) {
        const char* m = strchr(u_env, '_');
        if (!UPSCALE_STRICMP(u_env, "dlaa") || !UPSCALE_STRNICMP(u_env, "dlss", 4)) {
            g_cfg.mode = 1;
            g_cfg.backend = 0;
            if (m) g_cfg.up = parse_quality_mode(m + 1);
        } else if (!UPSCALE_STRNICMP(u_env, "fsr", 3)) {
            g_cfg.mode = 1;
            g_cfg.backend = 1;
            if (m && UPSCALE_STRICMP(m + 1, "native") != 0)
                g_cfg.up = parse_quality_mode(m + 1);
        }
    }

    if (d_env && *d_env && *d_env != '0' && UPSCALE_STRICMP(d_env, "off") != 0) {
        g_cfg.mode = 1;
        g_cfg.backend = 0;
    }

    const char* e = env_pair("FSR_JITTER", "DLSS_JITTER");
    if (e && *e == '0') g_cfg.jitter = 0;
    if ((e = env_pair("FSR_MV", "DLSS_MV")) && *e == '0') g_cfg.mv = 0;
    if ((e = env_pair("FSR_MVFIX", "DLSS_MVFIX")) && *e == '0') g_cfg.mvfix = 0;
    if ((e = env_pair("FSR_DEBUG", "DLSS_DEBUG"))) g_cfg.debug_mv = parse_debug_mode(e);
    if ((e = env_pair("FSR_MIPBIAS", "DLSS_MIPBIAS"))) {
        g_cfg.mipbias = (float)atof(e);
        g_cfg.mipbias_set = 1;
    }
    if ((e = getenv("DLSS_MODE")) && *e) g_cfg.up = parse_quality_mode(e);
    if ((e = env_pair("FSR_JSIGN", "DLSS_JSIGN")) && *e == '-') g_cfg.jsign = -1.0f;
    if ((e = env_pair("FSR_PROBE", "DLSS_PROBE"))) {
        const int v = atoi(e);
        g_cfg.probe = v > 0 ? v : 1;
    }
    if ((e = getenv("DLSS_PRESET")) && *e) {
        const char c = (char)toupper((unsigned char)*e);
        if (c >= 'A' && c <= 'O') g_cfg.preset = c - 'A' + 1;
    }
    if ((e = getenv("FSR_MIN_720P")) && *e) g_cfg.min_720p = (*e != '0') ? 1 : 0;

    if (f_env && *f_env && *f_env != '0' && UPSCALE_STRICMP(f_env, "off") != 0) {
        g_cfg.mode = 1;
        g_cfg.backend = 1;
        g_cfg.up = 0;
        if ((e = getenv("FSR_MODE")) && *e) g_cfg.up = parse_quality_mode(e);
        if ((e = getenv("FSR_SHARPNESS")) && *e) {
            g_cfg.fsr_sharp = (float)atof(e);
            if (g_cfg.fsr_sharp < 0.0f) g_cfg.fsr_sharp = 0.0f;
            if (g_cfg.fsr_sharp > 1.0f) g_cfg.fsr_sharp = 1.0f;
        }
        if ((e = getenv("FSR_VERSION")) && *e) {
            strncpy(g_cfg.fsr_ver, e, sizeof g_cfg.fsr_ver - 1);
            g_cfg.fsr_ver[sizeof g_cfg.fsr_ver - 1] = '\0';
        }
    }
    if ((e = getenv("FSR_FG")) && *e && *e != '0' && UPSCALE_STRICMP(e, "off") != 0)
        g_cfg.fg = 1;

    s_pskip = g_cfg.probe > 0 ? g_cfg.probe - 1 : 0;
    s_rec.last_q = 0xFFFFFFFFu;
    s_rec.composite = 0xFFFFFFFFu;
    s_exec.last_q = 0xFFFFFFFFu;
    s_exec.composite = 0xFFFFFFFFu;
}

float halton(int i, int b)
{
    float f = 1.0f, r = 0.0f;
    while (i > 0) {
        f /= (float)b;
        r += f * (float)(i % b);
        i /= b;
    }
    return r;
}

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull)
{
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

inline int uses(const uint32_t* cu, int i)
{
    return (cu[i >> 5] >> (i & 31)) & 1u;
}

void cam_note(const rsx_upscale_draw_info* di)
{
    if (!di->cuse || !di->consts) return;
    const float* c = di->consts;
    if (uses(di->cuse, 263) && uses(di->cuse, 264) && uses(di->cuse, 265) && uses(di->cuse, 266)) {
        const uint64_t h = fnv(c + 263 * 4, 64);
        CamVote& v = s_cam_votes[h];
        if (!v.n) {
            for (int r = 0; r < 4; r++)
                for (int k = 0; k < 4; k++)
                    v.m[r * 4 + k] = c[(263 + k) * 4 + r];
            v.zs = di->posscale[2];
            v.zo = di->posoffset[2];
        }
        v.n++;
    }
    if (uses(di->cuse, 256) && uses(di->cuse, 257) && uses(di->cuse, 258) &&
        uses(di->cuse, 259) && c[259 * 4 + 3] == 1.0f) {
        uint64_t h = fnv(c + 256 * 4, 48);
        h = fnv(&di->vs_hash, 4, h);
        h = fnv(&di->fp_addr, 4, h);
        h = fnv(&di->vcount, 4, h);
        ObjT& o = s_obj_cur[h];
        if (!o.n) {
            o.t[0] = c[259 * 4 + 0];
            o.t[1] = c[259 * 4 + 1];
            o.t[2] = c[259 * 4 + 2];
        }
        o.n++;
    }
}

void cam_finish(rsx_upscale_batch& b)
{
    b.cam_valid = 0;
    b.delta_valid = 0;
    const CamVote* best = nullptr;
    uint32_t total = 0;
    for (const auto& kv : s_cam_votes) {
        total += kv.second.n;
        if (!best || kv.second.n > best->n) best = &kv.second;
    }
    if (best && best->n >= 20 && best->n * 4 >= total) {
        memcpy(b.cam, best->m, sizeof b.cam);
        for (int k = 0; k < 4; k++)
            b.cam[8 + k] = best->zs * best->m[8 + k] + best->zo * best->m[12 + k];
        b.cam_valid = 1;
    }
    if (s_obj_prev_valid) {
        std::vector<std::array<float, 3>> d;
        for (const auto& kv : s_obj_cur) {
            if (kv.second.n != 1) continue;
            auto it = s_obj_prev.find(kv.first);
            if (it == s_obj_prev.end() || it->second.n != 1) continue;
            d.push_back({
                it->second.t[0] - kv.second.t[0],
                it->second.t[1] - kv.second.t[1],
                it->second.t[2] - kv.second.t[2]
            });
        }
        if (d.size() >= 8) {
            size_t bi = 0, bn = 0;
            const size_t step = d.size() > 400 ? d.size() / 400 : 1;
            for (size_t i = 0; i < d.size(); i += step) {
                size_t n = 0;
                for (const auto& e : d)
                    if (fabsf(e[0] - d[i][0]) < 0.5f &&
                        fabsf(e[1] - d[i][1]) < 0.5f &&
                        fabsf(e[2] - d[i][2]) < 0.5f)
                        n++;
                if (n > bn) { bn = n; bi = i; }
            }
            if (bn >= 6 && bn * 3 >= d.size()) {
                double s[3] = { 0, 0, 0 };
                size_t n = 0;
                for (const auto& e : d) {
                    if (fabsf(e[0] - d[bi][0]) < 0.5f &&
                        fabsf(e[1] - d[bi][1]) < 0.5f &&
                        fabsf(e[2] - d[bi][2]) < 0.5f) {
                        s[0] += e[0]; s[1] += e[1]; s[2] += e[2];
                        n++;
                    }
                }
                for (int k = 0; k < 3; k++) b.delta[k] = (float)(s[k] / (double)n);
                b.delta_valid = 1;
            }
            static int nlog = 0;
            if (nlog < 12 || (nlog % 600) == 0)
                fprintf(stderr, "%s camera: %s, moved %s (%.2f %.2f %.2f cm; %zu of %zu matched objects agree)\n",
                        rsx_upscale_tag(),
                        b.cam_valid ? "found" : "NOT found",
                        b.delta_valid ? "ok" : "UNKNOWN",
                        b.delta[0], b.delta[1], b.delta[2], bn, d.size());
            nlog++;
        }
    }
    s_obj_prev.swap(s_obj_cur);
    s_obj_cur.clear();
    s_obj_prev_valid = 1;
    s_cam_votes.clear();
}

void probe_write()
{
    FILE* f = fopen("dlss_probe.txt", "w");
    if (!f) return;
    fprintf(f, "# Upscaler probe: frame %u, %zu draws (q = qualifies for jitter)\n", s_pframe, s_pd.size());
    for (size_t i = 0; i < s_pd.size(); i++) {
        const ProbeDraw& p = s_pd[i];
        const rsx_upscale_draw_info& d = p.di;
        fprintf(f, "d %zu vs=%d h=%08X q=%d rt=%X %ux%u z=%X %ux%u vp=%u,%u,%u,%u dt=%d dm=%d ct=%X mrt=%u bl=%d fp=%X vc=%u ps=%g,%g,%g po=%g,%g,%g\n",
                i, d.vs_idx, d.vs_hash, p.q, d.rt_off, d.rt_w, d.rt_h, d.zeta_off, d.zw, d.zh,
                d.vp_x, d.vp_y, d.vp_w, d.vp_h, d.depth_test, d.depth_mask, d.color_target,
                d.mrt, d.blend, d.fp_addr, d.vcount,
                d.posscale[0], d.posscale[1], d.posscale[2],
                d.posoffset[0], d.posoffset[1], d.posoffset[2]);
        for (const auto& c : p.c)
            fprintf(f, " c%d=%.9g,%.9g,%.9g,%.9g\n", c.first, c.second[0], c.second[1], c.second[2], c.second[3]);
    }
    fclose(f);
    for (const auto& v : s_pvs) {
        char name[64];
        snprintf(name, sizeof name, "dlss_probe_vs_%d.hlsl", v.first);
        FILE* h = fopen(name, "w");
        if (h) { fwrite(v.second.data(), 1, v.second.size(), h); fclose(h); }
    }
    fprintf(stderr, "%s probe: wrote dlss_probe.txt (%zu draws) and %zu vertex programs\n",
            rsx_upscale_tag(), s_pd.size(), s_pvs.size());
}

void probe_draw(const rsx_upscale_draw_info* di, int q)
{
    if (di->frame != s_pframe) {
        if (s_pstate == 1 && !s_pd.empty()) { probe_write(); s_pstate = 2; s_pd.clear(); }
        else if (s_pstate == 0 && s_pq >= 300) { if (--s_pskip <= 0) s_pstate = 1; }
        s_pframe = di->frame;
        s_pq = 0;
    }
    s_pq += (uint32_t)q;
    if (s_pstate != 1 || s_pd.size() >= 60000) return;
    ProbeDraw p;
    p.di = *di;
    p.q = q;
    p.di.vs_src = nullptr;
    p.di.cuse = nullptr;
    p.di.consts = nullptr;
    if (di->cuse && di->consts) {
        for (int i = 0; i < 512; i++)
            if ((di->cuse[i >> 5] >> (i & 31)) & 1u)
                p.c.push_back({ i, std::vector<float>(di->consts + i * 4, di->consts + i * 4 + 4) });
    }
    s_pd.push_back(std::move(p));
    if (di->vs_src && di->vs_idx >= 0 && !s_pvs.count(di->vs_idx))
        s_pvs[di->vs_idx] = di->vs_src;
}

} // namespace

extern "C" {

const rsx_upscale_cfg* rsx_upscale_get_cfg(void)
{
    cfg_load();
    return &g_cfg;
}

float rsx_upscale_render_ratio(void)
{
    cfg_load();
    if (!g_cfg.mode) return 1.0f;
    const int m = (g_cfg.up >= 0 && g_cfg.up < 5) ? g_cfg.up : 0;
    return g_cfg.backend ? kFsrRatio[m] : kDlssRatio[m];
}

const char* rsx_upscale_mode_name(int mode_idx, int backend)
{
    const int m = (mode_idx >= 0 && mode_idx < 5) ? mode_idx : 0;
    return backend ? kFsrName[m] : kDlssName[m];
}

const char* rsx_upscale_tag(void)
{
    cfg_load();
    return g_cfg.backend ? "[FSR]" : "[DLSS]";
}

void rsx_upscale_set_sizes(uint32_t render_w, uint32_t render_h,
                           uint32_t out_w, uint32_t out_h)
{
    cfg_load();
    s_render_w = render_w;
    s_render_h = render_h;
    s_out_w = out_w ? out_w : render_w;
    s_out_h = out_h ? out_h : render_h;
    if (s_render_w && s_out_w) {
        const double ratio = (double)s_out_w / (double)s_render_w;
        const uint32_t ph = (uint32_t)ceil(8.0 * ratio * ratio);
        s_jphases = ph > 16u ? ph : 16u;
        if (!g_cfg.mipbias_set)
            g_cfg.mipbias = (float)(log2((double)s_render_w / (double)s_out_w) - 1.0);
    }
}

uint32_t rsx_upscale_render_w(void) { return s_render_w; }
uint32_t rsx_upscale_render_h(void) { return s_render_h; }
uint32_t rsx_upscale_out_w(void)    { return s_out_w; }
uint32_t rsx_upscale_out_h(void)    { return s_out_h; }
uint32_t rsx_upscale_jitter_phases(void) { return s_jphases; }
int      rsx_upscale_is_upscaling(void)  { return g_active && (s_out_w != s_render_w || s_out_h != s_render_h); }

void  rsx_upscale_set_active(int active) { g_active = active ? 1 : 0; }
int   rsx_upscale_is_active(void)        { return g_active; }
int   rsx_upscale_hooks_on(void)         { cfg_load(); return (g_cfg.mode || g_cfg.probe) ? 1 : 0; }
int   rsx_upscale_probe_on(void)         { cfg_load(); return (g_cfg.probe && s_pstate < 2) ? 1 : 0; }
float rsx_upscale_mip_bias(void)         { cfg_load(); return (g_cfg.mode && g_active) ? g_cfg.mipbias : 0.0f; }

int rsx_upscale_draw_qualifies(int depth_test, uint32_t zeta_off, uint32_t zw, uint32_t zh,
                               uint32_t rt_off, uint32_t rt_w, uint32_t rt_h,
                               uint32_t vp_w, uint32_t vp_h, uint32_t guest_w, uint32_t guest_h)
{
    if (!depth_test || !zeta_off) return 0;
    if (zw != guest_w || zh != guest_h) return 0;
    if (!rt_off) return 0;
    if (rt_w != guest_w || rt_h != guest_h) return 0;
    if (vp_w && vp_h && (vp_w != guest_w || vp_h != guest_h)) return 0;
    return 1;
}

void rsx_upscale_jitter_px(uint32_t frame_idx, float* jx, float* jy)
{
    const int i = (int)(frame_idx % (s_jphases ? s_jphases : 16u)) + 1;
    *jx = halton(i, 2) - 0.5f;
    *jy = halton(i, 3) - 0.5f;
}

void rsx_upscale_jitter_ndc(float* dx, float* dy)
{
    cfg_load();
    if (!g_active || !g_cfg.jitter || !s_render_w || !s_render_h) return;
    float jx = 0.0f, jy = 0.0f;
    rsx_upscale_jitter_px(s_rec_jidx, &jx, &jy);
    *dx += 2.0f * jx / (float)s_render_w;
    *dy -= 2.0f * jy / (float)s_render_h;
}

int rsx_upscale_note_draw(const rsx_upscale_draw_info* di, int qualifies)
{
    if (!di) return 0;
    cfg_load();
    if (g_cfg.probe && s_pstate < 2) probe_draw(di, qualifies);
    if (!qualifies) {
        if (s_rec.qualifying && s_rec.composite == 0xFFFFFFFFu && di->rt_off == 0 &&
            !di->blend && !di->depth_test &&
            di->vcount >= 3 && di->vcount <= 6 &&
            di->vp_w == di->guest_w && di->vp_h == di->guest_h) {
            s_rec.composite = di->rec_index;
            return RSX_UPSCALE_EVT_COMPOSITE;
        }
        return 0;
    }
    s_rec.qualifying++;
    s_rec.last_q = di->rec_index;
    s_rec.composite = 0xFFFFFFFFu;
    s_zeta_votes[di->zeta_off]++;
    if (g_active && g_cfg.mv) cam_note(di);
    return RSX_UPSCALE_EVT_QUALIFIED;
}

uint32_t rsx_upscale_rec_qualifying(void) { return s_rec.qualifying; }

void rsx_upscale_handoff(int present)
{
    if (!rsx_upscale_hooks_on()) return;
    if (present && g_active && g_cfg.mv) cam_finish(s_rec);
    s_exec = s_rec;
    uint32_t best = 0, bz = 0;
    for (const auto& v : s_zeta_votes) {
        if (v.second > best) { best = v.second; bz = v.first; }
    }
    s_exec.zeta = bz;
    if (g_active && g_cfg.jitter)
        rsx_upscale_jitter_px(s_rec_jidx, &s_exec.jx, &s_exec.jy);
    else
        s_exec.jx = s_exec.jy = 0.0f;

    s_rec.last_q = 0xFFFFFFFFu;
    s_rec.composite = 0xFFFFFFFFu;
    if (present) {
        s_rec_jidx++;
        memset(&s_rec, 0, sizeof s_rec);
        s_rec.last_q = 0xFFFFFFFFu;
        s_rec.composite = 0xFFFFFFFFu;
        s_zeta_votes.clear();
    }
}

const rsx_upscale_batch* rsx_upscale_exec(void) { return &s_exec; }

int rsx_upscale_invert4(const float* m, float* o)
{
    float inv[16];
    inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
    inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
    inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
    inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];
    const double det = (double)m[0]*inv[0] + (double)m[1]*inv[4] + (double)m[2]*inv[8] + (double)m[3]*inv[12];
    if (!(fabs(det) > 1e-30)) return 0;
    const float id = (float)(1.0 / det);
    for (int i = 0; i < 16; i++) o[i] = inv[i] * id;
    return 1;
}

void rsx_upscale_cam_planes(const float* m, float* nearp, float* farp, float* fov)
{
    const float lw = sqrtf(m[12]*m[12] + m[13]*m[13] + m[14]*m[14]);
    const float ly = sqrtf(m[4]*m[4] + m[5]*m[5] + m[6]*m[6]);
    if (lw < 1e-6f || ly < 1e-6f) return;
    const float a = (m[8]*m[12] + m[9]*m[13] + m[10]*m[14]) / lw;
    const float b = m[11];
    if (fabsf(a) > 1e-9f && fabsf(lw - a) > 1e-9f) {
        const float n = -b / a, f = b / (lw - a);
        if (n > 0.0f && f > n) { *nearp = n; *farp = f; }
    }
    *fov = 2.0f * atanf(lw / ly);
}

int rsx_upscale_pack_mv_consts(float k[44], int hist_ok, int* out_cut)
{
    memset(k, 0, 44 * sizeof(float));
    int mv_on = 0, cut = 0;
    if (g_cfg.mv) {
        const float* d = s_exec.delta;
        if (!s_exec.cam_valid || !s_exec.delta_valid ||
            d[0]*d[0] + d[1]*d[1] + d[2]*d[2] > 1000.0f * 1000.0f)
            cut = 1;
        float inv[16];
        if (!cut && s_prev_cam_valid && hist_ok && rsx_upscale_invert4(s_exec.cam, inv)) {
            float pv[16];
            memcpy(pv, s_prev_cam, sizeof pv);
            for (int r = 0; r < 4; r++)
                pv[r * 4 + 3] += s_prev_cam[r * 4 + 0] * d[0] +
                                 s_prev_cam[r * 4 + 1] * d[1] +
                                 s_prev_cam[r * 4 + 2] * d[2];
            memcpy(k, inv, 64);
            memcpy(k + 16, pv, 64);
            mv_on = 1;
        }
    }
    k[32] = (float)s_render_w;
    k[33] = (float)s_render_h;
    k[34] = s_exec.jx;
    k[35] = s_exec.jy;
    k[36] = mv_on ? (g_cfg.mvfix ? 2.0f : 1.0f) : 0.0f;
    k[37] = (float)g_cfg.debug_mv;
    k[38] = s_prev_jx;
    k[39] = s_prev_jy;
    k[40] = (float)s_out_w;
    k[41] = (float)s_out_h;
    if (out_cut) *out_cut = cut;
    return mv_on;
}

void rsx_upscale_commit_history(void)
{
    s_prev_jx = s_exec.jx;
    s_prev_jy = s_exec.jy;
    if (s_exec.cam_valid) {
        memcpy(s_prev_cam, s_exec.cam, sizeof s_prev_cam);
        s_prev_cam_valid = 1;
    } else {
        s_prev_cam_valid = 0;
    }
}

void rsx_upscale_reset_history(void)
{
    s_prev_cam_valid = 0;
}

} // extern "C"
