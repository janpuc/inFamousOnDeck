/*
 * ps3recomp - HLSL -> SPIR-V through glslang, for the Vulkan backend
 *
 * See rsx_shader_spirv.h for the binding contract. The front end is driven
 * exactly as rsx_shader_msl.cpp drives it (C++ API for setEntryPoint, Vulkan
 * rules, auto-mapped bindings and locations, then mapIO, legalized SPIR-V),
 * with three differences, all because the output goes to a Vulkan driver
 * rather than to spirv-cross:
 *
 *   - Vulkan 1.0 client, SPIR-V 1.0: the backend creates a 1.0 instance and
 *     promises to run on 1.0-level drivers. (The MSL path targets 1.3 because
 *     that is spirv-cross's floor, not a driver's.)
 *   - Texture and sampler registers are shifted apart (setShiftBinding),
 *     because Vulkan has one binding namespace per set.
 *   - The vertex stage flips Y (setInvertY): D3D clip space is +y up.
 *
 * Translations are cached. glslang takes 20-150 ms per shader, on the thread
 * that walks the FIFO, and the engine asks for the same HLSL again for every
 * pipeline that shares a program (Twisted Metal: 236 translations in its
 * first minute, 74 distinct). The cache key is the stage plus the whole HLSL
 * text, so a hit returns exactly the words that text translated to. With
 * PS3RECOMP_SPIRV_CACHE=<existing directory> the cache is also kept on disk,
 * one file per shader, and the next run starts warm. A file whose key,
 * glslang version or translator build differs from the request is a miss,
 * and is rewritten.
 */

#include "rsx_shader_spirv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(PS3RECOMP_HAVE_SPIRV_TRANSLATION)

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

void put_log(char* log, u32 log_size, const char* what, const char* detail)
{
    if (!log || log_size == 0) return;
    snprintf(log, log_size, "%s%s%s", what ? what : "",
             (what && detail && *detail) ? ": " : "", detail ? detail : "");
}

bool glslang_ready()
{
    static bool inited = false;
    if (!inited) {
        if (!glslang::InitializeProcess()) return false;
        inited = true;
    }
    return true;
}

bool is_ident_char(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
}

std::string replace_ident(const std::string& src, const std::string& name, const std::string& repl)
{
    std::string out;
    out.reserve(src.size() + 256);
    size_t pos = 0;
    const size_t nlen = name.size();
    while (pos < src.size()) {
        const size_t at = src.find(name, pos);
        if (at == std::string::npos) {
            out.append(src, pos, std::string::npos);
            break;
        }
        const bool left_ok = (at == 0) || !is_ident_char(src[at - 1]);
        const bool right_ok = (at + nlen >= src.size()) || !is_ident_char(src[at + nlen]);
        if (left_ok && right_ok) {
            out.append(src, pos, at - pos);
            out.append(repl);
            pos = at + nlen;
        } else {
            out.append(src, pos, at - pos + 1);
            pos = at + 1;
        }
    }
    return out;
}

std::string replace_indexed(const std::string& src, const std::string& name, const std::string& prefix)
{
    std::string out;
    out.reserve(src.size() + 512);
    size_t pos = 0;
    const size_t nlen = name.size();
    while (pos < src.size()) {
        const size_t at = src.find(name, pos);
        if (at == std::string::npos) {
            out.append(src, pos, std::string::npos);
            break;
        }
        const bool left_ok = (at == 0) || !is_ident_char(src[at - 1]);
        size_t j = at + nlen;
        while (j < src.size() && (src[j] == ' ' || src[j] == '\t')) j++;
        if (left_ok && j < src.size() && src[j] == '[') {
            int depth = 1;
            size_t k = j + 1;
            while (k < src.size() && depth > 0) {
                if (src[k] == '[') depth++;
                else if (src[k] == ']') depth--;
                if (depth > 0) k++;
            }
            if (depth == 0) {
                out.append(src, pos, at - pos);
                out.append(prefix);
                out.append("(uint)(");
                out.append(src, j + 1, k - (j + 1));
                out.append(")]");
                pos = k + 1;
                continue;
            }
        }
        out.append(src, pos, at - pos + 1);
        pos = at + 1;
    }
    return out;
}

std::string shift_registers_to_space1(const std::string& src)
{
    std::string out;
    out.reserve(src.size() + 128);
    size_t pos = 0;
    const std::string needle = "register(";
    while (pos < src.size()) {
        const size_t at = src.find(needle, pos);
        if (at == std::string::npos) {
            out.append(src, pos, std::string::npos);
            break;
        }
        const size_t close = src.find(')', at + needle.size());
        if (close == std::string::npos) {
            out.append(src, pos, std::string::npos);
            break;
        }
        const std::string inside = src.substr(at + needle.size(), close - (at + needle.size()));
        size_t p = 0;
        while (p < inside.size() && (inside[p] == ' ' || inside[p] == '\t')) p++;
        if (p < inside.size() && (inside[p] == 't' || inside[p] == 's') &&
            inside.find(',') == std::string::npos) {
            out.append(src, pos, close - pos);
            out.append(", space1)");
            pos = close + 1;
        } else {
            out.append(src, pos, close + 1 - pos);
            pos = close + 1;
        }
    }
    return out;
}

/* Rewrite VPConst/PSConstants cbuffers to a single persistently bound set-0
 * StructuredBuffer<float4> indexed by a per-draw push-constant float4 offset
 * (or SV_InstanceID / BuiltIn InstanceIndex for vertex shaders when use_inst
 * is true, enabling vkCmdDrawIndexedIndirect batching via firstInstance), and
 * move texture/sampler registers to descriptor set 1 (space1). */
std::string rewrite_hlsl_const_ssbo(const char* hlsl, int stage, bool use_inst)
{
    static const char* kSsboHeader =
        "[[vk::binding(0, 0)]] StructuredBuffer<float4> _rsx_cbuf;\n"
        "struct _RsxPc { uint vp_off; uint ps_off; };\n"
        "[[vk::push_constant]] ConstantBuffer<_RsxPc> _rsx_pc;\n";
    std::string s(hlsl);
    if (stage == RSX_SHADER_STAGE_VERTEX) {
        const size_t cb = s.find("cbuffer VPConst");
        if (cb != std::string::npos) {
            const size_t open = s.find('{', cb);
            const size_t close = (open != std::string::npos) ? s.find("};", open) : std::string::npos;
            if (close != std::string::npos) {
                const std::string blk = s.substr(open, close - open);
                const size_t p_scale = blk.find("vp_posscale");
                const size_t p_vc = blk.find("vp_c[");
                const bool packed = (p_scale != std::string::npos && p_vc != std::string::npos && p_scale < p_vc);
                std::string head = s.substr(0, cb) + kSsboHeader;
                std::string tail = s.substr(close + 2);
                std::string vp_base = "_rsx_pc.vp_off";
                if (use_inst) {
                    const std::string sig = "VSOutput main(VSInput input)";
                    const size_t mp = tail.find(sig);
                    if (mp != std::string::npos) {
                        tail.replace(mp, sig.size(),
                                     "VSOutput main(VSInput input, uint _rsx_inst : SV_InstanceID)");
                        vp_base = "_rsx_inst";
                    }
                }
                if (packed) {
                    tail = replace_indexed(tail, "vp_c", "_rsx_cbuf[" + vp_base + " + 2u + ");
                    tail = replace_ident(tail, "vp_posscale", "(_rsx_cbuf[" + vp_base + " + 0u])");
                    tail = replace_ident(tail, "vp_posoffset", "(_rsx_cbuf[" + vp_base + " + 1u])");
                } else {
                    tail = replace_indexed(tail, "vp_c", "_rsx_cbuf[" + vp_base + " + ");
                    tail = replace_ident(tail, "vp_posscale", "(_rsx_cbuf[" + vp_base + " + 512u])");
                    tail = replace_ident(tail, "vp_posoffset", "(_rsx_cbuf[" + vp_base + " + 513u])");
                }
                s = head + tail;
            }
        }
    } else if (stage == RSX_SHADER_STAGE_FRAGMENT) {
        const size_t cb = s.find("cbuffer PSConstants");
        if (cb != std::string::npos) {
            const size_t open = s.find('{', cb);
            const size_t close = (open != std::string::npos) ? s.find("};", open) : std::string::npos;
            if (close != std::string::npos) {
                const std::string blk = s.substr(open, close - open);
                unsigned long nconst = 1ul;
                const size_t fc = blk.find("fp_constants[");
                if (fc != std::string::npos) {
                    nconst = strtoul(blk.c_str() + fc + 13, nullptr, 10);
                    if (nconst == 0ul) nconst = 1ul;
                }
                std::string head = s.substr(0, cb) + kSsboHeader;
                std::string tail = s.substr(close + 2);
                tail = replace_indexed(tail, "fp_constants", "_rsx_cbuf[_rsx_pc.ps_off + ");
                tail = replace_ident(tail, "fp_alpha",
                    "(_rsx_cbuf[_rsx_pc.ps_off + " + std::to_string(nconst) + "u])");
                s = head + tail;
            }
        }
    }
    return shift_registers_to_space1(s);
}

int translate(const char* hlsl, int stage, u32 flags, std::vector<unsigned int>& spirv,
              char* log, u32 log_size)
{
    if (!glslang_ready()) {
        put_log(log, log_size, "glslang::InitializeProcess failed", nullptr);
        return -1;
    }

    const bool ssbo_inst = (flags & RSX_SPIRV_FLAG_CONST_SSBO_INST) != 0u;
    const bool ssbo = ((flags & RSX_SPIRV_FLAG_CONST_SSBO) != 0u) || ssbo_inst;
    std::string rewritten;
    const char* src = hlsl;
    if (ssbo) {
        rewritten = rewrite_hlsl_const_ssbo(hlsl, stage, ssbo_inst);
        src = rewritten.c_str();
    }

    const EShLanguage lang = (stage == RSX_SHADER_STAGE_VERTEX) ? EShLangVertex
                                                                : EShLangFragment;
    const EShMessages msgs =
        (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules | EShMsgReadHlsl);

    glslang::TShader shader(lang);
    const char* strings[1] = { src };
    shader.setStrings(strings, 1);
    shader.setEnvInput(glslang::EShSourceHlsl, lang, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);
    /* SPIR-V FragCoord.w is 1/w (FXC's SV_POSITION.w is w): tells the FP
     * decompiler's _rsx_inv_w helper which one it gets. */
    shader.setPreamble("#define RSX_FRAGCOORD_W_IS_RCP 1\n");
    shader.setEntryPoint("main");
    shader.setAutoMapBindings(true);
    shader.setAutoMapLocations(true);
    if (ssbo) {
        shader.setShiftBindingForSet(glslang::EResTexture, RSX_SPIRV_TEXTURE_BINDING, 1);
        shader.setShiftBindingForSet(glslang::EResSampler, RSX_SPIRV_SAMPLER_BINDING, 1);
    } else {
        shader.setShiftBinding(glslang::EResTexture, RSX_SPIRV_TEXTURE_BINDING);
        shader.setShiftBinding(glslang::EResSampler, RSX_SPIRV_SAMPLER_BINDING);
    }
    if (lang == EShLangVertex) shader.setInvertY(true);

    if (!shader.parse(GetDefaultResources(), 100, false, msgs)) {
        put_log(log, log_size, "HLSL parse", shader.getInfoLog());
        return -1;
    }
    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(msgs)) {
        put_log(log, log_size, "HLSL link", program.getInfoLog());
        return -1;
    }
    if (!program.mapIO()) {
        put_log(log, log_size, "HLSL io map", program.getInfoLog());
        return -1;
    }

    spv::SpvBuildLogger logger;
    glslang::SpvOptions opts;
    opts.disableOptimizer = false;     /* run the HLSL legalization passes */
    glslang::GlslangToSpv(*program.getIntermediate(lang), spirv, &logger, &opts);
    if (spirv.empty()) {
        put_log(log, log_size, "SPIR-V generation", logger.getAllMessages().c_str());
        return -1;
    }
    return 0;
}

/* ---- translation cache ------------------------------------------------- */

const u32 kCacheFormat = 1;
const u32 kCacheMagic = 0x56505352u;   /* "RSPV" */
/* This file's compile time stands for translate()'s options and the binding
 * contract in rsx_shader_spirv.h: when either changes, this file is rebuilt,
 * and the files an older build wrote become misses. */
const char kTranslatorBuild[24] = __DATE__ " " __TIME__;

struct Cache {
    std::mutex lock;
    std::unordered_map<std::string, std::vector<unsigned int>> map;
    std::string dir;
    bool dir_read = false;
    RsxSpirvCacheStats stats = {};
};

Cache& cache()
{
    static Cache c;
    return c;
}

const std::string& cache_dir(Cache& c)
{
    if (!c.dir_read) {
        const char* e = getenv("PS3RECOMP_SPIRV_CACHE");
        c.dir = (e && *e) ? e : "";
        c.dir_read = true;
    }
    return c.dir;
}

/* The key is the mode+stage, then the HLSL text. */
std::string cache_key(const char* hlsl, int stage, u32 flags)
{
    const char tag = ((flags & RSX_SPIRV_FLAG_CONST_SSBO_INST) && stage == RSX_SHADER_STAGE_VERTEX)
        ? (char)('I' + stage)
        : ((flags & (RSX_SPIRV_FLAG_CONST_SSBO | RSX_SPIRV_FLAG_CONST_SSBO_INST))
            ? (char)('A' + stage) : (char)('0' + stage));
    std::string key(1, tag);
    key += hlsl;
    return key;
}

std::string cache_path(const std::string& dir, const std::string& key)
{
    unsigned long long h = 0xcbf29ce484222325ull;       /* FNV-1a 64 */
    for (unsigned char ch : key) { h ^= ch; h *= 0x100000001b3ull; }
    char name[40];
    snprintf(name, sizeof name, "/%016llx.spv", h);
    return dir + name;
}

/* Header: magic, format, glslang major/minor/patch, translator build (six
 * words), key bytes, word count; then the key and the words. */
const int kHeaderWords = 13;

void cache_header(u32 hdr[kHeaderWords], const std::string& key, size_t nwords)
{
    const glslang::Version v = glslang::GetVersion();
    hdr[0] = kCacheMagic; hdr[1] = kCacheFormat;
    hdr[2] = (u32)v.major; hdr[3] = (u32)v.minor; hdr[4] = (u32)v.patch;
    memcpy(&hdr[5], kTranslatorBuild, sizeof kTranslatorBuild);
    hdr[11] = (u32)key.size(); hdr[12] = (u32)nwords;
}

bool cache_load(const std::string& path, const std::string& key,
                std::vector<unsigned int>& spirv)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    u32 want[kHeaderWords], hdr[kHeaderWords];
    cache_header(want, key, 0);
    bool ok = fread(hdr, sizeof hdr, 1, f) == 1 &&
              memcmp(hdr, want, (kHeaderWords - 1) * sizeof(u32)) == 0 &&
              hdr[kHeaderWords - 1] != 0 && hdr[kHeaderWords - 1] <= (1u << 24);
    if (ok) {
        std::string stored(key.size(), '\0');
        ok = fread(&stored[0], 1, stored.size(), f) == stored.size() && stored == key;
    }
    if (ok) {
        spirv.resize(hdr[kHeaderWords - 1]);
        ok = fread(spirv.data(), sizeof(u32), spirv.size(), f) == spirv.size();
    }
    fclose(f);
    if (!ok) spirv.clear();
    return ok;
}

/* Written under a temporary name and renamed, so a reader never sees half a file. */
void cache_store(const std::string& path, const std::string& key,
                 const std::vector<unsigned int>& spirv)
{
    char suffix[48];
    snprintf(suffix, sizeof suffix, ".%zx.tmp", std::hash<std::thread::id>{}(std::this_thread::get_id()));
    const std::string tmp = path + suffix;
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) return;
    u32 hdr[kHeaderWords];
    cache_header(hdr, key, spirv.size());
    bool ok = fwrite(hdr, sizeof hdr, 1, f) == 1 &&
              fwrite(key.data(), 1, key.size(), f) == key.size() &&
              fwrite(spirv.data(), sizeof(u32), spirv.size(), f) == spirv.size();
    ok = (fclose(f) == 0) && ok;
#ifdef _WIN32
    if (ok) remove(path.c_str());      /* rename does not replace on Windows */
#endif
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) remove(tmp.c_str());
}

} // namespace

extern "C" int rsx_hlsl_to_spirv_available(void) { return 1; }

extern "C" int rsx_hlsl_to_spirv_ex(const char* hlsl, int stage, u32 flags,
                                    u32* words, u32 max_words, u32* out_words,
                                    char* log, u32 log_size)
{
    if (out_words) *out_words = 0;
    if (!hlsl || !words || !out_words || max_words == 0) {
        put_log(log, log_size, "bad arguments", nullptr);
        return -1;
    }
    if (stage != RSX_SHADER_STAGE_VERTEX && stage != RSX_SHADER_STAGE_FRAGMENT) {
        put_log(log, log_size, "unknown shader stage", nullptr);
        return -1;
    }

    /* One lock around the whole lookup and translation: glslang's process
     * state is not documented as thread-safe either. */
    Cache& c = cache();
    std::lock_guard<std::mutex> guard(c.lock);
    const std::string key = cache_key(hlsl, stage, flags);
    auto it = c.map.find(key);
    if (it != c.map.end()) {
        c.stats.memory_hits++;
    } else {
        std::vector<unsigned int> spirv;
        const std::string& dir = cache_dir(c);
        const std::string path = dir.empty() ? std::string() : cache_path(dir, key);
        if (!dir.empty() && cache_load(path, key, spirv)) {
            c.stats.disk_hits++;
        } else {
            if (translate(hlsl, stage, flags, spirv, log, log_size)) return -1;
            c.stats.translations++;
            if (!dir.empty()) cache_store(path, key, spirv);
        }
        it = c.map.emplace(key, std::move(spirv)).first;
    }
    const std::vector<unsigned int>& spirv = it->second;
    if (spirv.size() > max_words) {
        char n[64];
        snprintf(n, sizeof n, "%zu words, room for %u", spirv.size(), max_words);
        put_log(log, log_size, "SPIR-V output too large", n);
        return -1;
    }
    memcpy(words, spirv.data(), spirv.size() * sizeof(u32));
    *out_words = (u32)spirv.size();
    return 0;
}

extern "C" int rsx_hlsl_to_spirv(const char* hlsl, int stage,
                                 u32* words, u32 max_words, u32* out_words,
                                 char* log, u32 log_size)
{
    return rsx_hlsl_to_spirv_ex(hlsl, stage, 0u, words, max_words, out_words, log, log_size);
}

extern "C" void rsx_hlsl_to_spirv_cache_stats(RsxSpirvCacheStats* out)
{
    Cache& c = cache();
    std::lock_guard<std::mutex> guard(c.lock);
    if (out) *out = c.stats;
}

extern "C" void rsx_hlsl_to_spirv_cache_reset(void)
{
    Cache& c = cache();
    std::lock_guard<std::mutex> guard(c.lock);
    c.map.clear();
    c.dir_read = false;
    c.stats = {};
}

#else /* !PS3RECOMP_HAVE_SPIRV_TRANSLATION */

extern "C" int rsx_hlsl_to_spirv_available(void) { return 0; }

extern "C" int rsx_hlsl_to_spirv_ex(const char* hlsl, int stage, u32 flags,
                                    u32* words, u32 max_words, u32* out_words,
                                    char* log, u32 log_size)
{
    (void)hlsl; (void)stage; (void)flags; (void)words; (void)max_words;
    if (out_words) *out_words = 0;
    if (log && log_size)
        snprintf(log, log_size, "HLSL -> SPIR-V translation not built (glslang not found)");
    return -1;
}

extern "C" int rsx_hlsl_to_spirv(const char* hlsl, int stage,
                                 u32* words, u32 max_words, u32* out_words,
                                 char* log, u32 log_size)
{
    return rsx_hlsl_to_spirv_ex(hlsl, stage, 0u, words, max_words, out_words, log, log_size);
}

extern "C" void rsx_hlsl_to_spirv_cache_stats(RsxSpirvCacheStats* out)
{
    if (out) memset(out, 0, sizeof *out);
}

extern "C" void rsx_hlsl_to_spirv_cache_reset(void) {}

#endif
