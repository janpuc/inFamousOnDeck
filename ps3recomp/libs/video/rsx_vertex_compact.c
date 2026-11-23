#include "rsx_vertex_compact.h"
#include "rsx_commands.h"
#include "rsx_vertex_formats.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static float rsx_compact_be_f32(const u8* p)
{
    /* One bswap rather than four byte loads and shifts: the renderer decodes
     * millions of these per city frame and the shift chain was a measurable
     * part of the FIFO walker's time. */
    u32 w;
    memcpy(&w, p, sizeof(w));
    w = __builtin_bswap32(w);
    float f;
    memcpy(&f, &w, sizeof(f));
    return f;
}

static float rsx_compact_be_f16(const u8* p)
{
    const u16 h = (u16)((p[0] << 8) | p[1]);
    const u32 sign = (u32)(h >> 15) << 31;
    const u32 exp = (h >> 10) & 0x1F;
    const u32 man = h & 0x3FF;
    u32 out;
    if (exp == 0)
        out = sign;
    else if (exp == 31)
        out = sign | 0x7F800000u | (man << 13);
    else
        out = sign | ((exp + 112) << 23) | (man << 13);
    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

int rsx_vertex_decode_element(
    u32 type, u32 size, const u8* source, float out[4])
{
    if (!source || !out)
        return 0;
    out[0] = out[1] = out[2] = 0.0f;
    out[3] = 1.0f;
    for (u32 component = 0; component < size && component < 4;
         component++) {
        switch (type) {
        case RSX_VTX_TYPE_FLOAT:
            out[component] = rsx_compact_be_f32(source + component * 4);
            break;
        case RSX_VTX_TYPE_HALF:
            out[component] = rsx_compact_be_f16(source + component * 2);
            break;
        case RSX_VTX_TYPE_UNORM8:
            out[component] = source[component] / 255.0f;
            break;
        case RSX_VTX_TYPE_UINT8:
            out[component] = (float)source[component];
            break;
        case RSX_VTX_TYPE_SNORM16: {
            const s16 value = (s16)(
                (source[component * 2] << 8) | source[component * 2 + 1]);
            out[component] = value / 32767.0f;
            break;
        }
        case RSX_VTX_TYPE_SINT16: {
            const s16 value = (s16)(
                (source[component * 2] << 8) | source[component * 2 + 1]);
            out[component] = (float)value;
            break;
        }
        case RSX_VTX_TYPE_CMP32: {
            const u32 word =
                ((u32)source[0] << 24) | ((u32)source[1] << 16) |
                ((u32)source[2] << 8) | (u32)source[3];
            s32 x = (s32)(word & 0x7FF);
            s32 y = (s32)((word >> 11) & 0x7FF);
            s32 z = (s32)((word >> 22) & 0x3FF);
            if (x & 0x400) x -= 0x800;
            if (y & 0x400) y -= 0x800;
            if (z & 0x200) z -= 0x400;
            out[0] = (float)x / 1023.0f;
            out[1] = (float)y / 1023.0f;
            out[2] = (float)z / 511.0f;
            out[3] = 1.0f;
            return 1;
        }
        default:
            return 0;
        }
    }
    return 1;
}

void rsx_vertex_layout_plan_init(
    rsx_vertex_layout_plan* plan, u32 input_mask)
{
    if (!plan)
        return;
    memset(plan, 0, sizeof(*plan));
    plan->mask = input_mask & 0xFFFFu;
    for (u32 attr = 0; attr < RSX_DSP_NUM_VERTEX_ATTR; attr++)
        plan->slot_by_attr[attr] = -1;
    for (u32 attr = 0; attr < RSX_DSP_NUM_VERTEX_ATTR; attr++) {
        if (!(plan->mask & (1u << attr)))
            continue;
        const u32 slot = plan->count++;
        plan->attrs[slot] = (u8)attr;
        plan->slot_by_attr[attr] = (s8)slot;
    }
    plan->stride = (u32)plan->count * 16u;
}

void rsx_vertex_fetch_plan_init(
    rsx_vertex_fetch_plan* plan, const rsx_dispatch* rsx,
    const rsx_vertex_layout_plan* layout,
    rsx_vertex_guest_ptr_fn guest_ptr, void* guest_user)
{
    if (!plan || !rsx || !layout)
        return;
    memset(plan, 0, sizeof(*plan));
    plan->layout = *layout;
    plan->base_offset = rsx_dsp_vertex_data_base_offset(rsx);
    plan->divider_mask = rsx_dsp_reg(rsx, 0x1FC0);
    plan->uniform_element = 1;
    plan->guest_ptr = guest_ptr;
    plan->guest_user = guest_user;
    for (u32 attr = 0; attr < RSX_DSP_NUM_VERTEX_ATTR; attr++) {
        rsx_vertex_fetch_attr* fetch = &plan->attr[attr];
        rsx_dsp_get_vertex_attr(rsx, attr, &fetch->desc);
        fetch->elem_size =
            rsx_vertex_attrib_size(fetch->desc.type, fetch->desc.size);
        fetch->stride =
            fetch->desc.stride ? fetch->desc.stride : fetch->elem_size;
        if (fetch->desc.frequency > 1)
            plan->uniform_element = 0;
        rsx_dsp_vertex_default(rsx, attr, fetch->default_value);
        if (attr == 3 &&
            fetch->default_value[0] == 0.0f &&
            fetch->default_value[1] == 0.0f &&
            fetch->default_value[2] == 0.0f &&
            fetch->default_value[3] == 1.0f) {
            fetch->default_value[0] = 1.0f;
            fetch->default_value[1] = 1.0f;
            fetch->default_value[2] = 1.0f;
        }
    }
}

u32 rsx_vertex_inline_layout(
    const rsx_dispatch* rsx, u32 offsets[RSX_DSP_NUM_VERTEX_ATTR])
{
    u32 packed = 0, declared = 0;
    for (u32 attr = 0; attr < RSX_DSP_NUM_VERTEX_ATTR; attr++) {
        rsx_dsp_vertex_attr desc;
        rsx_dsp_get_vertex_attr(rsx, attr, &desc);
        if (offsets)
            offsets[attr] = packed;
        if (!desc.type || !desc.size)
            continue;
        packed += rsx_vertex_attrib_size(desc.type, desc.size);
        if (desc.stride > declared)
            declared = desc.stride;
    }
    return declared ? declared : packed;
}

void rsx_vertex_fetch_plan_set_inline(
    rsx_vertex_fetch_plan* plan, const rsx_dispatch* rsx,
    const u8* data, u32 bytes)
{
    if (!plan)
        return;
    plan->inline_data = NULL;
    plan->inline_bytes = 0;
    if (!data || !bytes || !rsx)
        return;
    plan->inline_stride = rsx_vertex_inline_layout(rsx, plan->inline_off);
    plan->inline_data = data;
    plan->inline_bytes = bytes;
}

void rsx_vertex_fetch_plan_prepare(
    rsx_vertex_fetch_plan* plan, const rsx_vertex_ref* refs, u32 count)
{
    if (!plan)
        return;
    /* Every attribute of a uniform plan resolves the same element for a given
     * reference, so its span is one pass over the references instead of one
     * pass per attribute. */
    int shared = plan->uniform_element && !plan->inline_data && plan->guest_ptr &&
                 refs && count && plan->layout.count;
    u32 shared_first = 0, shared_last = 0;
    if (shared) {
        shared_first = UINT_MAX;
        for (u32 i = 0; i < count; i++) {
            const u32 element = rsx_vertex_element_index(
                refs[i].vertex_id, refs[i].base_index, 1, 0);
            if (element < shared_first) shared_first = element;
            if (element > shared_last) shared_last = element;
        }
    }
    for (u32 slot = 0; slot < plan->layout.count; slot++) {
        const u32 attr = plan->layout.attrs[slot];
        rsx_vertex_fetch_attr* fetch = &plan->attr[attr];
        fetch->stable_base = NULL;
        fetch->stable_first = 0;
        fetch->stable_last = 0;
        if (!refs || !count ||
            !fetch->desc.type || !fetch->desc.size || !fetch->elem_size)
            continue;

        /* INLINE_ARRAY: the stream is already one contiguous, correctly
         * strided array per attribute, which is exactly what stable_base
         * describes -- so it needs no resolver and no bounds fallback, only
         * the right base. */
        if (plan->inline_data) {
            const u32 stride =
                plan->inline_stride ? plan->inline_stride : fetch->stride;
            const u32 off = plan->inline_off[attr];
            if (!stride || off + fetch->elem_size > stride)
                continue;
            const u32 verts = plan->inline_bytes / stride;
            if (!verts)
                continue;
            fetch->stride = stride;
            fetch->stable_base = plan->inline_data + off;
            fetch->stable_first = 0;
            fetch->stable_last = verts - 1;
            continue;
        }

        if (!plan->guest_ptr)
            continue;

        u32 first = shared_first;
        u32 last = shared_last;
        if (!shared) {
            first = UINT_MAX;
            last = 0;
            for (u32 i = 0; i < count; i++) {
                const u32 element = rsx_vertex_element_index(
                    refs[i].vertex_id, refs[i].base_index,
                    fetch->desc.frequency,
                    (plan->divider_mask >> attr) & 1u);
                if (element < first) first = element;
                if (element > last) last = element;
            }
        }

        const u64 prefix =
            (u64)plan->base_offset + (u64)fetch->desc.offset;
        const u64 start = prefix + (u64)first * fetch->stride;
        const u64 end =
            prefix + (u64)last * fetch->stride + fetch->elem_size;
        if (start > UINT_MAX || end > 0x100000000ull || end <= start ||
            end - start > UINT_MAX)
            continue;

        const u8* base = plan->guest_ptr(
            plan->guest_user, fetch->desc.location, (u32)start,
            (u32)(end - start));
        if (!base)
            continue;
        fetch->stable_base = base;
        fetch->stable_first = first;
        fetch->stable_last = last;
    }
}

int rsx_vertex_fetch_one(
    const rsx_vertex_fetch_plan* plan, const rsx_vertex_ref* ref, u8* out)
{
    if (!plan || !ref || (!out && plan->layout.stride))
        return 0;
    for (u32 slot = 0; slot < plan->layout.count; slot++) {
        const u32 attr = plan->layout.attrs[slot];
        const rsx_vertex_fetch_attr* fetch = &plan->attr[attr];
        float* value = (float*)(out + slot * 16u);
        if (!fetch->desc.type || !fetch->desc.size) {
            memcpy(value, fetch->default_value, 16);
            continue;
        }

        const u32 element = rsx_vertex_element_index(
            ref->vertex_id, ref->base_index, fetch->desc.frequency,
            (plan->divider_mask >> attr) & 1u);
        const u8* source = NULL;
        if (fetch->stable_base &&
            element >= fetch->stable_first && element <= fetch->stable_last) {
            source = fetch->stable_base +
                (u64)(element - fetch->stable_first) * fetch->stride;
        } else if (plan->inline_data) {
            /* An inline draw has no vertex array to fall back to: the array
             * offsets still hold whatever the last ordinary draw bound, and
             * reading them would feed this draw another mesh's vertices.
             * An attribute the stream does not supply gets its default,
             * which is what the hardware feeds. */
            source = NULL;
        } else if (plan->guest_ptr) {
            const u32 offset =
                plan->base_offset + fetch->desc.offset +
                element * fetch->stride;
            source = plan->guest_ptr(
                plan->guest_user, fetch->desc.location, offset,
                fetch->elem_size);
        }

        if (source && rsx_vertex_decode_element(
                fetch->desc.type, fetch->desc.size, source, value))
            continue;
        if (attr == 0)
            return 0;
        memcpy(value, fetch->default_value, 16);
    }
    return 1;
}

int rsx_vertex_fetch_all(
    const rsx_vertex_fetch_plan* plan, const rsx_vertex_ref* refs,
    u32 count, u8* out)
{
    if (!plan || (!refs && count))
        return 0;
    const u32 stride = plan->layout.stride;
    if (!plan->uniform_element) {
        for (u32 i = 0; i < count; i++)
            if (!rsx_vertex_fetch_one(plan, &refs[i],
                                      stride ? out + (u64)i * stride : NULL))
                return 0;
        return 1;
    }
    const u32 slots = plan->layout.count;
    for (u32 slot = 0; slot < slots; slot++) {
        const u32 attr = plan->layout.attrs[slot];
        const rsx_vertex_fetch_attr* fetch = &plan->attr[attr];
        u8* dst_base = out + (size_t)slot * 16u;
        if (!fetch->desc.type || !fetch->desc.size) {
            const float d0 = fetch->default_value[0], d1 = fetch->default_value[1];
            const float d2 = fetch->default_value[2], d3 = fetch->default_value[3];
            for (u32 i = 0; i < count; i++) {
                float* v = (float*)(dst_base + (u64)i * stride);
                v[0] = d0; v[1] = d1; v[2] = d2; v[3] = d3;
            }
            continue;
        }
        if (fetch->stable_base) {
            const u8* const sb = fetch->stable_base;
            const u32 sfirst = fetch->stable_first;
            const u32 slast = fetch->stable_last;
            const u32 fstride = fetch->stride;
            const u32 type = fetch->desc.type;
            const u32 sz = fetch->desc.size < 4u ? fetch->desc.size : 4u;
            int all_in = 1;
            for (u32 i = 0; i < count; i++) {
                const u32 el = (refs[i].vertex_id + refs[i].base_index) & 0x000FFFFFu;
                if (el < sfirst || el > slast) { all_in = 0; break; }
            }
            if (all_in) {
                switch (type) {
                case RSX_VTX_TYPE_FLOAT:
                    for (u32 i = 0; i < count; i++) {
                        const u32 el = (refs[i].vertex_id + refs[i].base_index) & 0x000FFFFFu;
                        const u8* src = sb + (u64)(el - sfirst) * fstride;
                        float* v = (float*)(dst_base + (u64)i * stride);
                        v[0] = rsx_compact_be_f32(src);
                        v[1] = sz > 1u ? rsx_compact_be_f32(src + 4) : 0.0f;
                        v[2] = sz > 2u ? rsx_compact_be_f32(src + 8) : 0.0f;
                        v[3] = sz > 3u ? rsx_compact_be_f32(src + 12) : 1.0f;
                    }
                    continue;
                case RSX_VTX_TYPE_HALF:
                    for (u32 i = 0; i < count; i++) {
                        const u32 el = (refs[i].vertex_id + refs[i].base_index) & 0x000FFFFFu;
                        const u8* src = sb + (u64)(el - sfirst) * fstride;
                        float* v = (float*)(dst_base + (u64)i * stride);
                        v[0] = rsx_compact_be_f16(src);
                        v[1] = sz > 1u ? rsx_compact_be_f16(src + 2) : 0.0f;
                        v[2] = sz > 2u ? rsx_compact_be_f16(src + 4) : 0.0f;
                        v[3] = sz > 3u ? rsx_compact_be_f16(src + 6) : 1.0f;
                    }
                    continue;
                case RSX_VTX_TYPE_UNORM8:
                    for (u32 i = 0; i < count; i++) {
                        const u32 el = (refs[i].vertex_id + refs[i].base_index) & 0x000FFFFFu;
                        const u8* src = sb + (u64)(el - sfirst) * fstride;
                        float* v = (float*)(dst_base + (u64)i * stride);
                        v[0] = (float)src[0] * (1.0f / 255.0f);
                        v[1] = sz > 1u ? (float)src[1] * (1.0f / 255.0f) : 0.0f;
                        v[2] = sz > 2u ? (float)src[2] * (1.0f / 255.0f) : 0.0f;
                        v[3] = sz > 3u ? (float)src[3] * (1.0f / 255.0f) : 1.0f;
                    }
                    continue;
                case RSX_VTX_TYPE_UINT8:
                    for (u32 i = 0; i < count; i++) {
                        const u32 el = (refs[i].vertex_id + refs[i].base_index) & 0x000FFFFFu;
                        const u8* src = sb + (u64)(el - sfirst) * fstride;
                        float* v = (float*)(dst_base + (u64)i * stride);
                        v[0] = (float)src[0];
                        v[1] = sz > 1u ? (float)src[1] : 0.0f;
                        v[2] = sz > 2u ? (float)src[2] : 0.0f;
                        v[3] = sz > 3u ? (float)src[3] : 1.0f;
                    }
                    continue;
                case RSX_VTX_TYPE_SNORM16:
                    for (u32 i = 0; i < count; i++) {
                        const u32 el = (refs[i].vertex_id + refs[i].base_index) & 0x000FFFFFu;
                        const u8* src = sb + (u64)(el - sfirst) * fstride;
                        float* v = (float*)(dst_base + (u64)i * stride);
                        v[0] = (float)(s16)((src[0] << 8) | src[1]) * (1.0f / 32767.0f);
                        v[1] = sz > 1u ? (float)(s16)((src[2] << 8) | src[3]) * (1.0f / 32767.0f) : 0.0f;
                        v[2] = sz > 2u ? (float)(s16)((src[4] << 8) | src[5]) * (1.0f / 32767.0f) : 0.0f;
                        v[3] = sz > 3u ? (float)(s16)((src[6] << 8) | src[7]) * (1.0f / 32767.0f) : 1.0f;
                    }
                    continue;
                case RSX_VTX_TYPE_SINT16:
                    for (u32 i = 0; i < count; i++) {
                        const u32 el = (refs[i].vertex_id + refs[i].base_index) & 0x000FFFFFu;
                        const u8* src = sb + (u64)(el - sfirst) * fstride;
                        float* v = (float*)(dst_base + (u64)i * stride);
                        v[0] = (float)(s16)((src[0] << 8) | src[1]);
                        v[1] = sz > 1u ? (float)(s16)((src[2] << 8) | src[3]) : 0.0f;
                        v[2] = sz > 2u ? (float)(s16)((src[4] << 8) | src[5]) : 0.0f;
                        v[3] = sz > 3u ? (float)(s16)((src[6] << 8) | src[7]) : 1.0f;
                    }
                    continue;
                case RSX_VTX_TYPE_CMP32:
                    for (u32 i = 0; i < count; i++) {
                        const u32 el = (refs[i].vertex_id + refs[i].base_index) & 0x000FFFFFu;
                        const u8* src = sb + (u64)(el - sfirst) * fstride;
                        u32 word;
                        memcpy(&word, src, 4);
                        word = __builtin_bswap32(word);
                        s32 x = (s32)(word & 0x7FFu);
                        s32 y = (s32)((word >> 11) & 0x7FFu);
                        s32 z = (s32)((word >> 22) & 0x3FFu);
                        if (x & 0x400) x -= 0x800;
                        if (y & 0x400) y -= 0x800;
                        if (z & 0x200) z -= 0x400;
                        float* v = (float*)(dst_base + (u64)i * stride);
                        v[0] = (float)x * (1.0f / 1023.0f);
                        v[1] = (float)y * (1.0f / 1023.0f);
                        v[2] = (float)z * (1.0f / 511.0f);
                        v[3] = 1.0f;
                    }
                    continue;
                default:
                    break;
                }
            }
        }
        for (u32 i = 0; i < count; i++) {
            const rsx_vertex_ref* ref = &refs[i];
            const u32 element = rsx_vertex_element_index(ref->vertex_id,
                                                        ref->base_index, 1, 0);
            float* value = (float*)(dst_base + (u64)i * stride);
            const u8* source = NULL;
            if (fetch->stable_base && element >= fetch->stable_first &&
                element <= fetch->stable_last) {
                source = fetch->stable_base +
                    (u64)(element - fetch->stable_first) * fetch->stride;
            } else if (!plan->inline_data && plan->guest_ptr) {
                const u32 offset = plan->base_offset + fetch->desc.offset +
                    element * fetch->stride;
                source = plan->guest_ptr(plan->guest_user,
                                         fetch->desc.location, offset,
                                         fetch->elem_size);
            }
            if (source && rsx_vertex_decode_element(fetch->desc.type,
                                                    fetch->desc.size, source,
                                                    value))
                continue;
            if (attr == 0)
                return 0;
            memcpy(value, fetch->default_value, 16);
        }
    }
    return 1;
}

static u64 rsx_vertex_remap_hash(u64 value)
{
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdull;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ull;
    value ^= value >> 33;
    return value;
}

int rsx_vertex_remap_build(
    rsx_vertex_remap* remap, rsx_vertex_ref* refs, u32 count,
    u32* unique_count)
{
    if (!remap || (!refs && count) || !unique_count)
        return 0;
    *unique_count = count;
    if (!count)
        return 1;

    if (count > remap->occurrence_capacity) {
        u32* occurrences = (u32*)realloc(
            remap->occurrence_to_unique, (size_t)count * sizeof(u32));
        if (!occurrences)
            return 0;
        remap->occurrence_to_unique = occurrences;
        remap->occurrence_capacity = count;
    }

    /* Fast path: when all refs share one base_index and span a compact
     * vertex_id range (<= 65536, true for almost every PS3 u16 mesh), use a
     * flat direct-indexed lookup table instead of a hash table. */
    {
        const u32 base0 = refs[0].base_index;
        u32 min_v = refs[0].vertex_id, max_v = min_v;
        int same_base = 1;
        for (u32 i = 1; i < count; i++) {
            if (refs[i].base_index != base0) { same_base = 0; break; }
            const u32 v = refs[i].vertex_id;
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
        const u64 span = (u64)max_v - (u64)min_v + 1ull;
        if (same_base && span <= 65536ull) {
            u32 need = 16u;
            while (need < (u32)span) need *= 2u;
            if (need > remap->slot_capacity) {
                rsx_vertex_remap_slot* slots = (rsx_vertex_remap_slot*)realloc(
                    remap->slots, (size_t)need * sizeof(*slots));
                if (!slots)
                    return 0;
                remap->slots = slots;
                remap->slot_capacity = need;
                memset(remap->slots, 0,
                       (size_t)remap->slot_capacity * sizeof(*remap->slots));
                remap->generation = 1u;
            } else {
                remap->generation++;
                if (!remap->generation) {
                    memset(remap->slots, 0,
                           (size_t)remap->slot_capacity * sizeof(*remap->slots));
                    remap->generation = 1u;
                }
            }
            const u32 gen = remap->generation;
            rsx_vertex_remap_slot* const slots = remap->slots;
            u32* const occ = remap->occurrence_to_unique;
            u32 unique = 0;
            for (u32 i = 0; i < count; i++) {
                const u32 idx = refs[i].vertex_id - min_v;
                rsx_vertex_remap_slot* slot = &slots[idx];
                if (slot->generation != gen) {
                    slot->generation = gen;
                    slot->value = unique;
                    refs[unique].vertex_id = refs[i].vertex_id;
                    refs[unique].base_index = base0;
                    occ[i] = unique;
                    unique++;
                } else {
                    occ[i] = slot->value;
                }
            }
            *unique_count = unique;
            return 1;
        }
    }

    if (count > UINT_MAX / 2u)
        return 0;
    u32 slot_capacity = 16u;
    const u32 required_slots = count * 2u;
    while (slot_capacity < required_slots) {
        if (slot_capacity > UINT_MAX / 2u)
            return 0;
        slot_capacity *= 2u;
    }
    if (slot_capacity > remap->slot_capacity) {
        rsx_vertex_remap_slot* slots = (rsx_vertex_remap_slot*)realloc(
            remap->slots, (size_t)slot_capacity * sizeof(*slots));
        if (!slots)
            return 0;
        remap->slots = slots;
        remap->slot_capacity = slot_capacity;
        memset(remap->slots, 0,
               (size_t)remap->slot_capacity * sizeof(*remap->slots));
        remap->generation = 1u;
    } else {
        remap->generation++;
        if (!remap->generation) {
            memset(remap->slots, 0,
                   (size_t)remap->slot_capacity * sizeof(*remap->slots));
            remap->generation = 1u;
        }
    }

    u32 unique = 0;
    const u32 mask = remap->slot_capacity - 1u;
    for (u32 occurrence = 0; occurrence < count; occurrence++) {
        const rsx_vertex_ref ref = refs[occurrence];
        const u64 key = ((u64)ref.base_index << 32) | ref.vertex_id;
        u32 slot_index = (u32)rsx_vertex_remap_hash(key) & mask;
        for (;;) {
            rsx_vertex_remap_slot* slot = &remap->slots[slot_index];
            if (slot->generation != remap->generation) {
                slot->key = key;
                slot->value = unique;
                slot->generation = remap->generation;
                refs[unique] = ref;
                remap->occurrence_to_unique[occurrence] = unique;
                unique++;
                break;
            }
            if (slot->key == key) {
                remap->occurrence_to_unique[occurrence] = slot->value;
                break;
            }
            slot_index = (slot_index + 1u) & mask;
        }
    }
    *unique_count = unique;
    return 1;
}

u32 rsx_vertex_remap_index(
    const rsx_vertex_remap* remap, u32 occurrence)
{
    return remap && occurrence < remap->occurrence_capacity
        ? remap->occurrence_to_unique[occurrence] : occurrence;
}

void rsx_vertex_remap_destroy(rsx_vertex_remap* remap)
{
    if (!remap)
        return;
    free(remap->occurrence_to_unique);
    free(remap->slots);
    memset(remap, 0, sizeof(*remap));
}

int rsx_vertex_topology_plan(
    u32 primitive, int refs_remapped, int* indexed)
{
    if (!indexed)
        return 0;
    switch (primitive) {
    case RSX_PRIMITIVE_TRIANGLES:
        *indexed = refs_remapped != 0;
        return 1;
    case RSX_PRIMITIVE_TRIANGLE_STRIP:
    case RSX_PRIMITIVE_TRIANGLE_FAN:
    case RSX_PRIMITIVE_QUADS:
        *indexed = 1;
        return 1;
    default:
        *indexed = 0;
        return 0;
    }
}
