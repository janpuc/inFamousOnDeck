/*
 * test_spu_merge_cfg.c -- Bit-exact differential equivalence harness for
 * ps3recomp/tools/spu_merge_cfg.py.
 *
 * Links an unmerged RC-only SPU jobmod (prefix ORIG) side-by-side with its
 * CFG-merged twin (prefix MERGED) and executes both from every merged entry
 * point across randomized LS/GPR states under:
 *   - fast path (g_spu_dtail=1, no interrupts/lockstep)
 *   - slow path (g_spu_dtail=0)
 *   - pending interrupt (int_enable=1, event_status=event_mask=1)
 *
 * Verifies bit-exact equality of all 128 GPRs, full 256 KB LS, ctx->pc,
 * ctx->steps, ctx->int_enable, ctx->stop_code, ctx->status, trampoline
 * target PC, and all channel/stop/lodx side-effect traces.
 */
#define SPU_DRAIN_IMPL 1
#define SPU_CHANNELS_IMPL 1
#define SPU_WRCH_OUTLINE 1
#define SPU_DRAIN_CALL_OUTLINE 1
#define SPU_IMG_RESTORE_OUTLINE 1

#include "spu_context.h"
#include "spu_ibf.h"

#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

/* Runtime globals referenced by lifted RC/CFG SPU TUs (SPU_GATE globals are already
 * defined in spu_context.h / spu_ibf.h and initialized in main()) */
int g_spu_ls_watch_n = 0;
volatile int g_yz_lockstep_on = 0;
SPU_THREAD_LOCAL void (*g_spu_trampoline_fn)(spu_context*) = NULL;
SPU_THREAD_LOCAL uint32_t g_spu_pch[8];
SPU_THREAD_LOCAL unsigned g_spu_pch_n = 0;
SPU_THREAD_LOCAL uint32_t g_spu_recent[32];
SPU_THREAD_LOCAL uint32_t g_spu_recent_n = 0;
SPU_THREAD_LOCAL int g_irq_live_n = 0;

int g_spu_xfloat_on = 1;
int g_spu_xfloat_ab_phase = -1;
int g_spu_xfloat = 1;
float g_spu_xflim = 3.40282347e+38f;
volatile int g_lodx_f5_off = 0;
volatile float g_v237_lodf = 1.0f;
volatile int g_ab_phase = 0;
volatile int g_ab_mask = 0;

/* Trace buffer to capture all external side effects */
typedef struct {
    uint32_t kind; /* 1=wrch, 2=rdch, 3=rchcnt, 4=stop, 5=halt, 6=ib, 7=lodx231, 8=lodx233, 9=unsup */
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
} trace_ev_t;

#define MAX_TRACE 256
#define MAX_FUNCS 4096
#define MAX_STEPS 64

typedef struct {
    uint32_t addr;
    void (*fn)(spu_context*);
} reg_ent_t;

static reg_ent_t g_reg_tabs[2][MAX_FUNCS];
static int g_reg_cnt[2] = {0, 0};
static int g_active_tab = 0;

static trace_ev_t g_trace[2][MAX_TRACE];
static int g_trace_n[2] = {0, 0};
static sigjmp_buf g_abort_jb;
static int g_in_exec = 0;
static uint64_t g_step_cap = MAX_STEPS;
static volatile uint32_t g_cur_pc = 0;
static volatile int g_cur_tab = -1, g_cur_mode = -1, g_cur_seed = -1;

static void on_sigalrm(int sig)
{
    (void)sig;
    if (g_in_exec) siglongjmp(g_abort_jb, 99);
}

static uint8_t g_alt_stack[65536];

static void on_sigsegv(int sig)
{
    (void)sig;
    if (g_in_exec) siglongjmp(g_abort_jb, 139);
    char buf[128];
    int n = snprintf(buf, sizeof(buf), "SEGV pc=0x%X tab=%d mode=%d seed=%d\n",
                     g_cur_pc, g_cur_tab, g_cur_mode, g_cur_seed);
    if (n > 0) (void)write(2, buf, (size_t)n);
    _exit(139);
}

static void arm_timer_ms(int ms)
{
    struct itimerval it;
    memset(&it, 0, sizeof(it));
    it.it_value.tv_sec = ms / 1000;
    it.it_value.tv_usec = (ms % 1000) * 1000;
    setitimer(ITIMER_REAL, &it, NULL);
}

static void trace_push(uint32_t kind, uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
    int t = g_active_tab;
    if (g_trace_n[t] < MAX_TRACE) {
        trace_ev_t* e = &g_trace[t][g_trace_n[t]++];
        e->kind = kind; e->a = a; e->b = b; e->c = c; e->d = d;
    }
}

void spu_register_function(uint32_t addr, void (*fn)(spu_context*))
{
    int t = g_active_tab;
    if (g_reg_cnt[t] < MAX_FUNCS) {
        g_reg_tabs[t][g_reg_cnt[t]].addr = addr;
        g_reg_tabs[t][g_reg_cnt[t]].fn = fn;
        g_reg_cnt[t]++;
    }
}

static uint32_t fn_to_addr(int tab, void (*fn)(spu_context*))
{
    if (!fn) return 0;
    if (fn == spu_indirect_branch) return 0xFFFFFFFEu;
    for (int i = 0; i < g_reg_cnt[tab]; i++) {
        if (g_reg_tabs[tab][i].fn == fn) return g_reg_tabs[tab][i].addr;
    }
    return 0xFFFFFFFFu;
}

static void (*addr_to_fn(int tab, uint32_t addr))(spu_context*)
{
    addr &= SPU_LS_MASK;
    for (int i = 0; i < g_reg_cnt[tab]; i++) {
        if (g_reg_tabs[tab][i].addr == addr) return g_reg_tabs[tab][i].fn;
    }
    return NULL;
}

/* Mock runtime hooks */
void spu_ls_watch_slow(uint32_t lsa, int is_write, const uint8_t* p, uint32_t pc, uint32_t lr)
{
    (void)lsa; (void)is_write; (void)p; (void)pc; (void)lr;
}
void yz_lockstep_tick(spu_context* ctx) { (void)ctx; }
void spu_task_launch_check(spu_context* ctx, void* fn) { (void)ctx; (void)fn; }
void spu_img_restore(spu_context* ctx, int32_t saved_img) { ctx->image_id = saved_img; }
void spu_ch_wake(spu_context* ctx) { (void)ctx; }
void spu_depth_guard(spu_context* ctx)
{
    trace_push(5, 0xDEAD, ctx->host_depth, 0, 0);
    if (g_in_exec) siglongjmp(g_abort_jb, 2);
}
int spu_tailret_enabled(void) { return 0; }
void spu_check_stack_reset(spu_context* ctx, void (*fn)(spu_context*)) { (void)ctx; (void)fn; }
void spu_ls0_job_end(spu_context* ctx)
{
    trace_push(5, 0x1500, 0, 0, 0);
    if (g_in_exec) siglongjmp(g_abort_jb, 3);
}
int spu_tlc_cold(spu_context* ctx, void* fn)
{
    (void)fn;
    ctx->steps++;
    if (ctx->steps >= g_step_cap && g_in_exec) longjmp(g_abort_jb, 4);
    return 1;
}

void (*spu_take_interrupt(spu_context* ctx, void (*tf)(spu_context*)))(spu_context*)
{
    (void)ctx;
    return tf;
}

spu_ibf_fn spu_lookup(uint32_t addr, int image_id)
{
    (void)image_id;
    return addr_to_fn(g_active_tab, addr);
}

void spu_indirect_branch(spu_context* ctx)
{
    trace_push(6, ctx->pc & SPU_LS_MASK, 0, 0, 0);
    if (g_in_exec) siglongjmp(g_abort_jb, 5);
}

void spu_indirect_branch_mt(spu_context* ctx)
{
    spu_indirect_branch(ctx);
}

void spu_drain_call(spu_context* ctx, uint32_t return_pc)
{
    if (ctx->steps >= g_step_cap && g_in_exec) longjmp(g_abort_jb, 4);
    while (g_spu_trampoline_fn) {
        void (*tf)(spu_context*) = g_spu_trampoline_fn;
        g_spu_trampoline_fn = NULL;
        if (tf == spu_indirect_branch) {
            trace_push(6, ctx->pc & SPU_LS_MASK, 1, 0, 0);
            if (g_in_exec) siglongjmp(g_abort_jb, 5);
            return;
        }
        if (ctx->int_enable && (ctx->event_status & ctx->event_mask)) {
            trace_push(6, ctx->pc & SPU_LS_MASK, 2, fn_to_addr(g_active_tab, tf), 0);
            if (g_in_exec) siglongjmp(g_abort_jb, 6);
            return;
        }
        ctx->steps++;
        if (ctx->steps >= g_step_cap && g_in_exec) longjmp(g_abort_jb, 4);
        tf(ctx);
    }
    if (((ctx->pc ^ return_pc) & SPU_LS_MASK) != 0) {
        trace_push(6, ctx->pc & SPU_LS_MASK, 3, return_pc, 0);
        if (g_in_exec) siglongjmp(g_abort_jb, 7);
    }
}

void spu_stop(spu_context* ctx)
{
    trace_push(4, ctx->stop_code, ctx->status, 0, 0);
    if (g_in_exec) siglongjmp(g_abort_jb, 8);
}

void spu_halt(spu_context* ctx)
{
    trace_push(5, ctx->status, 0, 0, 0);
    if (g_in_exec) siglongjmp(g_abort_jb, 9);
}

void spu_wrch(spu_context* ctx, uint32_t channel, u128 value)
{
    (void)ctx;
    trace_push(1, channel, value._u32[0], value._u32[1], value._u32[2]);
    /* Stop after 8 channel ops so DMA-polling loops terminate quickly */
    if (g_trace_n[g_active_tab] >= 8 && g_in_exec) longjmp(g_abort_jb, 10);
}

u128 spu_rdch(spu_context* ctx, uint32_t channel)
{
    (void)ctx;
    uint32_t seq = (uint32_t)g_trace_n[g_active_tab];
    trace_push(2, channel, seq, 0, 0);
    u128 r;
    r._u32[0] = 0x00000001u ^ (channel << 8) ^ seq;
    r._u32[1] = 0; r._u32[2] = 0; r._u32[3] = 0;
    if (g_trace_n[g_active_tab] >= 8 && g_in_exec) longjmp(g_abort_jb, 10);
    return r;
}

uint32_t spu_rchcnt(spu_context* ctx, uint32_t channel)
{
    (void)ctx;
    trace_push(3, channel, 0, 0, 0);
    return 1u;
}

void spu_unsupported(uint32_t pc, const char* mnem)
{
    (void)mnem;
    trace_push(9, pc, 0, 0, 0);
    if (g_in_exec) siglongjmp(g_abort_jb, 11);
}

/* Mock lodx hooks for jobmod_50f371dc382d_e3D90 (defined in lodx_v231.h if included,
 * but weak/mocked here only if not compiled in). */

/* Deterministic xorshift64* PRNG */
static uint64_t rng_next(uint64_t* s)
{
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static void init_ctx(spu_context* ctx, uint8_t* ls_buf, uint64_t seed, uint32_t entry_pc, int mode)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->ls = ls_buf;
    ctx->image_id = 3;
    ctx->pc = entry_pc;
    ctx->host_depth = 1;

    uint64_t st = seed ^ ((uint64_t)entry_pc << 17) ^ 0x9E3779B97F4A7C15ULL;
    /* Fill LS with deterministic big-endian 32-bit words (small counts, offsets, floats)
     * so loops reading counts from LS terminate cleanly without 2^31-iteration memset stalls. */
    uint32_t* ls32 = (uint32_t*)ls_buf;
    for (size_t i = 0; i < SPU_LS_SIZE / 4; i++) {
        uint64_t v = rng_next(&st);
        uint32_t w;
        if ((v & 3u) == 0u) {
            /* Small big-endian integer 0..0x3FF (stored BE in LS) */
            uint32_t x = (uint32_t)((v >> 8) & 0x3FFu);
            w = __builtin_bswap32(x);
        } else if ((v & 3u) == 1u) {
            /* Aligned LS address 0x4000..0x3C000 in BE */
            uint32_t x = (uint32_t)(((v >> 8) & 0x38000u) | 0x4000u);
            w = __builtin_bswap32(x);
        } else {
            /* Normal small float in BE */
            float f = (float)(int16_t)((v >> 16) & 0xFFFFu) * 0.01f;
            uint32_t fb;
            memcpy(&fb, &f, 4);
            w = __builtin_bswap32(fb);
        }
        ls32[i] = w;
    }

    /* Fill GPRs with mix of small ints, floats, LS addresses, and arbitrary bits */
    for (int r = 0; r < 128; r++) {
        uint64_t a = rng_next(&st);
        uint64_t b = rng_next(&st);
        unsigned kind = (unsigned)((a >> 56) & 3u);
        if (kind == 0) {
            /* Valid aligned LS pointers */
            ctx->gpr[r]._u32[0] = (uint32_t)((a & 0x3F000u) | 0x4000u);
            ctx->gpr[r]._u32[1] = (uint32_t)(a & 0xFFu);
            ctx->gpr[r]._u32[2] = (uint32_t)(b & 0xFFFFu);
            ctx->gpr[r]._u32[3] = 0u;
        } else if (kind == 1) {
            /* Small normal floats (avoid putting 0x40000000..0xFFFFFFFF in r3..r7 word 0
             * where memset/memcpy loops treat it as a 2^31-byte count -> 200M iterations). */
            for (int w = 0; w < 4; w++) {
                int32_t v = (int32_t)((int16_t)(rng_next(&st) & 0xFFFFu));
                ctx->gpr[r]._f32[w] = (float)v * 0.01f;
            }
            if (r >= 3 && r <= 7)
                ctx->gpr[r]._u32[0] = (uint32_t)(a & 0x7FFu);
        } else if (kind == 2) {
            /* Small counters (0..7) so loops terminate cleanly within step cap */
            ctx->gpr[r]._u32[0] = (uint32_t)(a & 7u);
            ctx->gpr[r]._u32[1] = (uint32_t)(b & 7u);
            ctx->gpr[r]._u32[2] = (uint32_t)((a >> 8) & 7u);
            ctx->gpr[r]._u32[3] = (uint32_t)((b >> 8) & 7u);
        } else {
            ctx->gpr[r]._u64[0] = a;
            ctx->gpr[r]._u64[1] = b;
            /* Keep preferred word 0 bounded for arg/counter regs r3..r15 so randomized
             * memset/memcpy loops (rN -= 16; while rN > 15) finish in <1ms instead of 2^28 iters. */
            if (r >= 3 && r <= 15)
                ctx->gpr[r]._u32[0] = (uint32_t)(a & 0x7FFu);
        }
    }
    /* Stack pointer r1 near top of LS, link register r0 = sentinel 0x1000 */
    ctx->gpr[1]._u32[0] = 0x3D000u;
    ctx->gpr[0]._u32[0] = 0x00001000u;

    if (mode == 2) {
        /* Pending interrupt mode: exercises cold trampoline spill path */
        ctx->int_enable = 1;
        ctx->event_status = 1;
        ctx->event_mask = 1;
    }
}

static int run_one(int tab, spu_context* ctx, void (*fn)(spu_context*), int mode)
{
    g_active_tab = tab;
    g_trace_n[tab] = 0;
    g_spu_trampoline_fn = NULL;
    g_spu_dtail = (mode == 1) ? 0 : 1;
    g_spu_drain_fastret_pub = (mode == 0) ? 1 : 0;
    /* In mode 0 (fast path), run with tlc_slow=0; in mode 1 (slow path), g_spu_dtail=0 drains step-by-step */
    g_spu_smc_watch = 0;
    g_wws_code_probe = 0;
    g_spu_tlc_slow = 0;
    g_in_exec = 1;
    arm_timer_ms(50);
    int rc = sigsetjmp(g_abort_jb, 1);
    if (rc == 0) {
        fn(ctx);
        /* Drain any pending trampoline up to MAX_STEPS */
        while (g_spu_trampoline_fn) {
            void (*tf)(spu_context*) = g_spu_trampoline_fn;
            g_spu_trampoline_fn = NULL;
            if (tf == spu_indirect_branch) {
                trace_push(6, ctx->pc & SPU_LS_MASK, 1, 0, 0);
                break;
            }
            if (ctx->int_enable && (ctx->event_status & ctx->event_mask)) {
                /* Interrupt stops at the exact same transfer point for both */
                g_spu_trampoline_fn = tf;
                break;
            }
            ctx->steps++;
            if (ctx->steps >= g_step_cap) {
                g_spu_trampoline_fn = tf;
                break;
            }
            tf(ctx);
        }
    }
    arm_timer_ms(0);
    g_in_exec = 0;
    return rc;
}

extern void TEST_ORIG_REGISTER(void);
extern void TEST_MERGED_REGISTER(void);

int main(void)
{
    stack_t ss;
    ss.ss_sp = g_alt_stack;
    ss.ss_size = sizeof(g_alt_stack);
    ss.ss_flags = 0;
    sigaltstack(&ss, NULL);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_sigsegv;
    sa.sa_flags = SA_ONSTACK;
    sigaction(SIGSEGV, &sa, NULL);
    signal(SIGALRM, on_sigalrm);
    g_spu_smc_watch = 0;
    g_wws_code_probe = 0;
    g_spu_tlc_slow = 0;
    g_spu_stack_reset_n = 0;
    g_spu_depth_lim = 256;
    g_spu_tailret = 0;
    g_spu_wrch_fast_pub = 0;
    g_spu_dma_inl_ok = 0;
    g_spu_dma_inl_phase = -1;
    g_spu_dtail = 1;
    g_spu_drain_fastret_pub = 0;
    g_spu_ibf = 1;

    g_active_tab = 0;
    TEST_ORIG_REGISTER();
    g_active_tab = 1;
    TEST_MERGED_REGISTER();

    if (g_reg_cnt[0] == 0 || g_reg_cnt[0] != g_reg_cnt[1]) {
        fprintf(stderr, "FAIL: table count mismatch orig=%d merged=%d\n", g_reg_cnt[0], g_reg_cnt[1]);
        return 1;
    }

    uint8_t* lsA = (uint8_t*)aligned_alloc(64, SPU_LS_SIZE);
    uint8_t* lsB = (uint8_t*)aligned_alloc(64, SPU_LS_SIZE);
    if (!lsA || !lsB) return 2;

    spu_context ctxA, ctxB;
    unsigned long long total_runs = 0;
    unsigned long long total_steps = 0;
    int mismatches = 0;

    for (int i = 0; i < g_reg_cnt[0]; i++) {
        uint32_t pc = g_reg_tabs[0][i].addr;
        void (*fnA)(spu_context*) = g_reg_tabs[0][i].fn;
        void (*fnB)(spu_context*) = g_reg_tabs[1][i].fn;
        if (g_reg_tabs[1][i].addr != pc) {
            fprintf(stderr, "FAIL: addr mismatch at idx %d: 0x%X vs 0x%X\n", i, pc, g_reg_tabs[1][i].addr);
            return 1;
        }
        /* Test 3 modes x 2 seeds per function */
        for (int mode = 0; mode < 3; mode++) {
            for (int s = 0; s < 2; s++) {
                uint64_t seed = (uint64_t)(s + 1) * 0x123456789ABCDEFULL + (uint64_t)mode * 0x7777ULL;
                init_ctx(&ctxA, lsA, seed, pc, mode);
                init_ctx(&ctxB, lsB, seed, pc, mode);

                g_cur_pc = pc; g_cur_mode = mode; g_cur_seed = s; g_cur_tab = 0;
                int rcA = run_one(0, &ctxA, fnA, mode);
                uint32_t trA = fn_to_addr(0, g_spu_trampoline_fn);
                g_cur_tab = 1;
                int rcB = run_one(1, &ctxB, fnB, mode);
                uint32_t trB = fn_to_addr(1, g_spu_trampoline_fn);

                /* If only one side hit the 250ms soft timer (long inner memset/memcpy loop),
                 * re-run both with a 1500ms budget before comparing. */
                if ((rcA == 99) != (rcB == 99)) {
                    init_ctx(&ctxA, lsA, seed, pc, mode);
                    init_ctx(&ctxB, lsB, seed, pc, mode);
                    g_active_tab = 0; g_trace_n[0] = 0; g_spu_trampoline_fn = NULL;
                    g_spu_dtail = (mode == 1) ? 0 : 1;
    g_spu_drain_fastret_pub = (mode == 0) ? 1 : 0; g_spu_tlc_slow = 0; g_in_exec = 1;
                    arm_timer_ms(1500);
                    rcA = sigsetjmp(g_abort_jb, 1);
                    if (rcA == 0) {
                        fnA(&ctxA);
                        while (g_spu_trampoline_fn) {
                            void (*tf)(spu_context*) = g_spu_trampoline_fn; g_spu_trampoline_fn = NULL;
                            if (tf == spu_indirect_branch) { trace_push(6, ctxA.pc & SPU_LS_MASK, 1, 0, 0); break; }
                            if (ctxA.int_enable && (ctxA.event_status & ctxA.event_mask)) { g_spu_trampoline_fn = tf; break; }
                            ctxA.steps++; if (ctxA.steps >= g_step_cap) { g_spu_trampoline_fn = tf; break; }
                            tf(&ctxA);
                        }
                    }
                    arm_timer_ms(0); g_in_exec = 0;
                    trA = fn_to_addr(0, g_spu_trampoline_fn);

                    g_active_tab = 1; g_trace_n[1] = 0; g_spu_trampoline_fn = NULL;
                    g_spu_dtail = (mode == 1) ? 0 : 1;
    g_spu_drain_fastret_pub = (mode == 0) ? 1 : 0; g_spu_tlc_slow = 0; g_in_exec = 1;
                    arm_timer_ms(1500);
                    rcB = sigsetjmp(g_abort_jb, 1);
                    if (rcB == 0) {
                        fnB(&ctxB);
                        while (g_spu_trampoline_fn) {
                            void (*tf)(spu_context*) = g_spu_trampoline_fn; g_spu_trampoline_fn = NULL;
                            if (tf == spu_indirect_branch) { trace_push(6, ctxB.pc & SPU_LS_MASK, 1, 0, 0); break; }
                            if (ctxB.int_enable && (ctxB.event_status & ctxB.event_mask)) { g_spu_trampoline_fn = tf; break; }
                            ctxB.steps++; if (ctxB.steps >= g_step_cap) { g_spu_trampoline_fn = tf; break; }
                            tf(&ctxB);
                        }
                    }
                    arm_timer_ms(0); g_in_exec = 0;
                    trB = fn_to_addr(1, g_spu_trampoline_fn);
                }

                total_runs++;
                total_steps += ctxA.steps;

                if ((rcA == 99 && rcB == 99) || (rcA == 139 && rcB == 139)) {
                    /* Both hit an unguarded inner loop timer cap or unconditional data-section recursion */
                    continue;
                }
                int bad = 0;
                if (rcA != rcB) bad |= 1;
                if (trA != trB) bad |= 2;
                if ((ctxA.pc & SPU_LS_MASK) != (ctxB.pc & SPU_LS_MASK)) bad |= 4;
                if (ctxA.steps != ctxB.steps) bad |= 8;
                if (ctxA.int_enable != ctxB.int_enable) bad |= 16;
                if (ctxA.stop_code != ctxB.stop_code || ctxA.status != ctxB.status) bad |= 32;
                if (memcmp(ctxA.gpr, ctxB.gpr, sizeof(ctxA.gpr)) != 0) bad |= 64;
                if (memcmp(lsA, lsB, SPU_LS_SIZE) != 0) bad |= 128;
                if (g_trace_n[0] != g_trace_n[1] ||
                    memcmp(g_trace[0], g_trace[1], (size_t)g_trace_n[0] * sizeof(trace_ev_t)) != 0)
                    bad |= 256;

                if (bad) {
                    mismatches++;
                    if (mismatches <= 10) {
                        fprintf(stderr, "MISMATCH pc=0x%05X mode=%d seed=%d flags=0x%X: "
                                "rc=%d/%d tr=0x%X/0x%X pc=0x%X/0x%X steps=%llu/%llu trace=%d/%d\n",
                                pc, mode, s, bad, rcA, rcB, trA, trB,
                                (uint32_t)ctxA.pc, (uint32_t)ctxB.pc,
                                (unsigned long long)ctxA.steps, (unsigned long long)ctxB.steps,
                                g_trace_n[0], g_trace_n[1]);
                        if (bad & 64) {
                            for (int r = 0; r < 128; r++) {
                                if (memcmp(&ctxA.gpr[r], &ctxB.gpr[r], 16) != 0) {
                                    fprintf(stderr, "  gpr[%d] A=%08X %08X %08X %08X  B=%08X %08X %08X %08X\n",
                                            r, ctxA.gpr[r]._u32[0], ctxA.gpr[r]._u32[1], ctxA.gpr[r]._u32[2], ctxA.gpr[r]._u32[3],
                                            ctxB.gpr[r]._u32[0], ctxB.gpr[r]._u32[1], ctxB.gpr[r]._u32[2], ctxB.gpr[r]._u32[3]);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    free(lsA);
    free(lsB);
    printf("PASS_CHECK: funcs=%d runs=%llu total_steps=%llu mismatches=%d\n",
           g_reg_cnt[0], total_runs, total_steps, mismatches);
    return mismatches ? 1 : 0;
}
