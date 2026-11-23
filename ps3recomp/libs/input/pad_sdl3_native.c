/* SDL3 gamepads, including Steam Input's virtual gamepad, plus a keyboard
 * fallback when no controller is connected. SDL3 owns hotplug enumeration. */
#ifdef PS3RECOMP_SDL3_NATIVE

#include "pad_sdl3_native.h"
#include "cellPad.h"

#include <SDL3/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NATIVE_PADS 4

/* Sensor scale: libpad reports the SIXAXIS accelerometer as 10 bit values, 512 at rest, about 113 counts per g, and
 * the yaw gyro on the same 512 centre. SDL3 reports m/s^2 and rad/s in the gamepad frame (x right, y toward the far
 * edge, z out of the face; flat on a table face up reads z = +1 g). Mapping to the DualShock 3 frame: X = right,
 * Y = into the face (so flat reads 512 - 113 = 399, the value cellPad always reported), Z = far edge. */
#define SIXAXIS_COUNTS_PER_G 113.0f
#define SIXAXIS_GYRO_COUNTS_PER_DPS 1.0f   /* best effort: games use tilt, the gyro word is rarely read */

typedef struct {
    SDL_Gamepad* gp;
    SDL_JoystickID id;
    int has_accel, has_gyro;
    uint16_t rumble_req;          /* small << 8 | large, what the title asked for */
    uint16_t rumble_sent;         /* what the pad was last told, only advanced when SDL accepted it */
    Uint64 rumble_at;             /* refresh time while a motor is on */
    Uint64 rumble_retry_at;       /* after a failed SDL_RumbleGamepad: do not hammer the driver before this */
    int has_rumble;               /* SDL_PROP_GAMEPAD_CAP_RUMBLE_BOOLEAN, logged on connect */
} NativePad;

static NativePad s_pads[NATIVE_PADS];
static PadSdlNativePad s_state[NATIVE_PADS];
static int s_inited;
static Uint64 s_next_scan;
static SDL_AtomicInt s_hotplug;
static int s_gyro = 1, s_rumble = 1;
static int s_rumble_first_ok, s_rumble_fail_logged, s_act_logged;   /* one-time diagnostics, under s_mu */

/* s_pads[] is written by the pad thread (poll: hotplug, disconnect) and read
 * by the guest thread (get, rumble from a HLE call), so every entry point
 * takes this lock. close_pad/scan_pads run only with it held. */
static SDL_Mutex* s_mu;

static void pad_lock(void)   { if (s_mu) SDL_LockMutex(s_mu); }
static void pad_unlock(void) { if (s_mu) SDL_UnlockMutex(s_mu); }

static uint8_t axis_u8(Sint16 v)
{
    const int x = ((int)v + 32768) * 255 / 65535;
    return (uint8_t)(x < 0 ? 0 : x > 255 ? 255 : x);
}

static uint8_t trigger_u8(Sint16 v)
{
    const int x = (int)v * 255 / 32767;
    return (uint8_t)(x < 0 ? 0 : x > 255 ? 255 : x);
}

static uint16_t sens10(float v)
{
    const int x = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
    return (uint16_t)(x < 0 ? 0 : x > 1023 ? 1023 : x);
}

/* SDL drops rumble after the duration it was given; a PS3 motor holds its setting until the next SetActDirect.
 * Send on change and refresh while it is on (400 ms refresh, 600 ms duration), as the Windows backend does. A change
 * is never coalesced: rumble() calls this for every SetActDirect, so a start and the stop right behind it both reach
 * the pad. rumble_sent only advances once SDL accepted the call, so a stop that failed is retried (not left running
 * until the refresh), and a failing pad is retried every 250 ms rather than every frame. */
static void apply_rumble(NativePad* p, Uint64 now)
{
    const uint16_t r = s_rumble ? p->rumble_req : 0;
    const int due = r && now >= p->rumble_at;
    if (r == p->rumble_sent && !due) return;
    if (now < p->rumble_retry_at) return;
    /* libpad: the small motor is on/off (0 or 1), the large one 0..255 */
    const Uint16 low = (Uint16)((r & 0xFF) * 257), high = (Uint16)((r >> 8) ? 0xFFFF : 0);
    const bool ok = SDL_RumbleGamepad(p->gp, low, high, r ? 600 : 0);
    if (r && !s_rumble_first_ok) {
        /* The first non-zero request is the end-to-end proof: the guest called cellPadSetActDirect, it reached SDL,
         * and the driver took it (or did not, with SDL's reason). */
        if (ok) { s_rumble_first_ok = 1; printf("[cellPad] SDL3: first rumble low=%u high=%u: SDL_RumbleGamepad returned true\n", low, high); }
        else if (s_rumble_fail_logged < 3) {
            s_rumble_fail_logged++;
            printf("[cellPad] SDL3: rumble low=%u high=%u: SDL_RumbleGamepad returned false (%s)\n", low, high, SDL_GetError());
        }
    }
    if (ok) {
        p->rumble_sent = r;
        p->rumble_at = now + 400;
        p->rumble_retry_at = 0;
    } else {
        p->rumble_retry_at = now + 250;
    }
}

static void close_pad(int i)
{
    const char* n = SDL_GetGamepadName(s_pads[i].gp);
    printf("[cellPad] SDL3: %s disconnected\n", n ? n : "gamepad");
    SDL_CloseGamepad(s_pads[i].gp);
    for (int k = i + 1; k < NATIVE_PADS; k++) s_pads[k - 1] = s_pads[k];
    memset(&s_pads[NATIVE_PADS - 1], 0, sizeof s_pads[0]);
}

static int pad_count(void)
{
    int n = 0;
    for (int i = 0; i < NATIVE_PADS; i++) if (s_pads[i].gp) n++;
    return n;
}

static void scan_pads(void)
{
    int count = 0;
    SDL_JoystickID* ids = SDL_GetGamepads(&count);
    if (!ids) return;
    int used = pad_count();
    for (int k = 0; k < count && used < NATIVE_PADS; k++) {
        int known = 0;
        for (int i = 0; i < NATIVE_PADS; i++) if (s_pads[i].gp && s_pads[i].id == ids[k]) known = 1;
        if (known) continue;
        SDL_Gamepad* gp = SDL_OpenGamepad(ids[k]);
        if (!gp) continue;
        int slot = 0;
        while (slot < NATIVE_PADS && s_pads[slot].gp) slot++;
        NativePad* p = &s_pads[slot];
        memset(p, 0, sizeof *p);
        p->gp = gp;
        p->id = ids[k];
        used++;
        if (s_gyro) {
            if (SDL_GamepadHasSensor(gp, SDL_SENSOR_ACCEL) && SDL_SetGamepadSensorEnabled(gp, SDL_SENSOR_ACCEL, true))
                p->has_accel = 1;
            if (SDL_GamepadHasSensor(gp, SDL_SENSOR_GYRO) && SDL_SetGamepadSensorEnabled(gp, SDL_SENSOR_GYRO, true))
                p->has_gyro = 1;
        }
        p->has_rumble = SDL_GetBooleanProperty(SDL_GetGamepadProperties(gp), SDL_PROP_GAMEPAD_CAP_RUMBLE_BOOLEAN, false);
        const char* name = SDL_GetGamepadName(gp);
        printf("[cellPad] SDL3: %s connected (slot %d, vid:pid %04x:%04x, motion: %s%s%s, rumble: %s)\n",
               name ? name : "gamepad", slot, SDL_GetGamepadVendor(gp), SDL_GetGamepadProduct(gp),
               p->has_accel ? "accel" : "", p->has_accel && p->has_gyro ? "+" : "",
               p->has_gyro ? "gyro" : (p->has_accel ? "" : "none"), p->has_rumble ? "yes" : "no");
    }
    SDL_free(ids);
}

static void read_sensors(NativePad* p, PadSdlNativePad* o)
{
    o->sens_x = 512; o->sens_y = 399; o->sens_z = 512; o->sens_g = 512;
    if (!p->has_accel) return;
    float a[3];
    if (SDL_GetGamepadSensorData(p->gp, SDL_SENSOR_ACCEL, a, 3)) {
        const float k = SIXAXIS_COUNTS_PER_G / SDL_STANDARD_GRAVITY;
        o->sens_x = sens10(512.0f + a[0] * k);
        o->sens_y = sens10(512.0f - a[2] * k);
        o->sens_z = sens10(512.0f + a[1] * k);
        o->has_sensors = 1;
    }
    float g[3];
    if (p->has_gyro && SDL_GetGamepadSensorData(p->gp, SDL_SENSOR_GYRO, g, 3)) {
        /* rotation about the face normal (SDL z); the DS3 axis points into the face, so the sign flips */
        o->sens_g = sens10(512.0f - g[2] * (180.0f / 3.14159265f) * SIXAXIS_GYRO_COUNTS_PER_DPS);
    }
}

static void read_gamepad(NativePad* p, PadSdlNativePad* o)
{
    SDL_Gamepad* gp = p->gp;
    memset(o, 0, sizeof *o);
#define BUTTON(sdl, ps3) if (SDL_GetGamepadButton(gp, sdl)) o->buttons |= ps3
    BUTTON(SDL_GAMEPAD_BUTTON_SOUTH, CELL_PAD_CTRL_CROSS);
    BUTTON(SDL_GAMEPAD_BUTTON_EAST, CELL_PAD_CTRL_CIRCLE);
    BUTTON(SDL_GAMEPAD_BUTTON_WEST, CELL_PAD_CTRL_SQUARE);
    BUTTON(SDL_GAMEPAD_BUTTON_NORTH, CELL_PAD_CTRL_TRIANGLE);
    BUTTON(SDL_GAMEPAD_BUTTON_BACK, CELL_PAD_CTRL_SELECT);
    BUTTON(SDL_GAMEPAD_BUTTON_START, CELL_PAD_CTRL_START);
    BUTTON(SDL_GAMEPAD_BUTTON_LEFT_STICK, CELL_PAD_CTRL_L3);
    BUTTON(SDL_GAMEPAD_BUTTON_RIGHT_STICK, CELL_PAD_CTRL_R3);
    BUTTON(SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, CELL_PAD_CTRL_L1);
    BUTTON(SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, CELL_PAD_CTRL_R1);
    BUTTON(SDL_GAMEPAD_BUTTON_DPAD_UP, CELL_PAD_CTRL_UP);
    BUTTON(SDL_GAMEPAD_BUTTON_DPAD_DOWN, CELL_PAD_CTRL_DOWN);
    BUTTON(SDL_GAMEPAD_BUTTON_DPAD_LEFT, CELL_PAD_CTRL_LEFT);
    BUTTON(SDL_GAMEPAD_BUTTON_DPAD_RIGHT, CELL_PAD_CTRL_RIGHT);
    BUTTON(SDL_GAMEPAD_BUTTON_TOUCHPAD, CELL_PAD_CTRL_SELECT);
#undef BUTTON
    o->l2 = trigger_u8(SDL_GetGamepadAxis(gp, SDL_GAMEPAD_AXIS_LEFT_TRIGGER));
    o->r2 = trigger_u8(SDL_GetGamepadAxis(gp, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER));
    if (o->l2 > 30) o->buttons |= CELL_PAD_CTRL_L2;
    if (o->r2 > 30) o->buttons |= CELL_PAD_CTRL_R2;
    o->lx = axis_u8(SDL_GetGamepadAxis(gp, SDL_GAMEPAD_AXIS_LEFTX));
    o->ly = axis_u8(SDL_GetGamepadAxis(gp, SDL_GAMEPAD_AXIS_LEFTY));
    o->rx = axis_u8(SDL_GetGamepadAxis(gp, SDL_GAMEPAD_AXIS_RIGHTX));
    o->ry = axis_u8(SDL_GetGamepadAxis(gp, SDL_GAMEPAD_AXIS_RIGHTY));
    read_sensors(p, o);
}

/* No controller: WASD/IJKL are the two sticks, arrows are the d-pad,
 * Space/C/F/E are Cross/Circle/Square/Triangle, Q/R are L1/R1, 1/2 are
 * L2/R2, V/B are L3/R3, Enter is Start and Tab is Select. */
static void read_keyboard(PadSdlNativePad* o)
{
    memset(o, 0, sizeof *o);
    if (getenv("PAD_NO_KEYBOARD") || !SDL_GetKeyboardFocus()) return;
    int count = 0;
    const bool* k = SDL_GetKeyboardState(&count);
    if (!k) return;
#define KEY(scan, ps3) if (k[scan]) o->buttons |= ps3
    KEY(SDL_SCANCODE_UP, CELL_PAD_CTRL_UP);
    KEY(SDL_SCANCODE_DOWN, CELL_PAD_CTRL_DOWN);
    KEY(SDL_SCANCODE_LEFT, CELL_PAD_CTRL_LEFT);
    KEY(SDL_SCANCODE_RIGHT, CELL_PAD_CTRL_RIGHT);
    KEY(SDL_SCANCODE_SPACE, CELL_PAD_CTRL_CROSS);
    KEY(SDL_SCANCODE_C, CELL_PAD_CTRL_CIRCLE);
    KEY(SDL_SCANCODE_F, CELL_PAD_CTRL_SQUARE);
    KEY(SDL_SCANCODE_E, CELL_PAD_CTRL_TRIANGLE);
    KEY(SDL_SCANCODE_Q, CELL_PAD_CTRL_L1);
    KEY(SDL_SCANCODE_R, CELL_PAD_CTRL_R1);
    KEY(SDL_SCANCODE_1, CELL_PAD_CTRL_L2);
    KEY(SDL_SCANCODE_2, CELL_PAD_CTRL_R2);
    KEY(SDL_SCANCODE_V, CELL_PAD_CTRL_L3);
    KEY(SDL_SCANCODE_B, CELL_PAD_CTRL_R3);
    KEY(SDL_SCANCODE_RETURN, CELL_PAD_CTRL_START);
    KEY(SDL_SCANCODE_TAB, CELL_PAD_CTRL_SELECT);
#undef KEY
    const uint8_t mid = 96, hi = 224, centre = 128;
    uint8_t step = (k[SDL_SCANCODE_LSHIFT] || k[SDL_SCANCODE_RSHIFT]) ? hi : mid;
    o->lx = o->ly = o->rx = o->ry = centre;
    if (k[SDL_SCANCODE_A]) o->lx = centre - step;
    if (k[SDL_SCANCODE_D]) o->lx = centre + step;
    if (k[SDL_SCANCODE_W]) o->ly = centre - step;
    if (k[SDL_SCANCODE_S]) o->ly = centre + step;
    if (k[SDL_SCANCODE_J]) o->rx = centre - step;
    if (k[SDL_SCANCODE_L]) o->rx = centre + step;
    if (k[SDL_SCANCODE_I]) o->ry = centre - step;
    if (k[SDL_SCANCODE_K]) o->ry = centre + step;
    o->l2 = (o->buttons & CELL_PAD_CTRL_L2) ? 255 : 0;
    o->r2 = (o->buttons & CELL_PAD_CTRL_R2) ? 255 : 0;
    o->sens_x = 512; o->sens_y = 399; o->sens_z = 512; o->sens_g = 512;
}

/* SDL reads most joystick hints when its drivers start, so they have to be in place before SDL_InitSubSystem. They go
 * in at SDL_HINT_DEFAULT priority: an SDL_JOYSTICK_* variable from the environment (or a launcher) still wins, and
 * nothing here touches SDL_GAMECONTROLLER_IGNORE_DEVICES or the Steam / Steam Deck HIDAPI hints. Under Steam Input
 * (Game Mode) Steam hides the physical Deck pad with that variable and we see only its virtual gamepad, so the pad is
 * opened once; outside Steam SDL's own drivers own the device. */
static void set_pad_hints(void)
{
    SDL_SetHintWithPriority(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1", SDL_HINT_DEFAULT);
    /* SDL3 folded the PS4 / PS5 rumble hints into SDL_HINT_JOYSTICK_ENHANCED_REPORTS (the legacy macros only exist as
     * rename shims). Set both: the new name for SDL 3.2 and later, the literal old names for any older SDL3 build. */
    SDL_SetHintWithPriority(SDL_HINT_JOYSTICK_ENHANCED_REPORTS, "1", SDL_HINT_DEFAULT);
    SDL_SetHintWithPriority("SDL_JOYSTICK_HIDAPI_PS4_RUMBLE", "1", SDL_HINT_DEFAULT);
    SDL_SetHintWithPriority("SDL_JOYSTICK_HIDAPI_PS5_RUMBLE", "1", SDL_HINT_DEFAULT);
}

void pad_sdl3_native_init(void)
{
    if (s_inited) return;
    set_pad_hints();
    if (!(SDL_WasInit(SDL_INIT_GAMEPAD) & SDL_INIT_GAMEPAD) &&
        !SDL_InitSubSystem(SDL_INIT_GAMEPAD)) {
        printf("[cellPad] SDL3 gamepad init failed: %s\n", SDL_GetError());
        return;
    }
    { const char* e = getenv("PAD_GYRO");   s_gyro = !(e && *e == '0'); }     /* SIXAXIS accel/gyro, default on */
    { const char* e = getenv("PAD_RUMBLE"); s_rumble = !(e && *e == '0'); }   /* default on */
    if (!s_mu) s_mu = SDL_CreateMutex();
    s_inited = 1;
    pad_lock();
    scan_pads();
    pad_unlock();
    printf("[cellPad] SDL3 gamepads on (SIXAXIS motion %s, rumble %s), keyboard fallback when port 0 is empty\n",
           s_gyro ? "on" : "off (PAD_GYRO=0)", s_rumble ? "on" : "off (PAD_RUMBLE=0)");
}

void pad_sdl3_native_shutdown(void)
{
    pad_lock();
    for (int i = 0; i < NATIVE_PADS; i++) {
        if (s_pads[i].gp) {
            SDL_RumbleGamepad(s_pads[i].gp, 0, 0, 0);
            SDL_CloseGamepad(s_pads[i].gp);
        }
        memset(&s_pads[i], 0, sizeof s_pads[i]);
    }
    pad_unlock();
}

void pad_sdl3_native_notify_hotplug(void)
{
    SDL_SetAtomicInt(&s_hotplug, 1);
}

void pad_sdl3_native_poll(void)
{
    if (!s_inited) return;
    pad_lock();
    SDL_UpdateGamepads();
    const Uint64 now = SDL_GetTicks();
    /* Rescan on the 500 ms timer, and at once when the window's event pump saw an add/remove event (a Deck waking
     * from sleep, a Bluetooth pad reconnecting). */
    if (now >= s_next_scan || SDL_SetAtomicInt(&s_hotplug, 0)) {
        s_next_scan = now + 500;
        scan_pads();
    }
    int out = 0;
    for (int i = 0; i < NATIVE_PADS; i++) {
        if (!s_pads[i].gp) continue;
        if (!SDL_GamepadConnected(s_pads[i].gp)) {
            close_pad(i);
            i--;
            continue;
        }
        if (out != i) {
            s_pads[out] = s_pads[i];
            memset(&s_pads[i], 0, sizeof s_pads[i]);
        }
        read_gamepad(&s_pads[out], &s_state[out]);
        apply_rumble(&s_pads[out], now);
        out++;
    }
    if (out == 0) read_keyboard(&s_state[0]);
    pad_unlock();
}

int pad_sdl3_native_get(int index, PadSdlNativePad* out)
{
    if (index < 0 || index >= NATIVE_PADS || !out) return 0;
    pad_lock();
    const int present = s_pads[index].gp || index == 0;
    if (present) *out = s_state[index];
    pad_unlock();
    return present;
}

void pad_sdl3_native_rumble(int index, uint8_t small_motor, uint8_t large_motor)
{
    if (index < 0 || index >= NATIVE_PADS) return;
    pad_lock();
    if (!s_act_logged && (small_motor || large_motor)) {
        s_act_logged = 1;
        printf("[cellPad] SDL3: first cellPadSetActDirect port %d small=%u large=%u (%s)\n", index, small_motor, large_motor,
               !s_rumble ? "PAD_RUMBLE=0, ignored" : s_pads[index].gp ? "forwarding to SDL" : "no gamepad on this port");
    }
    if (s_pads[index].gp) {
        s_pads[index].rumble_req = (uint16_t)((small_motor ? 0xFF00u : 0u) | large_motor);
        s_pads[index].rumble_retry_at = 0;   /* a new request is always tried at once; poll() keeps it alive */
        apply_rumble(&s_pads[index], SDL_GetTicks());
    }
    pad_unlock();
}

#endif /* PS3RECOMP_SDL3_NATIVE */
