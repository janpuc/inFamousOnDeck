/* Native SDL3 gamepads and keyboard fallback for non-Windows cellPad. */
#ifndef PS3RECOMP_PAD_SDL3_NATIVE_H
#define PS3RECOMP_PAD_SDL3_NATIVE_H

#include <stdint.h>

typedef struct PadSdlNativePad {
    uint16_t buttons;
    uint8_t lx, ly, rx, ry, l2, r2;
    /* SIXAXIS in libpad units (10 bit, 512 = rest, 113 counts per g): sens_x lateral, sens_y vertical (399 = flat,
     * face up), sens_z front/back, sens_g yaw rate. has_sensors = 0 -> no accelerometer on this pad, PAD_GYRO=0, or
     * the keyboard fallback; cellPad then reports the resting values. */
    uint16_t sens_x, sens_y, sens_z, sens_g;
    uint8_t has_sensors;
} PadSdlNativePad;

void pad_sdl3_native_init(void);
void pad_sdl3_native_shutdown(void);
void pad_sdl3_native_poll(void);
int  pad_sdl3_native_get(int index, PadSdlNativePad* out);
void pad_sdl3_native_rumble(int index, uint8_t small_motor, uint8_t large_motor);
/* Any thread: a gamepad was added or removed (SDL_EVENT_GAMEPAD_ADDED/REMOVED seen by the window's event pump);
 * the next poll rescans at once instead of waiting for the 500 ms timer. */
void pad_sdl3_native_notify_hotplug(void);

#endif
