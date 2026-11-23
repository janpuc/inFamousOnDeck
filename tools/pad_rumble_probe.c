/* Rumble probe for the SDL3 gamepad path: opens every gamepad SDL sees, prints what it reports and sends a short
 * rumble pattern, so a controller can be checked without starting the game.
 *
 *   clang -O1 -o pad_rumble_probe tools/pad_rumble_probe.c -lSDL3     (or point -I and -L at an SDL3 build)
 *   ./pad_rumble_probe            two short pulses, then a double pulse 60 ms apart
 *
 * It sets the same joystick hints as ps3recomp/libs/input/pad_sdl3_native.c, before the gamepad subsystem starts. */
#include <SDL3/SDL.h>
#include <stdio.h>

static int rumble(SDL_Gamepad* gp, Uint16 low, Uint16 high, Uint32 ms)
{
    const bool ok = SDL_RumbleGamepad(gp, low, high, ms);
    printf("  SDL_RumbleGamepad(low=%u, high=%u, %u ms) -> %s%s%s\n", low, high, ms, ok ? "true" : "false",
           ok ? "" : ": ", ok ? "" : SDL_GetError());
    return ok;
}

int main(void)
{
    SDL_SetHintWithPriority(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1", SDL_HINT_DEFAULT);
    SDL_SetHintWithPriority(SDL_HINT_JOYSTICK_ENHANCED_REPORTS, "1", SDL_HINT_DEFAULT);
    if (!SDL_Init(SDL_INIT_GAMEPAD)) {
        printf("SDL_Init(GAMEPAD) failed: %s\n", SDL_GetError());
        return 2;
    }
    int count = 0;
    SDL_JoystickID* ids = SDL_GetGamepads(&count);
    printf("SDL %d.%d.%d, %d gamepad(s)\n", SDL_MAJOR_VERSION, SDL_MINOR_VERSION, SDL_MICRO_VERSION, count);
    int ok_all = count > 0;
    for (int i = 0; i < count; i++) {
        SDL_Gamepad* gp = SDL_OpenGamepad(ids[i]);
        if (!gp) { printf("[%d] open failed: %s\n", i, SDL_GetError()); ok_all = 0; continue; }
        const bool cap = SDL_GetBooleanProperty(SDL_GetGamepadProperties(gp), SDL_PROP_GAMEPAD_CAP_RUMBLE_BOOLEAN, false);
        printf("[%d] %s, vid:pid %04x:%04x, type %d, rumble: %s\n", i, SDL_GetGamepadName(gp), SDL_GetGamepadVendor(gp),
               SDL_GetGamepadProduct(gp), (int)SDL_GetGamepadType(gp), cap ? "yes" : "no");
        int ok = 1;
        ok &= rumble(gp, 0x6000, 0x0000, 400);   /* the large (low frequency) motor */
        SDL_Delay(600);
        ok &= rumble(gp, 0x0000, 0xC000, 400);   /* the small (high frequency) motor */
        SDL_Delay(600);
        /* a PS3 title sends a start and a stop back to back: both must reach the pad */
        ok &= rumble(gp, 0xFFFF, 0xFFFF, 600);
        SDL_Delay(60);
        ok &= rumble(gp, 0, 0, 0);
        SDL_Delay(60);
        ok &= rumble(gp, 0xFFFF, 0xFFFF, 600);
        SDL_Delay(60);
        ok &= rumble(gp, 0, 0, 0);
        SDL_Delay(200);
        printf("[%d] result: %s\n", i, ok ? "all rumble calls returned true" : "a rumble call failed");
        ok_all &= ok;
        SDL_CloseGamepad(gp);
    }
    SDL_free(ids);
    SDL_Quit();
    return ok_all ? 0 : 1;
}
