#!/usr/bin/env python3
"""Controller input for the Steam Deck launcher.

Three backends are tried in this order, and the first one that opens at
least one device is used:

  evdev   reads /dev/input/event* directly with the Linux ioctl interface.
          Stdlib only, and it is what works on a stock SteamOS Deck: the
          "Valve Software Steam Deck Controller" and the Steam Input virtual
          pad are both readable by the desktop user, while the devices that
          belong to the Steam runtime are not.
  pygame  used when the player installed it (optional, never required).
  sdl     SDL3 or SDL2 through ctypes. The SteamOS runtime ships both, but
          they open the same device nodes as evdev, so this is a fallback for
          systems where the raw nodes are not readable.

Every backend reports the same events:

  ("nav", "up" | "down" | "left" | "right")   a direction edge
  ("press", "a" | "b" | "start" | "lb" | "rb") a button press edge

The launcher never requires a controller: without one it falls back to the
keyboard and the touch screen.
"""

from __future__ import annotations

import ctypes
import glob
import os
import struct

# Linux input event codes.
EV_KEY = 0x01
EV_ABS = 0x03

ABS_X = 0x00
ABS_Y = 0x01
ABS_HAT0X = 0x10
ABS_HAT0Y = 0x11

BTN_SOUTH = 0x130  # A / Cross
BTN_EAST = 0x131  # B / Circle
BTN_NORTH = 0x132  # Y / Triangle
BTN_WEST = 0x133  # X / Square
BTN_TL = 0x134  # LB / L1
BTN_TR = 0x135  # RB / R1
BTN_SELECT = 0x136  # Back / Share
BTN_START = 0x137  # Start / Options

_BUTTON_NAMES = {
    BTN_SOUTH: "a",
    BTN_EAST: "b",
    BTN_NORTH: "y",
    BTN_WEST: "x",
    BTN_TL: "lb",
    BTN_TR: "rb",
    BTN_SELECT: "back",
    BTN_START: "start",
}

# The stick dead zone, in raw evdev units (a full axis is about 32767).
DEFAULT_DEADZONE = 12000
# A held direction repeats after this many seconds, so a long list can be
# scrolled without letting go.
REPEAT_SECONDS = 0.28

BACKENDS = ("evdev", "pygame", "sdl", "none")


def _ioc(direction: int, typ: int, nr: int, size: int) -> int:
    """The Linux _IOC encoding (asm-generic/ioctl.h)."""
    return (direction << 30) | (size << 16) | (typ << 8) | nr


_IOC_READ = 2
_EVIOCGNAME = _ioc(_IOC_READ, ord("E"), 0x06, 256)
_EVIOCGBIT = lambda ev, size: _ioc(_IOC_READ, ord("E"), 0x20 + ev, size)  # noqa: E731

_INPUT_EVENT = struct.Struct("llHHi")  # timeval, type, code, value


class Controller:
    """Base class. Subclasses fill in the device list and the poll logic."""

    backend = "none"

    def __init__(self, deadzone: int = DEFAULT_DEADZONE) -> None:
        self.deadzone = deadzone
        self.devices: list[str] = []
        self._closed = False

    def open(self) -> bool:
        raise NotImplementedError

    def poll(self) -> list[tuple[str, str]]:
        raise NotImplementedError

    def rumble(self, low: int = 0x4000, high: int = 0x8000, duration_ms: int = 200) -> bool:
        """A short pulse on the motors. False when the backend cannot do it."""
        return False

    def close(self) -> None:
        self._closed = True

    @property
    def available(self) -> bool:
        return bool(self.devices)


class NullController(Controller):
    """No controller. The launcher keeps working with keyboard and touch."""

    backend = "none"

    def open(self) -> bool:
        return False

    def poll(self) -> list[tuple[str, str]]:
        return []


class EvdevController(Controller):
    """Raw Linux input devices, read without any third party module."""

    backend = "evdev"

    def __init__(self, deadzone: int = DEFAULT_DEADZONE) -> None:
        super().__init__(deadzone)
        self._fds: dict[int, str] = {}
        self._axes: dict[int, dict[int, int]] = {}
        self._buttons: dict[int, set[int]] = {}
        self._stick: dict[int, tuple[int, int]] = {}
        self._repeat_at: dict[str, float] = {}

    # -- device discovery ---------------------------------------------------
    @staticmethod
    def _name(fd: int) -> str:
        import fcntl

        buf = fcntl.ioctl(fd, _EVIOCGNAME, b"\0" * 256)
        return buf.split(b"\0")[0].decode("utf-8", "replace")

    @staticmethod
    def _bits(fd: int, ev: int, size: int = 128) -> int:
        import fcntl

        buf = fcntl.ioctl(fd, _EVIOCGBIT(ev, size), b"\0" * size)
        value = 0
        for index, byte in enumerate(bytearray(buf)):
            value |= byte << (8 * index)
        return value

    def open(self) -> bool:
        for path in sorted(glob.glob("/dev/input/event*")):
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError:
                continue
            try:
                name = self._name(fd)
                axes = self._bits(fd, EV_ABS)
                keys = self._bits(fd, EV_KEY)
            except OSError:
                os.close(fd)
                continue
            # A pad has an absolute stick, or the buttons of a gamepad. The
            # keyboard, the lid switch and the power button have neither.
            has_stick = bool(axes & ((1 << ABS_X) | (1 << ABS_Y)))
            has_buttons = any(
                keys & (1 << code)
                for code in (BTN_SOUTH, BTN_EAST, BTN_NORTH, BTN_WEST, BTN_TL, BTN_TR, BTN_SELECT, BTN_START)
            )
            if not has_stick and not has_buttons:
                os.close(fd)
                continue
            self._fds[fd] = name
            self._axes[fd] = {}
            self._buttons[fd] = set()
            self._stick[fd] = (0, 0)
            self.devices.append(f"{name} ({os.path.basename(path)})")
        return bool(self._fds)

    # -- input --------------------------------------------------------------
    def _handle_event(self, fd: int, code: int, value: int, events: list) -> None:
        if code == ABS_X:
            self._axes[fd][ABS_X] = value
        elif code == ABS_Y:
            self._axes[fd][ABS_Y] = value
        elif code == ABS_HAT0X:
            self._axes[fd][ABS_HAT0X] = value
        elif code == ABS_HAT0Y:
            self._axes[fd][ABS_HAT0Y] = value
        elif code in _BUTTON_NAMES:
            if value == 1:
                self._buttons[fd].add(code)
                events.append(("press", _BUTTON_NAMES[code]))
            elif value == 0:
                self._buttons[fd].discard(code)
        else:
            return

    def poll(self) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        for fd in list(self._fds):
            try:
                data = os.read(fd, _INPUT_EVENT.size * 64)
            except BlockingIOError:
                data = b""
            except OSError:
                data = b""
            offset = 0
            while offset + _INPUT_EVENT.size <= len(data):
                _time, etype, code, value = _INPUT_EVENT.unpack_from(data, offset)
                offset += _INPUT_EVENT.size
                if etype == EV_ABS:
                    self._handle_event(fd, code, value, events)
                elif etype == EV_KEY:
                    self._handle_event(fd, code, value, events)
        events.extend(self._direction_events())
        return events

    def _direction_events(self) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        for fd in list(self._fds):
            axes = self._axes[fd]
            # The D-pad reports absolute positions on the hat axes.
            hat_x = axes.get(ABS_HAT0X, 0)
            hat_y = axes.get(ABS_HAT0Y, 0)
            stick_x = axes.get(ABS_X, 0)
            stick_y = axes.get(ABS_Y, 0)
            previous = self._stick[fd]
            current = (hat_x, hat_y, stick_x, stick_y)
            if current == previous:
                continue
            self._stick[fd] = current
            for direction, value in self._directions(hat_x, hat_y, stick_x, stick_y):
                events.append(("nav", direction))
        return events

    def _directions(self, hat_x, hat_y, stick_x, stick_y) -> list[str]:
        directions = []
        if hat_y < 0:
            directions.append("up")
        elif hat_y > 0:
            directions.append("down")
        if hat_x < 0:
            directions.append("left")
        elif hat_x > 0:
            directions.append("right")
        if not directions:
            if stick_y < -self.deadzone:
                directions.append("up")
            elif stick_y > self.deadzone:
                directions.append("down")
            if stick_x < -self.deadzone:
                directions.append("left")
            elif stick_x > self.deadzone:
                directions.append("right")
        return directions

    def close(self) -> None:
        for fd in list(self._fds):
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()
        super().close()


class PygameController(Controller):
    """Optional backend. pygame is not part of SteamOS, so it stays optional."""

    backend = "pygame"

    def __init__(self, deadzone: int = DEFAULT_DEADZONE) -> None:
        super().__init__(deadzone)
        self._pygame = None
        self._joysticks: list = []
        self._prev: list[list[bool]] = []

    def open(self) -> bool:
        try:
            import pygame
        except Exception:
            return False
        try:
            pygame.init()
            pygame.joystick.init()
            count = pygame.joystick.get_count()
            for index in range(count):
                pad = pygame.joystick.Joystick(index)
                pad.init()
                self._joysticks.append(pad)
                self._prev.append([False] * 16)
                self.devices.append(pad.get_name())
        except Exception:
            self._joysticks = []
            return False
        return bool(self._joysticks)

    def poll(self) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        if self._pygame is None:
            try:
                import pygame
            except Exception:
                return events
            self._pygame = pygame
        try:
            self._pygame.event.pump()
        except Exception:
            return events
        for index, pad in enumerate(self._joysticks):
            try:
                buttons = [bool(pad.get_button(i)) for i in range(16)]
            except Exception:
                continue
            previous = self._prev[index]
            for button, pressed in enumerate(buttons):
                if pressed and not previous[button]:
                    if button == 0:
                        events.append(("press", "a"))
                    elif button == 1:
                        events.append(("press", "b"))
                    elif button == 6:
                        events.append(("press", "start"))
                    elif button == 9:
                        events.append(("press", "lb"))
                    elif button == 10:
                        events.append(("press", "rb"))
            self._prev[index] = buttons
            try:
                axis_x = pad.get_axis(0)
                axis_y = pad.get_axis(1)
                hat = pad.get_hat(0)
            except Exception:
                continue
            if hat[1] == 1:
                events.append(("nav", "up"))
            elif hat[1] == -1:
                events.append(("nav", "down"))
            if hat[0] == -1:
                events.append(("nav", "left"))
            elif hat[0] == 1:
                events.append(("nav", "right"))
            if not hat or hat == (0, 0):
                if axis_y < -0.5:
                    events.append(("nav", "up"))
                elif axis_y > 0.5:
                    events.append(("nav", "down"))
                if axis_x < -0.5:
                    events.append(("nav", "left"))
                elif axis_x > 0.5:
                    events.append(("nav", "right"))
        return events

    def close(self) -> None:
        try:
            import pygame

            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass
        super().close()


class SdlController(Controller):
    """SDL3 or SDL2 through ctypes. No python binding needed."""

    backend = "sdl"

    def __init__(self, deadzone: int = DEFAULT_DEADZONE) -> None:
        super().__init__(deadzone)
        self._lib = None
        self._sdl3 = False
        self._joysticks: list = []
        self._prev: list[list[bool]] = []

    def _load(self, name: str):
        try:
            return ctypes.CDLL(name)
        except OSError:
            return None

    def open(self) -> bool:
        for candidate in ("libSDL3.so.0", "libSDL3.so", "libSDL2.so.0", "libSDL2.so"):
            lib = self._load(candidate)
            if lib is None:
                continue
            try:
                if self._setup(lib, candidate.startswith("libSDL3")):
                    return True
            except Exception:
                continue
        return False

    def _setup(self, lib, sdl3: bool) -> bool:
        lib.SDL_Init.restype = ctypes.c_int
        lib.SDL_Init.argtypes = [ctypes.c_uint32]
        if lib.SDL_Init(0x00002000) != 0:  # SDL_INIT_JOYSTICK
            return False
        if sdl3:
            lib.SDL_GetJoysticks.restype = ctypes.POINTER(ctypes.c_int)
            lib.SDL_GetJoysticks.argtypes = [ctypes.POINTER(ctypes.c_int)]
            count = ctypes.c_int(0)
            ids = lib.SDL_GetJoysticks(ctypes.byref(count))
            lib.SDL_OpenJoystick.restype = ctypes.c_void_p
            lib.SDL_OpenJoystick.argtypes = [ctypes.c_int]
            lib.SDL_GetJoystickAxis.restype = ctypes.c_float
            lib.SDL_GetJoystickAxis.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.SDL_GetJoystickButton.restype = ctypes.c_bool
            lib.SDL_GetJoystickButton.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.SDL_GetJoystickName.restype = ctypes.c_char_p
            lib.SDL_GetJoystickName.argtypes = [ctypes.c_void_p]
            for index in range(count.value):
                handle = lib.SDL_OpenJoystick(int(ids[index]))
                if not handle:
                    continue
                self._joysticks.append(handle)
                self._prev.append([False] * 16)
                name = lib.SDL_GetJoystickName(handle)
                self.devices.append(name.decode("utf-8", "replace") if name else "joystick")
        else:
            lib.SDL_NumJoysticks.restype = ctypes.c_int
            lib.SDL_JoystickOpen.restype = ctypes.c_void_p
            lib.SDL_JoystickOpen.argtypes = [ctypes.c_int]
            lib.SDL_JoystickGetAxis.restype = ctypes.c_int16
            lib.SDL_JoystickGetAxis.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.SDL_JoystickGetButton.restype = ctypes.c_int
            lib.SDL_JoystickGetButton.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.SDL_JoystickName.restype = ctypes.c_char_p
            lib.SDL_JoystickName.argtypes = [ctypes.c_void_p]
            for index in range(lib.SDL_NumJoysticks()):
                handle = lib.SDL_JoystickOpen(index)
                if not handle:
                    continue
                self._joysticks.append(handle)
                self._prev.append([False] * 16)
                name = lib.SDL_JoystickName(handle)
                self.devices.append(name.decode("utf-8", "replace") if name else "joystick")
        self._lib = lib
        self._sdl3 = sdl3
        return bool(self._joysticks)

    def poll(self) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        if self._lib is None:
            return events
        lib = self._lib
        for index, handle in enumerate(self._joysticks):
            buttons = []
            for button in range(16):
                try:
                    if self._sdl3:
                        pressed = bool(lib.SDL_GetJoystickButton(handle, button))
                    else:
                        pressed = lib.SDL_JoystickGetButton(handle, button) != 0
                except Exception:
                    pressed = False
                buttons.append(pressed)
            previous = self._prev[index]
            for button, pressed in enumerate(buttons):
                if pressed and not previous[button]:
                    if button == 0:
                        events.append(("press", "a"))
                    elif button == 1:
                        events.append(("press", "b"))
                    elif button == 6:
                        events.append(("press", "start"))
                    elif button == 9:
                        events.append(("press", "lb"))
                    elif button == 10:
                        events.append(("press", "rb"))
            self._prev[index] = buttons
            try:
                if self._sdl3:
                    axis_x = float(lib.SDL_GetJoystickAxis(handle, 0))
                    axis_y = float(lib.SDL_GetJoystickAxis(handle, 1))
                else:
                    axis_x = float(lib.SDL_JoystickGetAxis(handle, 0))
                    axis_y = float(lib.SDL_JoystickGetAxis(handle, 1))
            except Exception:
                continue
            if axis_y < -0.5:
                events.append(("nav", "up"))
            elif axis_y > 0.5:
                events.append(("nav", "down"))
            if axis_x < -0.5:
                events.append(("nav", "left"))
            elif axis_x > 0.5:
                events.append(("nav", "right"))
        return events

    def rumble(self, low: int = 0x4000, high: int = 0x8000, duration_ms: int = 200) -> bool:
        """A short pulse through SDL, when the library has the call."""
        if self._lib is None or not self._joysticks:
            return False
        lib = self._lib
        try:
            if self._sdl3:
                call = getattr(lib, "SDL_SendJoystickRumble", None)
                if call is None:
                    return False
                call.restype = ctypes.c_bool
                call.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint32]
                return bool(call(self._joysticks[0], low, high, duration_ms))
            call = getattr(lib, "SDL_JoystickRumble", None)
            if call is None:
                return False
            call.restype = ctypes.c_int
            call.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint32]
            return call(self._joysticks[0], low, high, duration_ms) == 0
        except Exception:
            return False

    def close(self) -> None:
        if self._lib is not None:
            try:
                if self._sdl3:
                    self._lib.SDL_Quit()
                else:
                    self._lib.SDL_Quit()
            except Exception:
                pass
        super().close()


def open_controller(preferred: str | None = None, deadzone: int = DEFAULT_DEADZONE) -> Controller:
    """Open the best available backend.

    `preferred` forces one backend ("evdev", "pygame", "sdl" or "none"), which
    is what the launcher uses for --controller-backend and what the tests use.
    """
    order = list(BACKENDS)
    if preferred:
        if preferred not in order:
            raise ValueError(f"unknown controller backend: {preferred}")
        order = [preferred] + [name for name in order if name != preferred]
    for name in order:
        if name == "none":
            break
        if name == "evdev":
            controller = EvdevController(deadzone)
        elif name == "pygame":
            controller = PygameController(deadzone)
        else:
            controller = SdlController(deadzone)
        try:
            if controller.open():
                return controller
        except Exception:
            pass
        controller.close()
    return NullController(deadzone)


if __name__ == "__main__":
    pad = open_controller()
    print(f"backend: {pad.backend}")
    for name in pad.devices:
        print(f"device: {name}")
    if not pad.available:
        print("no controller found, the launcher will use the keyboard")
        raise SystemExit(0)
    print("press a button or move the stick, Ctrl-C to stop")
    try:
        while True:
            for event in pad.poll():
                print(event, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        pad.close()
