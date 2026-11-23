#!/usr/bin/env python3
"""Steam Deck launcher and settings for the native Linux (Vulkan) inFamous port.

The launcher is the only part of the port that a player touches. It keeps the
settings in an INI file under the XDG config directory, turns them into the
environment the port understands, and starts the game.

  ~/.config/inFamousOnDeck/settings.ini

Everything here is stdlib only. The optional controller backends live in
controller.py next to this file and are used when they are available; without
one the window is driven with the keyboard, the mouse or the touch screen.

Command line:

  launcher.py                    open the settings window
  launcher.py --play             start the game with the saved settings
  launcher.py --steam            start the game and close Steam when it exits
  launcher.py --print-env        print the resolved environment (KEY=value)
  launcher.py --list-presets     print the preset names
  launcher.py --apply-preset NAME  store a preset in the settings file
  launcher.py --set KEY=VALUE    override one setting for this run
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import controller as controller_mod  # noqa: E402

try:
    from project import (  # noqa: E402
        CONFIG_DIR_NAME,
        GAME_TITLE,
        INSTALL_DIR_NAME,
        PROJECT_NAME,
    )
except ImportError:  # the launcher runs without its project file
    CONFIG_DIR_NAME = "inFamousOnDeck"
    GAME_TITLE = "inFamous"
    INSTALL_DIR_NAME = "inFamousOnDeck"
    PROJECT_NAME = "inFamousOnDeck"

APP_NAME = GAME_TITLE
SETTINGS_FILE_NAME = "settings.ini"
INSTALL_INFO_NAME = "inFamousOnDeck.json"

DEFAULT_INSTALL_ROOT = Path.home() / ".local" / "share" / INSTALL_DIR_NAME

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------

VIDEO_KEYS = (
    "fsr",
    "fsr_mode",
    "fsr_fg",
    "fps_cap",
    "res_scale",
    "tex_aniso",
    "lod_class_scale",
    "depth_blur_smooth",
)
INPUT_KEYS = ("pad_gyro", "pad_rumble")
PATH_KEYS = ("savedata_root",)
ALL_KEYS = VIDEO_KEYS + INPUT_KEYS + PATH_KEYS

SECTION_OF = {}
for _key in VIDEO_KEYS:
    SECTION_OF[_key] = "video"
for _key in INPUT_KEYS:
    SECTION_OF[_key] = "input"
for _key in PATH_KEYS:
    SECTION_OF[_key] = "paths"

DEFAULTS: Dict[str, str] = {
    "fsr": "0",
    "fsr_mode": "native",
    "fsr_fg": "0",
    "fps_cap": "0",
    "res_scale": "1",
    "tex_aniso": "0",
    "lod_class_scale": "1.0",
    "depth_blur_smooth": "1",
    "pad_gyro": "0",
    "pad_rumble": "1",
    "savedata_root": "",
}

# The setting name and the environment variable the port reads.
ENV_OF = {
    "fsr": "FSR",
    "fsr_mode": "FSR_MODE",
    "fsr_fg": "FSR_FG",
    "fps_cap": "FPS_CAP",
    "res_scale": "RES_SCALE",
    "tex_aniso": "TEX_ANISO",
    "lod_class_scale": "LOD_CLASS_SCALE",
    "depth_blur_smooth": "DEPTH_BLUR_SMOOTH",
    "pad_gyro": "PAD_GYRO",
    "pad_rumble": "PAD_RUMBLE",
    "savedata_root": "SAVEDATA_ROOT",
}

BOOL_KEYS = {"fsr", "fsr_fg", "depth_blur_smooth", "pad_gyro", "pad_rumble"}
INT_KEYS = {"fps_cap", "res_scale", "tex_aniso"}
FLOAT_KEYS = {"lod_class_scale"}

FSR_MODES = ("native", "quality", "balanced", "performance", "ultra")

# The six profiles. Every number comes from a measured run: the Steam Deck
# ones at the 15 W power limit in the city at 720p, the AMD BC-250 ones on the
# console hardware. The comments are the measurement.
PRESETS: Dict[str, Dict[str, str]] = {
    # 29.93 real / ~59.86 displayed fps locked. The real 1% low goes from
    # 16.14 (the same profile uncapped) to 21.22, which is 31.5% smoother.
    "Steam Deck OLED: Smooth 60 (30+FG)": {
        "fsr": "1",
        "fsr_mode": "quality",
        "fsr_fg": "1",
        "fps_cap": "30",
        "res_scale": "1",
        "tex_aniso": "0",
        "lod_class_scale": "1.0",
        "depth_blur_smooth": "1",
        "pad_gyro": "0",
        "pad_rumble": "1",
    },
    # 34.43 real / ~68.86 displayed fps, the fastest measured Deck profile.
    "Steam Deck OLED: Max FG (~69 fps)": {
        "fsr": "1",
        "fsr_mode": "quality",
        "fsr_fg": "1",
        "fps_cap": "0",
        "res_scale": "1",
        "tex_aniso": "0",
        "lod_class_scale": "1.0",
        "depth_blur_smooth": "1",
        "pad_gyro": "0",
        "pad_rumble": "1",
    },
    # 40.77 real fps uncapped, locked to 40 by the cap. The OLED panel runs
    # it at 80 Hz, the LCD one at 40 Hz.
    "Steam Deck OLED: Native 40 (80 Hz)": {
        "fsr": "0",
        "fsr_mode": "native",
        "fsr_fg": "0",
        "fps_cap": "40",
        "res_scale": "1",
        "tex_aniso": "0",
        "lod_class_scale": "1.0",
        "depth_blur_smooth": "1",
        "pad_gyro": "0",
        "pad_rumble": "1",
    },
    # 62.19 fps on the BC-250 with the 16x filtering, which is free there.
    "AMD BC-250: Native 60+ + 16x AF": {
        "fsr": "0",
        "fsr_mode": "native",
        "fsr_fg": "0",
        "fps_cap": "0",
        "res_scale": "1",
        "tex_aniso": "16",
        "lod_class_scale": "1.0",
        "depth_blur_smooth": "1",
        "pad_gyro": "0",
        "pad_rumble": "1",
    },
    # 54.67 real / ~109.34 displayed fps on the BC-250.
    "AMD BC-250: 100+ FSR FG + 16x AF": {
        "fsr": "1",
        "fsr_mode": "quality",
        "fsr_fg": "1",
        "fps_cap": "0",
        "res_scale": "1",
        "tex_aniso": "16",
        "lod_class_scale": "1.0",
        "depth_blur_smooth": "1",
        "pad_gyro": "0",
        "pad_rumble": "1",
    },
    # 58.40 fps on the BC-250 with 10% more of the city drawn.
    "AMD BC-250: Extended LOD 1.5 + 16x AF": {
        "fsr": "0",
        "fsr_mode": "native",
        "fsr_fg": "0",
        "fps_cap": "0",
        "res_scale": "1",
        "tex_aniso": "16",
        "lod_class_scale": "1.5",
        "depth_blur_smooth": "1",
        "pad_gyro": "0",
        "pad_rumble": "1",
    },
}

PRESET_ORDER = list(PRESETS)

PRESET_DESCRIPTIONS = {
    "Steam Deck OLED: Smooth 60 (30+FG)": "locked 60 Hz picture, the best 1% lows",
    "Steam Deck OLED: Max FG (~69 fps)": "the fastest measured Deck profile",
    "Steam Deck OLED: Native 40 (80 Hz)": "no upscaler, no frame generation",
    "AMD BC-250: Native 60+ + 16x AF": "the console hardware, native and fast",
    "AMD BC-250: 100+ FSR FG + 16x AF": "over 100 displayed fps on the console",
    "AMD BC-250: Extended LOD 1.5 + 16x AF": "more of the city, a few fps less",
}


def truthy(value) -> bool:
    """INI values are strings, so "1", "true", "on" and "yes" all mean true."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "on", "yes"}


def format_number(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text or "0"


class Settings:
    """The launcher settings: the values in the INI file."""

    def __init__(self, values: Optional[Dict[str, str]] = None) -> None:
        self.values: Dict[str, str] = dict(DEFAULTS)
        if values:
            for key, value in values.items():
                if key in self.values:
                    self.values[key] = str(value)

    # -- file ---------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "Settings":
        parser = configparser.ConfigParser()
        settings = cls()
        try:
            if path.is_file():
                parser.read(path, encoding="utf-8")
        except (configparser.Error, OSError):
            return settings
        for key in ALL_KEYS:
            section = SECTION_OF[key]
            if parser.has_option(section, key):
                settings.values[key] = parser.get(section, key)
        return settings

    def save(self, path: Path) -> None:
        parser = configparser.ConfigParser()
        for section in ("video", "input", "paths", "launch"):
            parser.add_section(section)
        for key in ALL_KEYS:
            parser.set(SECTION_OF[key], key, self.values.get(key, DEFAULTS[key]))
        parser.set("launch", "preset", self.preset_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"# {APP_NAME} settings for the native Linux port.\n")
            handle.write("# The in-game pause menu has its own settings file.\n")
            parser.write(handle)

    # -- values -------------------------------------------------------------
    def get(self, key: str) -> str:
        return self.values.get(key, DEFAULTS.get(key, ""))

    def set(self, key: str, value) -> None:
        if key not in DEFAULTS:
            raise KeyError(f"unknown setting: {key}")
        if key in BOOL_KEYS:
            # The file always holds a 1 or a 0, whatever the widget returned.
            self.values[key] = "1" if truthy(value) else "0"
        else:
            self.values[key] = str(value)

    @property
    def preset_name(self) -> str:
        """The preset that matches the current values, if any."""
        for name in PRESET_ORDER:
            preset = PRESETS[name]
            if all(self.get(key) == value for key, value in preset.items()):
                return name
        return "Custom"

    def apply_preset(self, name: str) -> None:
        if name not in PRESETS:
            raise KeyError(f"unknown preset: {name}")
        self.values.update(PRESETS[name])

    # -- environment --------------------------------------------------------
    def as_env(self) -> Dict[str, str]:
        """The environment the port reads. Empty paths are not exported."""
        env: Dict[str, str] = {}
        for key in ALL_KEYS:
            name = ENV_OF[key]
            value = self.get(key)
            if key in BOOL_KEYS:
                env[name] = "1" if truthy(value) else "0"
            elif key in INT_KEYS:
                try:
                    env[name] = str(int(float(value)))
                except (TypeError, ValueError):
                    env[name] = DEFAULTS[key]
            elif key in FLOAT_KEYS:
                try:
                    env[name] = format_number(float(value))
                except (TypeError, ValueError):
                    env[name] = DEFAULTS[key]
            else:
                if str(value).strip():
                    env[name] = str(value)
        return env


def settings_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base).expanduser() / CONFIG_DIR_NAME / SETTINGS_FILE_NAME


# --------------------------------------------------------------------------------------
# The install
# --------------------------------------------------------------------------------------


def install_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    from_env = os.environ.get("INFAMOUS_NATIVE_ROOT")
    if from_env:
        return Path(from_env).expanduser()
    return DEFAULT_INSTALL_ROOT


def read_install_info(root: Path) -> Dict:
    path = root / INSTALL_INFO_NAME
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def find_binary(root: Path) -> Optional[Path]:
    info = read_install_info(root)
    candidates = []
    if info.get("binary"):
        candidates.append(Path(info["binary"]))
    candidates.append(root / "bin" / "inFamousRecomp")
    for candidate in sorted(root.glob("build-*/inFamousRecomp")):
        candidates.append(candidate)
    candidates.append(root / "inFamousRecomp")
    for candidate in candidates:
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        except OSError:
            continue
    return None


def find_game_elf(root: Path) -> Optional[Path]:
    info = read_install_info(root)
    if info.get("game_elf"):
        candidate = Path(info["game_elf"])
        if candidate.is_file():
            return candidate
    patterns = [
        root / "game" / "*" / "PS3_GAME" / "USRDIR" / "EBOOT.elf",
        root / "*" / "PS3_GAME" / "USRDIR" / "EBOOT.elf",
        root / "PS3_GAME" / "USRDIR" / "EBOOT.elf",
    ]
    for pattern in patterns:
        for candidate in sorted(root.glob(str(pattern))):
            if candidate.is_file():
                return candidate
    return None


def find_play_script(root: Path) -> Optional[Path]:
    for candidate in (root / "launcher" / "play.sh", root / "play.sh", Path(__file__).resolve().parent / "play.sh"):
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------------------
# Launching
# --------------------------------------------------------------------------------------


def close_steam() -> None:
    """Close the Steam client after the game exits (the --steam mode)."""
    for command in (["steam", "steam://exit"], ["steam", "-shutdown"]):
        if not shutil.which(command[0]):
            continue
        try:
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return
        except OSError:
            continue


def write_env_file(env: Dict[str, str]) -> str:
    handle = tempfile.NamedTemporaryFile("w", prefix="infamous-env-", suffix=".env", delete=False)
    with handle:
        for key in sorted(env):
            handle.write(f"{key}={env[key]}\n")
    return handle.name


def launch(
    settings: Settings,
    root: Path,
    steam: bool = False,
    dry_run: bool = False,
    extra_env: Optional[Dict[str, str]] = None,
) -> int:
    """Start the game with the settings. Returns the exit status."""
    script = find_play_script(root)
    if script is None:
        print("error: play.sh was not found next to the launcher", file=sys.stderr)
        return 2
    binary = find_binary(root)
    if binary is None and not dry_run:
        print(f"error: no inFamousRecomp binary under {root}", file=sys.stderr)
        return 2
    env = settings.as_env()
    if extra_env:
        env.update(extra_env)
    env_file = write_env_file(env)
    command = [str(script), "--env-file", env_file, "--root", str(root)]
    try:
        if dry_run:
            print(" ".join(command))
            for key in sorted(env):
                print(f"{key}={env[key]}")
            return 0
        status = subprocess.call(command)
    finally:
        try:
            os.unlink(env_file)
        except OSError:
            pass
    if steam and status == 0:
        time.sleep(1.0)
        close_steam()
    return status


# --------------------------------------------------------------------------------------
# The settings window
# --------------------------------------------------------------------------------------


class LauncherWindow:
    """The settings window. Tkinter, with controller navigation on top."""

    def __init__(self, settings: Settings, root: Path, backend: Optional[str] = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.settings = settings
        self.root_path = root
        self.backend_name = backend
        self.controller = None
        self._events: List[Tuple[str, str]] = []
        self._lock = threading.Lock()
        self._poller: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._focus_order: List[object] = []
        self._dirty = False

        self.window = tk.Tk()
        self.window.title(PROJECT_NAME)
        self.window.geometry("960x600")
        self.window.minsize(720, 460)
        self.window.protocol("WM_DELETE_WINDOW", self.quit)

        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill="both", expand=True)

        columns = ttk.Frame(outer)
        columns.pack(fill="both", expand=True)

        left = ttk.Frame(columns)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(columns)
        right.pack(side="right", fill="both", expand=True, padx=(12, 0))

        ttk.Label(left, text="Presets", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        self.preset_list = tk.Listbox(left, exportselection=False, activestyle="none", height=8)
        self.preset_list.pack(fill="both", expand=True, pady=(4, 8))
        for name in PRESET_ORDER:
            self.preset_list.insert("end", name)
        self.preset_list.bind("<<ListboxSelect>>", self._on_preset_selected)
        self.preset_description = ttk.Label(left, text="", wraplength=380)
        self.preset_description.pack(anchor="w")

        ttk.Label(right, text="Video", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        grid = ttk.Frame(right)
        grid.pack(fill="x", pady=(4, 8))
        grid.columnconfigure(1, weight=1)

        self.vars: Dict[str, object] = {}
        self._widgets: Dict[str, object] = {}
        row = 0

        self.vars["fsr"] = tk.BooleanVar(value=truthy(settings.get("fsr")))
        widget = ttk.Checkbutton(grid, text="FSR upscaling", variable=self.vars["fsr"], command=self._on_toggle)
        widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=1)
        self._widgets["fsr"] = widget
        row += 1

        self.vars["fsr_mode"] = tk.StringVar(value=settings.get("fsr_mode"))
        ttk.Label(grid, text="FSR mode").grid(row=row, column=0, sticky="w", pady=1)
        self.fsr_mode_box = ttk.Combobox(
            grid, textvariable=self.vars["fsr_mode"], values=list(FSR_MODES), state="readonly", width=14
        )
        self.fsr_mode_box.grid(row=row, column=1, sticky="ew", pady=1)
        row += 1

        self.vars["fsr_fg"] = tk.BooleanVar(value=truthy(settings.get("fsr_fg")))
        widget = ttk.Checkbutton(
            grid, text="Frame generation (FSR 3.1)", variable=self.vars["fsr_fg"], command=self._on_toggle
        )
        widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=1)
        self._widgets["fsr_fg"] = widget
        row += 1

        for key, label in (
            ("fps_cap", "Frame cap (0 = off)"),
            ("res_scale", "Resolution scale (1-3)"),
            ("tex_aniso", "Anisotropic filtering (0-16)"),
            ("lod_class_scale", "Level of detail scale"),
        ):
            ttk.Label(grid, text=label).grid(row=row, column=0, sticky="w", pady=1)
            variable = tk.StringVar(value=settings.get(key))
            self.vars[key] = variable
            entry = ttk.Entry(grid, textvariable=variable, width=14)
            entry.grid(row=row, column=1, sticky="ew", pady=1)
            entry.bind("<FocusOut>", lambda _event: self._on_toggle())
            self._widgets[key] = entry
            row += 1

        self.vars["depth_blur_smooth"] = tk.BooleanVar(value=truthy(settings.get("depth_blur_smooth")))
        widget = ttk.Checkbutton(
            grid, text="Smooth distance blur", variable=self.vars["depth_blur_smooth"], command=self._on_toggle
        )
        widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=1)
        self._widgets["depth_blur_smooth"] = widget
        row += 1

        ttk.Label(grid, text="Input").grid(row=row, column=0, sticky="w", pady=(8, 1))
        row += 1
        self.vars["pad_gyro"] = tk.BooleanVar(value=truthy(settings.get("pad_gyro")))
        widget = ttk.Checkbutton(grid, text="Sixaxis gyro", variable=self.vars["pad_gyro"], command=self._on_toggle)
        widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=1)
        self._widgets["pad_gyro"] = widget
        row += 1
        self.vars["pad_rumble"] = tk.BooleanVar(value=truthy(settings.get("pad_rumble")))
        widget = ttk.Checkbutton(
            grid, text="Controller rumble (haptics)", variable=self.vars["pad_rumble"], command=self._on_toggle
        )
        widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=1)
        self._widgets["pad_rumble"] = widget
        row += 1
        self.rumble_button = ttk.Button(grid, text="Test the rumble", command=self._test_rumble)
        self.rumble_button.grid(row=row, column=0, columnspan=2, sticky="w", pady=1)
        row += 1

        ttk.Label(grid, text="Saves").grid(row=row, column=0, sticky="w", pady=(8, 1))
        row += 1
        ttk.Label(grid, text="Save folder").grid(row=row, column=0, sticky="w", pady=1)
        self.vars["savedata_root"] = tk.StringVar(value=settings.get("savedata_root"))
        savedata = ttk.Entry(grid, textvariable=self.vars["savedata_root"], width=32)
        savedata.grid(row=row, column=1, sticky="ew", pady=1)
        savedata.bind("<FocusOut>", lambda _event: self._on_toggle())
        self._widgets["savedata_root"] = savedata

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(12, 0))
        self.play_button = ttk.Button(buttons, text="Play", command=self.play)
        self.play_button.pack(side="left")
        self.save_button = ttk.Button(buttons, text="Save", command=self.save)
        self.save_button.pack(side="left", padx=6)
        self.quit_button = ttk.Button(buttons, text="Quit", command=self.quit)
        self.quit_button.pack(side="left")
        self.status = ttk.Label(buttons, text="")
        self.status.pack(side="right")

        self._focus_order = [
            self.preset_list,
            self.vars_widget("fsr"),
            self.fsr_mode_box,
            self.vars_widget("fsr_fg"),
            self.vars_widget("fps_cap"),
            self.vars_widget("res_scale"),
            self.vars_widget("tex_aniso"),
            self.vars_widget("lod_class_scale"),
            self.vars_widget("depth_blur_smooth"),
            self.vars_widget("pad_gyro"),
            self.vars_widget("pad_rumble"),
            self.rumble_button,
            self.vars_widget("savedata_root"),
            self.play_button,
            self.save_button,
            self.quit_button,
        ]
        self.preset_list.focus_set()
        self._start_controller()
        self.window.after(50, self._drain_events)

    # -- helpers ------------------------------------------------------------
    def vars_widget(self, key: str):
        """The widget that edits a setting, so the focus order can address it."""
        return self._widgets.get(key)

    def _start_controller(self) -> None:
        try:
            pad = controller_mod.open_controller(self.backend_name)
        except Exception:
            pad = controller_mod.NullController()
        self.controller = pad
        if not pad.available:
            self.status.configure(text="no controller found, use the keyboard or the touch screen")
            return
        self.status.configure(text=f"controller: {pad.backend} ({', '.join(pad.devices)})")

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    events = pad.poll()
                except Exception:
                    events = []
                if events:
                    with self._lock:
                        self._events.extend(events)
                self._stop.wait(0.05)

        self._poller = threading.Thread(target=loop, name="infamous-controller", daemon=True)
        self._poller.start()

    def _drain_events(self) -> None:
        with self._lock:
            events, self._events = self._events, []
        for kind, value in events:
            if kind == "nav":
                self._navigate(value)
            elif kind == "press":
                self._press(value)
        self.window.after(50, self._drain_events)

    def _navigate(self, direction: str) -> None:
        if direction not in ("up", "down"):
            return
        current = self.window.focus_get()
        if current is self.preset_list:
            self._move_preset(-1 if direction == "up" else 1)
            return
        if current in self._focus_order:
            index = self._focus_order.index(current)
            step = -1 if direction == "up" else 1
            index = (index + step) % len(self._focus_order)
            self._focus_order[index].focus_set()
            return
        self.preset_list.focus_set()

    def _move_preset(self, step: int) -> None:
        selection = self.preset_list.curselection()
        index = selection[0] if selection else 0
        index = (index + step) % len(PRESET_ORDER)
        self.preset_list.selection_clear(0, "end")
        self.preset_list.selection_set(index)
        self.preset_list.activate(index)
        self.preset_list.see(index)
        self._on_preset_selected()

    def _press(self, button: str) -> None:
        if button == "a":
            current = self.window.focus_get()
            if current is self.preset_list:
                self._on_preset_selected()
                return
            if hasattr(current, "invoke"):
                current.invoke()
                return
            if hasattr(current, "focus_set"):
                current.focus_set()
        elif button == "b":
            self.preset_list.focus_set()
        elif button == "start":
            self.play()

    # -- settings -----------------------------------------------------------
    def _collect(self) -> Settings:
        for key in ALL_KEYS:
            variable = self.vars.get(key)
            if variable is None:
                continue
            value = variable.get()
            self.settings.set(key, value)
        return self.settings

    def _on_toggle(self) -> None:
        self._collect()
        self._dirty = True
        self.status.configure(text="changed, not saved yet")

    def _on_preset_selected(self) -> None:
        selection = self.preset_list.curselection()
        if not selection:
            return
        name = PRESET_ORDER[selection[0]]
        self.settings.apply_preset(name)
        for key in ALL_KEYS:
            variable = self.vars.get(key)
            if variable is None:
                continue
            value = self.settings.get(key)
            if isinstance(variable, self.tk.BooleanVar):
                variable.set(truthy(value))
            else:
                variable.set(value)
        self.preset_description.configure(text=PRESET_DESCRIPTIONS.get(name, ""))
        self._dirty = True
        self.status.configure(text=f"preset: {name} (not saved yet)")

    def save(self) -> None:
        self._collect()
        path = settings_path(None)
        try:
            self.settings.save(path)
        except OSError as error:
            self.status.configure(text=f"could not save: {error}")
            return
        self._dirty = False
        self.status.configure(text=f"saved to {path}")

    def play(self) -> None:
        self._collect()
        self.status.configure(text="starting the game")
        self.window.update_idletasks()
        launch(self.settings, self.root_path)

    def _test_rumble(self) -> None:
        """A short pulse on the motors, so the player can hear the pad."""
        if self.controller is not None and self.controller.rumble():
            self.status.configure(text="rumble sent")
            return
        # The raw input backend cannot drive the motors. SDL can, and it sees
        # the Steam Input virtual pad that carries the haptics of the Deck.
        try:
            pad = controller_mod.open_controller("sdl")
        except Exception:
            pad = None
        if pad is None:
            self.status.configure(text="no controller to buzz")
            return
        try:
            if pad.available and pad.rumble():
                self.status.configure(text=f"rumble sent ({pad.backend})")
            else:
                self.status.configure(text="this controller cannot rumble")
        finally:
            pad.close()

    def quit(self) -> None:
        self._stop.set()
        if self.controller is not None:
            self.controller.close()
        self.window.destroy()

    def run(self) -> None:
        self.window.mainloop()


def open_window(settings: Settings, root: Path, backend: Optional[str] = None) -> int:
    try:
        import tkinter  # noqa: F401
    except Exception as error:
        print(f"error: tkinter is not available ({error})", file=sys.stderr)
        return 2
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("error: no display, use --play or --print-env", file=sys.stderr)
        return 2
    window = LauncherWindow(settings, root, backend)
    window.run()
    return 0


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="launcher.py",
        description=f"{APP_NAME} launcher for the native Linux (Vulkan) port.",
    )
    parser.add_argument("--play", action="store_true", help="start the game with the saved settings")
    parser.add_argument(
        "--steam", action="store_true", help="start the game and close Steam when the game exits"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print what would run and exit, do not start the game"
    )
    parser.add_argument("--print-env", action="store_true", help="print the resolved environment")
    parser.add_argument("--list-presets", action="store_true", help="print the preset names")
    parser.add_argument("--apply-preset", metavar="NAME", help="store a preset in the settings file")
    parser.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="override one setting for this run (repeatable)",
    )
    parser.add_argument("--save", action="store_true", help="write the --set overrides into the settings file")
    parser.add_argument("--settings", metavar="PATH", help="use another settings file")
    parser.add_argument("--root", metavar="PATH", help="the install directory")
    parser.add_argument(
        "--controller-backend",
        choices=list(controller_mod.BACKENDS),
        help="force one controller backend instead of the automatic choice",
    )
    parser.add_argument("--no-gui", action="store_true", help="never open the settings window")
    parser.add_argument("--gui", action="store_true", help="always open the settings window")
    parser.add_argument("--list-devices", action="store_true", help="list the controller devices and exit")
    return parser


def parse_overrides(items: List[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in DEFAULTS:
            raise ValueError(f"unknown setting: {key}")
        overrides[key] = value.strip()
    return overrides


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path = settings_path(args.settings)
    root = install_root(args.root)

    if args.list_devices:
        pad = controller_mod.open_controller(args.controller_backend)
        print(f"backend: {pad.backend}")
        for device in pad.devices:
            print(f"device: {device}")
        if not pad.available:
            print("no controller found")
        pad.close()
        return 0

    settings = Settings.load(path)

    if args.list_presets:
        for name in PRESET_ORDER:
            print(f"{name}\t{PRESET_DESCRIPTIONS.get(name, '')}")
        return 0

    try:
        overrides = parse_overrides(args.set)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.apply_preset:
        if args.apply_preset not in PRESETS:
            print(f"error: unknown preset: {args.apply_preset}", file=sys.stderr)
            return 2
        settings.apply_preset(args.apply_preset)
        try:
            settings.save(path)
        except OSError as error:
            print(f"error: could not write {path}: {error}", file=sys.stderr)
            return 2
        print(f"preset stored in {path}: {args.apply_preset}")
        if not args.print_env and not args.play:
            return 0

    for key, value in overrides.items():
        settings.set(key, value)

    if args.save and (overrides or args.apply_preset):
        try:
            settings.save(path)
        except OSError as error:
            print(f"error: could not write {path}: {error}", file=sys.stderr)
            return 2
        print(f"settings written to {path}")

    if args.print_env:
        env = settings.as_env()
        for key in sorted(env):
            print(f"{key}={env[key]}")
        if not args.play:
            return 0

    if args.play or args.steam:
        return launch(settings, root, steam=args.steam, dry_run=args.dry_run)

    if args.no_gui:
        print(f"{APP_NAME} settings: {path}")
        print(f"install: {root}")
        return 0

    return open_window(settings, root, args.controller_backend)


if __name__ == "__main__":
    raise SystemExit(main())
