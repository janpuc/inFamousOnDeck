#!/usr/bin/env python3
"""Tests for the Steam Deck launcher.

Run them with:

    python3 -m unittest discover -s linux/steamdeck/tests -t linux/steamdeck
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = HERE.parent
sys.path.insert(0, str(DECK))

import launcher  # noqa: E402


class SettingsRoundTripTest(unittest.TestCase):
    def test_defaults(self):
        settings = launcher.Settings()
        self.assertEqual(settings.get("fsr"), "0")
        self.assertEqual(settings.get("res_scale"), "1")
        self.assertEqual(settings.get("depth_blur_smooth"), "1")
        self.assertEqual(settings.get("savedata_root"), "")
        self.assertEqual(settings.get("pad_rumble"), "1")
        self.assertEqual(settings.as_env()["PAD_RUMBLE"], "1")

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sub" / "settings.ini"
            settings = launcher.Settings()
            settings.set("fsr", "1")
            settings.set("fsr_mode", "quality")
            settings.set("fsr_fg", "1")
            settings.set("fps_cap", "30")
            settings.set("res_scale", "1")
            settings.set("tex_aniso", "8")
            settings.set("lod_class_scale", "1.5")
            settings.set("depth_blur_smooth", "0")
            settings.set("pad_gyro", "1")
            settings.set("pad_rumble", "0")
            settings.set("savedata_root", "/home/player/saves")
            settings.save(path)

            self.assertTrue(path.is_file())
            loaded = launcher.Settings.load(path)
            for key in launcher.ALL_KEYS:
                self.assertEqual(loaded.get(key), settings.get(key), key)

    def test_missing_file_gives_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = launcher.Settings.load(Path(folder) / "nope.ini")
            self.assertEqual(settings.values, launcher.DEFAULTS)

    def test_unknown_keys_are_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.ini"
            path.write_text("[video]\nfsr = 1\nnot_a_setting = 5\n", encoding="utf-8")
            settings = launcher.Settings.load(path)
            self.assertEqual(settings.get("fsr"), "1")

    def test_settings_file_uses_the_sections(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.ini"
            launcher.Settings().save(path)
            text = path.read_text(encoding="utf-8")
            for section in ("[video]", "[input]", "[paths]", "[launch]"):
                self.assertIn(section, text)

    def test_config_path_follows_xdg(self):
        with tempfile.TemporaryDirectory() as folder:
            old = os.environ.get("XDG_CONFIG_HOME")
            os.environ["XDG_CONFIG_HOME"] = folder
            try:
                path = launcher.settings_path()
            finally:
                if old is None:
                    os.environ.pop("XDG_CONFIG_HOME", None)
                else:
                    os.environ["XDG_CONFIG_HOME"] = old
            self.assertEqual(path, Path(folder) / "inFamousOnDeck" / "settings.ini")


class EnvTest(unittest.TestCase):
    def test_env_names(self):
        env = launcher.Settings().as_env()
        self.assertEqual(
            sorted(env),
            sorted(
                [
                    "FSR",
                    "FSR_MODE",
                    "FSR_FG",
                    "FPS_CAP",
                    "RES_SCALE",
                    "TEX_ANISO",
                    "LOD_CLASS_SCALE",
                    "DEPTH_BLUR_SMOOTH",
                    "PAD_GYRO",
                    "PAD_RUMBLE",
                ]
            ),
        )

    def test_bool_and_number_formatting(self):
        settings = launcher.Settings()
        settings.set("fsr", "1")
        settings.set("fsr_fg", "0")
        settings.set("fps_cap", "30")
        settings.set("res_scale", "1")
        settings.set("tex_aniso", "16")
        settings.set("lod_class_scale", "1.5")
        settings.set("depth_blur_smooth", "1")
        settings.set("pad_gyro", "0")
        env = settings.as_env()
        self.assertEqual(env["FSR"], "1")
        self.assertEqual(env["FSR_FG"], "0")
        self.assertEqual(env["FPS_CAP"], "30")
        self.assertEqual(env["TEX_ANISO"], "16")
        self.assertEqual(env["LOD_CLASS_SCALE"], "1.5")
        self.assertEqual(env["DEPTH_BLUR_SMOOTH"], "1")
        self.assertEqual(env["PAD_GYRO"], "0")
        self.assertEqual(env["PAD_RUMBLE"], "1")

    def test_the_rumble_can_be_turned_off(self):
        settings = launcher.Settings()
        settings.set("pad_rumble", "0")
        self.assertEqual(settings.as_env()["PAD_RUMBLE"], "0")
        settings.set("pad_rumble", False)
        self.assertEqual(settings.as_env()["PAD_RUMBLE"], "0")
        settings.set("pad_rumble", True)
        self.assertEqual(settings.as_env()["PAD_RUMBLE"], "1")

    def test_empty_save_path_is_not_exported(self):
        env = launcher.Settings().as_env()
        self.assertNotIn("SAVEDATA_ROOT", env)

    def test_save_path_is_exported(self):
        settings = launcher.Settings()
        settings.set("savedata_root", "/data/saves")
        self.assertEqual(settings.as_env()["SAVEDATA_ROOT"], "/data/saves")

    def test_bad_numbers_fall_back_to_the_default(self):
        settings = launcher.Settings()
        settings.set("fps_cap", "abc")
        settings.set("lod_class_scale", "x")
        env = settings.as_env()
        self.assertEqual(env["FPS_CAP"], launcher.DEFAULTS["fps_cap"])
        self.assertEqual(env["LOD_CLASS_SCALE"], launcher.DEFAULTS["lod_class_scale"])


class PresetTest(unittest.TestCase):
    def test_the_six_presets_exist(self):
        self.assertEqual(len(launcher.PRESETS), 6)
        for name in (
            "Steam Deck OLED: Smooth 60 (30+FG)",
            "Steam Deck OLED: Max FG (~69 fps)",
            "Steam Deck OLED: Native 40 (80 Hz)",
            "AMD BC-250: Native 60+ + 16x AF",
            "AMD BC-250: 100+ FSR FG + 16x AF",
            "AMD BC-250: Extended LOD 1.5 + 16x AF",
        ):
            self.assertIn(name, launcher.PRESETS)

    def test_smooth_60_preset(self):
        settings = launcher.Settings()
        settings.apply_preset("Steam Deck OLED: Smooth 60 (30+FG)")
        env = settings.as_env()
        self.assertEqual(env["FSR"], "1")
        self.assertEqual(env["FSR_MODE"], "quality")
        self.assertEqual(env["FSR_FG"], "1")
        self.assertEqual(env["FPS_CAP"], "30")
        self.assertEqual(env["RES_SCALE"], "1")
        self.assertEqual(env["TEX_ANISO"], "0")
        self.assertEqual(env["LOD_CLASS_SCALE"], "1")

    def test_max_fg_preset(self):
        settings = launcher.Settings()
        settings.apply_preset("Steam Deck OLED: Max FG (~69 fps)")
        env = settings.as_env()
        self.assertEqual(env["FSR"], "1")
        self.assertEqual(env["FSR_MODE"], "quality")
        self.assertEqual(env["FSR_FG"], "1")
        self.assertEqual(env["FPS_CAP"], "0")
        self.assertEqual(env["RES_SCALE"], "1")
        self.assertEqual(env["TEX_ANISO"], "0")

    def test_native_40_preset(self):
        settings = launcher.Settings()
        settings.apply_preset("Steam Deck OLED: Native 40 (80 Hz)")
        env = settings.as_env()
        self.assertEqual(env["FSR"], "0")
        self.assertEqual(env["FSR_FG"], "0")
        self.assertEqual(env["FPS_CAP"], "40")
        self.assertEqual(env["RES_SCALE"], "1")
        self.assertEqual(env["TEX_ANISO"], "0")

    def test_bc250_native_preset(self):
        settings = launcher.Settings()
        settings.apply_preset("AMD BC-250: Native 60+ + 16x AF")
        env = settings.as_env()
        self.assertEqual(env["FSR"], "0")
        self.assertEqual(env["FSR_FG"], "0")
        self.assertEqual(env["FPS_CAP"], "0")
        self.assertEqual(env["RES_SCALE"], "1")
        self.assertEqual(env["TEX_ANISO"], "16")
        self.assertEqual(env["LOD_CLASS_SCALE"], "1")

    def test_bc250_frame_generation_preset(self):
        settings = launcher.Settings()
        settings.apply_preset("AMD BC-250: 100+ FSR FG + 16x AF")
        env = settings.as_env()
        self.assertEqual(env["FSR"], "1")
        self.assertEqual(env["FSR_MODE"], "quality")
        self.assertEqual(env["FSR_FG"], "1")
        self.assertEqual(env["FPS_CAP"], "0")
        self.assertEqual(env["RES_SCALE"], "1")
        self.assertEqual(env["TEX_ANISO"], "16")

    def test_bc250_extended_lod_preset(self):
        settings = launcher.Settings()
        settings.apply_preset("AMD BC-250: Extended LOD 1.5 + 16x AF")
        env = settings.as_env()
        self.assertEqual(env["FSR"], "0")
        self.assertEqual(env["FSR_FG"], "0")
        self.assertEqual(env["RES_SCALE"], "1")
        self.assertEqual(env["TEX_ANISO"], "16")
        self.assertEqual(env["LOD_CLASS_SCALE"], "1.5")

    def test_every_preset_uses_known_keys(self):
        for name, values in launcher.PRESETS.items():
            for key in values:
                self.assertIn(key, launcher.DEFAULTS, name)

    def test_every_preset_has_the_rumble_on(self):
        for name in launcher.PRESETS:
            settings = launcher.Settings()
            settings.apply_preset(name)
            self.assertEqual(settings.as_env()["PAD_RUMBLE"], "1", name)

    def test_preset_name_is_recognised(self):
        settings = launcher.Settings()
        for name in launcher.PRESETS:
            settings.apply_preset(name)
            self.assertEqual(settings.preset_name, name)

    def test_unknown_preset_raises(self):
        with self.assertRaises(KeyError):
            launcher.Settings().apply_preset("nope")


class CommandLineTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.settings = Path(self.folder.name) / "settings.ini"
        self.root = Path(self.folder.name) / "install"
        self.root.mkdir()

    def tearDown(self):
        self.folder.cleanup()

    def run_main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = launcher.main(["--settings", str(self.settings), *argv])
        return status, out.getvalue()

    def test_list_presets(self):
        status, output = self.run_main("--list-presets")
        self.assertEqual(status, 0)
        for name in launcher.PRESETS:
            self.assertIn(name, output)

    def test_print_env(self):
        status, output = self.run_main("--print-env")
        self.assertEqual(status, 0)
        self.assertIn("RES_SCALE=1", output)
        self.assertIn("FSR=0", output)

    def test_set_override(self):
        status, output = self.run_main("--set", "fsr=1", "--set", "fps_cap=30", "--print-env")
        self.assertEqual(status, 0)
        self.assertIn("FSR=1", output)
        self.assertIn("FPS_CAP=30", output)

    def test_unknown_setting_is_refused(self):
        status, _output = self.run_main("--set", "nope=1", "--print-env")
        self.assertEqual(status, 2)

    def test_apply_preset_writes_the_file(self):
        status, _output = self.run_main("--apply-preset", "Steam Deck OLED: Native 40 (80 Hz)")
        self.assertEqual(status, 0)
        self.assertTrue(self.settings.is_file())
        settings = launcher.Settings.load(self.settings)
        self.assertEqual(settings.get("fsr"), "0")
        self.assertEqual(settings.get("fsr_fg"), "0")
        self.assertEqual(settings.get("fps_cap"), "40")

    def test_apply_preset_with_save(self):
        status, _output = self.run_main("--set", "tex_aniso=16", "--save", "--no-gui")
        self.assertEqual(status, 0)
        settings = launcher.Settings.load(self.settings)
        self.assertEqual(settings.get("tex_aniso"), "16")

    def test_dry_run_play_does_not_start_anything(self):
        status, output = self.run_main("--play", "--dry-run")
        self.assertEqual(status, 0)
        self.assertIn("play.sh", output)
        self.assertIn("RES_SCALE=1", output)

    def test_no_gui_does_not_open_a_window(self):
        status, output = self.run_main("--no-gui")
        self.assertEqual(status, 0)
        self.assertIn("settings:", output)

    def test_missing_binary_is_reported(self):
        status, _output = self.run_main("--play", "--root", str(self.root))
        self.assertEqual(status, 2)


class ControllerTest(unittest.TestCase):
    def test_backends_are_known(self):
        self.assertEqual(
            list(launcher.controller_mod.BACKENDS),
            ["evdev", "pygame", "sdl", "none"],
        )

    def test_null_backend_reports_nothing(self):
        pad = launcher.controller_mod.NullController()
        self.assertFalse(pad.available)
        self.assertEqual(pad.poll(), [])

    def test_opening_an_unknown_backend_fails(self):
        with self.assertRaises(ValueError):
            launcher.controller_mod.open_controller("nope")

    def test_forced_none_backend(self):
        pad = launcher.controller_mod.open_controller("none")
        self.assertEqual(pad.backend, "none")
        self.assertFalse(pad.available)
        pad.close()

    def test_the_rumble_of_a_backend_without_motors(self):
        pad = launcher.controller_mod.NullController()
        self.assertFalse(pad.rumble())
        self.assertFalse(pad.rumble(low=0, high=0, duration_ms=10))

    def test_every_backend_has_a_rumble(self):
        for name in ("evdev", "pygame", "sdl"):
            pad = launcher.controller_mod.open_controller(name)
            try:
                # Without a device it must not raise, whatever it returns.
                pad.rumble()
            finally:
                pad.close()

    def test_evdev_deadzone_direction(self):
        pad = launcher.controller_mod.EvdevController()
        self.assertEqual(pad._directions(0, 0, 0, 0), [])
        self.assertEqual(pad._directions(0, -1, 0, 0), ["up"])
        self.assertEqual(pad._directions(0, 1, 0, 0), ["down"])
        self.assertEqual(pad._directions(-1, 0, 0, 0), ["left"])
        self.assertEqual(pad._directions(1, 0, 0, 0), ["right"])
        self.assertEqual(pad._directions(0, 0, 30000, 0), ["right"])
        self.assertEqual(pad._directions(0, 0, 0, -30000), ["up"])


if __name__ == "__main__":
    unittest.main()
