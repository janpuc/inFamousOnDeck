#!/usr/bin/env python3
"""Tests for the Steam shortcut helper.

Run them with:

    python3 -m unittest discover -s linux/steamdeck/tests -t linux/steamdeck
"""

from __future__ import annotations

import contextlib
import io
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = HERE.parent
sys.path.insert(0, str(DECK))

import steam_shortcut as vdf  # noqa: E402


def reference_shortcuts() -> bytes:
    """A shortcuts.vdf in the exact layout Steam writes.

    Built by hand instead of with the writer, so the test does not only prove
    that the writer agrees with itself.
    """
    out = bytearray(b"\x00shortcuts\x00")
    out += b"\x00" + b"0\x00"
    out += b"\x02" + b"appid\x00" + struct.pack("<I", 4228000151)
    out += b"\x01" + b"AppName\x00" + b"Google Chrome\x00"
    out += b"\x01" + b"Exe\x00" + b'"/usr/bin/flatpak"\x00'
    out += b"\x01" + b"StartDir\x00" + b"/usr/bin/\x00"
    out += b"\x01" + b"icon\x00" + b"/home/player/icon.ico\x00"
    out += b"\x01" + b"ShortcutPath\x00" + b"/usr/share/applications/chrome.desktop\x00"
    out += b"\x01" + b"LaunchOptions\x00" + b'"run" "--branch=stable"\x00'
    out += b"\x02" + b"IsHidden\x00" + struct.pack("<I", 0)
    out += b"\x02" + b"AllowDesktopConfig\x00" + struct.pack("<I", 1)
    out += b"\x02" + b"AllowOverlay\x00" + struct.pack("<I", 1)
    out += b"\x02" + b"OpenVR\x00" + struct.pack("<I", 0)
    out += b"\x02" + b"Devkit\x00" + struct.pack("<I", 0)
    out += b"\x01" + b"DevkitGameID\x00" + b"\x00"
    out += b"\x02" + b"DevkitOverrideAppID\x00" + struct.pack("<I", 0)
    out += b"\x02" + b"LastPlayTime\x00" + struct.pack("<I", 1767476261)
    out += b"\x01" + b"FlatpakAppID\x00" + b"com.google.Chrome\x00"
    out += b"\x01" + b"sortas\x00" + b"\x00"
    out += b"\x00" + b"tags\x00" + b"\x08"
    out += b"\x08"
    out += b"\x08"
    out += b"\x08"
    return bytes(out)


class ParseTest(unittest.TestCase):
    def test_reads_the_reference_file(self):
        entries = vdf.parse_shortcuts(reference_shortcuts())
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["appid"], 4228000151)
        self.assertEqual(entry["AppName"], "Google Chrome")
        self.assertEqual(entry["Exe"], '"/usr/bin/flatpak"')
        self.assertEqual(entry["StartDir"], "/usr/bin/")
        self.assertEqual(entry["IsHidden"], 0)
        self.assertEqual(entry["AllowDesktopConfig"], 1)
        self.assertEqual(entry["LastPlayTime"], 1767476261)
        self.assertEqual(entry["FlatpakAppID"], "com.google.Chrome")
        self.assertEqual(entry["tags"], {})

    def test_refuses_a_file_that_is_not_a_shortcut_list(self):
        with self.assertRaises(vdf.VdfError):
            vdf.parse_shortcuts(b"nonsense")

    def test_refuses_a_damaged_file(self):
        broken = bytearray(reference_shortcuts())
        broken[20] = 0x7F  # an unknown value type
        with self.assertRaises(vdf.VdfError):
            vdf.parse_shortcuts(bytes(broken))

    def test_reads_several_entries(self):
        out = bytearray(b"\x00shortcuts\x00")
        for index in range(3):
            out += b"\x00" + str(index).encode() + b"\x00"
            out += b"\x02" + b"appid\x00" + struct.pack("<I", 100 + index)
            out += b"\x01" + b"AppName\x00" + f"Game {index}".encode() + b"\x00"
            out += b"\x08"
        out += b"\x08"
        out += b"\x08"
        entries = vdf.parse_shortcuts(bytes(out))
        self.assertEqual([entry["AppName"] for entry in entries], ["Game 0", "Game 1", "Game 2"])


class RoundTripTest(unittest.TestCase):
    def test_reference_file_survives_a_round_trip(self):
        original = reference_shortcuts()
        entries = vdf.parse_shortcuts(original)
        written = vdf.write_shortcuts(entries)
        self.assertEqual(written, original)

    def test_entries_survive_a_round_trip(self):
        entries = [
            vdf.make_entry("inFamousOnDeck", "/home/player/play.sh", "/home/player", tags=["Linux"]),
            vdf.make_entry("Other", "/usr/bin/other", "/usr/bin"),
        ]
        written = vdf.write_shortcuts(entries)
        parsed = vdf.parse_shortcuts(written)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["AppName"], "inFamousOnDeck")
        self.assertEqual(parsed[0]["Exe"], '"/home/player/play.sh"')
        self.assertEqual(parsed[0]["StartDir"], "/home/player")
        self.assertEqual(parsed[0]["tags"], {"0": "Linux"})
        self.assertEqual(parsed[1]["AppName"], "Other")
        self.assertEqual(parsed[0]["appid"], entries[0]["appid"])

    def test_writing_twice_gives_the_same_bytes(self):
        entries = [vdf.make_entry("inFamous", "/home/player/play.sh", "/home/player")]
        first = vdf.write_shortcuts(entries)
        second = vdf.write_shortcuts(vdf.parse_shortcuts(first))
        self.assertEqual(first, second)

    def test_unknown_fields_are_kept(self):
        entries = [vdf.make_entry("inFamous", "/home/player/play.sh", "/home/player")]
        entries[0]["SomeNewField"] = "value"
        parsed = vdf.parse_shortcuts(vdf.write_shortcuts(entries))
        self.assertEqual(parsed[0]["SomeNewField"], "value")


class FileTest(unittest.TestCase):
    def test_backup_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "shortcuts.vdf"
            path.write_bytes(reference_shortcuts())
            entries = vdf.read_shortcuts(path)
            entries.append(vdf.make_entry("inFamous", "/home/player/play.sh", "/home/player"))
            backup = vdf.write_shortcuts_file(path, entries)
            self.assertIsNotNone(backup)
            self.assertTrue(backup.is_file())
            self.assertEqual(backup.read_bytes(), reference_shortcuts())
            self.assertEqual(len(vdf.read_shortcuts(path)), 2)

    def test_no_backup_when_asked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "shortcuts.vdf"
            path.write_bytes(reference_shortcuts())
            backup = vdf.write_shortcuts_file(path, [], backup=False)
            self.assertIsNone(backup)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_a_missing_file_reads_as_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "shortcuts.vdf"
            self.assertEqual(vdf.read_shortcuts(path), [])


class AppIdTest(unittest.TestCase):
    def test_the_id_is_stable(self):
        first = vdf.compute_appid("/home/player/play.sh", "inFamous")
        second = vdf.compute_appid("/home/player/play.sh", "inFamous")
        self.assertEqual(first, second)

    def test_the_id_has_the_high_bit(self):
        value = vdf.compute_appid("/home/player/play.sh", "inFamous")
        self.assertTrue(value & 0x80000000)
        self.assertLess(value, 0xFFFFFFFF)

    def test_different_names_give_different_ids(self):
        self.assertNotEqual(
            vdf.compute_appid("/home/player/play.sh", "inFamous"),
            vdf.compute_appid("/home/player/play.sh", "Something else"),
        )


class CommandLineTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name) / "Steam"
        config = self.root / "userdata" / "12345678" / "config"
        config.mkdir(parents=True)
        self.path = config / "shortcuts.vdf"
        self.path.write_bytes(reference_shortcuts())

    def tearDown(self):
        self.folder.cleanup()

    def run_main(self, *argv):
        out = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(errors):
            status = vdf.main(["--steam-root", str(self.root), *argv])
        return status, out.getvalue(), errors.getvalue()

    def test_list(self):
        status, output, _errors = self.run_main("--list")
        self.assertEqual(status, 0)
        self.assertIn("Google Chrome", output)

    def test_the_dry_run_writes_nothing(self):
        before = self.path.read_bytes()
        status, output, _errors = self.run_main("--exe", "/home/player/play.sh")
        self.assertEqual(status, 0)
        self.assertIn("dry run", output)
        self.assertEqual(self.path.read_bytes(), before)

    def test_apply_writes_and_backs_up(self):
        status, output, _errors = self.run_main(
            "--exe", "/home/player/play.sh", "--app-name", "inFamous", "--apply"
        )
        self.assertEqual(status, 0)
        entries = vdf.read_shortcuts(self.path)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[1]["AppName"], "inFamous")
        self.assertIn("backup:", output)
        backups = list(self.path.parent.glob("shortcuts.vdf.bak-*"))
        self.assertEqual(len(backups), 1)

    def test_apply_replaces_a_shortcut_with_the_same_name(self):
        self.run_main("--exe", "/home/player/play.sh", "--app-name", "Google Chrome", "--apply")
        entries = vdf.read_shortcuts(self.path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["Exe"], '"/home/player/play.sh"')

    def test_remove_needs_apply(self):
        before = self.path.read_bytes()
        status, output, _errors = self.run_main("--remove", "Google Chrome")
        self.assertEqual(status, 0)
        self.assertIn("dry run", output)
        self.assertEqual(self.path.read_bytes(), before)

    def test_remove_with_apply(self):
        status, _output, _errors = self.run_main("--remove", "Google Chrome", "--apply")
        self.assertEqual(status, 0)
        self.assertEqual(vdf.read_shortcuts(self.path), [])

    def test_remove_of_an_unknown_name_fails(self):
        status, _output, errors = self.run_main("--remove", "Nothing", "--apply")
        self.assertEqual(status, 1)
        self.assertIn("no shortcut named", errors)

    def test_without_an_exe_it_fails(self):
        status, _output, errors = self.run_main("--apply")
        self.assertEqual(status, 2)
        self.assertIn("--exe", errors)


if __name__ == "__main__":
    unittest.main()
