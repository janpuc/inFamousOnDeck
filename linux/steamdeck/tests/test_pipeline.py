#!/usr/bin/env python3
"""Tests for the Linux pipeline driver.

Run them with:

    python3 -m unittest discover -s linux/steamdeck/tests -t linux/steamdeck
"""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = HERE.parent
REPO = DECK.parent.parent
sys.path.insert(0, str(DECK))

import pipeline  # noqa: E402


class PatchTest(unittest.TestCase):
    def test_a_simple_hunk(self):
        old = b"a\nb\nc\n"
        patch = b"--- a/f\n+++ b/f\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"
        self.assertEqual(pipeline.apply_patch(old, patch), b"a\nB\nc\n")

    def test_an_insertion(self):
        old = b"a\nb\n"
        patch = b"--- a/f\n+++ b/f\n@@ -1,2 +1,3 @@\n a\n+new\n b\n"
        self.assertEqual(pipeline.apply_patch(old, patch), b"a\nnew\nb\n")

    def test_a_deletion(self):
        old = b"a\nb\nc\n"
        patch = b"--- a/f\n+++ b/f\n@@ -1,3 +1,2 @@\n a\n-b\n c\n"
        self.assertEqual(pipeline.apply_patch(old, patch), b"a\nc\n")

    def test_the_wrong_base_is_refused(self):
        old = b"a\nq\nc\n"
        patch = b"--- a/f\n+++ b/f\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"
        with self.assertRaises(pipeline.PipelineError):
            pipeline.apply_patch(old, patch)

    def test_the_line_endings_are_converted(self):
        old = b"a\r\nb\r\n"
        patch = b"# eol lf\n--- a/f\n+++ b/f\n@@ -1,2 +1,2 @@\n a\n-b\n+B\n"
        self.assertEqual(pipeline.apply_patch(old, patch), b"a\nB\n")

    def test_the_real_patch_set(self):
        """Every patch of the port applies to the file it was made from.

        The old file is rebuilt from the hunk itself, so this checks the
        parser and the applier against the patches the port ships.
        """
        patch_dir = REPO / "installer" / "patches" / "infamous-recomp"
        patches = sorted(patch_dir.rglob("*.patch"))
        self.assertTrue(patches, "the port ships patches")
        for patch_path in patches:
            with self.subTest(patch=str(patch_path.relative_to(REPO))):
                data = patch_path.read_bytes()
                hunks = pipeline.parse_patch(data)
                if not hunks:
                    # A patch without hunks only converts the line endings.
                    self.assertTrue(
                        data.startswith(pipeline.EOL_LF) or data.startswith(pipeline.EOL_CRLF),
                        "a patch without hunks only converts the line endings",
                    )
                    continue
                old_lines: list[bytes] = []
                new_lines: list[bytes] = []
                for old_start, old_count, body in hunks:
                    at = old_start if old_count == 0 else old_start - 1
                    gap = at - len(old_lines)
                    self.assertGreaterEqual(gap, 0, "the hunks are in order")
                    filler = [b"filler %d\n" % index for index in range(gap)]
                    old_lines.extend(filler)
                    new_lines.extend(filler)
                    for tag, payload in body:
                        if tag in (b" ", b"-"):
                            old_lines.append(payload)
                        if tag in (b" ", b"+"):
                            new_lines.append(payload)
                old = b"".join(old_lines)
                expected = b"".join(new_lines)
                self.assertEqual(pipeline.apply_patch(old, data), expected)


class JobmodTest(unittest.TestCase):
    @staticmethod
    def synthetic_eboot(entry: int = 0x10, size: int = 0x40) -> bytes:
        data = bytearray(b"\x00" * 0x20)
        module = bytearray(size)
        for index in range(4):
            module[index * 4 : index * 4 + 4] = ((0x21 << 25) | index).to_bytes(4, "big")
        module[0x10:0x14] = entry.to_bytes(4, "big")
        module[0x14:0x18] = size.to_bytes(4, "big")
        module[0x20:0x24] = b"\xc0\xde\xc0\xde"
        data += module
        return bytes(data)

    def test_the_module_is_found_and_named(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            elf = root / "EBOOT.elf"
            elf.write_bytes(self.synthetic_eboot())
            out = root / "jobmods"
            names = pipeline.extract_jobmods(elf, out)
            self.assertEqual(len(names), 1)
            digest = hashlib.sha1(self.synthetic_eboot()[0x20:]).hexdigest()[:12]
            self.assertEqual(names[0], f"jobmod_{digest}_e0010")
            self.assertTrue((out / f"{names[0]}.bin").is_file())

    def test_an_eboot_without_modules_is_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            elf = root / "EBOOT.elf"
            elf.write_bytes(b"\x00" * 64)
            with self.assertRaises(pipeline.PipelineError):
                pipeline.extract_jobmods(elf, root / "jobmods")

    def test_carve(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "src.bin"
            source.write_bytes(b"0123456789")
            target = root / "out" / "dst.bin"
            pipeline.carve(source, 2, 4, target)
            self.assertEqual(target.read_bytes(), b"2345")

    def test_carve_past_the_end_is_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "src.bin"
            source.write_bytes(b"0123456789")
            with self.assertRaises(pipeline.PipelineError):
                pipeline.carve(source, 8, 8, root / "dst.bin")


class ManifestTest(unittest.TestCase):
    def test_the_manifest_has_the_linux_build_section(self):
        import tomllib

        with open(REPO / "installer" / "infamous.toml", "rb") as handle:
            manifest = tomllib.load(handle)
        self.assertIn("configure_unix", manifest["build"])
        self.assertIn("compile_unix", manifest["build"])
        self.assertIn("exe_unix", manifest["build"])
        flags = " ".join(manifest["build"]["configure_unix"])
        for flag in (
            "-DPS3RECOMP_RSX_VULKAN=ON",
            "-DPORT_SPU_RC_ONLY=ON",
            "-DPORT_SPU_CFG_MERGE=ON",
            "-DPORT_PPU_REGCACHE=ON",
            "-DPORT_PPU_VMB_LOCAL=ON",
            "-DPORT_PPU_VR=ALL",
            "-DPORT_PPU_MEMLEAN=ALL",
        ):
            self.assertIn(flag, flags, flag)

    def test_the_manifest_has_the_linux_launch_environment(self):
        import tomllib

        with open(REPO / "installer" / "infamous.toml", "rb") as handle:
            manifest = tomllib.load(handle)
        self.assertIn("launch_unix", manifest)
        env = manifest["launch_unix"]["env"]
        self.assertEqual(env["RES_SCALE"], "1")
        self.assertEqual(env["LOD_CLASS_SCALE"], "1")
        self.assertEqual(env["DEPTH_BLUR_SMOOTH"], "1")


if __name__ == "__main__":
    unittest.main()
