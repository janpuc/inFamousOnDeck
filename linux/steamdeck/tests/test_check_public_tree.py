#!/usr/bin/env python3
"""Tests for the public tree gate.

Run them with:

    python3 -m unittest discover -s linux/steamdeck/tests -t linux/steamdeck
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = HERE.parent
REPO = DECK.parent.parent
sys.path.insert(0, str(REPO / "tools"))

import check_public_tree as gate  # noqa: E402


class SyntheticTreeTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def write(self, relative: str, text: str = "x") -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def scan(self):
        return gate.scan(self.root)

    def categories(self, report):
        return {finding.category for finding in report.findings}

    def paths(self, report):
        return {finding.path for finding in report.findings}

    def test_a_clean_tree_has_no_findings(self):
        self.write("README.md", "A native port of inFamous for the Steam Deck.\n")
        self.write("src/main.cpp", "int main() { return 0; }\n")
        report = self.scan()
        self.assertEqual(report.findings, [])
        self.assertEqual(report.files, 2)

    def test_the_game_executable_is_found(self):
        self.write("game/PS3_GAME/USRDIR/EBOOT.BIN")
        self.write("game/PS3_GAME/USRDIR/EBOOT.elf")
        report = self.scan()
        self.assertEqual(self.paths(report), {"game/PS3_GAME/USRDIR/EBOOT.BIN", "game/PS3_GAME/USRDIR/EBOOT.elf"})

    def test_the_archives_are_found(self):
        self.write("cache/infamous1.psarc")
        self.write("cache/infamous1.psarc_s")
        self.write("cache/patch.sdat")
        self.write("work/ppu_0000.ppu")
        report = self.scan()
        self.assertEqual(len(report.findings), 4)
        self.assertEqual(self.categories(report), {"game data"})

    def test_the_lifted_code_is_found(self):
        self.write("infamous-recomp/recompiled/ppu_recomp_000.cpp", "// lifted\n")
        self.write("infamous-recomp/spu/jobmod_0308a0f1346d_e4E10/spu_recomp.c", "// lifted\n")
        self.write("infamous-recomp/spu-images/spu_0000_at_005F6500.elf")
        self.write("infamous-recomp/assets/ui_font.bin")
        report = self.scan()
        self.assertEqual(len(report.findings), 4)
        self.assertEqual(self.categories(report), {"game data"})

    def test_the_captures_are_found(self):
        self.write("artifacts/city-2200.ppm")
        self.write("artifacts/frame.bmp")
        report = self.scan()
        self.assertEqual(self.categories(report), {"capture"})

    def test_the_saves_and_the_logs_are_found(self):
        self.write("play.log", "log\n")
        self.write("savedata/BLES00000_INFAMOUS_0", "save\n")
        report = self.scan()
        self.assertEqual(self.categories(report), {"saves"})

    def test_the_identifiers_are_found(self):
        self.write("docs/notes.md", "the box is janpuc@192.168.69.250\n")
        self.write("scripts/run.sh", "BC250_REMOTE=1\n")
        self.write("scripts/other.sh", "the aether host and the homelab network\n")
        report = self.scan()
        self.assertEqual(self.categories(report), {"private"})
        # janpuc and the address of the network, then aether and homelab.
        self.assertEqual(len(report.findings), 4)

    def test_the_hardware_name_is_allowed(self):
        self.write("docs/profiles.md", "the AMD BC-250 living room box\n")
        report = self.scan()
        self.assertEqual(report.findings, [])

    def test_the_host_name_of_the_box_is_not_allowed(self):
        self.write("docs/profiles.md", "the box is bc250.local\n")
        report = self.scan()
        self.assertEqual(self.categories(report), {"private"})

    def test_the_home_folders_are_not_allowed(self):
        self.write("docs/profiles.md", "the save is in /home/janpuc/saves\n")
        report = self.scan()
        self.assertEqual(self.categories(report), {"private"})

    def test_the_public_repository_address_is_allowed(self):
        self.write(
            "README.md",
            "git clone https://github.com/janpuc/inFamousOnDeck.git\n"
            "Issues: https://github.com/janpuc/inFamousOnDeck/issues\n"
            "Source: https://github.com/janpuc/inFamousOnDeck/tree/main/linux/steamdeck\n"
            "Or github.com/janpuc/inFamousOnDeck and git@github.com:janpuc/inFamousOnDeck.git\n",
        )
        report = self.scan()
        self.assertEqual(report.findings, [])

    def test_the_public_repository_address_does_not_hide_other_names(self):
        self.write("a.md", "https://github.com/janpuc/BC250\n")
        self.write("b.md", "https://github.com/janpuc/inFamousOnDeck-private\n")
        self.write("c.md", "https://github.com/janpuc/inFamousOnDeck/../../janpuc/BC250\n")
        self.write("d.md", "https://github.com/janpuc/inFamousOnDeckX\n")
        self.write("e.md", "see https://github.com/janpuc/inFamousOnDeck and ssh janpuc@box\n")
        self.write("f.md", "https://github.com/janpuc/inFamousOnDeck\nthe save is in /home/janpuc/saves\n")
        self.write("g.md", "https://example.org/janpuc/inFamousOnDeck\n")
        report = self.scan()
        self.assertEqual(self.paths(report), {"a.md", "b.md", "c.md", "d.md", "e.md", "f.md", "g.md"})
        self.assertEqual(self.categories(report), {"private"})
        # Line numbers still point at the right line after the address is blanked out.
        finding = [f for f in report.findings if f.path == "f.md" and "line 2" in f.detail]
        self.assertTrue(finding)

    def test_a_checksum_is_not_an_identifier(self):
        self.write("Cargo.lock", '"bc250dcfd960262841" = "1.0.0"\n')
        report = self.scan()
        self.assertEqual(report.findings, [])

    def test_a_word_that_contains_an_identifier_is_not_an_identifier(self):
        self.write("notes.md", "the spores are everywhere\n")
        report = self.scan()
        self.assertEqual(report.findings, [])

    def test_a_binary_file_is_only_matched_by_its_path(self):
        path = self.root / "data.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00\x01janpuc\x00")
        report = self.scan()
        self.assertEqual(report.findings, [])

    def test_the_gate_does_not_report_itself(self):
        self.write("tools/check_public_tree.py", "IDENTIFIERS = ('janpuc', 'bc250')\n")
        report = self.scan()
        self.assertEqual(report.findings, [])


class CommandLineTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def test_a_clean_tree_exits_zero(self):
        (self.root / "README.md").write_text("hello\n", encoding="utf-8")
        self.assertEqual(gate.main([str(self.root)]), 0)

    def test_a_dirty_tree_exits_one(self):
        (self.root / "EBOOT.BIN").write_text("x", encoding="utf-8")
        self.assertEqual(gate.main([str(self.root)]), 1)

    def test_the_report_mode_always_exits_zero(self):
        (self.root / "EBOOT.BIN").write_text("x", encoding="utf-8")
        self.assertEqual(gate.main(["--report", str(self.root)]), 0)

    def test_a_missing_tree_exits_two(self):
        self.assertEqual(gate.main([str(self.root / "nope")]), 2)


if __name__ == "__main__":
    unittest.main()
