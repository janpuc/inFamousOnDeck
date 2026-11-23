#!/usr/bin/env python3
"""Check that a tree is free of game data and of anything private.

This is the gate in front of publishing the port. It walks a directory and
reports two kinds of problems:

  game data    the executable of the game, its archives, the lifted code that
               was generated from the player's own dump, the frames and the
               images that were captured from a run, the saves and the logs
  private      names and addresses of the machines and the people the port was
               developed on

The check is a gate by default: it exits non-zero when it finds something.
With --report it prints the same report and exits 0, which is what the
release checklist runs to see what a tree still holds.

    tools/check_public_tree.py                     gate: fails on any finding
    tools/check_public_tree.py --report            the same report, always 0
    tools/check_public_tree.py --report --json     the report as JSON
    tools/check_public_tree.py path/to/tree        another tree

Exit status: 0 clean, 1 findings, 2 the tree could not be read.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# The directories of the tree that never hold anything a player may publish.
SKIP_DIRECTORIES = {".git", ".github/workflows", "__pycache__", ".agent", ".worktrees"}

# The identifiers of the machines and the people the port was developed on.
# They are matched case insensitively, so BC250 and bc250 are both found.
# A hardware name that the profiles and the documentation use on purpose (the
# AMD BC-250 living room box) is not private and is not listed here.
IDENTIFIERS = (
    "janpuc",
    "/home/janpuc",
    "/var/home/janpuc",
    "192.168.69.",
    "bc250.local",
    "spore",
    "aether",
    "sisyphus",
    "homelab",
    "tailscale",
)

# The one address that names the account on purpose: the public repository of the port. Only this repository and its
# known pages are allowed, so another repository of the same account, a bare handle and a path that climbs out of it
# (github.com/janpuc/inFamousOnDeck/../other) are still found. The match is blanked out before the identifiers are
# searched for, so a private name on the same line is still reported.
PUBLIC_REPOSITORY_PATTERN = re.compile(
    r"(?<![\w.\-])"
    r"(?:(?:https?://)?(?:www\.)?github\.com[/:]|git@github\.com:)"
    r"janpuc/inFamousOnDeck(?![\w\-])"
    r"(?:\.git(?![\w\-]))?"
    r"(?:/(?:(?:issues|pulls|releases|discussions|wiki|actions|tree|blob)"
    r"(?:/(?!\.+(?:/|$))[\w%#.\-]+)*)?)?"
)

# Files bigger than this are not read for identifiers. The generated code is
# the only thing this big and it is caught by the path rules.
MAX_TEXT_BYTES = 16 * 1024 * 1024

# The gate does not report itself: it names the identifiers it looks for, and
# so does its test.
SELF_NAMES = {"check_public_tree.py", "test_check_public_tree.py"}


def identifier_pattern(identifier: str) -> "re.Pattern[str]":
    """The identifier as a word.

    A letter or a digit next to the identifier means it is part of a longer
    name (a checksum, a different word) and not the name itself. An
    underscore counts as a separator, so BC250_REMOTE is found.
    """
    escaped = re.escape(identifier)
    if identifier[0].isdigit():
        return re.compile(escaped, re.IGNORECASE)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


IDENTIFIER_PATTERN = {identifier: identifier_pattern(identifier) for identifier in IDENTIFIERS}


@dataclass
class Rule:
    category: str
    kind: str  # name, suffix, glob or text
    pattern: str
    note: str = ""


RULES: Tuple[Rule, ...] = (
    # The game itself.
    Rule("game data", "name", "eboot.bin", "the executable of the game"),
    Rule("game data", "name", "eboot.elf", "the decrypted executable of the game"),
    Rule("game data", "suffix", ".psarc", "a game archive"),
    Rule("game data", "suffix", ".psarc_s", "a game archive"),
    Rule("game data", "suffix", ".sdat", "a game archive"),
    Rule("game data", "suffix", ".ppu", "a lifted PowerPC function"),
    Rule("game data", "glob", "infamous-recomp/recompiled/*.cpp", "code lifted from the game"),
    Rule("game data", "glob", "infamous-recomp/recompiled/*.h", "code lifted from the game"),
    Rule("game data", "glob", "infamous-recomp/spu/jobmod_*/spu_recomp*.c", "code lifted from the game"),
    Rule("game data", "glob", "infamous-recomp/spu/jobmod_*/spu_recomp*.h", "code lifted from the game"),
    Rule("game data", "glob", "infamous-recomp/spu-images/*", "a program lifted from the game"),
    Rule("game data", "glob", "infamous-recomp/assets/*.bin", "an asset taken from the game"),
    Rule("game data", "glob", "infamous-recomp/assets/*.png", "an asset taken from the game"),
    # Captures of a run.
    Rule("capture", "suffix", ".ppm", "a frame that was captured from a run"),
    Rule("capture", "suffix", ".bmp", "a frame that was captured from a run"),
    # Saves and logs.
    Rule("saves", "glob", "**/savedata/*", "a save of the game"),
    Rule("saves", "glob", "**/dev_hdd0/*", "the disk image of the game"),
    Rule("saves", "name", "savedata", "a save of the game"),
    Rule("saves", "suffix", ".sav", "a save of the game"),
    Rule("saves", "suffix", ".log", "a log of a run"),
)


@dataclass
class Finding:
    category: str
    rule: str
    path: str
    detail: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {
            "category": self.category,
            "rule": self.rule,
            "path": self.path,
            "detail": self.detail,
        }


@dataclass
class Report:
    root: Path
    findings: List[Finding] = field(default_factory=list)
    files: int = 0
    directories: int = 0
    skipped: int = 0

    def by_category(self) -> Dict[str, List[Finding]]:
        grouped: Dict[str, List[Finding]] = {}
        for finding in self.findings:
            grouped.setdefault(finding.category, []).append(finding)
        return grouped


def is_skipped(relative: Path) -> bool:
    for part in relative.parts:
        if part in SKIP_DIRECTORIES:
            return True
    return False


def match_path(relative: str, name: str, rule: Rule) -> Optional[str]:
    if rule.kind == "name":
        if name.lower() == rule.pattern.lower():
            return rule.note or f"the name {rule.pattern}"
        return None
    if rule.kind == "suffix":
        if name.lower().endswith(rule.pattern.lower()):
            return rule.note or f"a file that ends with {rule.pattern}"
        return None
    if rule.kind == "glob":
        import fnmatch

        if fnmatch.fnmatch(relative, rule.pattern) or fnmatch.fnmatch(relative, f"*/{rule.pattern}"):
            return rule.note or f"the pattern {rule.pattern}"
        # A pattern with ** also matches the deeper folders of the tree.
        if rule.pattern.startswith("**/"):
            tail = rule.pattern[3:]
            if fnmatch.fnmatch(relative, tail) or fnmatch.fnmatch(relative, f"*/{tail}"):
                return rule.note or f"the pattern {rule.pattern}"
            parts = relative.split("/")
            for index in range(len(parts)):
                if fnmatch.fnmatch("/".join(parts[index:]), tail):
                    return rule.note or f"the pattern {rule.pattern}"
        return None
    return None


def scan_text(path: Path, relative: str) -> List[Finding]:
    findings: List[Finding] = []
    try:
        size = path.stat().st_size
    except OSError:
        return findings
    if size > MAX_TEXT_BYTES:
        return findings
    try:
        data = path.read_bytes()
    except OSError:
        return findings
    if b"\x00" in data[:8192]:
        return findings  # a binary file, the path rules are what apply
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return findings
    text = PUBLIC_REPOSITORY_PATTERN.sub(lambda m: " " * len(m.group(0)), text)
    for identifier in IDENTIFIERS:
        match = IDENTIFIER_PATTERN[identifier].search(text)
        if not match:
            continue
        line = text.count("\n", 0, match.start()) + 1
        excerpt = text[match.start() : match.start() + 60].split("\n")[0].strip()
        findings.append(
            Finding(
                category="private",
                rule=f"the identifier {identifier}",
                path=relative,
                detail=f"line {line}: {excerpt!r}",
            )
        )
    return findings


def scan(root: Path) -> Report:
    report = Report(root=root)
    for directory, subdirectories, filenames in os.walk(root):
        here = Path(directory)
        relative_dir = here.relative_to(root)
        if is_skipped(relative_dir):
            subdirectories[:] = []
            continue
        report.directories += 1
        subdirectories[:] = sorted(
            name for name in subdirectories if not is_skipped(relative_dir / name)
        )
        for filename in sorted(filenames):
            path = here / filename
            relative = str((relative_dir / filename).as_posix()).lstrip("./")
            if is_skipped(Path(relative)) or filename in SELF_NAMES:
                report.skipped += 1
                continue
            report.files += 1
            for rule in RULES:
                note = match_path(relative, filename, rule)
                if note:
                    report.findings.append(
                        Finding(category=rule.category, rule=rule.pattern, path=relative, detail=note)
                    )
            report.findings.extend(scan_text(path, relative))
    return report


def print_report(report: Report, limit: int = 0) -> None:
    grouped = report.by_category()
    print(f"tree: {report.root}")
    print(f"files: {report.files}, directories: {report.directories}")
    if not report.findings:
        print("findings: none")
        return
    for category in sorted(grouped):
        findings = grouped[category]
        print(f"\n{category}: {len(findings)}")
        for finding in findings[:limit] if limit else findings:
            detail = f" ({finding.detail})" if finding.detail else ""
            print(f"  {finding.path}: {finding.rule}{detail}")
        if limit and len(findings) > limit:
            print(f"  ... and {len(findings) - limit} more")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_public_tree.py",
        description="Check that a tree is free of game data and of anything private.",
    )
    parser.add_argument("tree", nargs="?", default=".", help="the directory to check")
    parser.add_argument(
        "--report", action="store_true", help="print the report and always exit 0"
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument(
        "--limit", type=int, default=0, help="show at most this many findings per category"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="only print the summary and the exit status"
    )
    args = parser.parse_args(argv)

    root = Path(args.tree).expanduser()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    report = scan(root)

    if args.json:
        print(
            json.dumps(
                {
                    "tree": str(root),
                    "files": report.files,
                    "directories": report.directories,
                    "findings": [finding.as_dict() for finding in report.findings],
                },
                indent=2,
            )
        )
    elif not args.quiet:
        print_report(report, args.limit)
    else:
        print(f"{report.files} files, {len(report.findings)} findings")

    if args.report:
        return 0
    return 1 if report.findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
