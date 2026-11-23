#!/usr/bin/env python3
"""Run the pipeline of installer/infamous.toml on Linux.

The Windows installer runs this pipeline in Rust. This script runs the same
steps from the same manifest, so a Linux build generates the same code as a
Windows build, and it applies the same patch set with the same strict rules:
every context line of a hunk has to match the file byte for byte, and a
mismatch stops the build instead of drifting.

It runs inside the build container, see linux/steamdeck/build.sh.

    pipeline.py --root DIR --game-elf PATH [--jobs N] [--with-assets]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The steps that only produce cosmetic assets. The port runs without them.
ASSET_STEPS = {"ui-font", "kbm-icons", "kbm-icons-build", "kbm-prompts-xpp", "kbm-prompts"}

NO_NEWLINE = b"\\ No newline at end of file\n"
EOL_LF = b"# eol lf\n"
EOL_CRLF = b"# eol crlf\n"


class PipelineError(Exception):
    pass


# --------------------------------------------------------------------------------------
# The patch applier (the same rules as installer/src/diff.rs)
# --------------------------------------------------------------------------------------


def split_lines(data: bytes) -> List[bytes]:
    """Split into lines that keep their terminator, like the Rust splitter."""
    lines: List[bytes] = []
    start = 0
    for index, byte in enumerate(data):
        if byte == 0x0A:
            lines.append(data[start : index + 1])
            start = index + 1
    if start < len(data):
        lines.append(data[start:])
    return lines


def convert_eol(data: bytes, crlf: bool) -> bytes:
    out = bytearray()
    index = 0
    while index < len(data):
        byte = data[index]
        if byte == 0x0D and index + 1 < len(data) and data[index + 1] == 0x0A:
            index += 1
            continue
        if byte == 0x0A:
            if crlf:
                out.append(0x0D)
            out.append(0x0A)
        else:
            out.append(byte)
        index += 1
    return bytes(out)


def _parse_range(text: str) -> Optional[Tuple[int, int]]:
    if "," in text:
        first, second = text.split(",", 1)
    else:
        first, second = text, "1"
    try:
        return int(first), int(second)
    except ValueError:
        return None


def parse_patch(patch: bytes) -> List[Tuple[int, int, List[Tuple[bytes, bytes]]]]:
    """The hunks of a patch: (old_start, old_count, [(tag, body), ...])."""
    raw = split_lines(patch)
    hunks: List[Tuple[int, int, List[Tuple[bytes, bytes]]]] = []
    index = 0
    while index < len(raw):
        line = raw[index]
        index += 1
        if line.startswith(b"--- ") or line.startswith(b"+++ ") or line.startswith(b"# "):
            continue
        if not line.startswith(b"@@ "):
            continue
        parts = line.decode("utf-8", "replace").split()
        if len(parts) < 4 or not parts[1].startswith("-") or not parts[2].startswith("+"):
            raise PipelineError(f"bad hunk header: {line!r}")
        old_range = _parse_range(parts[1][1:])
        new_range = _parse_range(parts[2][1:])
        if old_range is None or new_range is None:
            raise PipelineError(f"bad hunk header: {line!r}")
        old_start, old_count = old_range
        _new_start, new_count = new_range
        body: List[Tuple[bytes, bytes]] = []
        seen_old = 0
        seen_new = 0
        while (seen_old < old_count or seen_new < new_count) and index < len(raw):
            entry = raw[index]
            index += 1
            if not entry:
                continue
            tag = entry[0:1]
            payload = entry[1:]
            if index < len(raw) and raw[index] == NO_NEWLINE:
                index += 1
                payload = payload[:-1] if payload.endswith(b"\n") else payload
            if tag == b" ":
                seen_old += 1
                seen_new += 1
            elif tag == b"-":
                seen_old += 1
            elif tag == b"+":
                seen_new += 1
            else:
                raise PipelineError(f"bad line in hunk at -{old_start}")
            body.append((tag, payload))
        if seen_old != old_count or seen_new != new_count:
            raise PipelineError(f"truncated hunk at -{old_start}")
        hunks.append((old_start, old_count, body))
    return hunks


def apply_patch(old: bytes, patch: bytes) -> bytes:
    """Apply a patch to a file. Strict: every context line has to match."""
    if patch.startswith(EOL_LF):
        old = convert_eol(old, crlf=False)
    elif patch.startswith(EOL_CRLF):
        old = convert_eol(old, crlf=True)
    source = split_lines(old)
    out = bytearray()
    cursor = 0
    for old_start, old_count, body in parse_patch(patch):
        at = old_start if old_count == 0 else old_start - 1
        if at < cursor or at > len(source):
            raise PipelineError(f"hunk at line {old_start} is out of order or past the end")
        for line in source[cursor:at]:
            out += line
        cursor = at
        for tag, payload in body:
            if tag in (b" ", b"-"):
                if cursor >= len(source) or source[cursor] != payload:
                    raise PipelineError(f"context mismatch at line {cursor + 1} (hunk at {old_start})")
                if tag == b" ":
                    out += payload
                cursor += 1
            else:
                out += payload
    for line in source[cursor:]:
        out += line
    return bytes(out)


# --------------------------------------------------------------------------------------
# The builtins of the installer
# --------------------------------------------------------------------------------------


def _u32be(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        return 0
    return int.from_bytes(data[offset : offset + 4], "big")


def extract_jobmods(elf: Path, out_dir: Path) -> List[str]:
    """The WWS job modules of the EBOOT, named the way the port names them."""
    data = elf.read_bytes()
    out_dir.mkdir(parents=True, exist_ok=True)
    magic = b"\xc0\xde\xc0\xde"
    names: List[str] = []
    index = 0x20
    while index + 4 <= len(data):
        if data[index : index + 4] == magic and (index - 0x20) % 4 == 0:
            start = index - 0x20
            entry = _u32be(data, start + 0x10)
            size = _u32be(data, start + 0x14)
            ila = all((_u32be(data, start + 4 * k) >> 25) == 0x21 for k in range(4))
            if (
                ila
                and 0x24 <= size <= 0x40000
                and start + size <= len(data)
                and entry < size
            ):
                module = data[start : start + size]
                name = f"jobmod_{hashlib.sha1(module).hexdigest()[:12]}_e{entry:04X}"
                if name not in names:
                    (out_dir / f"{name}.bin").write_bytes(module)
                    names.append(name)
                index = start + max(size, 0x24)
                continue
        index += 4
    if not names:
        raise PipelineError(f"no WWS job modules (C0DEC0DE) found in {elf}")
    return names


def carve(src: Path, offset: int, length: int, dst: Path) -> None:
    data = src.read_bytes()
    if offset + length > len(data):
        raise PipelineError(f"carve {offset:#x}+{length:#x} is past the end of {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data[offset : offset + length])


# --------------------------------------------------------------------------------------
# The steps
# --------------------------------------------------------------------------------------


class Pipeline:
    def __init__(self, root: Path, manifest: Dict[str, Any], variables: Dict[str, str]) -> None:
        self.root = root
        self.manifest = manifest
        self.variables = variables
        self.failed_optional: List[str] = []

    def expand(self, text: str) -> str:
        result = text
        for key, value in self.variables.items():
            result = result.replace("{" + key + "}", value)
        if "{" in result and "}" in result:
            missing = result[result.index("{") + 1 : result.index("}")]
            raise PipelineError(f"unknown placeholder {{{missing}}} in {text!r}")
        return result

    def expand_all(self, items: Sequence[str]) -> List[str]:
        return [self.expand(item) for item in items]

    def run_builtin(self, name: str, args: List[str]) -> None:
        if name == "extract-jobmods":
            modules = extract_jobmods(Path(args[0]), Path(args[1]))
            print(f"  {len(modules)} WWS job modules -> {args[1]}", flush=True)
        elif name == "carve":
            offset = int(args[1], 0)
            length = int(args[2], 0)
            carve(Path(args[0]), offset, length, Path(args[3]))
        elif name == "copy":
            shutil.copyfile(args[0], args[1])
        else:
            raise PipelineError(f"unknown builtin '{name}'")

    def run_step(self, step: Dict[str, Any]) -> None:
        title = step.get("title", step.get("id", "step"))
        print(f"== {title}", flush=True)
        for directory in step.get("mkdirs", []):
            Path(self.expand(directory)).mkdir(parents=True, exist_ok=True)
        if step.get("builtin"):
            self.run_builtin(step["builtin"], self.expand_all(step.get("args", [])))
            return
        command = self.expand_all(step.get("run", []))
        if not command:
            return
        cwd = self.expand(step["cwd"]) if step.get("cwd") else str(self.root)
        environment = dict(os.environ)
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        for key, value in step.get("env", {}).items():
            environment[key] = self.expand(value)
        targets = [""]
        if step.get("for_each"):
            pattern = self.expand(step["for_each"])
            targets = sorted(str(path) for path in Path(self.root).glob(pattern))
            if not targets:
                print(f"  nothing matched {pattern}", flush=True)
                return
        for target in targets:
            variables = dict(self.variables)
            if target:
                variables["file"] = target
            saved = self.variables
            self.variables = variables
            try:
                argv = self.expand_all(command)
            finally:
                self.variables = saved
            print(f"  $ {shlex.join(argv)}", flush=True)
            result = subprocess.run(argv, cwd=cwd, env=environment)
            if result.returncode != 0:
                message = f"step {step.get('id', '?')} failed with {result.returncode}"
                if step.get("optional"):
                    self.failed_optional.append(str(step.get("id")))
                    print(f"  optional step failed, continuing: {message}", flush=True)
                    return
                raise PipelineError(message)

    def run(self, with_assets: bool) -> None:
        for step in self.manifest.get("step", []):
            identifier = str(step.get("id", ""))
            if identifier in ASSET_STEPS and not with_assets:
                print(f"== {step.get('title', identifier)} (skipped, cosmetic)", flush=True)
                continue
            self.run_step(step)


# --------------------------------------------------------------------------------------
# The patch set
# --------------------------------------------------------------------------------------


def apply_patch_set(root: Path, installer_dir: Path) -> Tuple[int, List[str]]:
    """Apply installer/patches to the generated files. Returns (count, failures)."""
    patch_dir = installer_dir / "patches" / "infamous-recomp"
    if not patch_dir.is_dir():
        return 0, []
    applied = 0
    failures: List[str] = []
    for patch_path in sorted(patch_dir.rglob("*.patch")):
        relative = patch_path.relative_to(patch_dir)
        target = root / "infamous-recomp" / str(relative)[: -len(".patch")]
        if not target.is_file():
            failures.append(f"{relative}: the generated file is missing")
            continue
        try:
            patched = apply_patch(target.read_bytes(), patch_path.read_bytes())
        except PipelineError as error:
            failures.append(f"{relative}: {error}")
            continue
        target.write_bytes(patched)
        applied += 1
    return applied, failures


def read_lock(installer_dir: Path) -> Dict[str, str]:
    lock: Dict[str, str] = {}
    path = installer_dir / "generated.lock"
    if not path.is_file():
        return lock
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("file "):
            continue
        parts = line.split()
        if len(parts) >= 4:
            lock[parts[3]] = parts[2]
    return lock


def verify_lock(root: Path, installer_dir: Path) -> List[str]:
    """Check the generated files against installer/generated.lock."""
    problems: List[str] = []
    for relative, expected in read_lock(installer_dir).items():
        path = root / relative
        if not path.is_file():
            problems.append(f"{relative}: missing")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            problems.append(f"{relative}: {digest} != {expected}")
    return problems


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pipeline of installer/infamous.toml on Linux.")
    parser.add_argument("--root", required=True, help="the checkout root")
    parser.add_argument("--game-elf", required=True, help="the decrypted EBOOT.elf of the player's own dump")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--python", default=sys.executable, help="the Python that runs the lift tools")
    parser.add_argument("--with-assets", action="store_true", help="also build the cosmetic assets")
    parser.add_argument("--verify-lock", action="store_true", help="fail when the generated files differ from the lock")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    installer_dir = root / "installer"
    manifest_path = installer_dir / "infamous.toml"
    if not manifest_path.is_file():
        print(f"error: {manifest_path} is missing", file=sys.stderr)
        return 2
    with open(manifest_path, "rb") as handle:
        manifest = tomllib.load(handle)

    elf = Path(args.game_elf).resolve()
    if not elf.is_file():
        print(f"error: {elf} is missing", file=sys.stderr)
        return 2
    usrdir = elf.parent
    game = usrdir.parent.parent

    variables = {
        "root": str(root),
        "port": str(root / "infamous-recomp"),
        "toolkit": str(root / "ps3recomp"),
        "game": str(game),
        "usrdir": str(usrdir),
        "elf": str(elf),
        "python": args.python,
        "work": str(root / ".installer" / "work"),
        "jobs": str(args.jobs),
        "file": "",
        "cmake": "cmake",
        "ninja": "ninja",
        "cc": os.environ.get("CC", "clang"),
        "cxx": os.environ.get("CXX", "clang++"),
        "build_jobs": str(args.jobs),
    }

    pipeline = Pipeline(root, manifest, variables)
    try:
        pipeline.run(args.with_assets)
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    applied, failures = apply_patch_set(root, installer_dir)
    print(f"== patches: {applied} applied", flush=True)
    for failure in failures:
        print(f"  FAILED {failure}", file=sys.stderr)
    if failures:
        return 1

    problems = verify_lock(root, installer_dir)
    if problems:
        print("== the generated files differ from installer/generated.lock:", flush=True)
        for problem in problems:
            print(f"  {problem}", flush=True)
        if args.verify_lock:
            return 1
    else:
        print("== the generated files match installer/generated.lock", flush=True)

    if pipeline.failed_optional:
        print(f"== optional steps that failed: {', '.join(pipeline.failed_optional)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
