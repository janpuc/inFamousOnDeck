#!/usr/bin/env python3
"""Add the native inFamous build to Steam as a non-Steam game.

Steam keeps the non-Steam games of a user in a binary VDF file:

    ~/.local/share/Steam/userdata/<userid>/config/shortcuts.vdf

The file is read and written here without any third party module. The writer
keeps the order of the fields and every field it does not know about, so a
shortcut that Steam added stays exactly as it was.

Nothing is written unless --apply is given. The default is a dry run that
prints what would change. When --apply is used, the file that is replaced is
first copied to a timestamped backup next to it.

    steam_shortcut.py                       show what would be added
    steam_shortcut.py --apply               write it (backs up the old file)
    steam_shortcut.py --list                list the existing shortcuts
    steam_shortcut.py --remove "inFamous"   remove a shortcut by name
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from project import STEAM_SHORTCUT_NAME  # noqa: E402
except ImportError:  # the helper runs without its project file
    STEAM_SHORTCUT_NAME = "inFamousOnDeck"

TYPE_MAP = 0x00
TYPE_STRING = 0x01
TYPE_UINT32 = 0x02
TYPE_UINT64 = 0x07
TYPE_END = 0x08

HEADER = b"\x00shortcuts\x00"

# The field order Steam itself writes.
FIELD_ORDER = (
    "appid",
    "AppName",
    "Exe",
    "StartDir",
    "icon",
    "ShortcutPath",
    "LaunchOptions",
    "IsHidden",
    "AllowDesktopConfig",
    "AllowOverlay",
    "OpenVR",
    "Devkit",
    "DevkitGameID",
    "DevkitOverrideAppID",
    "LastPlayTime",
    "FlatpakAppID",
    "sortas",
    "tags",
)

DEFAULT_APP_NAME = STEAM_SHORTCUT_NAME


class VdfError(Exception):
    """The file is not a shortcuts.vdf, or it is damaged."""


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


def _read_cstring(data: bytes, pos: int) -> Tuple[str, int]:
    end = data.find(b"\x00", pos)
    if end < 0:
        raise VdfError("unterminated string")
    return data[pos:end].decode("utf-8", "replace"), end + 1


def _read_map(data: bytes, pos: int) -> Tuple[Dict[str, Any], int]:
    result: Dict[str, Any] = {}
    while pos < len(data):
        kind = data[pos]
        pos += 1
        if kind == TYPE_END:
            return result, pos
        key, pos = _read_cstring(data, pos)
        if kind == TYPE_MAP:
            value, pos = _read_map(data, pos)
        elif kind == TYPE_STRING:
            value, pos = _read_cstring(data, pos)
        elif kind == TYPE_UINT32:
            if pos + 4 > len(data):
                raise VdfError("truncated uint32")
            value = struct.unpack_from("<I", data, pos)[0]
            pos += 4
        elif kind == TYPE_UINT64:
            if pos + 8 > len(data):
                raise VdfError("truncated uint64")
            value = struct.unpack_from("<Q", data, pos)[0]
            pos += 8
        else:
            raise VdfError(f"unknown value type {kind:#x} at offset {pos - 1}")
        result[key] = value
    raise VdfError("unterminated map")


def parse_shortcuts(data: bytes) -> List[Dict[str, Any]]:
    """Read the entries of a shortcuts.vdf."""
    if not data.startswith(HEADER):
        raise VdfError("not a shortcuts.vdf (bad header)")
    root, _pos = _read_map(data, len(HEADER))
    entries: List[Dict[str, Any]] = []
    for key in sorted(root, key=lambda item: (0, int(item)) if item.isdigit() else (1, 0)):
        value = root[key]
        if isinstance(value, dict):
            entries.append(value)
    return entries


def read_shortcuts(path: Path) -> List[Dict[str, Any]]:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    except OSError as error:
        raise VdfError(f"could not read {path}: {error}") from error
    if not data:
        return []
    return parse_shortcuts(data)


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------


def _write_cstring(value: str) -> bytes:
    return value.encode("utf-8") + b"\x00"


def _write_value(key: str, value: Any) -> bytes:
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        return bytes([TYPE_UINT32]) + _write_cstring(key) + struct.pack("<I", value & 0xFFFFFFFF)
    if isinstance(value, dict):
        out = bytes([TYPE_MAP]) + _write_cstring(key)
        for inner_key, inner_value in value.items():
            out += _write_value(inner_key, inner_value)
        return out + bytes([TYPE_END])
    return bytes([TYPE_STRING]) + _write_cstring(key) + _write_cstring(str(value))


def write_shortcuts(entries: List[Dict[str, Any]]) -> bytes:
    """Write the entries back into the binary form Steam reads."""
    out = bytearray(HEADER)
    for index, entry in enumerate(entries):
        out += bytes([TYPE_MAP]) + _write_cstring(str(index))
        written = set()
        for key in FIELD_ORDER:
            if key in entry:
                out += _write_value(key, entry[key])
                written.add(key)
        for key, value in entry.items():
            if key not in written:
                out += _write_value(key, value)
        out += bytes([TYPE_END])
    out += bytes([TYPE_END])  # end of the shortcuts map
    out += bytes([TYPE_END])  # end of the file
    return bytes(out)


def write_shortcuts_file(path: Path, entries: List[Dict[str, Any]], backup: bool = True) -> Optional[Path]:
    """Replace the file. The old file is copied to a timestamped backup first."""
    backup_path = None
    if backup and path.is_file():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = path.with_name(f"{path.name}.bak-{stamp}")
        index = 1
        while backup_path.exists():
            backup_path = path.with_name(f"{path.name}.bak-{stamp}-{index}")
            index += 1
        backup_path.write_bytes(path.read_bytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(write_shortcuts(entries))
    return backup_path


# --------------------------------------------------------------------------------------
# Shortcuts
# --------------------------------------------------------------------------------------


def compute_appid(exe: str, app_name: str) -> int:
    """The application id Steam gives a non-Steam game.

    Steam derives it from the shortcut, every known tool uses the CRC32 of the
    executable and the name with the high bit set. It only has to be stable
    and unique inside the user's shortcut list.
    """
    payload = exe.encode("utf-8") + app_name.encode("utf-8")
    return (zlib.crc32(payload) & 0x7FFFFFFF) | 0x80000000


def make_entry(
    app_name: str,
    exe: str,
    start_dir: str,
    icon: str = "",
    launch_options: str = "",
    shortcut_path: str = "",
    tags: Optional[List[str]] = None,
    appid: Optional[int] = None,
) -> Dict[str, Any]:
    """A shortcut entry with the fields Steam expects."""
    quoted = f'"{exe}"' if exe else ""
    return {
        "appid": appid if appid is not None else compute_appid(exe, app_name),
        "AppName": app_name,
        "Exe": quoted,
        "StartDir": start_dir,
        "icon": icon,
        "ShortcutPath": shortcut_path,
        "LaunchOptions": launch_options,
        "IsHidden": 0,
        "AllowDesktopConfig": 1,
        "AllowOverlay": 1,
        "OpenVR": 0,
        "Devkit": 0,
        "DevkitGameID": "",
        "DevkitOverrideAppID": 0,
        "LastPlayTime": 0,
        "FlatpakAppID": "",
        "sortas": "",
        "tags": {str(index): tag for index, tag in enumerate(tags or [])},
    }


def steam_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    from_env = os.environ.get("STEAM_ROOT")
    if from_env:
        return Path(from_env).expanduser()
    return Path.home() / ".local" / "share" / "Steam"


def find_users(root: Path) -> List[str]:
    userdata = root / "userdata"
    if not userdata.is_dir():
        return []
    return sorted(
        entry.name
        for entry in userdata.iterdir()
        if entry.is_dir() and entry.name.isdigit()
    )


def shortcuts_path(root: Path, user: Optional[str] = None) -> Optional[Path]:
    users = [user] if user else find_users(root)
    for candidate in users:
        path = root / "userdata" / candidate / "config" / "shortcuts.vdf"
        if user or path.exists():
            return path
    return None


def describe(entry: Dict[str, Any]) -> str:
    appid = entry.get("appid", 0)
    name = entry.get("AppName", "")
    exe = entry.get("Exe", "")
    return f"{appid:>11}  {name}  ->  {exe}"


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steam_shortcut.py",
        description="Add the native inFamous build to Steam as a non-Steam game.",
    )
    parser.add_argument("--steam-root", metavar="PATH", help="the Steam directory")
    parser.add_argument("--user", metavar="ID", help="the Steam user id")
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME, help="the name in the Steam library")
    parser.add_argument("--exe", metavar="PATH", help="the program Steam starts")
    parser.add_argument("--start-dir", metavar="PATH", help="the working directory")
    parser.add_argument("--icon", default="", help="an icon file")
    parser.add_argument("--launch-options", default="", help="arguments Steam appends")
    parser.add_argument("--tag", action="append", default=[], help="a library tag (repeatable)")
    parser.add_argument("--apply", action="store_true", help="write the file (the default is a dry run)")
    parser.add_argument("--list", action="store_true", help="list the shortcuts and exit")
    parser.add_argument("--remove", metavar="NAME", help="remove a shortcut by name")
    parser.add_argument(
        "--no-backup", action="store_true", help="do not keep a copy of the file that is replaced"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = steam_root(args.steam_root)
    path = shortcuts_path(root, args.user)
    if path is None:
        print(f"error: no Steam user found under {root / 'userdata'}", file=sys.stderr)
        return 2

    try:
        entries = read_shortcuts(path)
    except VdfError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.list:
        print(f"{path}")
        for entry in entries:
            print(describe(entry))
        return 0

    if args.remove:
        remaining = [entry for entry in entries if entry.get("AppName") != args.remove]
        removed = len(entries) - len(remaining)
        if not removed:
            print(f"error: no shortcut named {args.remove!r}", file=sys.stderr)
            return 1
        if not args.apply:
            print(f"dry run: would remove {removed} shortcut(s) from {path}")
            for entry in entries:
                if entry.get("AppName") == args.remove:
                    print(f"  {describe(entry)}")
            return 0
        backup = write_shortcuts_file(path, remaining, backup=not args.no_backup)
        print(f"removed {removed} shortcut(s) from {path}")
        if backup:
            print(f"backup: {backup}")
        return 0

    if not args.exe:
        print("error: --exe is required", file=sys.stderr)
        return 2
    exe = str(Path(args.exe).expanduser())
    start_dir = args.start_dir or str(Path(exe).parent)
    entry = make_entry(
        args.app_name,
        exe,
        start_dir,
        icon=args.icon,
        launch_options=args.launch_options,
        tags=args.tag,
    )

    existing = None
    for candidate in entries:
        if candidate.get("AppName") == entry["AppName"]:
            existing = candidate
            break

    if existing == entry:
        print(f"the shortcut {entry['AppName']!r} is already there")
        return 0

    new_entries = [item for item in entries if item.get("AppName") != entry["AppName"]]
    new_entries.append(entry)

    if not args.apply:
        print(f"dry run: would write {len(new_entries)} shortcut(s) to {path}")
        print(f"  {describe(entry)}")
        if existing is not None:
            print(f"  replaces {describe(existing)}")
        return 0

    backup = write_shortcuts_file(path, new_entries, backup=not args.no_backup)
    print(f"wrote {path}")
    print(f"  {describe(entry)}")
    if backup:
        print(f"backup: {backup}")
    print("restart Steam (or stop and start it again) to pick the shortcut up")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
