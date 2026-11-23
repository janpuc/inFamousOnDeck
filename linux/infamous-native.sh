#!/usr/bin/env bash
# Steam non-Steam-game launcher for the native Linux Vulkan build.
# Use the full path to this script as Steam's target. No arguments are needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAME_LINK="Infamous (Europe) (Fr,De,Es,It,Nl)"
if [[ -z "${NATIVE_BIN:-}" ]]; then
    for cand in \
        "$ROOT/bin/inFamousRecomp" \
        "$ROOT/build-steamrt4-vk/inFamousRecomp" \
        "$ROOT/build-native-vk/inFamousRecomp"; do
        if [[ -x "$cand" ]]; then
            NATIVE_BIN="$cand"
            break
        fi
    done
    NATIVE_BIN="${NATIVE_BIN:-$ROOT/build-native-vk/inFamousRecomp}"
fi
INFAMOUS_ELF="${INFAMOUS_ELF:-$ROOT/$GAME_LINK/PS3_GAME/USRDIR/EBOOT.elf}"
if [[ ! -f "$INFAMOUS_ELF" && -f "$HOME/Games/inFamous/$GAME_LINK/PS3_GAME/USRDIR/EBOOT.elf" ]]; then
    INFAMOUS_ELF="$HOME/Games/inFamous/$GAME_LINK/PS3_GAME/USRDIR/EBOOT.elf"
fi
if [[ ! -x "$NATIVE_BIN" ]]; then
    printf 'Native executable not found: %s\n' "$NATIVE_BIN" >&2
    exit 1
fi
if [[ ! -f "$INFAMOUS_ELF" ]]; then
    printf 'Decrypted retail EBOOT not found: %s\n' "$INFAMOUS_ELF" >&2
    exit 1
fi

# installer/infamous.toml is the source of truth for game runtime settings.
# [launch_unix] is the Linux environment; older trees only have [launch].
while IFS= read -r setting; do
    export "${setting?}"
done < <(python3 - "$ROOT/installer/infamous.toml" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as manifest:
    data = tomllib.load(manifest)
launch = data.get("launch_unix") or data["launch"]
env = launch["env"]
for key, value in env.items():
    if key in {"WINEDLLOVERRIDES", "PAD_RAWINPUT", "RES_SCALE", "LOD_CLASS_SCALE"}:
        continue
    print(f"{key}={value}")
PY
)

export RES_SCALE="${RES_SCALE:-1}"
export LOD_CLASS_SCALE="${LOD_CLASS_SCALE:-1}"
export PS3RECOMP_VK_GUEST_PROGRAMS="${PS3RECOMP_VK_GUEST_PROGRAMS:-1}"
export PS3RECOMP_RSX_ENGINE="${PS3RECOMP_RSX_ENGINE:-dispatch}"
export PS3RECOMP_VK_WINDOW=1
export PS3RECOMP_FULLSCREEN="${PS3RECOMP_FULLSCREEN:-1}"
export PS3RECOMP_VK_VSYNC="${PS3RECOMP_VK_VSYNC:-1}"
# Keep saves and shader cache outside the build/install tree (XDG dirs) so
# updates never wipe progress and SteamOS read-only layouts stay clean.
XDG_DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
XDG_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}"
XDG_STATE="${XDG_STATE_HOME:-$HOME/.local/state}"
export PS3_HDD0_ROOT="${PS3_HDD0_ROOT:-$XDG_DATA/infamous-native/dev_hdd0}"
export PS3RECOMP_SPIRV_CACHE="${PS3RECOMP_SPIRV_CACHE:-$XDG_CACHE/infamous-native/spirv}"
mkdir -p "$PS3_HDD0_ROOT" "$PS3RECOMP_SPIRV_CACHE"

# Fall back to bundled libSDL3.so.0 beside the binary only when the host or
# Steam Linux Runtime does not provide one (never bundle Mesa / libvulkan).
BIN_DIR="$(cd "$(dirname "$NATIVE_BIN")" && pwd)"
if [[ -f "$BIN_DIR/lib/libSDL3.so.0" ]]; then
    if ! ldconfig -p 2>/dev/null | grep -q 'libSDL3\.so\.0'; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$BIN_DIR/lib"
    fi
fi

LOG="${INFAMOUS_NATIVE_LOG:-$XDG_STATE/infamous-native/infamous-native.log}"
mkdir -p "$(dirname "$LOG")"
printf 'Launching native inFamous. Log: %s\n' "$LOG"
cd "$ROOT/infamous-recomp"
exec "$NATIVE_BIN" "$INFAMOUS_ELF" >>"$LOG" 2>&1
