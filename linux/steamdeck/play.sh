#!/usr/bin/env bash
# Start the native Linux (Vulkan) build of inFamous.
#
# This is the script the Steam shortcut, the desktop entry and the launcher
# all point at. It resolves the install, applies the launch environment and
# hands over to the game.
#
#   play.sh [--root DIR] [--env-file FILE] [--dry-run]
#
#   --root DIR      the install directory (default: $INFAMOUS_NATIVE_ROOT or
#                   the directory that holds inFamousOnDeck.json)
#   --env-file FILE KEY=value lines with the settings of the launcher. Without
#                   it the settings are read from the launcher itself.
#   --dry-run       print what would run and exit
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=""
ENV_FILE=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            ROOT="$2"
            shift 2
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            sed -n '2,20p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

# The install root: the explicit one, then the environment, then the marker
# file, then the default location.
if [[ -z "$ROOT" ]]; then
    if [[ -n "${INFAMOUS_NATIVE_ROOT:-}" ]]; then
        ROOT="$INFAMOUS_NATIVE_ROOT"
    elif [[ -f "$SELF/../inFamousOnDeck.json" ]]; then
        ROOT="$(cd "$SELF/.." && pwd)"
    elif [[ -f "$SELF/inFamousOnDeck.json" ]]; then
        ROOT="$SELF"
    else
        ROOT="$HOME/.local/share/inFamousOnDeck"
    fi
fi
ROOT="$(cd "$ROOT" 2>/dev/null && pwd || printf '%s' "$ROOT")"

# The launch environment that the install recorded from installer/infamous.toml.
if [[ -f "$ROOT/launch.env" ]]; then
    while IFS= read -r setting; do
        [[ -z "$setting" || "$setting" == \#* ]] && continue
        export "${setting?}"
    done < "$ROOT/launch.env"
fi

# The settings of the launcher. They win over the recorded environment.
LAUNCHER="$SELF/launcher.py"
if [[ -z "$ENV_FILE" ]]; then
    if [[ -f "$LAUNCHER" && -n "$(command -v python3 || true)" ]]; then
        ENV_FILE="$(mktemp -t infamous-env-XXXXXX)"
        trap 'rm -f "$ENV_FILE"' EXIT
        python3 "$LAUNCHER" --print-env > "$ENV_FILE"
    else
        ENV_FILE="/dev/null"
    fi
fi
if [[ -f "$ENV_FILE" ]]; then
    while IFS= read -r setting; do
        [[ -z "$setting" || "$setting" == \#* ]] && continue
        export "${setting?}"
    done < "$ENV_FILE"
fi

# The binary and the game.
BIN="${INFAMOUS_NATIVE_BIN:-}"
if [[ -z "$BIN" ]]; then
    for candidate in \
        "$ROOT/bin/inFamousRecomp" \
        "$ROOT/build-steamrt4/inFamousRecomp" \
        "$ROOT/build-native-vk/inFamousRecomp" \
        "$ROOT/inFamousRecomp"; do
        if [[ -x "$candidate" ]]; then
            BIN="$candidate"
            break
        fi
    done
    for candidate in "$ROOT"/build-*/inFamousRecomp; do
        if [[ -x "$candidate" ]]; then
            BIN="$candidate"
            break
        fi
    done
fi

ELF="${INFAMOUS_GAME_ELF:-}"
if [[ -z "$ELF" ]]; then
    if [[ -f "$ROOT/inFamousOnDeck.json" ]] && command -v python3 >/dev/null 2>&1; then
        ELF="$(python3 - "$ROOT/inFamousOnDeck.json" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        info = json.load(handle)
except (OSError, ValueError):
    info = {}
print(info.get("game_elf", ""))
PY
)"
    fi
    if [[ -z "$ELF" ]]; then
        for candidate in \
            "$ROOT"/game/*/PS3_GAME/USRDIR/EBOOT.elf \
            "$ROOT"/*/PS3_GAME/USRDIR/EBOOT.elf \
            "$ROOT"/PS3_GAME/USRDIR/EBOOT.elf; do
            if [[ -f "$candidate" ]]; then
                ELF="$candidate"
                break
            fi
        done
    fi
fi

if [[ -z "$BIN" || ! -x "$BIN" ]]; then
    printf 'Native executable not found under %s\n' "$ROOT" >&2
    exit 1
fi
if [[ -z "$ELF" || ! -f "$ELF" ]]; then
    printf 'Decrypted retail EBOOT not found under %s\n' "$ROOT" >&2
    printf 'Copy your own BCES00220 dump into %s/game and run the installer again.\n' "$ROOT" >&2
    exit 1
fi

# The window, the frame pacing and the save locations.
export PS3RECOMP_VK_WINDOW=1
export PS3RECOMP_FULLSCREEN="${PS3RECOMP_FULLSCREEN:-1}"
export PS3RECOMP_VK_VSYNC="${PS3RECOMP_VK_VSYNC:-1}"
export PS3RECOMP_VK_GUEST_PROGRAMS="${PS3RECOMP_VK_GUEST_PROGRAMS:-1}"
export PS3RECOMP_RSX_ENGINE="${PS3RECOMP_RSX_ENGINE:-dispatch}"

XDG_DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
XDG_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}"
XDG_STATE="${XDG_STATE_HOME:-$HOME/.local/state}"
export PS3_HDD0_ROOT="${PS3_HDD0_ROOT:-$XDG_DATA/inFamousOnDeck/dev_hdd0}"
export PS3RECOMP_SPIRV_CACHE="${PS3RECOMP_SPIRV_CACHE:-$XDG_CACHE/inFamousOnDeck/spirv}"
export SAVEDATA_ROOT="${SAVEDATA_ROOT:-$XDG_DATA/inFamousOnDeck/savedata}"
mkdir -p "$PS3_HDD0_ROOT" "$PS3RECOMP_SPIRV_CACHE" "$SAVEDATA_ROOT"

# The bundled SDL3, but only when the system has none.
BIN_DIR="$(cd "$(dirname "$BIN")" && pwd)"
if [[ -f "$BIN_DIR/lib/libSDL3.so.0" ]]; then
    if ! ldconfig -p 2>/dev/null | grep -q 'libSDL3\.so\.0'; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$BIN_DIR/lib"
    fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'binary: %s\n' "$BIN"
    printf 'game:   %s\n' "$ELF"
    printf 'root:   %s\n' "$ROOT"
    exit 0
fi

LOG="${INFAMOUS_NATIVE_LOG:-$XDG_STATE/inFamousOnDeck/inFamousOnDeck.log}"
mkdir -p "$(dirname "$LOG")"
printf 'Launching native inFamous. Log: %s\n' "$LOG"

cd "$ROOT"
exec "$BIN" "$ELF" >>"$LOG" 2>&1
