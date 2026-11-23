#!/usr/bin/env bash
# One click installer for the native Linux (Vulkan) build of inFamous on SteamOS.
#
#   install.sh --game ~/dumps/BCES00220
#
# The installer never needs root. It builds the port in a user space container,
# installs the runtime, the launcher and a desktop entry, and offers to add the
# game to the Steam library.
#
#   --game PATH        the player's own dump of the European release BCES00220: a
#                       folder with PS3_GAME/USRDIR/EBOOT.elf, or an ISO
#   --prefix DIR       where the runtime is installed
#                       (default: ~/.local/share/inFamousOnDeck)
#   --source DIR       the checkout of the port (default: this repository)
#   --jobs N           parallel compile jobs
#   --image NAME       the container image (default: localhost/infamous-steamrt4:latest)
#   --skip-build       install what is already there, do not build
#   --with-assets      also build the cosmetic assets (needs numpy and Pillow)
#   --dry-run          print what would happen and stop
#   --no-shortcut      do not touch the Steam library
#   --apply-shortcut   write the Steam shortcut (the default is a dry run)
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAME=""
PREFIX="$HOME/.local/share/inFamousOnDeck"
SOURCE=""
JOBS=""
IMAGE="localhost/infamous-steamrt4:latest"
SKIP_BUILD=0
WITH_ASSETS=0
DRY_RUN=0
NO_SHORTCUT=0
APPLY_SHORTCUT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --game)
            GAME="$2"
            shift 2
            ;;
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --with-assets)
            WITH_ASSETS=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --no-shortcut)
            NO_SHORTCUT=1
            shift
            ;;
        --apply-shortcut)
            APPLY_SHORTCUT=1
            shift
            ;;
        -h|--help)
            sed -n '2,26p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$GAME" ]]; then
    printf 'error: --game is required (the folder or the ISO of your own BCES00220 dump)\n' >&2
    exit 2
fi
if [[ -z "$SOURCE" ]]; then
    if [[ -f "$SELF/../../installer/infamous.toml" ]]; then
        SOURCE="$(cd "$SELF/../.." && pwd)"
    else
        printf 'error: --source is required (the checkout of the port)\n' >&2
        exit 2
    fi
fi
if [[ ! -f "$SOURCE/installer/infamous.toml" ]]; then
    printf 'error: %s is not a checkout of the port\n' "$SOURCE" >&2
    exit 1
fi
if [[ -z "$JOBS" ]]; then
    JOBS="$(nproc 2>/dev/null || echo 4)"
fi

command -v python3 >/dev/null 2>&1 || { printf 'error: python3 is required\n' >&2; exit 1; }

# The name the port gives the game folder, and the hash of the executable the
# port was verified with. The name has spaces in it, so it is read line by line.
mapfile -t MANIFEST_VALUES < <(python3 - "$SOURCE/installer/infamous.toml" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as handle:
    manifest = tomllib.load(handle)
game = manifest["game"]
print(game.get("link_name", "inFamous"))
print(game.get("eboot_sha256", [""])[0])
PY
)
LINK_NAME="${MANIFEST_VALUES[0]:-inFamous}"
EBOOT_SHA="${MANIFEST_VALUES[1]:-}"

printf '== the native Linux installer for inFamous\n'
printf '   source:  %s\n' "$SOURCE"
printf '   prefix:  %s\n' "$PREFIX"
printf '   game:    %s\n' "$GAME"
printf '   image:   %s\n' "$IMAGE"

# --------------------------------------------------------------------------------------
# The game
# --------------------------------------------------------------------------------------

WORK="${WORK:-$(mktemp -d -t infamous-game-XXXXXX)}"
cleanup() {
    if [[ "${KEEP_WORK:-0}" != "1" && -d "$WORK" && "$WORK" != / ]]; then
        rm -rf "$WORK"
    fi
}
trap cleanup EXIT

find_elf() {
    # The decrypted executable of the dump.
    for candidate in \
        "$1/PS3_GAME/USRDIR/EBOOT.elf" \
        "$1"/*/PS3_GAME/USRDIR/EBOOT.elf; do
        if [[ -f "$candidate" ]]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

GAME_DIR="$GAME"
if [[ -f "$GAME" ]]; then
    printf '== the ISO\n'
    GAME_DIR="$WORK/iso"
    mkdir -p "$GAME_DIR"
    if command -v bsdtar >/dev/null 2>&1; then
        bsdtar -xf "$GAME" -C "$GAME_DIR"
    elif command -v 7z >/dev/null 2>&1; then
        7z x -y -o"$GAME_DIR" "$GAME" >/dev/null
    elif command -v xorriso >/dev/null 2>&1; then
        xorriso -osirrox on -indev "$GAME" -extract / "$GAME_DIR"
    else
        printf 'error: no tool to read the ISO (bsdtar, 7z or xorriso)\n' >&2
        printf '       extract the ISO by hand and point --game at the folder\n' >&2
        exit 1
    fi
    printf '   extracted to %s\n' "$GAME_DIR"
fi

if [[ ! -d "$GAME_DIR" ]]; then
    printf 'error: %s is not a folder\n' "$GAME_DIR" >&2
    exit 1
fi

ELF="$(find_elf "$GAME_DIR" || true)"
if [[ -z "$ELF" ]]; then
    printf 'error: no EBOOT.elf in %s\n' "$GAME_DIR" >&2
    printf '       the dump needs the decrypted executable. Decrypt EBOOT.BIN with\n' >&2
    printf '       RPCS3 (File > Export > decrypted EBOOT.BIN) and put EBOOT.elf next\n' >&2
    printf '       to it, then run this installer again.\n' >&2
    exit 1
fi
printf '== the game\n'
printf '   executable: %s\n' "$ELF"

if [[ -n "$EBOOT_SHA" ]]; then
    ACTUAL_SHA="$(sha256sum "$ELF" | cut -d' ' -f1)"
    if [[ "$ACTUAL_SHA" == "$EBOOT_SHA" ]]; then
        printf '   the executable matches the one the port was verified with\n'
    else
        printf '   note: the executable is not the one the port was verified with\n'
        printf '         (expected %s)\n' "$EBOOT_SHA"
    fi
fi

# --------------------------------------------------------------------------------------
# The build
# --------------------------------------------------------------------------------------

GAME_ROOT="$PREFIX/game/$LINK_NAME"

if [[ "$SKIP_BUILD" == "0" ]]; then
    if [[ "$DRY_RUN" != "1" ]] && ! command -v podman >/dev/null 2>&1; then
        printf 'error: podman is required to build the port\n' >&2
        printf '       SteamOS has it built in (podman --version)\n' >&2
        exit 1
    fi
    printf '== the build container\n'
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '   would use %s\n' "$IMAGE"
    elif ! podman image exists "$IMAGE"; then
        podman build --tag "$IMAGE" --file "$SOURCE/linux/steamdeck/Containerfile" "$SOURCE/linux/steamdeck"
    else
        printf '   %s is there\n' "$IMAGE"
    fi

    printf '== the build\n'
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '   would build the port from %s in %s\n' "$ELF" "$IMAGE"
    else
        BUILD_ARGS=()
        if [[ "$WITH_ASSETS" == "1" ]]; then
            BUILD_ARGS+=(--with-assets)
        fi
        mkdir -p "$PREFIX" "$HOME/.cache/ccache"
        podman run --rm \
            --volume "$SOURCE:/work:rw" \
            --volume "$PREFIX:/prefix:rw" \
            --volume "$GAME_DIR:/game:ro" \
            --volume "$HOME/.cache/ccache:/ccache:rw" \
            --env "CCACHE_DIR=/ccache" \
            "$IMAGE" \
            bash /work/linux/steamdeck/build.sh \
                --source /work \
                --game-elf "/game${ELF#"$GAME_DIR"}" \
                --prefix /prefix \
                --jobs "$JOBS" \
                "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}"
    fi
fi

# --------------------------------------------------------------------------------------
# The runtime
# --------------------------------------------------------------------------------------

printf '== the runtime\n'
if [[ "$DRY_RUN" == "1" ]]; then
    printf '   would install into %s\n' "$PREFIX"
else
    mkdir -p "$PREFIX/bin" "$PREFIX/lib" "$PREFIX/launcher" "$GAME_ROOT/PS3_GAME/USRDIR"
    cp -f "$SOURCE/linux/steamdeck/launcher.py" "$PREFIX/launcher/launcher.py"
    cp -f "$SOURCE/linux/steamdeck/controller.py" "$PREFIX/launcher/controller.py"
    cp -f "$SOURCE/linux/steamdeck/project.py" "$PREFIX/launcher/project.py"
    cp -f "$SOURCE/linux/steamdeck/play.sh" "$PREFIX/launcher/play.sh"
    cp -f "$SOURCE/linux/steamdeck/steam_shortcut.py" "$PREFIX/launcher/steam_shortcut.py"
    chmod +x "$PREFIX/launcher/"*.py "$PREFIX/launcher/play.sh"

    # The game, if it is not there yet.
    if [[ ! -f "$GAME_ROOT/PS3_GAME/USRDIR/EBOOT.elf" ]]; then
        mkdir -p "$WORK/link"
        cp -a "$GAME_DIR/." "$WORK/link/"
        cp -a "$WORK/link/." "$GAME_ROOT/"
    fi

    # The launch environment of the manifest.
    python3 - "$SOURCE/installer/infamous.toml" "$PREFIX/launch.env" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as handle:
    manifest = tomllib.load(handle)
launch = manifest.get("launch_unix") or manifest["launch"]
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    handle.write("# The launch environment of the native Linux port.\n")
    handle.write("# Generated from installer/infamous.toml, do not edit by hand.\n")
    for key in sorted(launch["env"]):
        handle.write(f"{key}={launch['env'][key]}\n")
PY

    python3 - "$PREFIX/inFamousOnDeck.json" "$PREFIX" "$GAME_ROOT" <<'PY'
import json
import sys

target, prefix, game_root = sys.argv[1], sys.argv[2], sys.argv[3]
info = {
    "prefix": prefix,
    "binary": f"{prefix}/bin/inFamousRecomp",
    "game_root": game_root,
    "game_elf": f"{game_root}/PS3_GAME/USRDIR/EBOOT.elf",
    "launcher": f"{prefix}/launcher/launcher.py",
}
with open(target, "w", encoding="utf-8") as handle:
    json.dump(info, handle, indent=2)
    handle.write("\n")
PY
    printf '   installed into %s\n' "$PREFIX"
fi

# The desktop entry.
DESKTOP_FILE="$HOME/.local/share/applications/inFamous.desktop"
DESKTOP_NAME="inFamousOnDeck"
if [[ -f "$SOURCE/linux/steamdeck/project.py" ]]; then
    DESKTOP_NAME="$(python3 - "$SOURCE/linux/steamdeck/project.py" <<'PY'
import sys

values = {}
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    for line in handle:
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"')
print(values.get("DESKTOP_ENTRY_NAME", "inFamousOnDeck"))
PY
)"
fi
printf '== the desktop entry\n'
if [[ "$DRY_RUN" == "1" ]]; then
    printf '   would write %s\n' "$DESKTOP_FILE"
else
    mkdir -p "$(dirname "$DESKTOP_FILE")"
    cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=$DESKTOP_NAME
Comment=inFamous, native Linux (Vulkan) port for the Steam Deck
Exec=$PREFIX/launcher/launcher.py
Path=$PREFIX
Terminal=false
Categories=Game;
StartupNotify=false
EOF
    chmod +x "$DESKTOP_FILE" 2>/dev/null || true
    printf '   wrote %s\n' "$DESKTOP_FILE"
fi

# --------------------------------------------------------------------------------------
# The Steam library
# --------------------------------------------------------------------------------------

if [[ "$NO_SHORTCUT" == "1" ]]; then
    printf '== the Steam library\n   skipped\n'
else
    printf '== the Steam library\n'
    SHORTCUT_NAME="inFamousOnDeck"
    if [[ -f "$SOURCE/linux/steamdeck/project.py" ]]; then
        SHORTCUT_NAME="$(python3 - "$SOURCE/linux/steamdeck/project.py" <<'PY'
import sys

values = {}
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    for line in handle:
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"')
print(values.get("STEAM_SHORTCUT_NAME", "inFamousOnDeck"))
PY
)"
    fi
    SHORTCUT_ARGS=(--exe "$PREFIX/launcher/play.sh" --start-dir "$PREFIX" --app-name "$SHORTCUT_NAME")
    if [[ "$APPLY_SHORTCUT" == "1" ]]; then
        SHORTCUT_ARGS+=(--apply)
    fi
    if ! python3 "$SOURCE/linux/steamdeck/steam_shortcut.py" "${SHORTCUT_ARGS[@]}"; then
        printf '   note: the Steam shortcut was not written (is Steam installed for this user?)\n'
    fi
fi

printf '\n== done\n'
printf '   start it with %s/launcher/launcher.py\n' "$PREFIX"
printf '   or straight from the command line: %s/launcher/play.sh\n' "$PREFIX"
