#!/usr/bin/env bash
# Build the native Linux (Vulkan) port of inFamous inside the build container.
#
#   build.sh --source DIR --game-elf PATH [--prefix DIR] [--jobs N] [--build-dir DIR]
#
#   --source DIR     the checkout of the port (the folder with installer/, ps3recomp/ and
#                    infamous-recomp/)
#   --game-elf PATH  the decrypted EBOOT.elf of the player's own BCES00220 dump
#   --prefix DIR     where the runtime is installed (the binary, the library and the
#                    launch environment)
#   --jobs N         parallel compile jobs
#   --build-dir DIR  the CMake build folder (default: <source>/infamous-recomp/build-install)
#
# The script runs the pipeline of installer/infamous.toml, the same steps the
# Windows installer runs, applies the patch set and compiles with the flags from
# the manifest.
set -euo pipefail

SOURCE=""
GAME_ELF=""
PREFIX=""
JOBS=""
BUILD_DIR=""
WITH_ASSETS=0
VENV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --game-elf)
            GAME_ELF="$2"
            shift 2
            ;;
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --with-assets)
            WITH_ASSETS=1
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

if [[ -z "$SOURCE" || -z "$GAME_ELF" ]]; then
    printf 'error: --source and --game-elf are required\n' >&2
    exit 2
fi
if [[ ! -f "$SOURCE/installer/infamous.toml" ]]; then
    printf 'error: %s is not a checkout of the port\n' "$SOURCE" >&2
    exit 1
fi
if [[ ! -f "$GAME_ELF" ]]; then
    printf 'error: %s is missing\n' "$GAME_ELF" >&2
    exit 1
fi
if [[ -z "$JOBS" ]]; then
    JOBS="$(nproc 2>/dev/null || echo 4)"
fi
if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR="$SOURCE/infamous-recomp/build-install"
fi
if [[ -z "$PREFIX" ]]; then
    PREFIX="$HOME/.local/share/inFamousOnDeck"
fi
if [[ -z "$VENV" ]]; then
    # The environment lives in the prefix, which is a mounted folder, so it
    # survives the next build.
    VENV="${INFAMOUS_VENV:-$PREFIX/venv}"
fi

export CCACHE_DIR="${CCACHE_DIR:-/ccache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CCACHE_DIR/pip}"
mkdir -p "$CCACHE_DIR" 2>/dev/null || true

# The Python that runs the lift tools.
if [[ ! -x "$VENV/bin/python" ]]; then
    printf '== the Python environment\n'
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet -r "$SOURCE/ps3recomp/requirements.txt"
fi
PYTHON="$VENV/bin/python"

# The flags of the manifest, with the placeholders filled in.
manifest_flags() {
    "$PYTHON" - "$SOURCE" "$1" "$JOBS" "$PYTHON" <<'PY'
import os
import sys
import tomllib

root, key, jobs, python = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(os.path.join(root, "installer", "infamous.toml"), "rb") as handle:
    manifest = tomllib.load(handle)
values = {
    "{cmake}": "cmake",
    "{ninja}": "ninja",
    "{cc}": os.environ.get("CC", "clang"),
    "{cxx}": os.environ.get("CXX", "clang++"),
    "{toolkit}": os.path.join(root, "ps3recomp"),
    "{port}": os.path.join(root, "infamous-recomp"),
    "{installer}": os.path.join(root, "installer"),
    "{python}": python,
    "{build_jobs}": jobs,
}
for item in manifest["build"][key]:
    for name, value in values.items():
        item = item.replace(name, value)
    print(item)
PY
}

printf '== the pipeline of installer/infamous.toml\n'
PIPELINE_ARGS=(--root "$SOURCE" --game-elf "$GAME_ELF" --jobs "$JOBS" --python "$PYTHON")
if [[ "$WITH_ASSETS" == "1" ]]; then
    PIPELINE_ARGS+=(--with-assets)
fi
"$PYTHON" "$SOURCE/linux/steamdeck/pipeline.py" "${PIPELINE_ARGS[@]}"

printf '== the build\n'
mapfile -t CONFIGURE < <(manifest_flags configure_unix)
printf '  %s\n' "${CONFIGURE[*]}"
"${CONFIGURE[@]}"

mapfile -t COMPILE < <(manifest_flags compile_unix)
printf '  %s\n' "${COMPILE[*]}"
"${COMPILE[@]}"

# The runtime: the binary and the SDL3 it uses.
mkdir -p "$PREFIX/bin" "$PREFIX/lib"
cp -f "$BUILD_DIR/inFamousRecomp" "$PREFIX/bin/inFamousRecomp"
chmod +x "$PREFIX/bin/inFamousRecomp"
SDL3=""
for candidate in \
    /usr/lib/x86_64-linux-gnu/libSDL3.so.0 \
    /usr/lib64/libSDL3.so.0 \
    /usr/lib/x86_64-linux-gnu/libSDL3.so; do
    if [[ -e "$candidate" ]]; then
        SDL3="$candidate"
        break
    fi
done
if [[ -z "$SDL3" ]]; then
    printf 'error: libSDL3.so.0 was not found in the container\n' >&2
    exit 1
fi
cp -L "$SDL3" "$PREFIX/lib/libSDL3.so.0"
strip --strip-unneeded "$PREFIX/bin/inFamousRecomp" 2>/dev/null || true

printf '== the runtime\n'
printf '  %s\n' "$PREFIX/bin/inFamousRecomp"
ls -l "$PREFIX/bin/inFamousRecomp" "$PREFIX/lib/libSDL3.so.0"
sha256sum "$PREFIX/bin/inFamousRecomp" "$PREFIX/lib/libSDL3.so.0"
