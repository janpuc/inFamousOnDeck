#!/usr/bin/env bash
# Boot the retail lift with the same runtime knobs as the Windows launcher.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAME_LINK="Infamous (Europe) (Fr,De,Es,It,Nl)"
NATIVE_BIN="${NATIVE_BIN:-$ROOT/build-native-vk/inFamousRecomp}"
INFAMOUS_ELF="${INFAMOUS_ELF:-$ROOT/$GAME_LINK/PS3_GAME/USRDIR/EBOOT.elf}"
if [[ ! -f "$INFAMOUS_ELF" && -f "$HOME/Games/inFamous/$GAME_LINK/PS3_GAME/USRDIR/EBOOT.elf" ]]; then
    # The development rsync excludes and removes the checkout's game link.
    INFAMOUS_ELF="$HOME/Games/inFamous/$GAME_LINK/PS3_GAME/USRDIR/EBOOT.elf"
fi
NATIVE_TIMEOUT="${NATIVE_TIMEOUT:-180}"

# installer/infamous.toml is the launcher's source of truth. Print plain
# KEY=value records so values never need to be evaluated as shell code. With
# NATIVE_KEEP_ENV=1 a key this script was already given wins over the toml, so
# tools/bench-run.sh can pin its own measurement knobs on top of the same
# launch environment instead of editing it.
while IFS= read -r setting; do
    export "$setting"
done < <(python3 - "$ROOT/installer/infamous.toml" <<'PY'
import os
import sys
import tomllib

with open(sys.argv[1], "rb") as manifest:
    env = tomllib.load(manifest)["launch"]["env"]
keep = os.environ.get("NATIVE_KEEP_ENV") == "1"
for key, value in env.items():
    if key in {"WINEDLLOVERRIDES", "PAD_RAWINPUT", "RES_SCALE", "LOD_CLASS_SCALE"}:
        continue
    if keep and key in os.environ:
        continue
    print(f"{key}={value}")
PY
)

export LOD_CLASS_SCALE="${LOD_CLASS_SCALE:-1}"
# Retail rendering needs the guest programs and render surfaces. The
# register-file engine covers both the movie/UI immediate draws and city passes.
export PS3RECOMP_VK_GUEST_PROGRAMS="${PS3RECOMP_VK_GUEST_PROGRAMS:-1}"
export PS3RECOMP_RSX_ENGINE="${PS3RECOMP_RSX_ENGINE:-dispatch}"
# This stage is strictly offscreen. Never let main.cpp default to a window.
export PS3RECOMP_VK_WINDOW=0
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
# The title installs ~1.3 GB of game data into /dev_hdd0/game/<id> before it
# boots, and keeps its saves in /dev_hdd0/home/00000001/savedata. Without a
# HDD root that path resolves under the read-only disc tree, the install cannot
# write and the boot never finishes; gamedata/dev_hdd0 is the tree's HDD.
export PS3_HDD0_ROOT="${PS3_HDD0_ROOT:-$ROOT/infamous-recomp/gamedata/dev_hdd0}"
mkdir -p "$PS3_HDD0_ROOT"

cd "$ROOT/infamous-recomp"
exec timeout --signal=TERM --kill-after=10s "$NATIVE_TIMEOUT" \
    "$NATIVE_BIN" "$INFAMOUS_ELF"
