# Linux (native)

This is the native POSIX build of the toolkit and the port. It is the first
stage of Linux support and is deliberately headless by default:

- `ps3recomp` (the runtime library) builds and passes its tests with GCC and
  Clang.
- `ps3recomp_host` drives the `cellGcm` → RSX → backend bridge with no lifted
  game.
- The port's runner (`infamous-recomp/main.cpp` + `stubs.cpp` + the PPU
  scaffold) builds and boots the toolkit's synthetic smoke title end to end.
- With `PS3RECOMP_RSX_VULKAN=ON`, the opt-in Vulkan RSX backend renders the
  host harness's draw, texture, mip, render-to-texture, depth-texture and MRT
  scenes (with the guest's own shaders), and the port's smoke boot presents
  through Vulkan.
- Depth-bounds testing forwards the guest's interval on every draw. The
  `--depthbounds` host scene checks it on GPUs that expose Vulkan `depthBounds`;
  an unsupported device exits with status 77 and a skip message.

The Windows D3D12 backend is still the reference renderer. The Vulkan backend
is the Linux hardware path and is being brought up against the same host
harness; a retail game boot is the next milestone. Until it lands, the port's
Windows-only overlay and trainer page are unavailable. Smoke builds disable
the title-specific trainer with `PORT_BUILD_TRAINER=OFF`.

## Requirements

Debian/Ubuntu:

```sh
sudo apt install build-essential cmake ninja-build libsdl2-dev
# Clang instead of GCC:
sudo apt install clang
# Vulkan renderer (optional):
sudo apt install libvulkan-dev glslang-dev glslang-tools spirv-tools \
                 mesa-vulkan-drivers vulkan-validationlayers
# Vulkan window, gamepads and audio (optional but required for a window):
sudo apt install libsdl3-dev
```

SDL2 provides `cellPad` and `cellAudio` off Windows. A Vulkan build that finds
SDL3 uses SDL3 for the window, the controller and the audio output instead,
and then links no SDL2: the two libraries export the same symbol names with
different ABIs, so a process linking both calls into the wrong one. A Vulkan
build without SDL3 still compiles and runs, but cannot open a window. No sudo
is needed to build or run.

## Build and test the toolkit

```sh
cmake -S ps3recomp -B build-runtime -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPS3RECOMP_BUILD_TESTS=ON
cmake --build build-runtime --parallel
ctest --test-dir build-runtime --output-on-failure

./build-runtime/ps3recomp_host
./build-runtime/ps3recomp_boot_smoke build-runtime/smoke/smoke.elf
```

The host harness should print `OK` for the clear colour and presented pixel.
The smoke title should print `[smoke] PASS`, checking its 11 guest markers and
`sys_process_exit`. CTest also runs the draw-engine state tests without a GPU.

## Build the Vulkan renderer

```sh
cmake -S ps3recomp -B build-vk -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPS3RECOMP_RSX_VULKAN=ON
cmake --build build-vk --target ps3recomp_host ps3recomp_smoke_lift --parallel

./build-vk/ps3recomp_host --draw
PS3RECOMP_VK_GUEST_PROGRAMS=1 ./build-vk/ps3recomp_host --tex
PS3RECOMP_VK_GUEST_PROGRAMS=1 ./build-vk/ps3recomp_host --depthbounds
```

Without a GPU, Mesa's lavapipe runs the same host scenes:

```sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
PS3RECOMP_VK_GUEST_PROGRAMS=1 ./build-vk/ps3recomp_host --draw
```

With `PS3RECOMP_VK_WINDOW=1` the backend opens an SDL3 window and presents to
it; the harness otherwise renders offscreen and checks the centre pixel.
`PS3RECOMP_VK_DUMP=<path>` writes the presented frame as a binary PPM.

The port build takes the same switch:

```sh
cmake -S infamous-recomp -B build-port-vk -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPS3RECOMP_DIR="$PWD/ps3recomp" \
  -DRECOMP_DIR="$PWD/build-vk/smoke/lift" \
  -DPS3RECOMP_RSX_VULKAN=ON \
  -DPORT_ENABLE_SPU=OFF -DPORT_BUILD_TRAINER=OFF \
  -DPORT_PPU_REGCACHE=OFF -DPORT_PGO=OFF -DPORT_CPU_AUTO=OFF
cmake --build build-port-vk --parallel
PS3RECOMP_VK_WINDOW=0 ./build-port-vk/inFamousRecomp build-vk/smoke/smoke.elf
```

The port selects Vulkan automatically when it was compiled with
`PS3RECOMP_RSX_VULKAN=ON`; the launcher can set `PS3RECOMP_VK_WINDOW=0` for a
headless run.

## Build the port against the smoke lift

The port normally builds its lifted game code from your own dump. For a
compile/link check without game data, point it at the smoke lift written by the
runtime configure:

```sh
cmake -S infamous-recomp -B build-port -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPS3RECOMP_DIR="$PWD/ps3recomp" \
  -DRECOMP_DIR="$PWD/build-runtime/smoke/lift" \
  -DPORT_ENABLE_SPU=OFF \
  -DPORT_BUILD_TRAINER=OFF \
  -DPORT_PPU_REGCACHE=OFF \
  -DPORT_PGO=OFF \
  -DPORT_CPU_AUTO=OFF
cmake --build build-port --parallel
timeout 60 ./build-port/inFamousRecomp build-runtime/smoke/smoke.elf
```

## Building and booting the real game (stage 1)

The installer pipeline is cross-platform in principle (`configure_unix` /
`compile_unix` in `installer/infamous.toml`). The generated retail lift can now
also be built and booted directly on Linux. Retail-game testing requires the
decrypted European (`BCES00220`) game folder used for that lift, or an ISO plus
its matching disc key to generate it. An encrypted ISO without a key cannot be
used to generate the game sources. This is an intro-rendering milestone, not a
playability or performance claim.

Install glslang's development package as well as the requirements above so
that the guest HLSL to SPIR-V translator is available. On Fedora that is
`sudo dnf install glslang-devel`. Then configure against the existing lift:

```sh
cmake -S infamous-recomp -B build-native-vk -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DPS3RECOMP_DIR="$PWD/ps3recomp" \
  -DRECOMP_DIR="$PWD/infamous-recomp/recompiled" \
  -DPS3RECOMP_RSX_VULKAN=ON \
  -DPORT_ENABLE_SPU=ON -DPORT_PPU_REGCACHE=ON \
  -DPORT_PGO=OFF -DPORT_LTO=OFF -DPORT_CPU_AUTO=OFF
cmake --build build-native-vk --parallel 6
```

`tools/run-native.sh` applies the environment in `installer/infamous.toml`,
selects guest shader programs, forces headless presentation and starts the
retail EBOOT under `timeout`. It uses the checkout's game link, falling back to
`$HOME/Games/inFamous/<link_name>` on the BC-250. The register-file engine
(`PS3RECOMP_RSX_ENGINE=dispatch`) is the default: it decodes the immediate-mode
`SET_VERTEX_DATA*` vertices the title uses for its startup movie and overlays,
and owns the render surfaces. `PS3RECOMP_RSX_ENGINE=vtable` selects the older
draw-vtable path.

The script also sets `PS3_HDD0_ROOT` to `infamous-recomp/gamedata/dev_hdd0`.
The title copies ~1.3 GB of game data into `/dev_hdd0/game/BLES00000` on its
first boot and keeps saves under `/dev_hdd0/home/00000001/savedata`; with no
HDD root both land in the read-only disc tree, every install write fails and
the boot stops on its installation screen.

```sh
mkdir -p artifacts
NATIVE_TIMEOUT=120 \
PS3RECOMP_VK_DUMP="$PWD/artifacts/intro-%u.ppm" \
PS3RECOMP_VK_DUMP_EVERY=120 \
./tools/run-native.sh
```

A `%u` in the dump path is replaced by the presented-frame number. On the
BC-250, build in the container but run this command on the host so Vulkan uses
RADV rather than the container's llvmpipe. No window is opened.

## Windowed play and the Steam launch script

`linux/infamous-native.sh` is the non-Steam-game target. It applies the same
launch environment, opens the SDL3 window fullscreen with vsync, appends the
log to `artifacts/infamous-native.log`, and has no timeout. Point Steam at the
script's full path; nothing else is needed.

```sh
./linux/infamous-native.sh
```

The window is borderless fullscreen by default. `PS3RECOMP_FULLSCREEN=0` (or
`PS3RECOMP_VK_WINDOWED=1`) starts it windowed and resizable, F11 or Alt+Enter
toggles, and Esc closes. `PS3RECOMP_VK_VSYNC=0` asks for an immediate present
mode and falls back to FIFO with a log line when the driver has none. A
controller is read through SDL3 (Steam Input's virtual pad works); with no
controller, WASD/IJKL are the two sticks, the arrows are the d-pad, space is
Cross, enter is Start, and the rest is listed in `pad_sdl3_native.c`.

Two telemetry lines go to the log and to stderr. `[HEARTBEAT]` prints every
10 s with the flip, present, rendered-frame and draw counts plus the last
present age; `[FRAMEPCT]` prints the average fps, p50/p99 frametime and the
1%/0.1% low fps over each 300 rendered frames. `HEARTBEAT=0` and `FRAMEPCT=0`
turn them off.

The BC-250's TV is the console's only screen, so a window can be verified
without one reaching it: gamescope's headless backend still owns a real
compositor and a real Vulkan swapchain.

```sh
gamescope --backend headless -W 1280 -H 720 -r 60 -- \
  ./build-native-vk/inFamousRecomp "<game>/PS3_GAME/USRDIR/EBOOT.elf"
```

## What the retail boot renders today

Verified on the BC-250's RADV (GFX1013) through the register-file engine: the
Sony Computer Entertainment card, the Sucker Punch logo movie, the fade into
and out of the title's installation screen, and the window present itself.

Past the movie the boot answers the "Game data is automatically saved" dialog,
creates the guest's `Game State Thread` and starts loading an empire: it adds
SPURS workloads 1..10 (`Wws_Job`/`CTaskMgr`), opens the streaming grains and
`49_pleasant_empire_load_loop.mp3`, and its audio stops being silent. It then
shows the loading screen and, from about presented frame 7000, the title
screen: "Press the Start Button" over the inFAMOUS logo with its animated
smoke, drawing 10167 groups per frame (1.16 M vertices) and presenting at
5.9 fps on the BC-250. Nothing presses Start yet, so it stays there.

The frame the boot presents at that point is the title screen, not black. Two
frames in the sequence really are black (a transition after the loading bar,
frame ~7000 in one run) and one white with scan lines is the loading bar
itself; the capture set in `native-artifacts/m0-m1b/` has all of them. The
present path is sound: the presented image and the flipped colour target agree
to a mean absolute difference of 0.17/255 (`PS3RECOMP_VK_DUMP` against a
`PS3RECOMP_VK_DUMP_SURFACES` dump of the same surface one frame earlier).

Reaching that loading state needed one runtime fix: the first indexed draw
killed the FIFO thread with a NULL dereference, because `rsx_process_method`
called `s_backend->draw_indexed` outside its
`if (s_backend && s_backend->draw_indexed)` guard and `s_backend` is only
registered by a backend that also drives the vtable path
(`rsx_vulkan_backend_init` sets it only when the register-file engine failed
to start).

### Past the title screen: the menu and the city

`PAD_SCRIPT` drives the pad from wall-clock time (`160:0x0008` is Start held
for 40 polls, `230:0x4000` is Cross), which is enough to leave the title and
load the autosave that is already in `PS3_HDD0_ROOT`:

```sh
BENCH_TAG=city BENCH_DUMP_EVERY=50 BENCH_SECONDS=480 \
PAD_SCRIPT="160:0x0008,230:0x4000" ENG_VCACHE=1 tools/bench-run.sh
```

Start leads to the main menu, Cross takes the item under the cursor
("Continue" with an autosave present), and after about 150 s of wall clock at
these frame rates the opening mission runs: 13495 groups and 1.97 M vertices
per frame at 4.7 fps on the BC-250 at 1280x720 offscreen, with the tutorial
overlay, the controller hint and the mission text all correct. (4.7 fps is the
measurement before the world rendered; the same scene now costs 4.3 fps, see
`native-artifacts/m1-world/README.md` for both profiles and why.)

Both of those used to draw no 3D world at all: the menu presented a uniform
colour and the opening mission showed its HUD and additive fire embers over
black. `native-artifacts/m1cd/` has those frames and
`native-artifacts/m1-world/` the frames after the fix, with the numbers. Two
engine defects caused it, described with the mechanisms that replaced them
below.

The save itself is read-only material: the HDD root is the tree's own
`infamous-recomp/gamedata/dev_hdd0`, and the
`home/00000001/savedata/BLES00000_INFAMOUS_0` in it is a copy of the retail
folder's "(Autosave) 1. Genesis", byte for byte. Nothing writes to `~/Games`.

## Measuring a run

`tools/bench-run.sh` is the canonical measurement command: one game tree, one
HDD snapshot, one set of knobs, a bounded run, and a log whose header names
the commit and the binary hash.

```sh
cd native                         # on the BC-250 host, after bc250.sh sync/build
BENCH_TAG=baseline tools/bench-run.sh          # 180 s, stats on, no dumps
BENCH_TAG=ab BENCH_AB=1 tools/bench-run.sh     # in-run A/B, 20 s per phase
BENCH_DUMP_EVERY=600 tools/bench-run.sh        # frame dumps (not for timing)
```

It layers on `tools/run-native.sh` and `installer/infamous.toml`; with
`NATIVE_KEEP_ENV=1` (which the script sets) an exported knob wins over the
launcher's value, so a measurement never edits the play configuration. It
also sets `ulimit -c 0`: a core dump of this process is several GB, and `/tmp`
on the BC-250 is RAM.

`PS3RECOMP_ENG_STATS=1` prints three lines per 20th presented frame. All carry
the frame number and the A/B phase, and all describe **that one frame** (the
counters are reset on every present):

```
[eng stats] frame 7660: 10171 draws, 1163032 verts, 801 texture hashes (69 MB), 4 uploads,
           68274 tex lookups/46489873 probes, 10171 pipeline lookups/1353041 probes
[engtime] frame 7660 phase 0: group 222.09 ms, fetch 51.95, pipeline 16.51,
           textures 61.73 (hash 11.28), bind 6.28, draw 84.87 | 10171 groups,
           10171 draws/frame, 21.84 us/group
[engtime] gpu frame 7660 phase 0: fence wait 53.23 ms over 47 submissions, 1.132 ms/submission
```

`group` is the walker's whole time inside `sink_end` (fetch, pipeline, textures,
bind, backend draw), `draw` includes the fence waits the backend's batching adds,
and the `gpu` line separates those out. `probes` are entries a linear scan
walked; that is the number a hash table has to remove. The numbers above are
the frame as it was before the identity caches landed; the same frame now
reads:

```
[eng stats] frame 8660: 10167 draws, 1162996 verts, 801 texture hashes (9 MB, 100 full),
           4 uploads, 68254 tex lookups/90165 probes, 10167 pipeline lookups/10167 probes
[engtime] frame 8660 phase 0: group 137.6 ms, fetch 35.5, pipeline 3.4
           (program 1.1, constants 0.6), textures 11.1 (hash 3.6), bind 4.9,
           draw 81.0 | 10167 groups, 10167 draws/frame, 13.5 us/group
[ENGVC] frame 8660 phase 0: 10135 hits / 14 misses (99.9% of 1162897 verts),
           961 converted, 30.5 MB hashed, refs 24.00 ms, key 7.89, convert 0.03,
           0 evict, 2 unkeyed
```

`texture hashes` counts the textures revalidated in that frame and how many of
them needed the full hash rather than the sparse sample. `[ENGVC]` is the
converted-vertex cache (see below): groups served from it, groups that were
converted, the share of *vertices* that came from it, the bytes hashed for its
keys, and where the fetch phase went. `refs` is building the reference list
from the index array and de-duplicating it, `key` is the plan, the reference
list and the guest bytes, `convert` is the attribute decode of the misses.

`PS3RECOMP_ENG_FRAME_LOG=1` prints what a flip resolves to and every registered
target with its guest address, every 120th frame; `=2` prints every present,
which is what a double-buffered title needs (the guest alternates its render
target between the two display buffers, so sampling every 120th frame aliases
with that alternation).

The second half of the methodology is in-run A/B: `BENCH_AB=1` starts a POSIX
phase driver (`PS3_AB_PHASE=1`, `PS3_AB_SEC`, default 20 s) that advances
`g_ab_phase`, which every `*_PHASE` knob already reads — even phases are the
experimental path, odd the old one. `PS3_ABX_SEQ=<x,x,...>` also sets the
whole `g_abx` bitmask to the listed value each phase (the Windows thread's
semantics), so list the default `0x83F` with an experiment's bit added rather
than the bit alone. Each phase change prints
what it closed:

```
[ABFPS] phase 10: 84 frames, 236.44 ms/frame, 4.23 fps, 10167 draws/frame, odd: experiments off
```

Never compare two boots for a performance claim; with `BENCH_AB=1` the same
scene runs both configurations inside one process.

Frame and surface captures:

- `PS3RECOMP_VK_DUMP=<path>` writes each presented frame as binary PPM; a `%u`
  in the path receives the frame number. `PS3RECOMP_VK_DUMP_EVERY=<n>` thins it.
  Leave both off while timing: a dump of every frame stalls this title's
  walker around frame 6500.
- `PS3RECOMP_VK_DUMP_SURFACES=<dir>` writes every registered colour target as
  PPM every `PS3RECOMP_VK_DUMP_SURFACES_EVERY` presents (default 600), named
  `surface-f<frame>-<handle>-<w>x<h>.ppm`. Every format the engine can render
  into is covered: RGBA8 as it is, and the fp16/float targets converted and
  clamped, with `-f16`, `-r32f` or `-f32` in the name. That is how a frame is
  attributed: if the presented target holds the scene and the display still
  shows black, the present path is wrong; if every surface is black, the frame
  really is black; and a G-buffer target that holds the geometry while the
  display target does not says the composite is the broken pass.
- `PS3RECOMP_VK_DUMP_SURFACES_FRAME=<n>[,<n>...]` dumps the targets of those
  presented frames instead of every Nth, which is what a defect report needs
  and costs one readback of each target rather than the whole run's. Naming
  two (the menu and a gameplay frame) is what a scene report wants: one boot,
  one binary, both frames.
- `PS3RECOMP_VK_DUMP_FRAME=<n>[,<n>...]` (up to 8) dumps exactly those
  presented frames instead of every Nth. A scene capture wants the same frame
  numbers on every run, which the thinning knob cannot name, and each dump
  costs a readback plus a 2.7 MB write. `tools/scene.sh` uses it.
- `convert` or `magick` turns either into PNG for review.

### The Proton reference

The reference frames for a scene come from the BC-250 itself, not from the
Deck, so the comparison shares machine, driver and scene. `tools/proton-ref.sh`
(below) does all of this in one command; the manual recipe and every trap are
kept here because they are what the script encodes. `native-artifacts/m1e/`
has a captured set and the numeric comparison, and its `HANDOFF.md` has the
longer procedure. The short version:

```sh
# hardlinked copy of the retail checkout without its Wine prefix or logs, its
# own prefix, and the same HDD snapshot (save) the native run loads
src="$HOME/Games/inFamous"; dst="$HOME/dev/proton-ref"
rm -rf "$dst" "$HOME/dev/proton-ref-prefix"; mkdir -p "$dst" "$HOME/dev/proton-ref-prefix"
cd "$src"
ls -A | grep -v -E '^(\.prefix|\.git|gamedata|.*\.log)$' | while IFS= read -r x; do cp -al "$x" "$dst/"; done
mkdir -p "$dst/infamous-recomp/gamedata"
cp -a "$HOME/dev/infamous-native/infamous-recomp/gamedata/dev_hdd0" \
      "$dst/infamous-recomp/gamedata/dev_hdd0"

# the copy's launcher must not open a blocking dialog in a headless run
cp --remove-destination "$dst/installer/launch.txt" "$dst/installer/launch.txt.new"
grep -v -E '^env=(MSGDIALOG_NATIVE|SAVEDATA_NATIVE_UI)=' \
     "$dst/installer/launch.txt.new" > "$dst/installer/launch.txt"

WINEPREFIX=$HOME/dev/proton-ref-prefix GAMEID=0 PROTON_USE_XALIA=0 \
PROTONPATH="$HOME/.local/share/Steam/steamapps/common/Proton 11.0" \
CELLMARK_DUMP=3 CELLMARK_DUMP_MINDRAWS=13000 CELLMARK_DUMP_DIR='Z:\tmp\refbmp' \
  gamescope --backend headless -W 1280 -H 720 -- \
  /usr/bin/umu-run "$HOME/dev/proton-ref/inFamous.exe" --play
```

`gamescope --backend headless` gives a real compositor and swapchain without
reaching the TV. The launcher reads `installer/launch.txt` from the copy, so
edit that copy only, and copy-then-edit (`cp --remove-destination`) because the
hardlinks are shared with the retail folder.

`umu-run` is `/usr/bin/umu-run`, not the path under `~/.local/share/umu`. Proton
11.0 logs two harmless errors while it builds the prefix (`amdxcffx64.dll`,
"unable to use parent for game drive") and then runs at 30 fps. The Windows
D3D12 build has no `LD_FRAME_DUMP`: it captures with `CELLMARK_DUMP=<n>`
(exactly n presented frames, once), `CELLMARK_DUMP_DIR` (a Windows path),
`CELLMARK_DUMP_SKIP` and `CELLMARK_DUMP_MINDRAWS=<n>`, which is how a gameplay
frame is reached without counting presents. Input comes from the copy's
`launch.txt` environment or `PAD_FILE`, never from the title's dialogs: with
`MSGDIALOG_NATIVE` set, the in-game dialog only XInput or `PAD_FILE` can
dismiss, and `SAVEDATA_NATIVE_UI` opens a Win32 MessageBox. A backgrounded
process does not survive the `ssh` session that started it, so run the driver
as `systemd-run --user --unit=<name> --collect`.

## Dev tooling: scenes, goldens, references

Four tools, each with a usage line in its own `--help`. They exist so that a
claim in this file can be re-measured with one command, and so that a change
which moves the picture fails a check instead of passing a review.

| tool | what it does |
| --- | --- |
| `tools/framestat.py` | numbers for one frame, or a comparison against a reference |
| `tools/scene.sh` | boots to a scene on the BC-250, captures its frames at fixed frame numbers, writes `summary.json` |
| `tools/golden.sh` | captures or checks the golden frames of the scenes |
| `tools/proton-ref.sh` | the Proton reference frames of a scene, into a refs directory |

`docs/native-status.md` is the living state of the port: what is verified, with
which numbers, which defects are open and what the next step is.

### tools/framestat.py

One frame, or a frame against a reference. Stdlib only (PNG through `zlib`,
Pillow used only for a PNG variant the built-in decoder does not cover), reads
PPM (P6/P3), BMP (24/32-bit) and 8-bit PNG. `-I` keeps the interpreter out of
the environment.

```sh
python3 -I tools/framestat.py artifacts/scene-city-run/city-2200.ppm
python3 -I tools/framestat.py native-artifacts/golden/frames/city.ppm \
        native-artifacts/m1e/frame_001.bmp --ref-crop auto --json
```

One frame prints the per-channel mean/min/max (of 255), the fraction of pixels
that are black (all three channels below 4/255), the fraction that is the
magenta clear colour (r and b at least 200, g at most 60), and a 4x4 grid of
region mean luminances (Rec.601, sampled every 4th pixel). Two frames instead
print the mean absolute difference per channel and over all channels, the share
of channels more than 8/255 apart, and the per-region luminance delta grid.

`--json` puts one JSON object on stdout and the human line on stderr, so a
script can read the numbers and a person still sees the sentence. `--crop
X,Y,W,H` crops both sides, `--frame-crop` and `--ref-crop` one side, and
`auto` trims a letterbox off that side (border rows and columns whose mean
level is below `--auto-level`, default 2/255, because the Windows capture's bars
carry faint dither noise). The exit status is the measurement: 0 when every
threshold asked for passed, 1 when one did not, 2 on bad input. The thresholds
are `--min-mean`, `--max-dark`, `--max-magenta` for one frame and
`--max-mean-diff` (default 2.0, the tolerance section 3 of PLAN-native.md sets),
`--max-off8` (default 10) and `--max-region-d` (default 8) for a comparison.

The reference comparison that settled the lighting question
(`native-artifacts/m1e/`) was made with an ad-hoc version of this script,
`native-artifacts/m1e/frame-stat.py`. That file stays as the record of that
report; `tools/framestat.py` reproduces its numbers exactly and adds the rest.

### tools/scene.sh

One command from a cold shell to a measured scene.

```sh
tools/scene.sh city            # boot to gameplay, capture frame 2200, ~8 min
tools/scene.sh title           # the title screen, frame 800
tools/scene.sh menu            # gameplay, then START, the pause menu, frame 2600
tools/scene.sh city --ab VK_DRAW_BATCH_BARRIER_PHASE=0   # in-run A/B
```

| scene | pad script | frame | draws/frame | seconds |
| --- | --- | --- | --- | --- |
| `title` | none | 800 | 10167 | 140 |
| `menu` | `150:0x0008,170:0x4000,300:0x0008` | 2600 | pause menu | 470 |
| `city` | `150:0x0008,170:0x4000` | 2200 | 13500 | 430 |

START on the title screen does not open a main menu: it goes straight into the
opening mission's own intro cinematic, which is a wall clock animation and so is
not something a golden can hold (measured: `PAD_SCRIPT` with START held, and
`PAD_FILE` with a 2-poll press, both land in the cinematic). The `menu` scene is
the pause menu instead, opened by START during gameplay, and it stays open.

On the BC-250 it runs `tools/bench-run.sh` under `setsid`, in its own process
group, with a `trap` that TERMs and then KILLs the group on any exit, so no run
leaves a `inFamousRecomp` behind and nothing is drawn to the TV. From the Mac it
syncs the tree with `scripts/bc250.sh`, runs the same script over ssh, and copies
the artifacts back, so both hosts have one implementation.

It writes `artifacts/scene-<scene>-<tag>/summary.json` and prints the path:

```json
{
 "scene": "city",
 "boot_to_scene": { "draws": 13461, "frame": 1940, "ms": 274107, "seconds": 274.1 },
 "run": { "seconds": 420, "status": 0, "frames": [1600, 1800, 2000, 2200, 2400],
          "draws_window": [12000, 20000], "skip_intro": "1", "ab": null },
 "fps": { "avg_fps": 4.59, "p50_ms": 217.235, "p99_ms": 244.61,
          "low1_fps": 4.0, "low01_fps": 3.95, "windows": [ ... ] },
 "engtime_ms": { "frames": 36, "group": 204.481, "fetch": 57.193,
                 "pipeline": 2.396, "textures": 21.049, "bind": 6.908,
                 "draw": 112.294, "fence_wait": 75.444, "submissions": 31.389, ... },
 "eng_stats": { "frames": 36, "draws": 13499.222, "verts": 1966619.306,
                "texture_hashes": 595.028, "hash_mb": 9.5, "full_hashes": 71.333,
                "tex_lookups": 80879.417, "tex_probes": 100857.889,
                "handle0_binds": 0, "pipeline_full": 0, ... },
 "engvc": { "frames": 36, "hit_verts_pct": 98.197, "hits": 13325.75, "misses": 155.472 },
 "captures": [ { "frame": 2200, "width": 1280, "height": 720,
                 "mean": [0.084, 0.076, 0.072], "dark_fraction": 0.672,
                 "regions_luma": [ ... ] } ]
}
```

Every average comes from frames that belong to the scene: at or after the first
captured frame, and carrying the scene's draw count, so the boot, the loading
screens and the mission's cutscene never enter one. `[FRAMEPCT]` carries no
frame number, so its windows are taken by their mean draw count, the last three
in range. `boot_to_scene` is stamped by watching the log for the first frame
with the scene's draw count, which is the only way to time a boot: the
`[eng stats]` line names its frame but not its clock.

`--ab KEY=VALUE` exports `PS3_AB_PHASE=1` and `KEY`, and the summary then
reports even (experiment on) and odd (off) phases from the `[ABFPS]` lines.

### tools/golden.sh

```sh
tools/golden.sh capture          # title, menu and city, ~18 min
tools/golden.sh check            # run each scene and compare, OK/FAIL per scene
tools/golden.sh check city       # one scene
tools/golden.sh manifest         # rewrite the manifest from the frames
```

`capture` plays each scene through `tools/scene.sh`, keeps the one frame the
scene is defined by, and stores the Proton reference of the same scene next to
it. `check` replays the scenes and compares each capture against its golden with
`tools/framestat.py`, so a failure names the numbers:

```
    artifacts/scene-city-chk/city-2200.ppm vs /path/to/golden/city.ppm ...:
    mean abs diff 0.97 (limit 2.00), channels >8 apart 2.65% (limit 10.00%), ...
OK   city (city-2200.ppm vs city.ppm)
```

A PPM frame is 2.7 MB, so the frames live outside git: `~/dev/golden` on the
BC-250 (which is where they are captured), a copy under
`native-artifacts/golden/` on the Mac, and a manifest committed at
`artifacts/golden/manifest.json` naming every file with its size and sha256. The
city golden holds the known lighting defect on purpose: the check guards against
a regression, it does not certify the picture.

The tolerances come from section 3 of PLAN-native.md, widened for what moves
inside a still scene (fire embers, the title's smoke, the pause menu's
silhouette): mean absolute difference at most 2/255, at most 10 % of channels
more than 8/255 apart, and no 4x4 region more than 10 luminance levels away.
Two runs of the same binary measured 0.88 / 2.83 % / 6.2 on the title (the
smoke), 0.93 / 2.62 % / 2.1 on the pause menu and 0.97 / 2.65 % / 3.4 on the
city. A structural change is nowhere near these: the missing lighting term is a
region delta of 60-80.

### tools/proton-ref.sh

The reference frames of a scene from the Windows D3D12 build under Proton on the
BC-250 itself: same machine, same driver, same save. The traps are all in the
section above and in `native-artifacts/m1e/HANDOFF.md`; this script encodes them
so they are not learned twice.

```sh
tools/proton-ref.sh city                 # ~3 gameplay frames into ~/dev/refs
tools/proton-ref.sh title --min-draws 9000
tools/proton-ref.sh --setup-only         # just the copy and its prefix
```

It builds the hardlinked copy at `~/dev/proton-ref` with its own prefix
`~/dev/proton-ref-prefix`, copies the native run's HDD snapshot in so both are
the same save, replaces the copy's `installer/launch.txt` without the two
variables that turn a headless run into an undismissable dialog, and then runs
`gamescope --backend headless` + `umu-run` with the D3D12 build's own
`CELLMARK_DUMP*` knobs. Input is the closed loop `PAD_FILE` file, not a fixed
schedule: START when the frame is drawing, CROSS a few seconds later. It never
runs in `~/Games` and never writes there.

A scene is reached by draw count, not by present count:
`CELLMARK_DUMP_MINDRAWS=<n>` only dumps frames that draw at least that much, so
`--min-draws 13000` lands on gameplay. `title` and `menu` need a threshold and a
present skip that no run has pinned down yet; treat their defaults as a starting
point and read the driver log.

### The two knobs the tooling added

`PS3RECOMP_VK_DUMP_FRAME=<n>[,<n>...]` is above: the same frame numbers on every
run.

`DEV_SKIP_INTRO=1` makes the boot's own splash and logo movies finish
immediately. inFamous plays `movies/cutscene/splash_*.bik` and
`infamous_logo.bik` to their end before the title, and the player's clock is the
guest's, so the 353-frame `splash_america.bik` costs 12 s of wall clock however
fast the port renders.

Ending the file early does not work, and the measurement is the reason: the
loader that reads the `.bik` wants the whole 13.5 MB and answers a short read
with `Short read at offset 0: xfer was only 0 of 65536 bytes ... Retrying`
forever, presenting one draw at 2800 fps, with the boot never finishing. The
seam that does work is the Bink header's frame count, which Bink1 keeps at
offset 8 and again at 16 (little endian): the header is served with a count of
1, the player stops after the first frame, and the stream stays intact so the
loader is satisfied. `ppu_fs.cpp` patches those two words in the guest's copy of
the header, for those two file names only, and only when the knob is set.
Cutscenes are untouched.

Measured on the BC-250, same tree and binary otherwise:

| | with `DEV_SKIP_INTRO=1` | without |
| --- | --- | --- |
| boot to the title screen | 12.0-12.9 s (first title frame 240-260) | SKIPNOSKIP_TITLE |
| boot to gameplay, same pad script | 274.1 s (first gameplay frame 1940) | SKIPNOSKIP_CITY |

The knob saves the movie's own length and nothing else: the pad script is wall
clock, so a scene run that waits for a pad press keeps waiting. Its value is the
12 s of boot, and the ~7000 presented frames it removes from the front of every
run, which is why the scenes above are captured at frames in the low thousands
instead of at 9500.

What the skip does not change is the picture. SKIPNOKNOWN_FRAME_CHECK

## What the identity caches do

The engine used to derive, per draw, everything a pipeline needs from the
microcode: two end-marker scans, the transform-program input analysis, the
fragment structural hash, the inline-constant collection (which zeroed and
rescanned a 32 KB block), a byte-at-a-time hash of both programs, and a linear
scan of up to 8192 pipeline keys. The texture side hashed every texel of every
bound texture and then scanned up to 1024 entries to find the slot. On the
title screen that was 79 ms of a 236 ms frame, 1.35 M pipeline-key probes and
47 M texture-slot probes.

Now:

- **Program identity** (`eng_program`): keyed on the registers an upload
  changes (the dispatcher's `vp_gen`, `VP_START_FROM_ID`, `SHADER_PROGRAM`,
  `SHADER_CONTROL`, the cube and vertex-texture masks) and *validated* against
  a hash of both programs' bytes, because a guest can rewrite fragment
  microcode in place without touching `SHADER_PROGRAM`. On a hit the per-draw
  cost is the two content hashes; the scans, the input analysis, the
  structural hash and the constants ran once per program.
- **Content hashing** is XXH3 (`libs/video/xxhash.h`) instead of byte-at-a-time
  FNV: 69 MB of texture content per frame at ~15 GB/s instead of ~1.2 GB/s.
  The span is sampled now, see "Sparse texture validation" below.
- **Indexes instead of scans**: open-addressed tables for pipeline keys,
  texture slots (with tombstones, since a slot can be evicted), colour targets
  and depth targets by `(location, offset)`. The key is always compared in
  full, never trusted from the hash.
- **Render state** is decoded once per `pipe_gen` (the dispatcher bumps it on a
  write to any register the decode reads) and its structural hash is taken with
  it, instead of 30 register reads plus a 33-field hash per draw.
- **One decode of the 16 fragment and 4 vertex texture units per draw**, shared
  by the cube mask, the bind and the vertex-unit bind, instead of three.

Measured on the title screen, same scene and knobs before and after: pipeline
16.5 -> 3.4 ms (4.8x), textures 61.7 -> 12.1 ms (5.1x), the pair 78.2 -> 15.5
ms (5.0x; 4.8-5.9x frame by frame over four boots), the walker's group 222 ->
153 ms, and 4.19 -> 5.85 fps at 10167 draws and 1.16 M vertices per frame,
unchanged. Frames captured at the same frame numbers are bit-identical where
the scene is not animated (the SCE card, the transition frames); where it is
(the smoke, the loading bar) the two builds are at different points of a
wall-clock animation because one of them presents 40% more frames per second.

## Sparse texture validation

A cached texture has to be re-read every frame it is bound, or an animated UI,
a video or a water surface freezes. Hashing every texel of every bound texture
was 801 hashes and 69-73 MB a frame on the preload, 5.7-6.5 ms of the walker.
The D3D12 cache's policy is now the engine's (`tex_csum_step` and the
`VC_SWD_VERIFY` rule, `rsx_d3d12_backend.c:5254-5338`, `:5653-5682`):

- every frame, ~1024 words spread across the span (4 KB of a large texture),
  FNV-1a with the reference's eight-independent-loads batching, compared
  against the sparse hash stored with the last full verification;
- the full XXH3 hash when the entry is new, when the samples disagree, and at
  least every `ENG_TEX_VERIFY_FRAMES` (8) frames per texture regardless.

The last rule is what keeps a change the samples cannot see, one texel between
two sample points, from freezing a surface forever, which is why it is a rule
and not an optimization. It is staggered by the entry's key: every cached
texture is created in the same couple of frames, and an unstaggered counter put
all 69 MB into one frame of eight.

`ENG_TEX_SPARSE=0` restores the old policy, `ENG_TEX_SPARSE_PHASE=0|1` runs the
sparse path only on that A/B parity. Measured in-run on the title screen:
72 MB and 801 full hashes per frame against 8-10 MB and about 110, the hash
5.8 -> 3.7 ms and the texture phase 13.5 -> 11.6 ms. The 8-10 MB is where the
mandatory verification puts the floor: 69 MB divided by 8 is 8.6 MB before any
sampling.

## The converted-vertex cache

A group's converted vertices are a pure function of three things: the fetch
plan (which attributes, from which offsets and strides, decoded as what), the
reference list (which elements, in which order) and the guest bytes those
references read. `ENG_VCACHE=1` keys all three with XXH3 and keeps the
converted block in a persistent arena, so a group whose key is already there
skips the attribute decode entirely and hands the backend the stored block.

This is the D3D12 cache's policy (`rsx_d3d12_backend.c:13924-14000`) in the
shared engine instead of one backend: the same key shape (plan plus index range
plus a hash of the raw guest bytes), the same per-size-class LRU with a CLOCK
second chance that never evicts a block the frame being recorded touched, and
the same refusal to key a group whose plan cannot be hashed exactly. A group
whose attribute the resolver could only read element by element is converted
every frame as before, because a key that does not see those bytes could hand
back another mesh's vertices. The key carries the full layout, every attribute
descriptor and default, the base offset, the frequency divider mask, the
inline-stream state and the reference list, not a sample of any of them.

`ENG_VCACHE_MB` sizes the arena (512 MB default), `ENG_VCACHE_PHASE=0|1` runs
the cache only on that A/B parity. Measured in-run on the title screen over
five phase pairs: 99.2-99.9 % of vertices served from the cache, 30.5 MB of key
hashing a frame, fetch 51.9 -> 35.6 ms, the walker's group 153 -> 137 ms and
5.83 -> 6.38 fps (+9.3 %). The attribute decode itself is gone (23.0 -> 0.03
ms). What is left of the fetch phase is building the reference list from the
index array and de-duplicating it, 23-29 ms, which this does not touch, and
the key hashing, 8 ms.

## Render targets, and why the 3D world was black

Until this was fixed the title screen, the menu and the opening mission drew no
3D world at all. The surfaces were there and the draws were not being dropped:
`PS3RECOMP_ENG_STATS=1` reported 13495 groups and 1.97 M vertices a frame, every
one of them with a guest pipeline, and the per-surface dump showed the shadow
atlas holding content while every colour target that mattered was zero. Two
engine defects caused it.

**A render target was keyed by its guest address alone.** A guest that declares
one address at two sizes (inFamous points `0x0F340000` at a 1024x1024 shadow
atlas and at a 512x288 depth copy, and it renders its world at 1280x720 into
addresses it also declares smaller) made each declaration recreate the other's
resource: the first pass's frame was destroyed before the second read it, every
frame, for both. The D3D12 engine records the same defect and its fix
(`rsx_live_draw.c`'s `surface_get`, 2823-2900): key the target on
`(location, offset, width, height, format)` and let both declarations keep their
contents. That is what `eng_surface_get` does now. Two declarations of one
address still do not alias, which is the reference's stated tradeoff -- a pass
that renders at one size and samples at the other sees the other's contents,
not its own -- but nothing destroys anybody's frame any more. The sampling
lookup prefers the declaration whose size the texture unit declares, so a unit
reads the one it means.

**A draw that samples the target it writes sampled guest memory.** In-place
lighting (read the accumulation buffer, blend the next light into it) cannot be
one host image on any API, and the engine's answer was to exclude the target
from the surface lookup and fall back to the guest's bytes at that address,
which hold whatever was last stored there. `rsx_draw_backend.color_snapshot`
is that copy: the backend keeps one per target and refills it in command order,
so the draws that sample it see the target as it was just before each of them.
`ENG_FEEDBACK=0` turns it off (guest memory instead), `ENG_FEEDBACK_PHASE=0|1`
does that on one A/B parity. It is 251 copies in a title-screen frame, and
because the copies join the open batch they cost no submission of their own: a
per-call image and a per-call fence wait was 47 ms of the same frame and 400
submissions.

A unit whose format is `DEPTH24_D8` is not that case even when the address
matches a colour target: a title points its zeta at a colour address for passes
that do not use depth (inFamous: `0x1734000` is a display buffer and a zeta),
and those reads are answered by the depth snapshot path.

Both are visible in `[eng stats]`: the last field is the frame's feedback
copies. The per-surface dump is what attributes a frame, and it now covers
every colour format, because the world pass of this title writes 8-bit targets
at 1280x720 while the composite lands in the display target.

## The FIFO fast paths on the engine path

The walker has a family of fast paths (`GCM_GEN_FAST`, `GCM_FIFO_DIRECT`,
`GCM_TC_FAST`, `GCM_INDEX_SETUP`, `GCM_STATE_PACKET`, `GCM_STATE_PACKET2`)
that store state straight into `rsx_state` instead of calling
`rsx_process_method` per word. That store is what the vtable path reads, and
the generic loop is what mirrors each word into the register-file draw engine
(`rsx_draw_engine_method`). They used to switch themselves off whenever the
engine was the active path, because otherwise the engine's register file would
silently miss the state those packets carry. They now hand each consumed word
to the engine themselves (`GCM_ENGINE_FEED`), which is what lets them run here
at all; their remaining saving is the generic loop's per-word IO lookup and
debug gates, and on the preload that is about 3 ms of a 236 ms frame
(`BENCH_AB=1` with `PS3_ABX_SEQ=0x83F,0x283F`).

## The game data must be complete

`PS3_GAME/USRDIR/cache/load_menu_catalog.log` is a disc asset (1143 bytes: the
list of `.xpp` character files the load menu previews). A game folder that is
missing it boots exactly as far as the save dialog and then parks: the guest
never creates `Game State Thread`, never submits a world draw, and keeps
presenting thirteen post-process passes per frame forever. This is not a
runtime fault and no runtime knob changes it, so check the file first when a
boot stops right after the movie. Adding it to an otherwise identical tree
(the copy under `~/dev/game-shadow` on the BC-250 is one) is what moves the
boot past that point.


## What is different on Linux

- The Windows D3D12 renderer is replaced by the opt-in Vulkan backend. Without
  `PS3RECOMP_RSX_VULKAN=ON`, Linux still builds and runs headless through the
  null backend.
- The Win32 UI overlay (`rsx_ui_overlay.c`), the F10 trainer page and the raw
  mouse path are Windows-only. SDL2 supplies the controller and audio
  backends, but their behavior with the retail game has not been verified.
- Native save/message dialogs fall back to the non-interactive behaviour
  already used by the runtime off Windows.
- The runtime's write-watch vertex cache is inert; the D3D12 backend owns it.
- `spu_plug_lookup` has a weak fallback so a build without the WWS plugin lift
  still links.
