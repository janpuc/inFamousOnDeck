# The Steam Deck

The native Linux build of inFamous runs on the Steam Deck. This page is the
whole procedure: what you need, how to install it, what the launcher does,
what the numbers behind the presets are, and what to do when something does
not work.

The names of the project live in one file,
[`linux/steamdeck/project.py`](../linux/steamdeck/project.py), and everything
that names the project reads it from there.

- [What you need](#what-you-need)
- [Your own dump](#your-own-dump)
- [Install](#install)
- [The launcher](#the-launcher)
- [The presets](#the-presets)
- [The controller](#the-controller)
- [The Steam library](#the-steam-library)
- [Where things are](#where-things-are)
- [Troubleshooting](#troubleshooting)
- [Build it yourself](#build-it-yourself)
- [What was verified](#what-was-verified)

## What you need

- A Steam Deck with SteamOS 3.9 or newer, or any Linux machine with a Vulkan
  1.3 driver.
- Your own dump of the European release `BCES00220` of inFamous.
- About 15 GB of free disk space and an hour for the first build.
- `podman`. SteamOS has it built in, so nothing has to be installed and
  nothing needs root.

## Your own dump

The port ships no game code and no game data. You bring the game.

1. Dump the disc with your own hardware. Any blue ray disc dumper works, the
   result is the usual `PS3_GAME` folder.
2. Decrypt `PS3_GAME/USRDIR/EBOOT.BIN` with
   [RPCS3](https://rpcs3.net): open the executable with File, Export, and pick
   the decrypted `EBOOT.BIN`. Put the decrypted `EBOOT.elf` next to it in the
   same folder.
3. Check that the folder looks like this:

   ```
   BCES00220/
     PS3_GAME/
       USRDIR/
         EBOOT.BIN
         EBOOT.elf     <- the decrypted one
   ```

The installer accepts the folder or an ISO of the disc. For an ISO it needs
`bsdtar`, `7z` or `xorriso` to read it, and the decrypted `EBOOT.elf` has to
be in the folder it extracts to.

The port was verified with one executable, the SHA-256 of which the installer
prints. A different one is not refused, but the port is only known to work
with that release.

## Install

```sh
git clone https://github.com/janpuc/inFamousOnDeck.git
cd inFamousOnDeck
./linux/steamdeck/install.sh --game ~/dumps/BCES00220
```

The installer:

1. checks the dump and prints the hash of the executable,
2. builds the container image from `linux/steamdeck/Containerfile` (once),
3. builds the port inside it: it runs the pipeline of
   `installer/infamous.toml`, applies the patch set and compiles,
4. installs the runtime, the launcher and a desktop entry under
   `~/.local/share/inFamousOnDeck`,
5. offers to add the game to the Steam library.

Useful options:

| option | what it does |
| --- | --- |
| `--game PATH` | the folder or the ISO of your dump |
| `--prefix DIR` | where the runtime goes (default `~/.local/share/inFamousOnDeck`) |
| `--source DIR` | the checkout to build from (default: this repository) |
| `--jobs N` | parallel compile jobs (default: the number of cores) |
| `--image NAME` | the container image (default `localhost/infamous-steamrt4:latest`) |
| `--skip-build` | install what is already there, do not build |
| `--dry-run` | print what would happen and stop |
| `--no-shortcut` | do not touch the Steam library |
| `--apply-shortcut` | write the Steam shortcut (the default is a dry run) |

Nothing is written outside your home directory. The container runs with your
user, so there is no root and no `sudo` anywhere in the procedure.

## The launcher

```sh
~/.local/share/inFamousOnDeck/launcher/launcher.py
```

The window has the presets on the left and the individual settings on the
right. The settings are saved in `~/.config/inFamousOnDeck/settings.ini`:

```ini
[video]
fsr = 1
fsr_mode = quality
fsr_fg = 1
fps_cap = 30
res_scale = 1
tex_aniso = 0
lod_class_scale = 1.0
depth_blur_smooth = 1

[input]
pad_gyro = 0
pad_rumble = 1

[paths]
savedata_root =
```

| setting | what it does |
| --- | --- |
| `fsr` | FSR 3.1 upscaling on or off |
| `fsr_mode` | `native`, `quality`, `balanced`, `performance` or `ultra` |
| `fsr_fg` | FSR 3.1 frame generation on or off |
| `fps_cap` | the frame limiter of the port, in frames per second (0 is off) |
| `res_scale` | the internal resolution scale, 1 to 3 (1 is 720p) |
| `tex_aniso` | the anisotropic filtering, 0 to 16 |
| `lod_class_scale` | how much of the city is drawn, 1 to 2 |
| `depth_blur_smooth` | the smooth distance blur that hides the pop-in |
| `pad_gyro` | the Sixaxis gyro of the controller |
| `pad_rumble` | the controller rumble and haptics, on by default (`PAD_RUMBLE=1`), 0 turns the motors off |
| `savedata_root` | where the saves go (empty means the default) |

The command line does the same without a window:

```sh
launcher.py --list-presets
launcher.py --apply-preset "Steam Deck OLED: Native 40 (80 Hz)"
launcher.py --set fps_cap=40 --play
launcher.py --print-env
launcher.py --play --dry-run
```

## The presets

The presets carry the profiles that were measured, in the city (about 13,500
draws a frame), at 720p. The Steam Deck ones at the 15 W power limit, the AMD
BC-250 ones on the console hardware:

| preset | hardware | real fps | displayed fps | 1% low | note |
| --- | --- | --- | --- | --- | --- |
| `Steam Deck OLED: Smooth 60 (30+FG)` | Steam Deck, 15 W | 29.93 | ~59.86 | 21.22 | the locked 60 Hz picture, the best 1% lows |
| `Steam Deck OLED: Max FG (~69 fps)` | Steam Deck, 15 W | 34.43 | ~68.86 | 16.14 | the fastest measured Deck profile |
| `Steam Deck OLED: Native 40 (80 Hz)` | Steam Deck, 15 W | 40.77 | 40.77 | 24.28 | no upscaler, no frame generation, locked to 40 |
| `AMD BC-250: Native 60+ + 16x AF` | AMD BC-250 | 62.19 | 62.19 | 37.54 | native, with the 16x filtering |
| `AMD BC-250: 100+ FSR FG + 16x AF` | AMD BC-250 | 54.67 | ~109.34 | 32.67 | FSR Quality with frame generation |
| `AMD BC-250: Extended LOD 1.5 + 16x AF` | AMD BC-250 | 58.40 | 58.40 | | 10% more of the city is drawn |

The Steam Deck profiles work on both panels: the OLED one runs at 90 Hz and
the LCD one at 60 Hz, so the locked 30 fps profile with frame generation gives
a steady 60 Hz picture on either, and the native profile locks to 40 fps
(80 Hz on the OLED panel, 40 Hz on the LCD one).

The frame cap is the limiter of the port itself (`FPS_CAP`), not the one of the
pause menu. With frame generation on, a cap of 30 gives a steady 60 Hz
picture, because every rendered frame becomes two displayed ones. Capping it
that way also makes the picture smoother: the real 1% low goes from 16.14 fps
of the uncapped FSR Quality + frame generation profile to 21.22 fps, which is
31.5% better.

The anisotropic filtering is off by default on the Deck on purpose: 16x
filtering costs about 4.5 fps of the 1% low at 15 W. On the BC-250 it is free,
so the profiles of that hardware turn it on.

The level of detail scale is 1 by default. 1.5 draws about 10% more of the
city and costs a few frames per second.

## The controller

The window reads the controller directly, without any extra package:

1. the raw Linux input devices (what a stock SteamOS Deck offers),
2. pygame, if it is installed,
3. SDL3 or SDL2 through ctypes.

The Steam Deck controller and the Steam Input virtual pad are both found. The
window is driven with:

| input | what it does |
| --- | --- |
| D-pad up and down, left stick up and down | move between the controls |
| A | activate the control under the cursor |
| B | back to the presets |
| Start | start the game |

Without a controller the same window works with the keyboard, the mouse and
the touch screen. `launcher.py --list-devices` prints what was found.

### Rumble and haptics

The game asks the pad for the small and the big motor, and the port passes
them on:

| controller | how it is read | the haptics |
| --- | --- | --- |
| the built-in Steam Deck controller | Steam Input, as a virtual gamepad | the haptics of the Deck |
| DualSense (PS5) | SDL3 over Bluetooth or USB | its own rumble motors |
| DualShock 4 (PS4) | SDL3 over Bluetooth or USB | its own rumble motors |
| Xbox controller | SDL3 over Bluetooth or USB | its own rumble motors |
| Switch Pro | SDL3 over Bluetooth or USB | its own rumble motors |
| any other SDL3 gamepad | SDL3 | its own rumble motors |

The port keeps the motors running instead of pulsing them, because a
PlayStation motor holds its setting until the game changes it.

The launcher has a `Controller rumble` switch next to the Sixaxis gyro. It is
**enabled by default in every preset**, which writes `PAD_RUMBLE=1`. Turning it
off writes `PAD_RUMBLE=0` and the game then runs without any haptics. The same
works from the command line:

```sh
launcher.py --set pad_rumble=0 --play
```

## The Steam library

`linux/steamdeck/steam_shortcut.py` writes the non-Steam shortcut:

```sh
python3 linux/steamdeck/steam_shortcut.py \
    --exe ~/.local/share/inFamousOnDeck/launcher/play.sh \
    --start-dir ~/.local/share/inFamousOnDeck \
    --app-name "inFamousOnDeck"
```

It prints what it would do and writes nothing. Add `--apply` to write it. The
file it replaces is copied to a timestamped backup first, and the tool reads
and writes the binary format of Steam directly, so the other shortcuts of the
library stay as they are.

Stop and start Steam afterwards to pick the shortcut up.

## Where things are

| what | where |
| --- | --- |
| the runtime | `~/.local/share/inFamousOnDeck` |
| the binary | `~/.local/share/inFamousOnDeck/bin/inFamousRecomp` |
| the SDL3 it uses | `~/.local/share/inFamousOnDeck/lib/libSDL3.so.0` |
| the launcher | `~/.local/share/inFamousOnDeck/launcher` |
| the settings | `~/.config/inFamousOnDeck/settings.ini` |
| the saves | `~/.local/share/inFamousOnDeck/savedata` |
| the disk image of the game | `~/.local/share/inFamousOnDeck/dev_hdd0` |
| the shader cache | `~/.cache/inFamousOnDeck/spirv` |
| the log | `~/.local/state/inFamousOnDeck/inFamousOnDeck.log` |

The game keeps its saves in the disk image, the same way the console did. The
first start installs about 1.3 GB of game data there, which takes a moment.

## Troubleshooting

**The window does not open.** The launcher needs Tkinter. SteamOS ships it.
Check with `python3 -c "import tkinter"`.

**The game starts and shows a black screen.** The log says what happened:
`~/.local/state/inFamousOnDeck/inFamousOnDeck.log`. A missing executable
shows up there as "Decrypted retail EBOOT not found".

**The frame rate is low.** The Deck runs at 15 W by default in a game session.
The preset `Steam Deck OLED: Smooth 60 (30+FG)` caps the frame rate, which keeps
the picture steady and the battery longer. `RES_SCALE=1` is the 720p mode, and
the level of detail scale of 1 is the fast one.

**The first boot takes a while.** The game plays its intro movies before the
title screen, and the movies run at the speed of the guest. `DEV_SKIP_INTRO=1`
in the environment skips them. It is not the default because it changes how
the game starts.

**The game does not start from Steam.** The shortcut points at
`play.sh`, which needs a terminal free environment. Start it once from the
desktop entry first, so the runtime exists and the settings file is written.

## Build it yourself

The container image is the Steam Runtime 4 SDK image (Clang 19.1.7, CMake,
Ninja, glslang, SPIRV-Tools, SDL3, Python 3.13) with lld, ccache and pip
added.

```sh
podman build --tag infamous-steamrt4 --file linux/steamdeck/Containerfile linux/steamdeck
podman run --rm \
    --volume "$PWD:/work:rw" \
    --volume "$HOME/.local/share/inFamousOnDeck:/prefix:rw" \
    --volume "$HOME/dumps/BCES00220:/game:ro" \
    --volume "$HOME/.cache/ccache:/ccache:rw" \
    infamous-steamrt4 \
    bash /work/linux/steamdeck/build.sh \
        --source /work \
        --game-elf /game/PS3_GAME/USRDIR/EBOOT.elf \
        --prefix /prefix
```

The build has three parts:

1. the pipeline of `installer/infamous.toml`: the PPU and the SPU programs of
   your dump are lifted to C, the same steps the Windows installer runs,
2. the patch set of `installer/patches`, applied with the same strict rules as
   the Windows installer (every context line has to match byte for byte),
3. the compile, with the flags of `[build].configure_unix`.

`installer/generated.lock` holds the hash of every generated file before and
after the patches. The pipeline prints whether the result still matches it.

The unit tests of the launcher, the shortcut helper, the patch applier and the
public tree gate:

```sh
python3 -m unittest discover -s linux/steamdeck/tests -t linux/steamdeck
```

## What was verified

On a Steam Deck at the 15 W power limit, with the final build:

- the numbers of the table above, in the city, with 13,500 draws a frame and
  no dropped or repeated audio,
- the Khronos Vulkan Validation Layers with `VK_LAYER_KHRONOS_validation`
  enabled and **0 errors** in the native profile, in the FSR Quality profile,
  in the native plus frame generation profile and in the locked 30 fps
  profile,
- the unit tests of the runtime (`ctest`),
- the profile with the frame generation, checked against the frames of the
  reference to make sure the interpolated frames are the ones the engine
  produced.
