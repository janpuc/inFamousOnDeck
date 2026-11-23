# inFamousOnDeck

**A native Steam Deck and Linux (Vulkan) port of inFamous** (PlayStation 3,
2009, European release `BCES00220`).

The game's PowerPC code is statically recompiled to x86-64 C++ with
[ps3recomp](https://github.com/sp00nznet/ps3recomp), so the game runs as a
normal program instead of under an emulator. On the Steam Deck it runs on the
Vulkan backend, at 15 W, with a controller navigable launcher and a one click
installer that builds everything in a user space container. No root, no `sudo`,
no Windows.

**Controller rumble and haptics work.** The motor requests of the game reach
the controller: the haptics of the built-in Steam Deck controller come through
Steam Input, and a Bluetooth or USB controller (DualSense, DualShock 4, Xbox,
Switch Pro, or any SDL3 gamepad) is read with its own rumble motors. The
launcher has a switch for it, and `PAD_RUMBLE=0` turns the motors off.

**This repository contains no game code and no game data.** You build the game
yourself, on your own machine, from your own copy of the disc.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the build checks, Vulkan validation requirements, and the strict no-game-data rule.

## License

This project is released under the MIT License. The full text is in
[LICENSE](LICENSE).

The parts that come from other projects keep their own licenses, and every one
of them is credited in the
[third party notices](THIRD_PARTY_NOTICES.md): ps3recomp (MIT), the
inFAMOUSreborn project (no license file in its repository), FireBurn
FSR-Vulkan and the AMD FidelityFX SDK (MIT), SDL (zlib), glslang (Apache 2.0
with the LLVM exception) and the ATRAC3 decoders (LGPL 2.1).

## What this is built on

This port stands on the work of several projects, and every one of them is
credited in the source, in the
[third party notices](THIRD_PARTY_NOTICES.md) and here:

- **[whompay/inFAMOUSreborn](https://github.com/whompay/inFAMOUSreborn)**, the
  Windows port of inFamous. It contributes the game specific sources, the
  installer, its manifest and the patch set that keeps the lifted code in
  shape. The Linux port follows it step by step.
- **[sp00nznet/ps3recomp](https://github.com/sp00nznet/ps3recomp)**, the
  recompiler toolkit and runtime. The PPU and SPU lifters, the runtime
  libraries and the Vulkan backend come from it.
- **[FireBurn/FSR-Vulkan](https://github.com/FireBurn/UniversalScalers-Vulkan)**
  and the **[AMD FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK)**,
  for FSR 3.1.5 upscaling and FSR 3.1.6 frame generation.
- **[SDL](https://libsdl.org)** for the window, the controllers and the audio.
- **[glslang](https://github.com/KhronosGroup/glslang)** for the guest shaders.
- **[RPCS3](https://rpcs3.net)**, which the player uses once to decrypt the
  executable of their own disc.

## The Steam Deck

Measured on a Steam Deck at the 15 W power limit, 720p, in the city:

| profile | real fps | displayed fps | 1% low |
| --- | --- | --- | --- |
| native, no upscaler | **40.77** | 40.77 | 24.28 |
| FSR 3.1 Quality + frame generation | 34.43 | **~68.86** | 16.14 |
| FSR 3.1 Quality + frame generation, locked at 30 | 29.93 | **~59.86** | 21.22 |

The last row is the one to use on the Deck: the picture is locked to a steady
60 Hz, and the real 1% low goes from 16.14 to 21.22 fps against the same
uncapped FSR Quality + frame generation profile, which is 31.5% smoother.

The build passes the Khronos Vulkan Validation Layers with **zero errors** in
every one of those profiles.

### Install it

You need your own dump of the European release `BCES00220` and a Steam Deck
(or any Linux machine with a Vulkan driver).

1. Dump the disc with your own hardware. Decrypt `EBOOT.BIN` with
   [RPCS3](https://rpcs3.net) (File, Export, decrypted `EBOOT.BIN`) and keep
   the decrypted `EBOOT.elf` next to it.
2. Clone this repository and run the installer:

   ```sh
   git clone https://github.com/janpuc/inFamousOnDeck.git
   cd inFamousOnDeck
   ./linux/steamdeck/install.sh --game ~/dumps/BCES00220
   ```

   The installer builds the port in a container, installs the runtime under
   `~/.local/share/inFamousOnDeck`, writes a desktop entry and offers to add
   the game to the Steam library. It never touches anything outside your home
   directory and it never needs root.

3. Start it from the desktop entry, from the Steam library, or directly:

   ```sh
   ~/.local/share/inFamousOnDeck/launcher/launcher.py
   ```

The whole procedure, the launcher, the presets and the numbers behind them are
in [docs/steamdeck.md](docs/steamdeck.md).

### The launcher

The launcher is a small window with the settings of the port. It is written in
Python and uses nothing but the standard library, so it runs on a stock SteamOS.

- Six presets carry the measured profiles, three for the Steam Deck and three
  for the AMD BC-250 console hardware.
- Every value can be changed on its own: FSR, the FSR mode, frame generation,
  the frame cap, the resolution scale, the anisotropic filtering, the level of
  detail scale, the smooth distance blur, the Sixaxis gyro, the controller
  rumble and the save folder.
- The settings live in `~/.config/inFamousOnDeck/settings.ini`.
- The window is driven with the D-pad, the left stick and the A and B buttons.
  Without a controller it works with the keyboard, the mouse and the touch
  screen.

### Controller rumble and haptics

The game asks for the small and the big motor of the pad, and the port passes
them on:

- the **built-in Steam Deck controller** gets its haptics through Steam Input,
- a **Bluetooth or USB controller** (DualSense, DualShock 4, Xbox, Switch Pro,
  or any other SDL3 gamepad) is read by SDL3 and gets its own rumble motors,
- the port keeps the motors running, because a PlayStation motor holds its
  setting until the game changes it and a rumble call is only a pulse.

The launcher has a `Controller rumble` switch next to the Sixaxis gyro. It is
**enabled by default in every preset**, which writes `PAD_RUMBLE=1`. Turning it
off writes `PAD_RUMBLE=0`, and the game then runs without any haptics.

The launcher also has a command line for the people who prefer one:

```sh
launcher.py --list-presets          # the presets
launcher.py --apply-preset "Steam Deck OLED: Native 40 (80 Hz)"
launcher.py --print-env              # the environment the port reads
launcher.py --play                   # start the game with the saved settings
launcher.py --steam                  # start the game and close Steam afterwards
```

### Add it to Steam

`linux/steamdeck/steam_shortcut.py` writes the non-Steam shortcut of the game.
It is a dry run by default and needs `--apply` to write anything, and it keeps
a timestamped backup of the file it replaces.

```sh
python3 linux/steamdeck/steam_shortcut.py --exe ~/.local/share/inFamousOnDeck/launcher/play.sh
python3 linux/steamdeck/steam_shortcut.py --list
```

## The presets

Every preset is a measured profile, not a guess.

| preset | hardware | real fps | displayed fps |
| --- | --- | --- | --- |
| `Steam Deck OLED: Smooth 60 (30+FG)` | Steam Deck, 15 W | 29.93 | ~59.86 |
| `Steam Deck OLED: Max FG (~69 fps)` | Steam Deck, 15 W | 34.43 | ~68.86 |
| `Steam Deck OLED: Native 40 (80 Hz)` | Steam Deck, 15 W | 40.77 | 40.77 |
| `AMD BC-250: Native 60+ + 16x AF` | AMD BC-250 | 62.19 | 62.19 |
| `AMD BC-250: 100+ FSR FG + 16x AF` | AMD BC-250 | 54.67 | ~109.34 |
| `AMD BC-250: Extended LOD 1.5 + 16x AF` | AMD BC-250 | 58.40 | 58.40 |

## The Windows port (Direct3D 12)

The Windows build is the reference renderer of the project. It is the port the
Linux one follows.

### Requirements

- Your own inFamous disc dump, European release (`BCES00220`).
- Windows 10 or 11 (64-bit) and a GPU with Direct3D 12 support.
- A CPU with SSE4.1 (2008 or newer). On CPUs without AVX2 the Setup builds a
  slower compatibility version automatically.
- [Python 3.10+](https://www.python.org/downloads/) with "Add python.exe to
  PATH" ticked.
- [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/)
  with the **Desktop development with C++** workload and the **C++ Clang
  Compiler for Windows** component.
- [RPCS3](https://rpcs3.net), only if your `PS3_GAME\USRDIR` has `EBOOT.BIN`
  but no decrypted `EBOOT.elf`. Setup uses it to decrypt the executable.
- About 10 GB of free disk space and 16 GB of RAM.

### Easy install

Download **inFamous Setup.exe** from the
[releases page](https://github.com/whompay/inFAMOUSreborn/releases), run it,
choose your `.iso` (an encrypted image works with its `.dkey` next to it) or
your game folder (the one with `PS3_GAME` in it), pick an install location and
press **Install**.

Setup checks for Python and the C++ build tools and offers to install them,
copies the game out of the disc image, regenerates the recompiled code from
your copy, builds it and adds an inFamous shortcut. Running Setup again on the
same folder updates or repairs the install and only redoes what changed.

### Playing

Run **inFamous** (the shortcut Setup made). It opens a small options window
first (vsync, frame cap, resolution, language and more, saved to
`settings.ini`); **Play** starts the game. The same settings are in the game's
pause menu under **Settings**. Controllers work as on the console, and
keyboard and mouse are supported.

The distance blur that hides pop-in is upsampled smoothly instead of showing
blocks at high resolution (`DEPTH_BLUR_SMOOTH=0` for the original). In blackout
areas the game starts it about four times closer; `DEPTH_BLUR=2` keeps it at
the powered-area distance there, and `DEPTH_BLUR=0` turns it off entirely.

Parked cars and wrecks are placed by a spawner that fills parking spots in a
region around the camera, so they appear and vanish a few car lengths away.
`CAR_PARK_SCALE=2` doubles that region (cars appear and vanish twice as far
out). It only scales the spawner that places cars; `LOD_DIST_SCALE` above 1
breaks the street lights, so keep that one at 1.

Frames with more draws than the renderer records at once (28672, reached with a
high `LOD_CLASS_SCALE` in the city) are split into parts.

**F10** opens the in-game trainer: God Mode, Infinite Energy, Refill Health,
Max Health x2/x5/x10, Add XP, Karma +/-, Kill Enemies / Civilians, Super Speed
and Player Model. Set `TRAINER=0` to turn it off.

### DLSS and FSR (optional)

The game can use NVIDIA DLSS / DLAA or AMD FSR for upscaling, and AMD FSR
frame generation. They are not included in the download because NVIDIA and AMD
distribute their SDKs themselves, but both are free and take a few minutes to
add.

| For | Download | Rename the extracted folder to |
| --- | --- | --- |
| DLSS / DLAA | [github.com/NVIDIA/DLSS](https://github.com/NVIDIA/DLSS) | `dlss-sdk` |
| FSR and frame generation | [github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK) | `fsr-sdk` |

Put the folder in your inFamous install folder next to `infamous-recomp` and
`ps3recomp`, rebuild the game once, and turn the feature on in the pause menu
under **Settings**, **Upscaling** and **Frame Generation**.

On Linux the FSR 3.1 path is built in: the temporal upscaler and the frame
generation come from the Vulkan implementation that is vendored with the
runtime, so no SDK has to be downloaded.

## How it works

| Path | Contents |
| --- | --- |
| `ps3recomp` | The recompiler toolkit and runtime: PPU and SPU lifters, HLE system libraries, RSX on Direct3D 12 and on Vulkan, audio and input |
| `infamous-recomp` | The inFamous specific parts: program entry, game stubs and fixes, asset tools |
| `installer` | The installer: reads the disc, regenerates the code from your game, applies `installer/patches`, builds |
| `installer/infamous.toml` | Everything about the game and the port: pipeline steps, generated files, build and launch settings |
| `installer/generated.lock` | Hashes of each generated file before and after the port's patches |
| `linux/steamdeck` | The Steam Deck launcher, the installer, the build container and the shortcut helper |
| `linux/steamdeck/project.py` | The names of the project, in one place |
| `tools/check_public_tree.py` | The gate that keeps game data and private names out of a published tree |

## Building the Linux port

The build container carries the toolchain: Clang 19 with lld, CMake, Ninja,
glslang, SPIRV-Tools, SDL3 and Python 3.

```sh
podman build --tag infamous-steamrt4 --file linux/steamdeck/Containerfile linux/steamdeck
podman run --rm --volume "$PWD:/work:rw" --volume "$PREFIX:/prefix:rw" \
    --volume "$GAME_DIR:/game:ro" infamous-steamrt4 \
    bash /work/linux/steamdeck/build.sh --source /work --game-elf /game/PS3_GAME/USRDIR/EBOOT.elf --prefix /prefix
```

`linux/steamdeck/pipeline.py` runs the steps of `installer/infamous.toml`, the
same ones the Windows installer runs, and applies the patch set with the same
strict rules.

The flags of the build are in `installer/infamous.toml` under
`[build].configure_unix`: the Vulkan RSX backend, the register-cached PPU with
a per function local guest base, the RC-only SPU with merged CFG components and
x86-64-v3 / znver2 codegen.

## Tests

```sh
python3 -m unittest discover -s linux/steamdeck/tests -t linux/steamdeck
```

The tests cover the settings of the launcher, the environment it generates, the
presets, the command line, the binary `shortcuts.vdf` of Steam, the patch
applier of the port against every patch it ships, and the public tree gate.

## Credits

- **[whompay/inFAMOUSreborn](https://github.com/whompay/inFAMOUSreborn)**, the
  Windows port of inFamous, the project this one follows. The game specific
  sources, the installer, the manifest and the patch set come from it.
- **[ps3recomp](https://github.com/sp00nznet/ps3recomp)** by sp00nznet and
  contributors (see
  [ps3recomp/CONTRIBUTORS.md](ps3recomp/CONTRIBUTORS.md)), the recompiler
  toolkit and the runtime.
- **[FireBurn/FSR-Vulkan](https://github.com/FireBurn/UniversalScalers-Vulkan)**
  and the **[AMD FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK)**,
  for FSR 3.1.5 upscaling and FSR 3.1.6 frame generation.
- **[SDL](https://libsdl.org)**, for the window, the controllers and the audio.
- **[glslang](https://github.com/KhronosGroup/glslang)**, for the guest shaders.
- **[RPCS3](https://rpcs3.net)**, used to decrypt the game executable.
- Sucker Punch Productions, for the game.

The licenses of everything that is used here are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
