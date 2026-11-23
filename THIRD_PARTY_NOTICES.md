# Third party notices

The port is built on the work of other people. This page lists everything
that is used here, who wrote it, and under which terms.

The project itself is published under the MIT License. The full text is in
[LICENSE](LICENSE), with the copyright of the inFamousOnDeck contributors. The
parts that come from other projects keep their own licenses, and they are
listed here.

The game itself is not part of this repository. inFamous is a trademark and
copyright of Sucker Punch Productions. The repository carries no game code and
no game data: the executable of the game is lifted to C on the machine of the
player, from the player's own dump, and the result is never published.

## ps3recomp

The recompiler toolkit and runtime: the PPU and SPU disassemblers and lifters,
the ELF and SELF parsing, the NID system, the runtime libraries, the system
call layer, the recompilation pipeline and the RSX backends.

- Project: <https://github.com/sp00nznet/ps3recomp>
- Author: Ned Heller (sp00nznet) and the contributors listed in
  [ps3recomp/CONTRIBUTORS.md](ps3recomp/CONTRIBUTORS.md)
- License: MIT, the full text is in [ps3recomp/LICENSE](ps3recomp/LICENSE)

The MIT license of ps3recomp covers the parts of this repository that come
from it, which is the runtime and the toolkit.

## inFAMOUSreborn

The Windows port of inFamous, the project the Linux one follows. It contributes
the game specific sources (`infamous-recomp/`), the installer and its manifest
(`installer/`), the patch set (`installer/patches/`) and the lift of the game
that the Linux port reproduces.

- Project: <https://github.com/whompay/inFAMOUSreborn>
- Author: whompay

The upstream repository carries no license file, so the terms of the reuse of
its parts are not stated there. This repository credits the project in every
file that comes from it, in the README and here, and the terms of the reuse
are the terms of the upstream project.

## FireBurn FSR-Vulkan

The Vulkan implementation of FSR 3.1 that the port uses for the upscaling and
the frame generation. A subset is vendored with the runtime in
`ps3recomp/third_party/ffx_vk`, with local fixes that are listed in
[ps3recomp/third_party/ffx_vk/README.md](ps3recomp/third_party/ffx_vk/README.md).

- Project: <https://github.com/FireBurn/UniversalScalers-Vulkan>
  (the repository was called FSR-Vulkan before it moved)
- Version: commit `d38393c`
- License: MIT, the full text is in
  [ps3recomp/third_party/ffx_vk/LICENSE.txt](ps3recomp/third_party/ffx_vk/LICENSE.txt)

## AMD FidelityFX SDK

The source of the FSR 3.1.5 temporal upscaler and of the FSR 3.1.6 frame
generation that the vendored subset above comes from.

- Project: <https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK>
- Version: 2.3.0 (the subset the port needs)
- License: MIT, Copyright (C) Advanced Micro Devices, Inc. The notice is at the
  top of every header of the SDK, for example
  [ffx_api.h](ps3recomp/third_party/ffx_vk/upstream/ffx-2.3.0/Kits/FidelityFX/api/include/ffx_api.h)

## SDL

The window, the controllers and the audio of the port. The headers are
vendored in `sdl3/include`, the library is loaded at run time.

- Project: <https://github.com/libsdl-org/SDL>
- License: zlib, the full text is in
  [sdl3/LICENSE.txt](sdl3/LICENSE.txt)

## glslang

The HLSL to SPIR-V translator that turns the shaders of the guest into the
SPIR-V the Vulkan backend draws. It is used by the build container and is not
part of this repository.

- Project: <https://github.com/KhronosGroup/glslang>
- License: Apache License 2.0 with the LLVM exception, the full text is in
  the `LICENSE.txt` of the project

## The ATRAC3 decoders

The ATRAC3 and ATRAC3plus decoder of the runtime. It is the standalone
extraction of the decoders of FFmpeg that PPSSPP vendors, kept as a separate
component.

- Location: [ps3recomp/third_party/at3_standalone](ps3recomp/third_party/at3_standalone)
- Project: <https://github.com/hrydgard/ppsspp> (the `ext/at3_standalone`
  folder) and <https://ffmpeg.org/>
- License: LGPL-2.1-or-later, the full text is in
  [COPYING.LGPLv2.1](ps3recomp/third_party/at3_standalone/COPYING.LGPLv2.1)

## RPCS3

Not part of the port and not linked into it. The player uses it once, to
decrypt the executable of their own disc.

- Project: <https://rpcs3.net>
- License: GPL-2.0

## The build container

The Linux build runs in a container that is built from the Steam Runtime 4 SDK
image of Valve. The image is not part of this repository and is not
redistributed with it. It provides Clang 19, CMake, Ninja, glslang, SPIRV-Tools,
SDL3 and Python 3.

- Project: <https://github.com/ValveSoftware/steam-linuxruntime>

## The games and the tools that are not part of the port

- **inFamous** (2009, Sucker Punch Productions) is the game the port runs. It
  is not included and it is not redistributed. Every player needs their own
  copy of the European release `BCES00220`.
- **PlayStation 3** is a trademark of Sony Interactive Entertainment.
- **Steam** and **Steam Deck** are trademarks of Valve Corporation. The
  shortcut helper talks to the local Steam installation through its own
  configuration file and is not affiliated with Valve.
