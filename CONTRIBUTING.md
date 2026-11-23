# Contributing to inFamousOnDeck

Contributions are welcome, whether you are improving Steam Deck performance, Vulkan correctness, controller support, packaging, or documentation.

## Non-negotiable rule: no copyrighted game or firmware files

This repository is strictly bring-your-own-game (BYOG). Pull requests must **never** include:

- Disc dumps, ISOs, `.dkey`, `EBOOT.BIN`, decrypted `EBOOT.elf`, `.psarc`, or `.sfo` files
- Recompiled game sources (`infamous-recomp/recompiled/`, `infamous-recomp/spu/` except the tracked map/helpers)
- Sony PlayStation 3 firmware modules (`*.prx`, `ps3recomp/fw_spu/`)
- Captured game frames (`*.ppm`, `*.bmp`, `*.png` screenshots), saves (`dev_hdd0/`, `savedata/`), or logs

Run the public-tree gate before opening a PR:

```sh
python3 tools/check_public_tree.py .
```

## Checks before opening a PR

1. **Public-tree gate**: `python3 tools/check_public_tree.py .` must report `findings: none`.
2. **Launcher and packaging unit tests**:
   ```sh
   python3 -m unittest discover -s linux/steamdeck/tests
   ```
3. **Shell scripts**: `bash -n linux/steamdeck/*.sh linux/infamous-native.sh` (and `shellcheck -S style` when available).
4. **Runtime tests (when touching C/C++ or `ps3recomp/`)**:
   ```sh
   cmake -S ps3recomp -B build-runtime -G Ninja -DCMAKE_BUILD_TYPE=Release -DPS3RECOMP_BUILD_TESTS=ON
   cmake --build build-runtime --parallel
   ctest --test-dir build-runtime --output-on-failure
   ```
5. **Vulkan changes**: verify with real Khronos Validation Layers (`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`, `VK_LOADER_DEBUG=layer` confirming `Insert instance layer "VK_LAYER_KHRONOS_validation"`) and **zero validation errors** across both native (`FSR=0`) and FSR 3.1.6 Frame Generation (`FSR=1 FSR_MODE=quality FSR_FG=1`).

## Target hardware profiles

Built-in launcher presets target:

- **Steam Deck OLED (90 Hz)** and **Steam Deck LCD** at the 15 W APU envelope
- **AMD BC-250** (and general Linux Vulkan GPUs)

Keep Steam Deck 15 W power-envelope regressions in mind: anything expensive on GPU clocks (such as anisotropic filtering) should stay opt-in for Deck while remaining available in desktop / BC-250 presets.

## License

Contributions to this repository are licensed under the [MIT License](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream component licenses.
