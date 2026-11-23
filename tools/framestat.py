#!/usr/bin/env python3
"""Numeric frame statistics and reference comparison for the native port.

One frame: per-channel mean/min/max, the fraction of pixels that are black
(<4/255 on all three channels), the fraction that is the magenta clear colour,
and a 4x4 grid of region mean luminances. Two frames: mean absolute difference
per channel, the share of channels more than 8/255 apart, and the per-region
luminance delta grid, optionally after cropping a letterbox off either side.

Reads binary/ASCII PPM, 24/32-bit BMP and 8-bit PNG with the standard library
only (PNG through zlib); Pillow, when installed, is used only for a PNG variant
this decoder does not cover (16-bit, palette, interlaced).

Usage:
  framestat.py FRAME [--json] [--min-mean L] [--max-dark P] [--max-magenta P]
  framestat.py FRAME REFERENCE [--crop X,Y,W,H | --frame-crop ... --ref-crop ...]
                              [--max-mean-diff D] [--max-off8 P] [--max-region-d]
  framestat.py FRAME:REFERENCE        (the older m1e/frame-stat.py spelling)

  FRAME/REFERENCE      .ppm, .bmp or .png, anywhere on either host
  --json               print one JSON object on stdout, the human line on stderr
  --min-mean L         fail unless the frame's mean luminance is >= L
  --max-dark P         fail if more than P percent of pixels are <4/255
  --max-magenta P      fail if more than P percent of pixels are clear-magenta
  --crop X,Y,W,H       crop both images to that rectangle before anything else
  --frame-crop / --ref-crop   the same rectangle for one side only, or 'auto' to
                       trim a letterbox off that image: border rows and columns
                       whose mean level is below --auto-level are dropped
  --max-mean-diff D    compare: fail if the mean abs difference (all channels)
                       is > D, default 2.0 (PLAN-native.md section 3)
  --max-off8 P         compare: fail if more than P percent of channels are
                       more than 8/255 apart (default 10.0)
  --max-region-d L     compare: fail if any 4x4 region's mean luminance is more
                       than L away from the reference's (default 8.0)

Exit status: 0 the measurement passed its thresholds, 1 it failed, 2 bad input.
"""

import array
import json
import struct
import sys
import zlib

DARK_LEVEL = 4          # a pixel is "dark" when all three channels are below this
DIFF_LEVEL = 8          # a channel "differs" from the reference above this
MAGENTA_MIN = 200       # r and b at least this, g at most MAGENTA_MAX_LOW
MAGENTA_MAX_LOW = 60
GRID = 4                # region grid, GRID x GRID
REGION_SAMPLE = 4       # region means sample every Nth pixel on both axes


class Image:
    __slots__ = ("width", "height", "px")

    def __init__(self, width, height, px):
        self.width = width
        self.height = height
        self.px = px            # bytes/bytearray, RGB, row-major, top row first


# --------------------------------------------------------------------------
# Decoders. Each returns an Image with tightly packed RGB bytes.
# --------------------------------------------------------------------------

def _ppm(data, path):
    if not data.startswith(b"P"):
        raise ValueError("%s: not a PPM" % path)
    kind = data[1:2]
    fields = []
    i = 2
    while len(fields) < 3:
        while i < len(data) and data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b"#":
            while i < len(data) and data[i:i + 1] not in (b"\n", b"\r"):
                i += 1
            continue
        start = i
        while i < len(data) and not data[i:i + 1].isspace():
            i += 1
        fields.append(int(data[start:i]))
    i += 1                                  # single whitespace after maxval
    width, height, maxval = fields
    if maxval != 255:
        raise ValueError("%s: %d levels per channel, only 255 supported" % (path, maxval))
    if kind == b"6":
        px = data[i:i + width * height * 3]
        if len(px) < width * height * 3:
            raise ValueError("%s: short PPM body" % path)
        return Image(width, height, px)
    if kind == b"3":
        raw = array.array("B", (int(v) for v in data[i:].split()))
        return Image(width, height, raw.tobytes())
    raise ValueError("%s: PPM variant %s" % (path, kind.decode("latin-1")))


def _bmp(data, path):
    if data[:2] != b"BM":
        raise ValueError("%s: not a BMP" % path)
    offset = struct.unpack_from("<I", data, 10)[0]
    header = struct.unpack_from("<I", data, 14)[0]
    if header < 40:
        raise ValueError("%s: BMP header %d, only BITMAPINFOHEADER supported" % (path, header))
    width, height = struct.unpack_from("<ii", data, 18)
    bpp = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]
    if compression != 0:
        raise ValueError("%s: compressed BMP (mode %d)" % (path, compression))
    if bpp not in (24, 32):
        raise ValueError("%s: %d bpp BMP" % (path, bpp))
    top_down = height < 0
    height = abs(height)
    stride = ((width * bpp // 8) + 3) & ~3
    step = bpp // 8
    px = bytearray(width * height * 3)
    for y in range(height):
        src = offset + (y if top_down else height - 1 - y) * stride
        row = data[src:src + stride]
        if step == 3:
            # BMP stores BGR; every other path here is RGB.
            out = px[y * width * 3:(y + 1) * width * 3]
            out[0::3] = row[2:width * 3:3]
            out[1::3] = row[1:width * 3:3]
            out[2::3] = row[0:width * 3:3]
            px[y * width * 3:(y + 1) * width * 3] = out
        else:
            out = px[y * width * 3:(y + 1) * width * 3]
            out[0::3] = row[2:width * 4:4]
            out[1::3] = row[1:width * 4:4]
            out[2::3] = row[0:width * 4:4]
            px[y * width * 3:(y + 1) * width * 3] = out
    return Image(width, height, px)


def _png(data, path):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("%s: not a PNG" % path)
    pos = 8
    width = height = depth = colour = interlace = None
    idat = bytearray()
    while pos + 8 <= len(data):
        length, kind = struct.unpack_from(">I4s", data, pos)
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, depth, colour, _comp, _filt, interlace = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
    if interlace:
        raise ValueError("%s: interlaced PNG" % path)
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(colour)
    if channels is None:
        raise ValueError("%s: PNG colour type %d" % (path, colour))
    if depth != 8 or colour in (3,):
        raise ValueError("%s: PNG depth %d colour %d" % (path, depth, colour))
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    out = bytearray(width * height * channels)
    prev = bytearray(stride)
    pos = 0
    for y in range(height):
        filter_type = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        pos += stride
        if filter_type == 1:                        # Sub
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif filter_type == 2:                      # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif filter_type == 3:                      # Average
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif filter_type == 4:                      # Paeth
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                p = a + b - c
                pa = abs(p - a)
                pb = abs(p - b)
                pc = abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 0xFF
        elif filter_type != 0:
            raise ValueError("%s: PNG row filter %d" % (path, filter_type))
        out[y * stride:(y + 1) * stride] = line
        prev = line
    if channels == 3:
        return Image(width, height, out)
    px = bytearray(width * height * 3)
    if channels == 4:
        px[0::3] = out[0::4]
        px[1::3] = out[1::4]
        px[2::3] = out[2::4]
    elif channels == 2:
        px[0::3] = out[0::2]
        px[1::3] = out[0::2]
        px[2::3] = out[0::2]
    else:
        px[0::3] = out
        px[1::3] = out
        px[2::3] = out
    return Image(width, height, px)


def load(path):
    with open(path, "rb") as f:
        data = f.read()
    lower = path.lower()
    if lower.endswith(".ppm") or lower.endswith(".pgm") or data[:2] in (b"P6", b"P3"):
        return _ppm(data, path)
    if lower.endswith(".bmp") or data[:2] == b"BM":
        return _bmp(data, path)
    if lower.endswith(".png") or data[:8] == b"\x89PNG\r\n\x1a\n":
        try:
            return _png(data, path)
        except ValueError:
            # Pillow covers the PNG variants (16-bit, palette, interlaced) and
            # any other container a capture tool produced. Optional, never
            # required.
            try:
                from PIL import Image as PILImage
            except ImportError:
                raise
            with PILImage.open(path) as im:
                rgb = im.convert("RGB")
                return Image(rgb.width, rgb.height, rgb.tobytes())
    raise ValueError("%s: unknown image type" % path)


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------

def crop(image, box):
    x, y, w, h = box
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image.width or y + h > image.height:
        raise ValueError("crop %d,%d,%dx%d outside a %dx%d image"
                         % (x, y, w, h, image.width, image.height))
    out = bytearray(w * h * 3)
    for row in range(h):
        src = ((y + row) * image.width + x) * 3
        out[row * w * 3:(row + 1) * w * 3] = image.px[src:src + w * 3]
    return Image(w, h, out)


def auto_crop(image, level=2.0):
    """Trim border rows and columns whose mean luma is below `level` (of 255).

    Less than an exact zero because the Windows capture's letterbox bars carry
    faint dither noise (max 6-10/255, row mean 0.2). A real frame row is a
    couple of levels brighter even in the native build's darkest place.
    """
    w, h, px = image.width, image.height, image.px
    stride = w * 3

    def row_mean(y):
        return sum(px[y * stride:(y + 1) * stride]) / float(stride)

    def col_mean(x, top, bottom):
        total = 0
        for y in range(top, bottom):
            o = y * stride + x * 3
            total += px[o] + px[o + 1] + px[o + 2]
        return total / (3.0 * (bottom - top))

    top = 0
    while top < h and row_mean(top) < level:
        top += 1
    bottom = h
    while bottom > top and row_mean(bottom - 1) < level:
        bottom -= 1
    left = 0
    while left < w and col_mean(left, top, bottom) < level:
        left += 1
    right = w
    while right > left and col_mean(right - 1, top, bottom) < level:
        right -= 1
    if (left, top, right - left, bottom - top) == (0, 0, w, h):
        return image
    return crop(image, (left, top, right - left, bottom - top))


def _presence_mask(channel, accept):
    table = bytes(1 if accept(v) else 0 for v in range(256))
    return int.from_bytes(bytes(channel).translate(table), "big")


def regions_luma(image):
    """GRID x GRID region mean luminance (Rec.601), every REGION_SAMPLE-th pixel."""
    w, h, px = image.width, image.height, image.px
    rows = []
    for ry in range(GRID):
        row = []
        y0, y1 = ry * h // GRID, (ry + 1) * h // GRID
        for rx in range(GRID):
            x0, x1 = rx * w // GRID, (rx + 1) * w // GRID
            total = 0
            count = 0
            for y in range(y0, y1, REGION_SAMPLE):
                base = (y * w) * 3
                for x in range(x0, x1, REGION_SAMPLE):
                    o = base + x * 3
                    total += 299 * px[o] + 587 * px[o + 1] + 114 * px[o + 2]
                    count += 1
            row.append(round(total / (1000.0 * count), 1) if count else 0.0)
        rows.append(row)
    return rows


def frame_stats(image, path):
    px = image.px
    n = image.width * image.height
    red, green, blue = px[0::3], px[1::3], px[2::3]
    mean = [round(sum(red) / (255.0 * n), 3), round(sum(green) / (255.0 * n), 3),
            round(sum(blue) / (255.0 * n), 3)]
    lo = [min(red), min(green), min(blue)]
    hi = [max(red), max(green), max(blue)]
    dark = _presence_mask(red, lambda v: v < DARK_LEVEL)
    dark &= _presence_mask(green, lambda v: v < DARK_LEVEL)
    dark &= _presence_mask(blue, lambda v: v < DARK_LEVEL)
    magenta = _presence_mask(red, lambda v: v >= MAGENTA_MIN)
    magenta &= _presence_mask(green, lambda v: v <= MAGENTA_MAX_LOW)
    magenta &= _presence_mask(blue, lambda v: v >= MAGENTA_MIN)
    regions = regions_luma(image)
    luma = round(sum(row[i] for row in regions for i in range(GRID)) / (GRID * GRID), 2)
    return {
        "file": path,
        "width": image.width,
        "height": image.height,
        "mean": mean,
        "min": lo,
        "max": hi,
        "luma_mean": luma,
        "dark_fraction": round(dark.bit_count() / float(n), 4),
        "magenta_fraction": round(magenta.bit_count() / float(n), 4),
        "regions_luma": regions,
    }


def compare_stats(frame, reference, frame_path, ref_path):
    if (frame.width, frame.height) != (reference.width, reference.height):
        raise ValueError("%s is %dx%d but %s is %dx%d after cropping"
                         % (frame_path, frame.width, frame.height,
                            ref_path, reference.width, reference.height))
    a, b = frame.px, reference.px
    n = frame.width * frame.height
    sums = [0, 0, 0]
    over = 0
    for r1, g1, b1, r2, g2, b2 in zip(a[0::3], a[1::3], a[2::3],
                                      b[0::3], b[1::3], b[2::3]):
        d = r1 - r2
        sums[0] += d if d >= 0 else -d
        if d > DIFF_LEVEL or d < -DIFF_LEVEL:
            over += 1
        d = g1 - g2
        sums[1] += d if d >= 0 else -d
        if d > DIFF_LEVEL or d < -DIFF_LEVEL:
            over += 1
        d = b1 - b2
        sums[2] += d if d >= 0 else -d
        if d > DIFF_LEVEL or d < -DIFF_LEVEL:
            over += 1
    a_regions = regions_luma(frame)
    b_regions = regions_luma(reference)
    delta = [[round(a_regions[r][c] - b_regions[r][c], 1) for c in range(GRID)]
             for r in range(GRID)]
    worst = max(abs(v) for row in delta for v in row)
    return {
        "frame": frame_path,
        "reference": ref_path,
        "width": frame.width,
        "height": frame.height,
        "mean_abs_diff": [round(s / float(n), 3) for s in sums],
        "mean_abs_diff_all": round(sum(sums) / (3.0 * n), 3),
        "off8_fraction": round(over / (3.0 * n), 4),
        "regions_luma_frame": a_regions,
        "regions_luma_reference": b_regions,
        "regions_delta": delta,
        "max_region_delta": round(worst, 1),
    }


# --------------------------------------------------------------------------
# Output and thresholds
# --------------------------------------------------------------------------

def grid_lines(rows):
    return ["    " + " ".join("%6.1f" % v for v in row) for row in rows]


def stats_line(stats):
    return ("%s %dx%d mean %.1f/%.1f/%.1f (of 255) dark %.1f%% magenta %.1f%% luma %.1f"
            % (stats["file"], stats["width"], stats["height"],
               255 * stats["mean"][0], 255 * stats["mean"][1], 255 * stats["mean"][2],
               100 * stats["dark_fraction"], 100 * stats["magenta_fraction"],
               stats["luma_mean"]))


def compare_line(cmp, limits):
    return ("%s vs %s %dx%d: mean abs diff %.2f (limit %.2f), "
            "channels >%d apart %.2f%% (limit %.2f%%), worst region delta %.1f (limit %.1f)"
            % (cmp["frame"], cmp["reference"], cmp["width"], cmp["height"],
               cmp["mean_abs_diff_all"], limits["max_mean_diff"], DIFF_LEVEL,
               100 * cmp["off8_fraction"], limits["max_off8"], cmp["max_region_delta"],
               limits["max_region_delta"]))


def check_stats(stats, args):
    failures = []
    if args.min_mean is not None:
        luma = 255.0 * (0.299 * stats["mean"][0] + 0.587 * stats["mean"][1]
                        + 0.114 * stats["mean"][2])
        if luma < args.min_mean:
            failures.append("mean luminance %.1f < %.1f" % (luma, args.min_mean))
    if args.max_dark is not None and 100 * stats["dark_fraction"] > args.max_dark:
        failures.append("dark %.1f%% > %.1f%%" % (100 * stats["dark_fraction"], args.max_dark))
    if args.max_magenta is not None and 100 * stats["magenta_fraction"] > args.max_magenta:
        failures.append("magenta %.1f%% > %.1f%%"
                        % (100 * stats["magenta_fraction"], args.max_magenta))
    return failures


def check_compare(cmp, args):
    failures = []
    if cmp["mean_abs_diff_all"] > args.max_mean_diff:
        failures.append("mean abs diff %.2f > %.2f"
                        % (cmp["mean_abs_diff_all"], args.max_mean_diff))
    if 100 * cmp["off8_fraction"] > args.max_off8:
        failures.append("channels off >%d %.2f%% > %.2f%%"
                        % (DIFF_LEVEL, 100 * cmp["off8_fraction"], args.max_off8))
    if cmp["max_region_delta"] > args.max_region_delta:
        failures.append("worst region delta %.1f > %.1f"
                        % (cmp["max_region_delta"], args.max_region_delta))
    return failures


def parse_crop(text):
    if text.lower() == "auto":
        return "auto"
    parts = [p for p in text.replace("x", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("crop '%s' is not X,Y,W,H" % text)
    return tuple(int(p) for p in parts)


def main(argv):
    args = {"json": False, "min_mean": None, "max_dark": None, "max_magenta": None,
            "crop": None, "frame_crop": None, "ref_crop": None, "auto_level": 2.0,
            "max_mean_diff": 2.0, "max_off8": 10.0, "max_region_delta": 8.0};
    positional = []
    i = 0
    takes_value = {"--min-mean": "min_mean", "--max-dark": "max_dark",
                   "--max-magenta": "max_magenta", "--crop": "crop",
                   "--frame-crop": "frame_crop", "--ref-crop": "ref_crop",
                   "--auto-level": "auto_level",
                   "--max-mean-diff": "max_mean_diff", "--max-off8": "max_off8",
                   "--max-region-d": "max_region_delta"}
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            print(__doc__.strip("\n"))
            return 0
        if arg == "--json":
            args["json"] = True
            i += 1
            continue
        key = takes_value.get(arg)
        if key is None and arg.startswith("-"):
            sys.stderr.write("framestat: unknown option %s\n" % arg)
            return 2
        if key is not None:
            if i + 1 >= len(argv):
                sys.stderr.write("framestat: %s needs a value\n" % arg)
                return 2
            value = argv[i + 1]
            try:
                args[key] = value if key.endswith("crop") else float(value)
            except ValueError:
                sys.stderr.write("framestat: %s wants a number, not '%s'\n" % (arg, value))
                return 2
            i += 2
            continue
        positional.append(arg)
        i += 1

    if not positional:
        sys.stderr.write(__doc__.strip("\n") + "\n")
        return 2
    if len(positional) == 1 and ":" in positional[0]:
        positional = positional[0].split(":", 1)
    if len(positional) > 2:
        sys.stderr.write("framestat: one frame, or a frame and a reference\n")
        return 2

    try:
        for key in ("crop", "frame_crop", "ref_crop"):
            if isinstance(args[key], str):
                args[key] = parse_crop(args[key])
        frame_path = positional[0]
        frame = load(frame_path)
        ref_path = positional[1] if len(positional) == 2 else None
        reference = load(ref_path) if ref_path else None
        if args["crop"]:
            frame = crop(frame, args["crop"])
            if reference is not None:
                reference = crop(reference, args["crop"])
        for side, image in (("frame", frame), ("ref", reference)):
            how = args[side + "_crop"]
            if image is None or not how:
                continue
            trimmed = auto_crop(image, args["auto_level"]) if how == "auto" else crop(image, how)
            if side == "frame":
                frame = trimmed
            else:
                reference = trimmed

        limits = {"max_mean_diff": args["max_mean_diff"], "max_off8": args["max_off8"],
                  "max_region_delta": args["max_region_delta"]}
        if reference is None:
            stats = frame_stats(frame, frame_path)
            failures = check_stats(stats, _Obj(args))
            report = {"mode": "stats", "path": frame_path, "pass": not failures,
                      "failures": failures, "stats": stats}
            human = [stats_line(stats)] + grid_lines(stats["regions_luma"]) + \
                    ["    regions are %dx%d mean luminance, top row first" % (GRID, GRID)]
        else:
            cmp = compare_stats(frame, reference, frame_path, ref_path)
            failures = check_compare(cmp, _Obj(args))
            report = {"mode": "compare", "path": frame_path, "reference": ref_path,
                      "pass": not failures, "failures": failures, "compare": cmp}
            human = [compare_line(cmp, limits)] + grid_lines(cmp["regions_delta"]) + \
                    ["    regions are %dx%d luminance deltas (frame - reference)"
                     % (GRID, GRID)]
        human.append("    " + ("PASS" if not failures else "FAIL: " + "; ".join(failures)))
        if args["json"]:
            print(json.dumps(report, sort_keys=True))
            for line in human:
                sys.stderr.write(line + "\n")
        else:
            for line in human:
                print(line)
        return 0 if not failures else 1
    except (OSError, ValueError) as exc:
        sys.stderr.write("framestat: %s\n" % exc)
        return 2


class _Obj:
    def __init__(self, mapping):
        self.__dict__.update(mapping)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))