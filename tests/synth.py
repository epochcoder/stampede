"""Synthetic flatbed scans with exact ground truth.

A print is made from a sample photo (OpenCV's sample images, downloaded once into tests/.cache),
optionally with an orange camera date stamp drawn in the bottom right of the camera's landscape
frame (7-segment or dot-matrix), then laid on a noisy white A4 bed: rotated 0/90/180/270, tilted
by a fraction of a degree, with a lid shadow/highlight around it, a soft edge, an optional lighter
paper rim, saved as JPEG. Truth: where the print is, how to rotate it back, the stamp text.

Set STAMPEDE_TEST_SAMPLES to a folder that already holds the sample images to skip the download."""
import io
import math
import os
import random
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SAMPLE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
SAMPLES = ["messi5.jpg", "baboon.jpg", "fruits.jpg", "building.jpg", "home.jpg", "aero3.jpg",
           "butterfly.jpg", "apple.jpg"]
CACHE = Path(__file__).resolve().parent / ".cache" / "samples"

SEG = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc",
       "7": "abc", "8": "abcdefg", "9": "abcdfg"}


def sample(name):
    """Path of an OpenCV sample image (downloaded once)."""
    env = os.environ.get("STAMPEDE_TEST_SAMPLES")
    if env and (Path(env) / name).exists():
        return Path(env) / name
    p = CACHE / name
    if not p.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        # explicit proxies: tests may set sys.platform = "darwin", where urllib's own lookup needs macOS
        proxies = {s: os.environ.get(f"{s}_proxy") or os.environ.get(f"{s.upper()}_PROXY") for s in ("http", "https")}
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({k: v for k, v in proxies.items() if v}))
        with opener.open(SAMPLE_URL + name, timeout=60) as resp:
            tmp.write_bytes(resp.read())
        os.replace(tmp, p)
    return p


def _fit(src, w, h):
    im = Image.open(sample(src)).convert("RGB")
    s = max(w / im.width, h / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    left, top = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((left, top, left + w, top + h))


def draw_stamp(img, text, dot=False):
    """Glowing orange 7-segment (or dot-matrix) date in the bottom right of a landscape frame."""
    W, H = img.size
    h = int(H * 0.035)
    w, t, gap = int(h * 0.55), max(2, int(h * 0.13)), int(h * 0.35)
    total = sum(w + gap if c.isdigit() else (gap * 2 if c == " " else gap) for c in text)
    x, y = W - int(W * 0.05) - total, H - int(H * 0.06) - h
    layer = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(layer)
    for c in text:
        if c == " ":
            x += gap * 2
            continue
        if c in "'/":
            d.line([(x + gap // 2, y), (x + gap // 3, y + (h if c == "/" else h // 4))], fill=255, width=t)
            x += gap
            continue
        segs = {"a": [(x, y), (x + w, y)], "b": [(x + w, y), (x + w, y + h // 2)],
                "c": [(x + w, y + h // 2), (x + w, y + h)], "d": [(x, y + h), (x + w, y + h)],
                "e": [(x, y + h // 2), (x, y + h)], "f": [(x, y), (x, y + h // 2)],
                "g": [(x, y + h // 2), (x + w, y + h // 2)]}
        for s in SEG[c]:
            (x0, y0), (x1, y1) = segs[s]
            if dot:
                for k in range(5):
                    px, py = x0 + (x1 - x0) * k / 4, y0 + (y1 - y0) * k / 4
                    d.ellipse([px - t * .6, py - t * .6, px + t * .6, py + t * .6], fill=255)
            else:  # segment ends shortened: separate segments, like a real LCD date back
                sx, sy = (x1 - x0) * 0.12, (y1 - y0) * 0.12
                d.line([(x0 + sx, y0 + sy), (x1 - sx, y1 - sy)], fill=255, width=t)
        x += w + gap
    a = np.asarray(img).astype(np.float32)
    glow = np.asarray(layer.filter(ImageFilter.GaussianBlur(h * 0.08)), np.float32)[..., None] / 255
    a = a * (1 - glow * 0.9) + np.array([255, 140, 40], np.float32) * glow * 0.9
    core = np.asarray(layer.filter(ImageFilter.GaussianBlur(1)), np.float32)[..., None] / 255
    a = a * (1 - core * 0.5) + np.array([255, 200, 120], np.float32) * core * 0.5
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def make_print(src, portrait=False, stamp="", dot=False, kind="plain", rim=0, dpi=300, seed=0):
    """The print as it should end up (content upright). kind: plain | sky (white sky into the top
    edge) | dark | whitewall (bright colourless area into the right edge) | orange (orange wood/
    tartan stripes away from the stamp corner)."""
    r = random.Random(seed)
    cw, ch = int(15 / 2.54 * dpi), int(10 / 2.54 * dpi)  # 10x15 cm print
    if portrait:
        cw, ch = ch, cw
    content = _fit(src, cw, ch)
    a = np.asarray(content).astype(np.float32)
    if kind == "sky":
        ramp = np.clip(1 - np.arange(ch) / (ch * 0.3), 0, 1)[:, None, None]
        a = a * (1 - ramp) + 250 * ramp
    elif kind == "dark":
        a = a * 0.25
    elif kind == "whitewall":
        yy = np.arange(ch)
        edge = (cw * (0.75 + 0.06 * np.sin(yy / ch * 9 * r.random() + r.random() * 6))).astype(int)
        for y_, e_ in zip(yy, edge):
            a[y_, e_:] = 243
    elif kind == "orange":  # thin bright orange lines (wood edges, tartan) where no camera prints
        for i in range(6):
            y = int(ch * (0.25 + 0.08 * i))
            a[y:y + max(2, ch // 150), int(cw * 0.05):int(cw * 0.55)] = (235, 120, 40)
        for i in range(4):
            x = int(cw * (0.1 + 0.1 * i))
            a[int(ch * 0.6):int(ch * 0.95), x:x + max(2, cw // 200)] = (240, 140, 50)
    content = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    # the camera's frame is landscape; a portrait shot is the content turned clockwise
    frame = content.rotate(-90, expand=True) if portrait else content
    if stamp:
        frame = draw_stamp(frame, stamp, dot)
    out = frame.rotate(90, expand=True) if portrait else frame
    if rim:  # the cut edge of the paper, slightly lighter
        b = np.asarray(out).copy()
        b[:rim] = b[-rim:] = 235
        b[:, :rim] = b[:, -rim:] = 235
        out = Image.fromarray(b)
    return out


def make_scan(path, print_img, placed_rot=0, tilt=0.0, dpi=300, seed=0, quality=88):
    """Lay the print on an A4 bed (clockwise placed_rot, then tilt degrees), save JPEG to path.
    Returns truth: box (axis-aligned bounds of the print), undo_rot (rotation that makes it upright),
    inprint (bool mask), size (print w, h as it lies on the bed, before the tilt), tilt."""
    r = random.Random(seed)
    im = print_img.rotate(-placed_rot, expand=True)
    pw, ph = im.size
    mask = Image.new("L", im.size, 255)
    if tilt:
        im = im.rotate(tilt, resample=Image.BICUBIC, expand=True)
        mask = mask.rotate(tilt, resample=Image.BILINEAR, expand=True)
    BW, BH = int(21.6 / 2.54 * dpi), int(29.7 / 2.54 * dpi)
    bed = Image.fromarray(np.clip(np.random.default_rng(seed).normal(249, 1.5, (BH, BW, 3)), 0, 255)
                          .astype(np.uint8))
    x = r.randint(dpi // 4, BW - im.width - dpi // 4)
    y = r.randint(dpi // 4, BH - im.height - dpi // 4)
    full = Image.new("L", bed.size, 0)
    full.paste(mask, (x, y))
    # lid shadow (bottom/right) and highlight (top/left) just outside the print, like a CCD flatbed
    shadow = np.asarray(full.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.5)),
                        np.float32) / 255
    hi = np.roll(np.roll(shadow, -3, 0), -3, 1)
    lo = np.roll(np.roll(shadow, 3, 0), 3, 1)
    b = np.asarray(bed, np.float32) * (1 - 0.25 * lo[..., None])
    b = np.minimum(255, b + 6 * hi[..., None])
    bed = Image.fromarray(b.astype(np.uint8))
    soft = full.filter(ImageFilter.GaussianBlur(0.8))  # optics: an edge is never a hard step
    canvas = bed.copy()
    canvas.paste(im, (x, y), mask)
    out = Image.composite(canvas, bed, soft)
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=quality)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(buf.getvalue())
    inprint = np.asarray(full) > 127
    ys, xs = np.nonzero(inprint)
    return dict(box=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                undo_rot=(360 - placed_rot) % 360, inprint=inprint, size=(pw, ph), tilt=tilt)


# the standard set used by the scan tests: (name, sample, portrait, stamp, placed clockwise, dot, kind)
STANDARD = [
    ("01_land_up", "messi5.jpg", False, "'87 6 14", 0, False, "plain"),
    ("02_land_180", "fruits.jpg", False, "'92 11 3", 180, False, "plain"),
    ("03_port_up", "butterfly.jpg", True, "'89 12 24", 0, False, "plain"),
    ("04_port_up_dmy", "messi5.jpg", True, "14 6 '87", 0, False, "plain"),
    ("05_land_90", "building.jpg", False, "'03 5 22", 90, False, "plain"),
    ("06_land_270_dot", "aero3.jpg", False, "'95 7 30", 270, True, "plain"),
    ("07_nostamp_90", "baboon.jpg", False, "", 90, False, "plain"),
    ("08_nostamp", "fruits.jpg", False, "", 0, False, "plain"),
    ("09_bright_sky", "home.jpg", False, "'85 8 2", 0, False, "sky"),
    ("10_noapos", "baboon.jpg", False, "88 4 9", 0, False, "plain"),
    ("11_port_up", "home.jpg", True, "'98 1 17", 0, False, "plain"),
    ("12_orange_nostamp", "apple.jpg", True, "", 0, False, "orange"),
]
STAMP_TO_DATE = {"'87 6 14": "1987-06-14", "'92 11 3": "1992-11-03", "'89 12 24": "1989-12-24",
               "14 6 '87": "1987-06-14", "'03 5 22": "2003-05-22", "'95 7 30": "1995-07-30", "": "",
               "'85 8 2": "1985-08-02", "88 4 9": "1988-04-09", "'98 1 17": "1998-01-17"}


def standard_set(folder, seed=4):
    """Write the standard scans into folder; returns {name: truth}."""
    truth = {}
    for i, (name, src, portrait, stamp, placed, dot, kind) in enumerate(STANDARD):
        pr = make_print(src, portrait, stamp, dot, kind, seed=seed + i)
        t = make_scan(Path(folder) / f"{name}.jpg", pr, placed, tilt=0.0, seed=seed + i)
        t.update(stamp=stamp, date=STAMP_TO_DATE[stamp])
        t.pop("inprint")
        truth[name] = t
    return truth


def ideal_trim(size, tilt):
    """Per-side shrink that a tilted print makes unavoidable for an axis-aligned crop."""
    pw, ph = size
    t = math.radians(abs(tilt))
    return pw * math.cos(t) - ph * math.sin(t), ph * math.cos(t) - pw * math.sin(t)
