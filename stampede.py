#!/usr/bin/env python3
"""
Stampede: crop, rotate and date scanned photo prints from their camera date stamps, locally on an
Apple Silicon Mac, ready for Immich.

  1. python stampede.py scan ~/Pictures/Scans
       Analyses every scan (subfolders included), saves the results to ~/Pictures/Scans-review/photos.csv
       and opens the review page in your browser: every photo exactly as it will be written. Fix dates,
       rotate, mark OK or skip; each change is saved to photos.csv immediately. Ctrl+C when done.
       Run scan again any time: it only analyses new (or previously failed) scans, then opens the page.
       To reopen the page without scanning: python stampede.py review ~/Pictures/Scans

  2. python stampede.py apply ~/Pictures/Scans --out ~/Pictures/Dated --dry-run
     python stampede.py apply ~/Pictures/Scans --out ~/Pictures/Dated
       Writes cropped, rotated, dated copies exactly as photos.csv says. Scans are never modified.
       Photos still flagged for review are left out unless you add --include-flagged.

Dates: a date is filled in automatically only when Apple Vision and the Qwen model read the same date.
Everything else is a guess on the review page, which you accept per photo or in bulk; guesses use what
the folder shows (e.g. 25/12/2006 proves the camera prints day/month/year) and neighbouring scans.
Vision runs on 4 photos at a time; the model reads only the small stamp crop. Undated photos get a year
from their folder name ("1987", "Summer 1987") or --fallback-date, and an Immich tag
"Scanned prints/Date estimated". Your edits on the review page are never overwritten by a later scan.

Setup:  brew install exiftool
        pip install opencv-python-headless pillow numpy pyobjc-framework-Vision mlx-vlm
"""
import argparse
import contextlib
import csv
import fcntl
import io
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
FIELDS = ["rel", "date", "rotate", "status", "date_guess", "flags", "orient_src", "ocr_raw", "ocr_conf",
          "ocr_backend", "crop", "source", "src_sig", "edited", "evidence"]
DATE_FLAGS = ("no-date", "stamp-unread", "date-unconfirmed", "check-date-order", "readers-disagree", "ambiguous",
              "date-order-learned", "date-out-of-sequence", "date-before-rescan")
YEARS = "1975-2008"  # plausible years for these prints; dates outside are rejected everywhere
WORK = 2000  # long side of the working copy used for analysis
# Bump when pass 1 changes in a way that makes old results wrong: scan re-analyses those rows
# (except photos you've settled). 2 = edge-fitted crop + stamp only in the camera's date position.
ANALYSIS_VERSION = 2


# ======================================================================= image helpers
def load(path):
    """RGB image (EXIF orientation applied) + an ICC profile that matches it (or None = sRGB)."""
    img = Image.open(path)
    icc = img.info.get("icc_profile")
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I"):
        # 16-bit greyscale: Pillow's convert() clips to white; scale down to 8 bits instead
        a = np.asarray(img).astype(np.uint32)
        img = Image.fromarray((a >> 8 if a.max() > 255 else a).clip(0, 255).astype(np.uint8), "L")
    if icc:
        try:
            from PIL import ImageCms
            prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            space = prof.profile.xcolor_space.strip()
            if space != "RGB" or img.mode != "RGB":
                # CMYK/grey/Lab profile: convert the pixels to sRGB, don't stamp that profile on RGB output
                src = img if img.mode in ("RGB", "CMYK", "L", "LAB") else img.convert("RGB")
                img = ImageCms.profileToProfile(src, prof, ImageCms.createProfile("sRGB"), outputMode="RGB")
                icc = None
        except Exception:
            icc = None  # unreadable/mismatched profile: plain conversion below, output tagged as sRGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img, icc


def oriented_size(path):
    """Image size after EXIF orientation, without decoding pixels."""
    with Image.open(path) as im:
        w, h = im.size
        o = im.getexif().get(0x0112, 1)
    return (h, w) if o in (5, 6, 7, 8) else (w, h)


def rot(a, deg):
    """Rotate numpy image clockwise by 0/90/180/270."""
    return {0: a, 90: cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE), 180: cv2.rotate(a, cv2.ROTATE_180),
            270: cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE)}[deg % 360]


def shrink(a, long_side):
    s = long_side / max(a.shape[:2])
    if s >= 1:
        return a, 1.0
    return cv2.resize(a, (round(a.shape[1] * s), round(a.shape[0] * s)), interpolation=cv2.INTER_AREA), s


def render(img, crop, rotate):
    """The exact transform used for both review thumbnails and final output."""
    x0, y0, x1, y1 = crop
    out = img.crop((x0, y0, x1, y1))
    if rotate:
        out = out.rotate(-rotate, expand=True)  # PIL rotates counter-clockwise
    return out


# ======================================================================= crop detection
STD_ASPECTS = [1.0, 1.25, 1.31, 1.33, 1.385, 1.43, 1.5, 1.78, 2.0, 3.0]


def fit_edges(rgb, coarse, bed):
    """Exact print edges at full resolution. `coarse` = (x0, y0, x1, y1) from the downscaled detection.
    For each side, ~80 lines across the edge are walked from the scanner bed inward to the first pixel
    that isn't bed (that's the print, or its shadow on the lid side); a straight line is fitted through
    those points robustly (dust, white sky or walls running into the edge are ignored as outliers).
    The crop then sits a small margin inside all four lines, so neither the shadow, the soft edge,
    the paper's cut rim nor the wedge of a slightly tilted print shows. Returns (box, flags)."""
    H, W = rgb.shape[:2]
    cx0, cy0, cx1, cy1 = coarse
    pw, ph = cx1 - cx0, cy1 - cy0
    margin = max(3, int(round(0.0045 * max(pw, ph))))  # ~0.7 mm on a 15 cm print
    out = int(0.05 * max(pw, ph))  # look this far outside the coarse edge (an edge it missed)
    inn = int(0.04 * min(pw, ph))  # and this far inside
    rng = np.random.default_rng(0)
    lines, flags = {}, []
    for side in ("top", "bottom", "left", "right"):
        along = (cx0, cx1) if side in ("top", "bottom") else (cy0, cy1)
        span = along[1] - along[0]
        pos = np.linspace(along[0] + 0.05 * span, along[1] - 0.05 * span, 80).astype(int)
        pts = []
        for q in pos:
            if side == "top":
                s0, s1 = max(0, int(cy0) - out), min(H, int(cy0) + inn)
                prof, start, step = rgb[s0:s1, q], s0, 1
            elif side == "bottom":
                s0, s1 = max(0, int(cy1) - inn), min(H, int(cy1) + out)
                prof, start, step = rgb[s0:s1, q][::-1], s1 - 1, -1
            elif side == "left":
                s0, s1 = max(0, int(cx0) - out), min(W, int(cx0) + inn)
                prof, start, step = rgb[q, s0:s1], s0, 1
            else:
                s0, s1 = max(0, int(cx1) - inn), min(W, int(cx1) + out)
                prof, start, step = rgb[q, s0:s1][::-1], s1 - 1, -1
            if len(prof) < 3:
                continue
            p16 = prof.astype(np.int16)
            off = (np.abs(p16.mean(axis=1) - bed) > 12) | ((p16.max(axis=1) - p16.min(axis=1)) > 25)
            hit = np.flatnonzero(off[:-1] & off[1:])  # two in a row: not a noise pixel
            if len(hit):
                pts.append((q, start + step * int(hit[0])))
        best = None
        if len(pts) >= 20:
            P = np.array(pts, np.float64)
            tol = max(1.5, 0.0008 * max(pw, ph))
            for _ in range(300):  # RANSAC: the line most edge points agree with
                i, j = rng.choice(len(P), 2, replace=False)
                if P[i, 0] == P[j, 0]:
                    continue
                k = (P[j, 1] - P[i, 1]) / (P[j, 0] - P[i, 0])
                if abs(k) > 0.05:  # more than ~3 degrees: not a print edge
                    continue
                inl = np.abs(P[:, 1] - (P[i, 1] + k * (P[:, 0] - P[i, 0]))) <= tol
                if best is None or inl.sum() > best.sum():
                    best = inl
        if best is not None and best.sum() >= 0.4 * len(pos) and np.ptp(P[best, 0]) >= 0.5 * span:
            k, c = np.polyfit(P[best, 0], P[best, 1], 1)
            lines[side] = (k, c)
        else:
            flags.append(f"edge-uncertain-{side}")  # fell back to the rough edge: check this side
    # the box inside all four lines (a tilted print's corners decide how far in each edge goes)
    def at(side, v):
        k, c = lines[side]
        return k * v + c
    ins = max(2, int(0.004 * min(pw, ph)))
    slope = float(np.median([abs(lines[k_][0]) for k_ in lines])) if lines else 0.0  # the print's tilt
    # a side whose edge couldn't be seen (white sky into white bed): rough edge, minus the tilt's wedge
    x0 = max(at("left", cy0), at("left", cy1)) + margin if "left" in lines else cx0 + ins + slope * ph
    x1 = min(at("right", cy0), at("right", cy1)) - margin if "right" in lines else cx1 - ins - slope * ph
    y0 = max(at("top", x0), at("top", x1)) + margin if "top" in lines else cy0 + ins + slope * pw
    y1 = min(at("bottom", x0), at("bottom", x1)) - margin if "bottom" in lines else cy1 - ins - slope * pw
    tilt = [math.degrees(math.atan(lines[sd_][0])) for sd_ in lines]
    if tilt and max(abs(t) for t in tilt) > 1.0:
        flags.append(f"print-tilted-{max(tilt, key=abs):.1f}deg")  # corners of the photo are lost
    box = (int(math.ceil(max(0, x0))), int(math.ceil(max(0, y0))), int(min(W, x1)), int(min(H, y1)))
    if box[2] - box[0] < 0.8 * pw or box[3] - box[1] < 0.8 * ph:  # implausible: keep the rough box
        return (int(cx0 + ins), int(cy0 + ins), int(cx1 - ins), int(cy1 - ins)), ["edge-uncertain"]
    return box, flags


def find_crop(rgb, bg_margin, inset_frac):
    """Axis-aligned bounding box of the print on a white flatbed. Returns (box, flags)."""
    H, W = rgb.shape[:2]
    small, s = shrink(rgb, 1500)
    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[..., 1]
    b = max(2, int(min(h, w) * 0.01))
    border = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
    bed = float(np.median(border))
    if bed < 200:
        return (0, 0, W, H), ["no-white-border"]
    thr = min(bed - bg_margin, 245)
    fg = ((cv2.GaussianBlur(gray, (5, 5), 0) < thr) | (sat > 40)).astype(np.uint8) * 255
    k = max(3, int(max(h, w) * 0.01)) | 1
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # dust/specks
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))  # bright patches inside photo
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    if not cnts or cv2.contourArea(cnts[0]) < 0.03 * h * w:
        return (0, 0, W, H), ["no-photo-found"]
    flags = []
    if len(cnts) > 1 and cv2.contourArea(cnts[1]) > 0.15 * cv2.contourArea(cnts[0]):
        flags.append("multiple-prints")
    x, y, bw, bh = cv2.boundingRect(cnts[0])
    if cv2.contourArea(cnts[0]) / (bw * bh) < 0.9:
        flags.append("irregular-edge")
    x0, y0, x1, y1 = x / s, y / s, (x + bw) / s, (y + bh) / s
    box, eflags = fit_edges(rgb, (x0, y0, x1, y1), bed)
    flags += eflags
    gfull = gray  # check strips just inside each detected edge (small-image coords)
    sx0, sy0, sx1, sy1 = (int(v * s) for v in box)
    strips = {"top": gfull[sy0:sy0 + 3, sx0:sx1], "bottom": gfull[sy1 - 3:sy1, sx0:sx1],
              "left": gfull[sy0:sy1, sx0:sx0 + 3], "right": gfull[sy0:sy1, sx1 - 3:sx1]}
    bright = [side for side, st in strips.items() if st.size and np.median(st) > thr - 10]
    ar = max(box[2] - box[0], box[3] - box[1]) / max(1, min(box[2] - box[0], box[3] - box[1]))
    if min(abs(ar - a) / a for a in STD_ASPECTS) > 0.025:
        flags.append(f"odd-aspect-{ar:.2f}")
    # near-white right at a cut: the photo may continue into what looked like scanner bed (white sky/wall)
    flags += [f"maybe-cut-{b}" for b in bright]
    return box, flags


# ======================================================================= date parsing
_MAP = str.maketrans({"O": "0", "D": "0", "Q": "0", "U": "0", "I": "1", "L": "1", "|": "1", "!": "1",
                      "]": "1", "[": "1", "Z": "2", "S": "5", "G": "6", "T": "7", "B": "8"})
_LOOKALIKE = set("ODQUILZSGTB|!][")


def _normalize(text):
    """Upper-case, unify apostrophes, and turn digit look-alikes (O->0, l->1, S->5 ...) into digits,
    but only inside chunks that look like digits: a chunk containing any other letter, or more
    letters than digits, is a word (AUG, SEP, JUL) and is dropped rather than turned into numbers."""
    t = unicodedata.normalize("NFKC", text).upper()
    t = re.sub(r"[`’‘´\"]", "'", t)

    def chunk(m):
        c = m.group(0)
        letters = [ch for ch in c if not ch.isdigit()]
        if any(ch not in _LOOKALIKE for ch in letters):
            return " "
        if len(c) >= 2 and len(letters) > sum(ch.isdigit() for ch in c):
            return " "
        return c.translate(_MAP)
    return re.sub(r"[A-Z0-9|!\[\]]+", chunk, t)


def _year(s):
    y = int(s)
    return y if len(s) == 4 else y + (1900 if y >= 50 else 2000)


def _try(vals, order, ymin, ymax):
    p = {c: i for i, c in enumerate(order)}
    try:
        ys = vals[p["y"]]
        if len(ys) not in (2, 4) or len(vals[p["m"]]) > 2 or len(vals[p["d"]]) > 2:
            return None
        d = date(_year(ys), int(vals[p["m"]]), int(vals[p["d"]]))
        return d if ymin <= d.year <= ymax else None
    except ValueError:
        return None


def parse_date(text, order, ymin, ymax, compact=True):
    """Lenient parser for camera date stamps. Returns (date|None, note).
    compact=False: don't read a 6-digit run as yymmdd (for whole-photo text: signs, phone numbers)."""
    if not text or "NONE" in text.upper():
        return None, ""
    t = _normalize(text)
    toks = []  # (year_marked, digits)
    for m in re.finditer(r"('?)(\d+)", t):
        mark, d = bool(m.group(1)), m.group(2)
        if len(d) == 3 and not mark and d[0] in "17":  # apostrophe misread as 1/7: '87 -> 187
            mark, d = True, d[1:]
        if len(d) == 4 and d[:2] in ("19", "20"):
            mark = True
        toks.append((mark, d))
    auto = order == "auto"
    orders = ["ymd", "dmy", "mdy"] if auto else [order] + [o for o in ("ymd", "dmy", "mdy") if o != order]

    def by_mark(tk):
        marked = [i for i, (m, _) in enumerate(tk) if m]
        if marked == [0]:
            return ["ymd"]
        if marked == [2]:
            return [order] if order in ("dmy", "mdy") else ["dmy", "mdy"]
        # no apostrophe: with --date-order given, trust it; on auto try every order and call
        # it ambiguous if more than one gives a real date (07 06 05 could be 2007 or 2005)
        return ["ymd", "dmy", "mdy"] if auto else [order]

    hits = set()
    if len(toks) == 3:  # strict: exactly three groups, anything else is not trusted
        for o in by_mark(toks):  # e.g. 5 6 '95 could be 5 June or May 6: both valid -> ambiguous
            d = _try([v for _, v in toks], o, ymin, ymax)
            if d:
                hits.add(d)
    elif compact and len(toks) == 1 and len(toks[0][1]) == 6:  # yymmdd printed without gaps
        v = toks[0][1]
        for o in (["ymd", "dmy", "mdy"] if auto else [order]):
            d = _try([v[i:i + 2] for i in (0, 2, 4)], o, ymin, ymax)
            if d:
                hits.add(d)
    if len(hits) == 1:
        return hits.pop(), ""
    if len(hits) > 1:
        return None, "ambiguous:" + "/".join(sorted(x.isoformat() for x in hits))
    # last resort: does another field order parse it? Report, but don't trust silently.
    if len(toks) == 3 and not auto:
        alt = {o: _try([v for _, v in toks], o, ymin, ymax) for o in orders[1:]}
        alt = {o: d for o, d in alt.items() if d}
        if alt:
            return None, "other-order?:" + ",".join(f"{o}={d}" for o, d in alt.items())
    return None, ""


def _fields(text, compact=True):
    """(orders the stamp's marks allow, three digit groups) or None."""
    if not text or "NONE" in text.upper():
        return None
    toks = []
    for m in re.finditer(r"('?)(\d+)", _normalize(text)):
        mark, d = bool(m.group(1)), m.group(2)
        if len(d) == 3 and not mark and d[0] in "17":
            mark, d = True, d[1:]
        if len(d) == 4 and d[:2] in ("19", "20"):
            mark = True
        toks.append((mark, d))
    if len(toks) == 3:
        marked = [i for i, (m, _) in enumerate(toks) if m]
        allowed = ["ymd"] if marked == [0] else ["dmy", "mdy"] if marked == [2] else ["ymd", "dmy", "mdy"]
        return allowed, [v for _, v in toks]
    if compact and len(toks) == 1 and len(toks[0][1]) == 6:
        return ["ymd", "dmy", "mdy"], [toks[0][1][i:i + 2] for i in (0, 2, 4)]
    return None


def date_options(text, ymin, ymax, compact=True):
    """{order: date} for every field order the stamp's marks allow that gives a real date, plus
    whether the reading was informative about the camera's order (more than one order allowed)."""
    f = _fields(text, compact)
    if not f:
        return {}, False
    allowed, vals = f
    opts = {o: d for o in allowed for d in [_try(vals, o, ymin, ymax)] if d}
    return opts, len(allowed) > 1


def dates_for(text, order, ymin, ymax, compact=True):
    """Possible dates of a reading given the camera's field order (None/auto = unknown)."""
    opts, _ = date_options(text, ymin, ymax, compact)
    if order and order != "auto":
        f = _fields(text, compact)
        if f and order in f[0]:
            # the camera's order applies; if it gives no real date, the reading is wrong, not the order
            return {opts[order]} if order in opts else set()
        # the marks rule this order out (e.g. '95 1 18 is year-first): the marks decide
    return set(opts.values())


# ======================================================================= stamp localisation
def stamp_mask(rgb, weak=False):
    """Thin, bright, orange strokes that stand out from their surroundings (white top-hat),
    so warm-toned photos (skin, sunsets, 80s colour casts) don't light up as a whole."""
    a = rgb.astype(np.int16)
    o = np.clip(a[..., 0] - a[..., 2], 0, 255).astype(np.uint8)  # R-B: orange-ness
    v = rgb.max(axis=2)
    k = max(5, int(min(rgb.shape[:2]) * 0.012)) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    th_o = cv2.morphologyEx(o, cv2.MORPH_TOPHAT, se)
    th_v = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, se)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue_ok = (hsv[..., 0] <= 30) | (hsv[..., 0] >= 170)
    if weak:
        m = hue_ok & (hsv[..., 1] >= 50) & (th_o >= 12) & (th_v >= 6)
    else:
        m = hue_ok & (hsv[..., 1] >= 70) & (th_o >= 25) & (th_v >= 15)
    return m.astype(np.uint8) * 255


STAMP_ZONE = dict(x=0.6, y=0.82, h=(0.02, 0.09))  # where camera date backs print: bottom right of the frame


def in_stamp_zone(cx, cy, W, H):
    """Centre (cx, cy) in a rotated frame W x H: is that where a camera prints its date? The camera's
    frame is always landscape (35 mm), and date backs print in its bottom-right corner. Orange
    things elsewhere (wood, clothes, tartan) are never treated as a stamp."""
    return W >= 0.95 * H and cx >= STAMP_ZONE["x"] * W and cy >= STAMP_ZONE["y"] * H


def stamp_candidates(work):
    """For each rotation, the best orange text-line blob in the stamp position (bottom right of the
    landscape frame). Returns [(score, rotation, (x0,y0,x1,y1) in rotated work coords)] best first."""
    mask0 = stamp_mask(work)
    out = []
    for r in (0, 90, 180, 270):
        m = rot(mask0, r)
        H, W = m.shape
        if W < 0.95 * H:
            continue  # portrait view: not the camera's frame
        top = int(H * 0.70)
        band = m[top:]
        kw = max(3, int(W * 0.03))
        merged = cv2.dilate(band, np.ones((max(3, int(H * 0.008)), kw), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(merged)
        best = None
        for i in range(1, n):
            x, y, w, h, _ = stats[i]
            if not (STAMP_ZONE["h"][0] * H <= h <= STAMP_ZONE["h"][1] * H and w >= 1.5 * h and w <= 0.6 * W):
                continue
            if not in_stamp_zone(x + w / 2, top + y + h / 2, W, H):
                continue
            sub = band[y:y + h, x:x + w]
            if not 0.03 <= sub.mean() / 255 <= 0.6:
                continue
            parts = cv2.connectedComponents(cv2.dilate(sub, np.ones((3, 3), np.uint8)))[0] - 1
            if parts < 2:  # a date is several glyphs/segments
                continue
            score = min(w / h, 8) * min(parts, 10) * (0.5 + (y + h) / band.shape[0])
            if not best or score > best[0]:
                best = (score, r, (x, y + top, x + w, y + top + h))
        if best:
            out.append(best)
    return sorted(out, reverse=True)


def line_crops(work_r, box):
    """Generous crop around a located stamp line. Returns (colour crop, OCR variants, binary masks, digit height)."""
    x0, y0, x1, y1 = box
    h = y1 - y0
    px, py = int(h * 2.5), int(h * 0.6)  # the strict mask often misses the ends of the line
    H, W = work_r.shape[:2]
    c = np.ascontiguousarray(work_r[max(0, y0 - py):min(H, y1 + py), max(0, x0 - px):min(W, x1 + px)])
    a = c.astype(np.int16)
    o = cv2.GaussianBlur(np.clip(a[..., 0] - a[..., 2], 0, 255).astype(np.uint8), (3, 3), 0)
    hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
    hue_ok = (((hsv[..., 0] <= 30) | (hsv[..., 0] >= 170)) & (hsv[..., 1] >= 50)).astype(np.uint8) * 255
    blk = max(15, int(h * 1.5)) | 1
    masks = [cv2.adaptiveThreshold(o, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk, C) & hue_ok
             for C in (-10, -6, -14)]
    masks = [clean_line(m, h) for m in masks]
    variants = [cv2.cvtColor(255 - cv2.dilate(m, np.ones((2, 2), np.uint8)), cv2.COLOR_GRAY2RGB)
                for m in masks] + [c]
    return c, variants, masks, h


def groups(mask, h):
    """Column ranges of the digit groups in a cleaned line mask (split at gaps wider than ~0.45h)."""
    cols = np.where(mask.sum(0) > 0)[0]
    if not len(cols):
        return []
    out, start, prev = [], cols[0], cols[0]
    for x in cols[1:]:
        if x - prev > 0.45 * h:
            out.append((start, prev + 1))
            start = x
        prev = x
    out.append((start, prev + 1))
    return out


def layouts(masks, h):
    """Candidate geometries of the date in a stamp line: [(group widths / digit height, apostrophe flags)]
    for every run of 3 consecutive digit-height groups, over all mask variants. Stray blobs next to the
    stamp (grass, a toy) differ in height or just add groups, so windows of 3 keep the real date."""
    out = []
    for mask in masks:
        grp = groups(mask, h)
        ext = []
        for ga, gb in grp:
            rows = np.where(mask[:, ga:gb].sum(1) > 0)[0]
            ext.append((rows.min(), rows.max() + 1) if len(rows) else (0, 0))
        hs = [e[1] - e[0] for e in ext]
        if not hs or max(hs) < 8:
            continue
        H = float(np.median([x for x in hs if x >= 8]))
        keep = [k for k, x in enumerate(hs) if 0.6 * H <= x <= 1.4 * H]
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for w0 in range(len(keep) - 2):
            idx = keep[w0:w0 + 3]
            if idx != list(range(idx[0], idx[0] + 3)):
                continue  # must be adjacent groups
            g3 = [grp[k] for k in idx]
            y0 = min(ext[k][0] for k in idx)
            apos = [False] * 3
            for i in range(1, n):
                x, y, w, hh, _ = st[i]
                if hh < 0.45 * H and w < 0.4 * H and y - y0 < 0.15 * H and y + hh - y0 < 0.6 * H:
                    for k, (ga, gb) in enumerate(g3):
                        if ga - 2 <= x < ga + 0.5 * H:  # small blob at the start-top of a group
                            apos[k] = True
            lay = ([(gb - ga) / H for ga, gb in g3], apos)
            if lay not in out:
                out.append(lay)
    return out


def regroup_any(text, lays, order, ymin, ymax):
    """regroup() over every candidate geometry; only answers if they all point to one date."""
    hits = set()
    for lay in lays:
        rg = regroup(text, lay, order, ymin, ymax)
        if rg:
            hits.add(rg.replace("'", ""))
    if len(hits) != 1:
        return None
    return next(rg for lay in lays for rg in [regroup(text, lay, order, ymin, ymax)] if rg)


def _pred_width(g, apos):
    """Predicted width/height of a group of 7-segment digits (a '1' is narrow)."""
    return sum(0.28 if c == "1" else 0.6 for c in g) + 0.22 * (len(g) - 1) + (0.3 if apos else 0)


def regroup(text, lay, order, ymin, ymax):
    """OCR often drops the gaps ('95 1 18 -> "195118"). Try every split of the digits into the
    line's 3 groups, keep those that are valid dates, and pick the split whose predicted group
    widths fit the measured ones. Returns the re-spaced text only if one split clearly wins."""
    if not lay or not text:
        return None
    widths, apos = lay
    digits = re.sub(r"\D", "", _normalize(text))
    tries = {digits}
    if any(apos) and len(digits) > 4 and digits[0] in "17" and apos[0]:
        tries.add(digits[1:])  # the apostrophe read as a leading 1 or 7
    scored = []
    for ds in filter(None, tries):
        L = len(ds)
        for l1 in (1, 2, 4):  # a group is a day/month (1-2 digits) or a year (2 or 4); never 3
            for l2 in (1, 2, 4):
                l3 = L - l1 - l2
                if l3 not in (1, 2, 4):
                    continue
                parts = [ds[:l1], ds[l1:l1 + l2], ds[l1 + l2:]]
                cand = " ".join(("'" if apos[k] else "") + parts[k] for k in range(3))
                d, note = parse_date(cand, order, ymin, ymax)
                if not d and note.startswith("ambiguous:"):
                    d = cand  # valid date, field order unclear: the split itself is still right or wrong
                if not d:
                    continue
                pred = [_pred_width(parts[k], apos[k]) for k in range(3)]
                scale = sum(p * m for p, m in zip(pred, widths)) / sum(p * p for p in pred)
                err = sum(((scale * p - m) / max(m, 0.2)) ** 2 for p, m in zip(pred, widths))
                scored.append((err, cand, d))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    best = scored[0]
    rival = next((x for x in scored[1:] if x[1] != best[1]), None)
    if best[0] < 0.15 and (rival is None or rival[0] > 3 * best[0]):
        return best[1]
    return None


def clean_line(m, h):
    """Drop blobs that can't be part of a single text line of height ~h centred in the crop."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    H = m.shape[0]
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, hh, area = st[i]
        if y == 0 or y + hh >= H:  # touches top/bottom of crop -> background structure
            continue
        if hh > 1.5 * h or w > 1.6 * h or area < 3:
            continue
        if abs(cen[i][1] - H / 2) > 0.75 * h:
            continue
        keep[i] = True
    return (keep[lab] * 255).astype(np.uint8)


# ======================================================================= OCR backends
PROMPT_LINE = ("This is a small crop from the corner of a photo print. Some cameras imprint the date there "
               "as glowing orange or yellow digits. If this crop shows such an imprinted date, transcribe it "
               "exactly as printed (digits, spaces, apostrophes), nothing else. If it doesn't, or you can't "
               "read it, answer NONE. Don't guess.")
PROMPT_FULL = ("This is a scanned photo print. If it has a camera-imprinted date stamp (usually orange "
               "digits in a corner, possibly sideways), transcribe it exactly as printed, nothing else. "
               "If there is none, answer NONE.")
PROMPT_ORIENT = ("These numbered tiles show the same photo rotated different ways. Which number shows it "
                 "the right way up (people upright, sky and ceilings at the top, ground at the bottom)? "
                 "Ignore any date stamp text. Answer with the number only.")


def to_h(a, h):
    return cv2.resize(a, (max(8, round(a.shape[1] * h / a.shape[0])), h), interpolation=cv2.INTER_CUBIC)


def jpeg_bytes(img_np, png=False):
    buf = io.BytesIO()
    if png:
        Image.fromarray(img_np).save(buf, "PNG")
    else:
        Image.fromarray(img_np).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def _vn_perform(request, img_np, png=False):
    """Run one Apple Vision request on a numpy RGB image; returns the results list."""
    import Vision
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(jpeg_bytes(img_np, png), None)
    ret = handler.performRequests_error_([request], None)
    ok, err = ret if isinstance(ret, tuple) else (bool(ret), None)  # PyObjC returns bool or (bool, err)
    if not ok or err is not None:
        raise RuntimeError(f"Vision request failed: {err}")
    return list(request.results() or [])


class AppleVision:
    """Apple's on-device OCR (Vision framework, Neural Engine). macOS only.
    pip install pyobjc-framework-Vision"""
    name, strong = "vision", False

    def __init__(self, **_):
        if sys.platform != "darwin" and not os.environ.get("STAMPEDE_MOCK_VISION"):
            raise SystemExit("this needs macOS (Apple Vision framework)")
        import importlib.util
        if not (importlib.util.find_spec("objc") and importlib.util.find_spec("Vision")):
            raise SystemExit("pip install pyobjc-framework-Vision")

    def _ocr(self, img_np, png=False, min_h=None):
        """[(text, confidence, (x, y, w, h) normalised, origin bottom-left)]"""
        import objc
        import Vision
        with objc.autorelease_pool():
            req = Vision.VNRecognizeTextRequest.alloc().init()
            req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            req.setUsesLanguageCorrection_(False)  # essential: otherwise digits get "corrected" into words
            req.setRecognitionLanguages_(["en-US"])
            if min_h is not None:
                req.setMinimumTextHeight_(min_h)
            out = []
            for o in _vn_perform(req, img_np, png):
                c = o.topCandidates_(1)
                if c:
                    bb = o.boundingBox()
                    out.append((str(c[0].string()), float(c[0].confidence()),
                                (bb.origin.x, bb.origin.y, bb.size.width, bb.size.height)))
            return out

    @staticmethod
    def _join(obs):
        """Join observations that sit on one text line, left to right."""
        obs = sorted(obs, key=lambda o: o[2][0])
        return " ".join(o[0] for o in obs), (float(np.mean([o[1] for o in obs])) if obs else 0.0)

    def line(self, variants, grp=None, h=None):
        for v in variants:
            if v.shape[0] < 48:
                v = to_h(v, 48)
            v = cv2.copyMakeBorder(v, 16, 16, 24, 24, cv2.BORDER_REPLICATE)  # native size: shrinking hurts
            obs = self._ocr(v, png=True)
            if obs:
                yield self._join(obs)
            if grp and len(grp) >= 2:  # read each digit group of the cleaned mask on its own too
                vv = variants[0]  # binarised mask, same geometry as grp
                parts = []
                for a_, b_ in grp:
                    piece = cv2.copyMakeBorder(vv[:, max(0, a_ - 6):b_ + 6], 16, 16, 24, 24, cv2.BORDER_REPLICATE)
                    o = self._ocr(piece, png=True)
                    ink = np.where((vv[:, a_:b_] < 128).any(axis=(1, 2)) if vv.ndim == 3
                                   else (vv[:, a_:b_] < 128).any(axis=1))[0]
                    tall = len(ink) and (ink[-1] - ink[0] + 1) >= 0.6 * h if h else False
                    if not o and tall and (b_ - a_) < 0.35 * h:
                        o = [("1", 0.5, (0, 0, 0, 0))]  # OCR skips a lone narrow glyph; in a date it's a 1
                    parts.append(self._join(o) if o else ("", 0.0))
                if all(t for t, _ in parts):
                    yield " ".join(re.sub(r"\s+", "", t) for t, _ in parts), float(np.mean([c for _, c in parts]))
                grp = None  # only once per line

    def full(self, work):
        """OCR the bottom band of each landscape rotation; yields the text found in the stamp position
        (bottom right), line by line plus joined."""
        for r in (0, 90, 180, 270):
            a = rot(work, r)
            H, W = a.shape[:2]
            if W < 0.95 * H:
                continue
            top = int(H * 0.70)
            band = np.ascontiguousarray(a[top:])
            obs = [o for o in self._ocr(band, min_h=0.05)  # box: normalised to the band, origin bottom-left
                   if in_stamp_zone((o[2][0] + o[2][2] / 2) * W, top + (1 - o[2][1] - o[2][3] / 2) * (H - top), W, H)]
            for t, c, _ in obs:
                yield r, t, c
            if len(obs) > 1:
                t, c = self._join(obs)
                yield r, t, c


class MLX:
    """Local vision-language model on Apple Silicon via MLX (no server). pip install mlx-vlm
    Default model ~5 GB, downloaded from Hugging Face on first use."""
    name, strong = "mlx", True

    def __init__(self, mlx_model, **_):
        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
        except ImportError:
            raise SystemExit("pip install mlx-vlm   (Apple Silicon only)")
        print(f"loading {mlx_model} (first run downloads it) ...")
        self.model, self.processor = load(mlx_model)
        self.config = load_config(mlx_model)
        self._gen, self._tmpl = generate, apply_chat_template
        import atexit
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="stampede_"))
        atexit.register(shutil.rmtree, self.tmp, True)

    def ask(self, prompt, img_np, max_tokens=40):
        a, _ = shrink(img_np, 1024)  # keeps image tokens (and time per call on an M1) down
        f = self.tmp / "q.jpg"
        Image.fromarray(a).save(f, "JPEG", quality=92)
        fp = self._tmpl(self.processor, self.config, prompt, num_images=1)
        out = self._gen(self.model, self.processor, fp, [str(f)], max_tokens=max_tokens, temperature=0.0,
                        verbose=False)
        return (out.text if hasattr(out, "text") else str(out)).strip()  # newer mlx-vlm returns an object

    def line(self, variants, grp=None, h=None):
        c = variants[-1]  # colour crop
        yield self.ask(PROMPT_LINE, to_h(c, 160) if c.shape[0] < 160 else c), 1.0

    def full(self, work):
        yield None, self.ask(PROMPT_FULL, work), 1.0


BACKENDS = {"vision": AppleVision, "mlx": MLX}


def make_read(txt, sc, eng, kind, r, lays, a, compact=True):
    """One reading of the stamp, kept if it can be a date under some field order. If the digits ran
    together ("195118"), the gaps are put back from the stamp's geometry and the read is marked rg."""
    t, rg = txt, False
    opts, _ = date_options(t, a.ymin, a.ymax, compact)
    if not opts and lays:
        r2 = regroup_any(txt, lays, "auto", a.ymin, a.ymax)
        if r2:
            t, rg = r2, True
            opts, _ = date_options(t, a.ymin, a.ymax)
    if not opts:
        return None
    # clean = exactly three digit groups with ordinary separators: nothing guessed, mapped or dropped
    clean = not rg and bool(re.fullmatch(r"\s*['`’‘]?\d{1,4}(\s*[\s./-]\s*['`’‘]?\d{1,4}){2}\s*", txt))
    return dict(t=t, raw=txt, eng=eng, kind=kind, rot=r, conf=round(float(sc), 2), rg=rg, clean=clean,
                compact=compact)


def vision_evidence(work, a):
    """Everything Apple Vision can tell about one photo, without the model (fast)."""
    eng = AppleVision()
    cands = stamp_candidates(work)
    reads, last, stamp_raw = [], "", ""
    top_lays, top_rot = [], (cands[0][1] if cands else None)
    for ci, (_, r, box) in enumerate(cands[:3]):
        c, variants, masks, h = line_crops(rot(work, r), box)
        lays = layouts(masks, h)
        if ci == 0:
            top_lays = lays
        n = len(reads)
        for txt, sc in eng.line(variants, groups(masks[0], h), h):
            last = txt.strip() or last
            if len(re.findall(r"\d", txt)) > len(re.findall(r"\d", stamp_raw)):
                stamp_raw = txt.strip()
            rd = make_read(txt, sc, "vision", "line", r, lays, a)
            if rd:
                rd["box"] = [int(v) for v in box]
                reads.append(rd)
        if len(reads) > n:
            break
    if not reads:  # nothing on a stamp line: read the photo's edges (Vision also reads sideways text)
        for r, txt, sc in eng.full(work):
            last = txt.strip() or last
            rd = make_read(txt, sc, "vision", "full", None, top_lays, a, compact=False)
            if rd:
                if rd["rg"]:
                    rd["rot"] = top_rot  # spaced like the stamp: the stamp's rotation applies
                reads.append(rd)
    top_box = [int(v) for v in cands[0][2]] if cands else None
    return dict(reads=reads, raw=last or stamp_raw, stamp_raw=stamp_raw, cand=top_rot, cand_box=top_box, faces=face_scores(work, a.faces))


# ======================================================================= orientation
_FACE = None
FACE_FILES = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
                                                "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
}


def face_model():
    """OpenCV's DNN face detector (downloaded once, ~10 MB), else Haar, else None if this
    OpenCV build has neither module. Only used when Apple Vision isn't available."""
    global _FACE
    if _FACE is None:
        _FACE = ("none", None)
        if hasattr(cv2, "dnn") and hasattr(cv2.dnn, "readNetFromCaffe"):
            d = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "stampede"
            try:
                d.mkdir(parents=True, exist_ok=True)
                for fn, url in FACE_FILES.items():
                    if not (d / fn).exists():
                        import urllib.request
                        print(f"downloading face model {fn} ...")
                        part = d / f"{fn}.{os.getpid()}.part"
                        urllib.request.urlretrieve(url, part)
                        os.replace(part, d / fn)
                _FACE = ("dnn", cv2.dnn.readNetFromCaffe(str(d / "deploy.prototxt"),
                                                         str(d / "res10_300x300_ssd_iter_140000.caffemodel")))
            except Exception as e:
                print(f"WARNING: OpenCV DNN face model unavailable ({e})")
        if _FACE[0] == "none" and hasattr(cv2, "CascadeClassifier"):
            _FACE = ("haar", cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"))
        if _FACE[0] == "none":
            print("WARNING: no face detector available; rotation relies on the date stamp and the model")
    return _FACE


def faces_dnn(work, r):
    """Confidence-weighted score of upright faces (OpenCV DNN) for the photo rotated by r."""
    kind, m = face_model()
    if kind == "none":
        return 0.0
    img = rot(work, r)
    if kind == "haar":
        g, _ = shrink(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 1200)
        ms = max(24, int(min(g.shape) * 0.04))
        return float(len(m.detectMultiScale(cv2.equalizeHist(g), 1.1, 7, minSize=(ms, ms))))
    score = 0.0
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    # whole frame + overlapping tiles so small faces in group shots still count
    tiles = [(0, 0, W, H)]
    t = int(min(H, W) * 0.6)
    tiles += [(x, y, x + t, y + t) for x in (0, W - t) for y in (0, H - t)]
    seen = []
    for x0, y0, x1, y1 in tiles:
        blob = cv2.dnn.blobFromImage(cv2.resize(bgr[y0:y1, x0:x1], (300, 300)), 1.0, (300, 300), (104, 177, 123))
        m.setInput(blob)
        for det in m.forward()[0, 0]:
            c = float(det[2])
            if c < 0.6:
                continue
            bx = (x0 + det[3] * (x1 - x0), y0 + det[4] * (y1 - y0), x0 + det[5] * (x1 - x0), y0 + det[6] * (y1 - y0))
            if any(abs(bx[0] - o[0]) < 0.5 * (bx[2] - bx[0]) and abs(bx[1] - o[1]) < 0.5 * (bx[3] - bx[1]) for o in seen):
                continue
            seen.append(bx)
            score += c
    return score


def _vision_faces(img):
    """[(confidence, roll_degrees|None)] from Apple Vision on the whole image plus 2x2 enlarged tiles
    (small faces in group shots). Every view's detections are returned: a real face shows up in
    several views at the same angle, a false positive usually in one, so callers sum the votes."""
    import objc
    import Vision
    full, _ = shrink(img, 1600)
    H, W = full.shape[:2]
    views = [full] + [cv2.resize(full[y:y + H * 6 // 10, x:x + W * 6 // 10], None, fx=1.6, fy=1.6)
                      for y in (0, H * 4 // 10) for x in (0, W * 4 // 10)]
    found = []
    for v in views:
        with objc.autorelease_pool():
            req = Vision.VNDetectFaceRectanglesRequest.alloc().init()
            try:
                req.setRevision_(3)  # revision 3 (macOS 12+) reports continuous roll
            except Exception:
                pass
            for o in _vn_perform(req, np.ascontiguousarray(v)):
                roll = o.roll()
                found.append((float(o.confidence()), None if roll is None else math.degrees(float(roll))))
    return found


def face_scores(work, how):
    """{rotation: score of faces that are upright after rotating the photo clockwise by it}.
    Apple Vision finds faces at any angle and reports their roll. Only its magnitude is used
    (the sign convention isn't relied on): |roll|~0 -> upright now, ~180 -> upside down, ~90 ->
    sideways, resolved by a second pass on the photo turned 90 degrees. At most 10 requests."""
    out = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
    if how == "vision":
        try:
            f0 = [x for x in _vision_faces(work) if x[0] >= 0.5]
            if f0 and all(rl is None for _, rl in f0):
                raise RuntimeError("this macOS reports no face roll")
            sideways = 0.0
            for c, rl in f0:
                if rl is None:
                    continue
                m = abs((rl + 180) % 360 - 180)
                if m < 30:
                    out[0] += c
                elif m > 150:
                    out[180] += c
                elif 60 < m < 120:
                    sideways += c
            if sideways >= 0.5:
                for c, rl in _vision_faces(rot(work, 90)):
                    if c < 0.5 or rl is None:
                        continue
                    m = abs((rl + 180) % 360 - 180)
                    if m < 30:
                        out[90] += c
                    elif m > 150:
                        out[270] += c
            return out
        except Exception as e:
            print(f"WARNING: Apple Vision face detection unusable ({e}); trying OpenCV")
    for r in out:
        try:
            out[r] = faces_dnn(work, r)
        except Exception as e:
            print(f"WARNING: OpenCV face detection failed ({e})")
            break
    return out


def vlm_pick(eng, work, cands, confirm_if):
    """Ask the model which rotation is upright. If its answer would settle the rotation (it matches
    the faces or the stamp, `confirm_if`), ask again with the tiles in another order and keep it only
    if it names the same rotation (a model that favours "1" can't fake agreement). Returns
    (pick or None, confirmed)."""
    first = _vlm_pick_once(eng, work, list(cands))
    if first is None or first not in confirm_if:
        return first, False
    order = list(cands)[1:] + list(cands)[:1]  # every candidate moves position
    return (first, True) if _vlm_pick_once(eng, work, order) == first else (None, False)


def _vlm_pick_once(eng, work, cands):
    """One question, candidates shown as a numbered 2x2 grid of 330 px tiles (small image = few image
    tokens = fast; enough to tell which way is up)."""
    tiles = []
    H, W = work.shape[:2]
    sq = min(H, W)  # centre square: every tile has the same shape, so shape can't give the answer away
    square = work[(H - sq) // 2:(H - sq) // 2 + sq, (W - sq) // 2:(W - sq) // 2 + sq]
    for i, r in enumerate(cands, 1):
        t, _ = shrink(rot(np.ascontiguousarray(square), r), 330)
        canvas = np.full((350, 350, 3), 255, np.uint8)
        y, x = (350 - t.shape[0]) // 2, (350 - t.shape[1]) // 2
        canvas[y:y + t.shape[0], x:x + t.shape[1]] = t
        cv2.rectangle(canvas, (0, 0), (44, 44), (0, 0, 0), -1)
        cv2.putText(canvas, str(i), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        tiles.append(canvas)
    while len(tiles) < 4:
        tiles.append(np.full((350, 350, 3), 255, np.uint8))
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
    ans = eng.ask(PROMPT_ORIENT, grid)
    m = re.search(r"\d", ans)
    if m and 1 <= int(m.group()) <= len(cands):
        return cands[int(m.group()) - 1]
    return None


# ======================================================================= batch reasoning
# Photos aren't judged alone: a folder of scans comes from the same camera(s), lay on the scanner the
# same way, and prints of one roll sit next to each other. That settles most questions without the
# model; only what's still open after this goes to Qwen.
NEAR_DAYS, FAR_DAYS = 60, 365
# informational only, no decision needed: undated (apply uses folder year / fallback), and rotation
# taken from the stamp + how this folder's prints lay on the scanner (a faceless portrait shot would
# be wrong: the review page has a filter to skim exactly these)
REVIEW_FREE = ("no-date", "landscape-assumed")


def evidence_of(r):
    import json
    try:
        return json.loads(r.get("evidence") or "{}")
    except ValueError:
        return {}


def folder_of(r):
    return str(Path(r["rel"]).parent)


def learn_orders(rows, a):
    """Camera field order per folder, from photos whose stamp fits only one order (25/12/2006 can
    only be day/month/year). One vote per photo; needs 3+ photos and no real dissent. A learned order
    only ever shapes a *guess* (you accept those explicitly on the review page), never `date`."""
    votes = defaultdict(lambda: defaultdict(int))
    for r in rows:
        mine = set()
        for rd in evidence_of(r).get("reads", []):
            if rd["eng"] != "vision":
                continue
            opts, informative = date_options(rd["t"], a.ymin, a.ymax, rd.get("compact", True))
            if informative and len(opts) == 1:
                mine.add(next(iter(opts)))
        if len(mine) == 1:
            votes[folder_of(r)][mine.pop()] += 1
    out = {}
    for folder, v in votes.items():
        o, n = max(v.items(), key=lambda kv: kv[1])
        if n >= 3 and n >= 0.9 * sum(v.values()):
            out[folder] = o
    return out


def placement_votes(rows):
    """How prints lay on the scanner, per folder: votes from the frame rotation of confirmed stamps."""
    votes = defaultdict(lambda: defaultdict(int))
    for r in rows:
        ev = evidence_of(r)
        if ev.get("stamp_frame") is not None:
            votes[folder_of(r)][ev["stamp_frame"]] += 1
    return votes


def placement_of(v):
    if not v:
        return None
    rr, n = max(v.items(), key=lambda kv: kv[1])
    return rr if n >= 5 and n >= 0.8 * sum(v.values()) else None


class Batch:
    """Reasoning over a whole folder. Hard rule: `date` is only ever filled when Apple Vision and the
    Qwen model read the same date (and not merely by re-spacing the same run-together digits).
    The batch (learned field order, neighbouring scans) only shapes guesses and flags."""

    def __init__(self, rows, a):
        self.a = a
        self.orders = learn_orders(rows, a)
        self.votes = placement_votes(rows)
        self.by_folder = defaultdict(list)
        for r in sorted(rows, key=lambda x: x["rel"]):
            self.by_folder[folder_of(r)].append(r)

    def learned(self, r):
        return self.orders.get(folder_of(r))

    def reads(self, r, eng, order):
        a = self.a
        return [(dates_for(rd["t"], order, a.ymin, a.ymax, rd.get("compact", True)), rd)
                for rd in evidence_of(r).get("reads", []) if rd["eng"] == eng]

    def neighbour_dates(self, r):
        folder = self.by_folder[folder_of(r)]
        i = next((k for k, x in enumerate(folder) if x["rel"] == r["rel"]), None)
        if i is None:
            return []
        out = []
        for x in folder[max(0, i - 4):i] + folder[i + 1:i + 5]:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", x.get("date", "")):
                out.append(date.fromisoformat(x["date"]))
            elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", x.get("date_guess", "")):
                out.append(date.fromisoformat(x["date_guess"]))
        return out

    def nearest(self, r, opts):
        nd = self.neighbour_dates(r)
        if not nd:
            return sorted(opts)[0]
        return min(sorted(opts), key=lambda d: min(abs((d - x).days) for x in nd))

    def out_of_sequence(self, r, d):
        nd = self.neighbour_dates(r)
        return bool(nd) and min(abs((d - x).days) for x in nd) > 365

    def resolve_date(self, r):
        """-> dict(date, guess, flags, frame, src, ask_model)"""
        a, ev = self.a, evidence_of(r)
        explicit = a.order if a.order != "auto" else None
        learned = self.learned(r)
        asked = ev.get("model_date", False)
        has_stamp = ev.get("cand") is not None or bool(ev.get("reads"))
        res = dict(date="", guess="", flags=[], frame=None, src="", ask_model=False)

        def frame_of(d, V):
            return next((rd["rot"] for ds, rd in V if d in ds and rd.get("rot") is not None), None)

        V = self.reads(r, "vision", explicit)
        # Qwen counts only where Vision saw digits on the stamp line: on a crop without a stamp it
        # can make a date up, and that must not even become a guess
        vision_digits = len(re.findall(r"\d", ev.get("stamp_raw", ""))) >= 2 or bool(V)
        Q = self.reads(r, "mlx", explicit) if vision_digits else []
        vall = set().union(*[d for d, _ in V]) if V else set()
        qall = set().union(*[d for d, _ in Q]) if Q else set()
        vdef = {next(iter(d)) for d, _ in V if len(d) == 1}
        qdef = {next(iter(d)) for d, _ in Q if len(d) == 1}

        # 1. confirmed: both readers give exactly the same single date, at least one read taken as printed
        if len(vdef) == 1 and vdef == qdef and vall == vdef and qall == qdef:
            d = next(iter(vdef))
            plain = any(not rd["rg"] for ds, rd in V + Q if ds == {d})
            if plain:
                res.update(date=d.isoformat(), frame=frame_of(d, V), src="vision+model")
                if self.out_of_sequence(r, d):
                    res["flags"].append("date-out-of-sequence")  # both read it, but it doesn't fit its neighbours
                return res

        # 2. both read the same digits but the field order is open (05/06/2006): a guess, using the
        #    folder's learned order if there is one (accept those in one go on the review page)
        if vall and qall and vall == qall and len(vall) > 1:
            if learned:
                Vl = self.reads(r, "vision", learned)
                ld = {next(iter(d)) for d, _ in Vl if len(d) == 1}
                if len(ld) == 1:
                    d = next(iter(ld))
                    res.update(guess=d.isoformat(), frame=frame_of(d, V), flags=[f"date-order-learned:{learned}"])
                    return res
            g = self.nearest(r, vall)
            res.update(guess=g.isoformat(), frame=frame_of(g, V),
                       flags=["ambiguous-date-order:" + "/".join(sorted(map(str, vall)))])
            return res

        # 3. anything else is a guess; ask the model once if a stamp is there and it hasn't been asked
        opts = vall | qall
        res["ask_model"] = has_stamp and not asked
        if not opts:
            res["flags"] = ["no-date"]
            seen = f"{ev.get('stamp_raw', '')}|{ev.get('model_raw', '') if vision_digits else ''}"
            if has_stamp and max(len(re.findall(r"\d", x)) for x in seen.split("|")) >= 3:
                res["flags"].append("stamp-unread")  # digits on the stamp line, but no date in them: look
            return res
        if learned and len(vall) > 1:  # Vision alone, order open: prefer the learned order for the guess
            ld = {next(iter(d)) for d, _ in self.reads(r, "vision", learned) if len(d) == 1}
            g = next(iter(ld)) if len(ld) == 1 else self.nearest(r, opts)
        elif len(vdef) == 1:
            g = next(iter(vdef))
        else:
            g = self.nearest(r, opts)
        res.update(guess=g.isoformat(), frame=frame_of(g, V))
        if (vall and qall and vall != qall) or len(vdef) > 1 or len(qdef) > 1:
            res["flags"] = ["readers-disagree:" + "/".join(sorted(map(str, opts)))]
        elif len(opts) > 1:
            res["flags"] = ["ambiguous-date-order:" + "/".join(sorted(map(str, opts)))]
        else:
            res["flags"] = ["date-unconfirmed"]
        return res

    def resolve_rotation(self, r, dres):
        """-> (rotation, source, flags, ask) ; ask = (candidates, answers-that-would-settle-it) or None.
        The frame (landscape-upright) comes from a *confirmed* date stamp, or the folder's scanner
        placement; faces reveal portrait shots. 'landscape-assumed' (no review needed, but its own
        filter) only when the stamp is confirmed, matches the placement and no face points elsewhere."""
        a = self.a
        if a.rotate != "auto":
            return int(a.rotate), "fixed", [], None
        ev = evidence_of(r)
        fs = {int(k): v for k, v in ev.get("faces", {}).items()}
        fc = {x: fs.get(x, 0.0) for x in a.allowed}
        best = max(fc, key=fc.get)
        face = best if fc[best] >= 0.8 and all(fc[best] >= fc[x] + 0.5 for x in fc if x != best) else None
        prior = placement_of(self.votes.get(folder_of(r)))
        stamp = dres["frame"] if dres["date"] else None  # only a confirmed stamp fixes the frame
        frame = stamp if stamp is not None else (dres["frame"] if dres["frame"] is not None else prior)
        cands = ([x for x in (frame, (frame + 270) % 360, (frame + 90) % 360) if x in a.allowed]
                 if frame is not None else list(a.allowed)) or list(a.allowed)
        m = ev.get("model_orient")
        stray_face = frame is not None and any(v >= 0.3 for x, v in fs.items() if x != frame)
        if face is not None:
            if face == frame and (stamp is not None or frame == prior):
                return face, "frame+faces", [], None
            if m and m.get("sure") and m["pick"] == face:
                return face, "faces+model", [], None
            ask = None if m else (cands, {x for x in (face, stamp) if x is not None})
            return face, "faces", ["stamp-orientation-conflict" if frame is not None and face not in cands
                                   else "rotation-check"], ask
        located = dres["frame"]  # where Vision found the date digits: real frame evidence, even if a
        # digit is misread (the reading may be a guess, the position isn't)
        if located is not None and prior in (None, located) and not stray_face:
            return located, "stamp", ["landscape-assumed"], None
        if m and m.get("sure") and frame is not None and m["pick"] == frame:
            return frame, "frame+model", [], None  # the model, asked twice, agrees with the frame
        ask = None if m else (cands, {frame} if frame is not None else set())
        if m and m.get("pick") is not None:
            return m["pick"], "model", ["rotation-check"], ask
        if frame is not None:
            return frame, "frame", ["rotation-check"], ask
        return a.allowed[0], "none", ["orientation-unknown"], ask

    def finish(self, r):
        """Turn the evidence into the row's decision (in place). Returns (ask_model_date, rotation_ask)."""
        import json
        ev = evidence_of(r)
        dres = self.resolve_date(r)
        rr, src, rflags, ask = self.resolve_rotation(r, dres)
        if dres["date"] and dres["frame"] is not None and ev.get("stamp_frame") != dres["frame"]:
            ev["stamp_frame"] = dres["frame"]  # teaches the folder's scanner placement
            self.votes[folder_of(r)][dres["frame"]] += 1
        elif not dres["date"]:
            ev.pop("stamp_frame", None)
        ev["settings"] = scan_settings(self.a)  # decided under these; changing them re-decides the row
        flags = ev.get("crop_flags", []) + rflags + dres["flags"]
        prev = ev.get("previous") or {}
        if prev.get("date") and prev["date"] != dres["date"]:
            flags.append(f"date-before-rescan:{prev['date']}")  # what you had entered for the old scan
            if not dres["date"] and not dres["guess"]:
                dres["guess"] = prev["date"] if re.fullmatch(r"\d{4}-\d{2}-\d{2}", prev["date"]) else ""
        chosen = dres["date"] or dres["guess"]
        best = None
        if chosen:
            cd = date.fromisoformat(chosen)
            best = next((rd for rd in ev.get("reads", [])
                         if cd in dates_for(rd["t"], None, self.a.ymin, self.a.ymax, rd.get("compact", True))), None)
        r.update(date=dres["date"], date_guess=dres["guess"], rotate=str(rr), orient_src=src,
                 flags=" ".join(flags), ocr_raw=(best["raw"] if best else ev.get("raw", "")),
                 ocr_backend=dres["src"] or (best["eng"] if best else ""), evidence=json.dumps(ev),
                 status="review" if any(f not in REVIEW_FREE for f in flags) else "ok")
        if prev.get("skip"):
            prev["skip"] = False  # applied once: if you un-skip it later, it stays un-skipped
            r.update(status="skip", _carried=("status",), evidence=json.dumps(ev))  # you skipped the old scan
        return dres["ask_model"], ask


def scan_settings(a):
    return f"{a.order}|{','.join(map(str, a.allowed))}"


def preview_path(wd, r):
    import hashlib
    content = r.get("src_sig", "").partition("@")[0]  # not the file time: copying the scans keeps previews
    return wd / "previews" / (hashlib.sha1(f"{r['rel']}|{content}".encode()).hexdigest()[:16] + ".jpg")


def pass1(todo, root, wd, a, jobs, crashed):
    """Run analyse() over the scans, `jobs` at a time. If a worker process dies (a scan that crashes
    the decoder or Vision), the photos that were in flight are recorded as errors and the rest go on;
    a photo that crashes again on its own is marked so it isn't retried forever. Photos that crashed
    before always run in their own worker process, never in this one."""
    suspects = [p for p in todo if str(p.relative_to(root)) in crashed]
    queue = [p for p in todo if p not in suspects]
    if jobs == 0:  # --jobs 0: no worker processes for normal photos (debugging/tests)
        for p in queue:
            yield analyse((p, root, wd, a))
        queue = []
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool

    def err(q, why):
        return dict(dict.fromkeys(FIELDS, ""), source=str(q), rel=str(q.relative_to(root)), rotate="0",
                    status="error", flags=f"error:{why}")

    for group, width in ((queue, max(1, min(jobs, len(queue)))), (suspects, 1)):
        if not group:
            continue
        group = list(group)
        ex = ProcessPoolExecutor(max_workers=width)
        inflight = {}
        try:
            while group or inflight:
                while group and len(inflight) < width:  # only running tasks in flight
                    p = group.pop(0)
                    inflight[ex.submit(analyse, (p, root, wd, a))] = p
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                broken = False
                for f in done:
                    p = inflight.pop(f)
                    try:
                        yield f.result()
                    except BrokenProcessPool:
                        broken = True
                        yield err(p, "crashes-the-analysis" if width == 1 else "worker-crashed")
                    except Exception as e:
                        yield err(p, f"{type(e).__name__}:{str(e)[:150]}")
                if broken:
                    for f, q in list(inflight.items()):  # finished ones keep their result
                        try:
                            yield f.result(timeout=0) if f.done() else err(q, "worker-crashed")
                        except BrokenProcessPool:
                            yield err(q, "worker-crashed")
                        except Exception as e:
                            yield err(q, f"{type(e).__name__}:{str(e)[:150]}")
                    inflight.clear()
                    ex.shutdown(wait=False, cancel_futures=True)
                    ex = ProcessPoolExecutor(max_workers=width)
        finally:
            ex.shutdown(wait=True, cancel_futures=True)


ROT_FLAGS = ("landscape-assumed", "orientation-unknown", "rotation-check", "stamp-orientation-conflict")
EDITABLE = ("date", "rotate", "status")


def edited_fields(r):
    """Which of date/rotate/status you set yourself ('1' from older versions = all of them)."""
    e = r.get("edited", "")
    f = set(EDITABLE) if e == "1" else set(e.split(",")) & set(EDITABLE)
    if r.get("status") == "skip":
        f.add("status")
    return f


def frozen(r):
    """You've settled this photo (typed/accepted a date, marked it OK, or skipped it): scans leave it alone."""
    return bool(edited_fields(r) & {"date", "status"})


def commit(out, decided):
    """Write scan decisions into photos.csv, never over your edits: what you set on the review page
    (date, rotation, OK/skip) stays; the scan only refreshes the rest."""
    def merge(rows):
        for i, x in enumerate(rows):
            r = decided.get(x["rel"])
            if r is None:
                continue
            merged = dict(x)
            merged.update({k: r[k] for k in FIELDS if k in r})
            mine = edited_fields(x)
            if r.get("_unfreeze"):  # the scan's date changed under your OK: back to you for a look
                mine.discard("status")
            theirs = set(r.get("_carried", ())) - mine  # the skip carried over from a replaced scan
            if r.get("_unfreeze"):
                theirs.discard("status")
            merged["edited"] = ",".join(sorted(mine | theirs))
            if mine or theirs:
                merged.update({k: x.get(k, "") for k in mine})
                fl = merged["flags"].split()
                if frozen(merged):
                    fl = [f for f in fl if f != "unresolved"]
                if "rotate" in mine:
                    fl = [f for f in fl if not f.startswith(ROT_FLAGS)]
                if "date" in mine:
                    fl = [f for f in fl if not is_date_flag(f)]
                merged["flags"] = " ".join(fl)
                if "status" not in mine | theirs and merged["status"] != "error":
                    merged["status"] = "review" if any(f not in REVIEW_FREE for f in fl) else "ok"
            rows[i] = merged
    update_csv(out, merge)


def analyse(job):
    """Pass 1 for one scan (runs in a worker process): crop + everything Apple Vision can tell,
    plus a small preview of the crop so the review page doesn't have to decode full scans."""
    import json
    p, root, wd, a = job
    row = dict.fromkeys(FIELDS, "")
    row.update(source=str(p), rel=str(p.relative_to(root)), rotate="0", status="error")
    try:
        row["src_sig"] = src_sig(p)
        img, _ = load(p)
        rgb = np.asarray(img)
        box, cflags = find_crop(rgb, 18, 0.004)
        crop = np.ascontiguousarray(rgb[box[1]:box[3], box[0]:box[2]])
        work, _ = shrink(crop, WORK)
        ev = vision_evidence(work, a)
        ev["crop_flags"] = cflags
        ev["av"] = ANALYSIS_VERSION
        row.update(crop=",".join(map(str, box)), evidence=json.dumps(ev), ocr_raw=ev["raw"],
                   flags=" ".join(cflags + ["unresolved"]), status="review")
        pv = preview_path(wd, row)
        pv.parent.mkdir(parents=True, exist_ok=True)
        small, _ = shrink(work, 1200)
        tmp = pv.with_name(f".{pv.stem}.{os.getpid()}.tmp.jpg")
        Image.fromarray(small).save(tmp, "JPEG", quality=88)
        os.replace(tmp, pv)
    except Exception as e:
        row["flags"] = f"error:{type(e).__name__}:{str(e)[:150]}"
    return row


def model_check(r, root, a, mlx, need_date):
    """Pass 3 for one photo: one Qwen read of the same stamp line Vision looked at (a small crop, so
    it's quick). Stores the answer in the row's evidence; returns the working image."""
    import json
    ev = evidence_of(r)
    img, _ = load(src_of(root, r))
    rgb = np.asarray(img)
    x0, y0, x1, y1 = parse_crop(r["crop"])
    work, _ = shrink(np.ascontiguousarray(rgb[y0:y1, x0:x1]), WORK)
    if need_date:
        where = next(((rd["rot"], rd["box"]) for rd in ev.get("reads", []) if rd.get("box")), None)
        if where is None and ev.get("cand_box") is not None:
            where = (ev["cand"], ev["cand_box"])
        if where is not None:
            rr, box = where
            c, variants, masks, h = line_crops(rot(work, rr), tuple(box))
            for txt, sc in mlx.line(variants):
                if len(re.findall(r"\d", txt)) > len(re.findall(r"\d", ev.get("model_raw", ""))):
                    ev["model_raw"] = txt.strip()[:80]
                rd = make_read(txt, sc, "mlx", "line", rr, layouts(masks, h), a)
                if rd:
                    ev.setdefault("reads", []).append(rd)
        ev["model_date"] = True
        r["evidence"] = json.dumps(ev)
    return work


# ======================================================================= CSV / review
def read_csv(p):
    # utf-8-sig: a CSV re-saved by Excel/Numbers starts with a byte-order mark
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@contextlib.contextmanager
def locked(csv_path):
    """Exclusive lock for read-modify-write of photos.csv, shared by scan and the review page
    (separate processes and threads)."""
    with open(str(csv_path) + ".lock", "a") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def update_csv(csv_path, fn):
    """Re-read photos.csv under the lock, let fn modify the row list, write it back atomically."""
    with locked(csv_path):
        rows = read_csv(csv_path) if csv_path.exists() else []
        result = fn(rows)
        write_csv(csv_path, rows)
        return result


def _content_hash(p, size):
    """Hash of the whole file (uncompressed scans of different prints can share identical white
    margins, so sampling the start and end isn't enough). Only runs for new or changed files."""
    import hashlib
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def src_sig(path):
    """'size-contenthash@mtime': the content part survives copies that drop timestamps and changes
    when a scan is really replaced; the mtime part lets unchanged files skip the hashing."""
    p = Path(path)
    st = p.stat()
    return f"{st.st_size}-{_content_hash(p, st.st_size)}@{st.st_mtime_ns}"


def scan_status(row, path):
    """'same', 'changed', or 'unsure' (old-style signature, same size, different time: can't tell).
    Also returns the up-to-date signature to store."""
    old = row.get("src_sig", "")
    p = Path(path)
    st = p.stat()
    if not old:
        return "same", src_sig(p)
    if ":" in old:  # legacy 'size:mtime'
        size, mtime = old.split(":", 1)
        if size == str(st.st_size) and mtime == str(st.st_mtime_ns):
            return "same", src_sig(p)
        return ("unsure" if size == str(st.st_size) else "changed"), src_sig(p)
    content, _, mtime = old.partition("@")
    if mtime == str(st.st_mtime_ns) and content.split("-")[0] == str(st.st_size):
        return "same", old  # untouched: no need to read the file
    new = src_sig(p)
    return ("same" if new.partition("@")[0] == content else "changed"), new


def same_scan(row, path):
    return scan_status(row, path)[0] == "same"


def row_id(r):
    """Stable id of a photo = its path inside the scans folder, so moving the folder or a NAS
    remounting under another name keeps every row and edit."""
    import hashlib
    return hashlib.sha1(r["rel"].encode()).hexdigest()[:12]


def src_of(root, r):
    return Path(root) / r["rel"]


def parse_crop(s):
    v = [int(x) for x in s.split(",")]
    if len(v) != 4:
        raise ValueError
    return tuple(v)


def thumb_key(r):
    import hashlib
    content = r.get("src_sig", "").partition("@")[0]
    return hashlib.sha1(f"{r['rel']}|{content}|{r['crop']}|{r['rotate']}".encode()).hexdigest()[:16]


def make_thumb(r, work_dir, root, img=None):
    """Cached thumbnail of exactly what apply will write (crop + rotate), from pass 1's small preview
    of the crop when there is one (fast), else from the scan."""
    t = work_dir / "thumbs" / f"{thumb_key(r)}.jpg"
    if not t.exists():
        t.parent.mkdir(parents=True, exist_ok=True)
        pv = preview_path(work_dir, r)
        if img is None and pv.exists():
            im = Image.open(pv)
            if int(r["rotate"] or 0):
                im = im.rotate(-int(r["rotate"]), expand=True)
        else:
            if img is None:
                img, _ = load(src_of(root, r))
            im = render(img, parse_crop(r["crop"]), int(r["rotate"] or 0))
        im.thumbnail((420, 420))
        tmp = t.with_name(f".{t.stem}.{os.getpid()}.{threading.get_ident()}.tmp.jpg")
        im.save(tmp, "JPEG", quality=85)
        os.replace(tmp, t)  # never a half-written thumbnail, even on Ctrl+C or parallel requests
    return t


def write_csv(path, rows):
    extra = []  # keep columns you added yourself
    for r in rows:
        extra += [k for k in r if k not in FIELDS and k not in extra and k]
    rows.sort(key=lambda r: r.get("rel", ""))  # stable order: by file name, not by when it was analysed
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS + extra, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())  # on disk before it replaces the old one (power loss)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written CSV


REVIEW_PAGE = r"""<!doctype html><meta charset="utf-8"><title>Scan review</title>
<style>
body{font:14px system-ui;margin:0;background:#eee;color:#222}
header{position:sticky;top:0;background:#fff;padding:10px 16px;border-bottom:1px solid #ccc;z-index:2;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
header button.on{background:#222;color:#fff}
main{display:flex;flex-wrap:wrap;gap:12px;padding:16px}
.card{background:#fff;width:430px;padding:8px;border-left:6px solid #3a3;border-radius:4px}
.card.review{border-color:#d33}.card.skip{opacity:.4;border-color:#999}
.card img{width:420px;height:300px;object-fit:contain;background:#ddd;cursor:zoom-in;display:block}
.row{display:flex;gap:6px;align-items:center;margin-top:6px;flex-wrap:wrap}
input.date{width:110px;font:inherit;padding:3px}input.bad{outline:2px solid #d33}
.meta{color:#666;font-size:12px;word-break:break-all}.flags{color:#c00;font-size:12px}.card.ok .flags{color:#aaa}
button{font:inherit;padding:3px 8px;cursor:pointer}.saved{color:#3a3;font-size:12px}
#big{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:5}
#big img{max-width:96vw;max-height:96vh}
</style>
<header><b>Scan review</b>
<button data-f="all">All</button><button data-f="review" class="on">Flagged</button>
<button data-f="nostamp" title="Flagged, no date and nothing read from a stamp: check the rotation, then Mark shown as OK">No stamp</button>
<button data-f="needdate" title="Flagged, a stamp was found but there's no date yet: pick or type it">Date to enter</button><button data-f="landscape" title="Rotation taken from the date stamp and how the prints lay on the scanner; a portrait shot without faces would be sideways">Assumed landscape</button><button data-f="skip">Skipped</button>
<span id="count"></span>
<span style="margin-left:auto;color:#666">Edits save to photos.csv immediately. Click a photo to enlarge.
Reload to see photos added by a running scan.</span>
<button id="guessall" title="Use the suggested date for every photo shown that has one">Use guesses shown</button>
<button id="okall" title="Mark every photo currently shown as OK">Mark shown as OK</button></header>
<main id="m"></main><div id="big" onclick="this.style.display='none'"><img></div>
<script>
let rows = ROWS, filter = 'review'; const TOKEN = "__TOKEN__";
const esc=t=>String(t??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const m = document.getElementById('m');
// a guess can be one date, or several when the field order is unclear / the readers disagree
function options(r){ const fl=(r.flags||'').split(' '), f=fl.find(x=>x.startsWith('ambiguous-date-order:')||x.startsWith('readers-disagree:'));
  const o=f?f.split(':')[1].split('/'):(r.date_guess?[r.date_guess]:[]);
  const p=fl.find(x=>x.startsWith('date-before-rescan:')); if(p&&!o.includes(p.split(':')[1])) o.push(p.split(':')[1]); return o; }
const coinflip = r => /(ambiguous-date-order|readers-disagree|date-before-rescan):/.test(r.flags||'');
// a stamp was seen: there's a date to pick or type (vs. photos with nothing on them to date from)
const stamped = r => !!r.date_guess || /(stamp-unread|date-before-rescan|readers-disagree|ambiguous-date-order)/.test(r.flags||'');
function show(r){ if(filter=='all') return true; if(filter=='review') return r.status=='review'||r.status=='error';
  if(filter=='nostamp') return r.status=='review' && !r.date && !stamped(r);
  if(filter=='needdate') return r.status=='review' && !r.date && stamped(r);
  if(filter=='landscape') return (r.flags||'').split(' ').includes('landscape-assumed') && r.status!='skip';
  return r.status==filter; }
function card(r){
  const d=document.createElement('div'); d.className='card '+r.status; d.id='c'+r.id;
  d.innerHTML=`<img loading="lazy" src="/thumb/${r.id}?k=${r.key}">
  <div class="row">Date <input class="date" value="${esc(r.date)}" placeholder="YYYY-MM-DD">
   ${r.date?'':options(r).map(d=>`<button class="guess" data-d="${esc(d)}">use ${esc(d)}</button>`).join('')}
   <button class="rl" title="rotate left">⟲</button><button class="rr" title="rotate right">⟳</button>
   ${r.status=='error'?'':`<button class="ok">${r.status=='ok'?'✓ OK':'OK'}</button>`}<button class="sk">${r.status=='skip'?'Unskip':'Skip'}</button>
   <span class="saved"></span></div>
  <div class="flags">${esc(r.flags)}</div>
  <div class="meta">${esc(r.rel)} · rotate ${esc(r.rotate)}° · read ${r.ocr_raw?esc(JSON.stringify(r.ocr_raw)):'—'}</div>`;
  const q=s=>d.querySelector(s);
  q('img').onclick=()=>{const b=document.getElementById('big'); b.querySelector('img').src='/large/'+r.id+'?k='+r.key; b.style.display='flex'};
  q('.date').onchange=e=>save(r,'date',e.target.value.trim());
  d.querySelectorAll('.guess').forEach(b=>b.onclick=()=>save(r,'date',b.dataset.d));
  q('.rl').onclick=()=>save(r,'rotate',String((+r.rotate+270)%360));
  q('.rr').onclick=()=>save(r,'rotate',String((+r.rotate+90)%360));
  if(q('.ok')) q('.ok').onclick=()=>save(r,'status','ok');
  q('.sk').onclick=()=>save(r,'status',r.status=='skip'?'review':'skip');
  return d;
}
function render(){ m.innerHTML=''; let n=0; for(const r of rows) if(show(r)){m.appendChild(card(r));n++}
  document.getElementById('count').textContent=n+' of '+rows.length+' shown'; }
async function save(r,field,value,bulk){
  const res=await fetch('/api/update',{method:'POST',headers:{'Content-Type':'application/json','X-Token':TOKEN},
    body:JSON.stringify({id:r.id,field,value,v:r.v})});
  const j=await res.json(); const c=document.getElementById('c'+r.id);
  if(j.stale){ Object.assign(r,j.row); if(c){const nc=card(r); c.replaceWith(nc); nc.querySelector('.saved').textContent='updated by the scan'}
    if(!bulk) alert(j.error); return false }
  if(!j.ok){ if(c&&field=='date') c.querySelector('.date').classList.add('bad'); if(!bulk) alert(j.error); return false }
  Object.assign(r,j.row);
  if(!c) return true;
  if(!show(r)){ c.style.transition='opacity .3s'; c.style.opacity='0'; setTimeout(()=>{c.remove(); count()},300); return true }
  const nc=card(r); c.replaceWith(nc); nc.querySelector('.saved').textContent='saved'; return true;
}
async function bulk(list,field,val){ let bad=0; for(const r of list) if(!await save(r,field,val(r),true)) bad++;
  render(); if(bad) alert(bad+' of '+list.length+' photos were not changed: the scan updated them meanwhile (or they failed). Have another look at those.'); }
function count(){ document.getElementById('count').textContent=rows.filter(show).length+' of '+rows.length+' shown'; }
document.querySelectorAll('header button[data-f]').forEach(b=>b.onclick=()=>{filter=b.dataset.f;
  document.querySelectorAll('header button[data-f]').forEach(x=>x.classList.toggle('on',x==b)); render()});
document.getElementById('okall').onclick=async()=>{const sh=rows.filter(show).filter(r=>r.status!='skip'&&r.status!='error');
  if(!confirm('Mark '+sh.length+' photos as OK?'))return; await bulk(sh,'status',()=>'ok')};
document.getElementById('guessall').onclick=async()=>{const sh=rows.filter(show).filter(r=>r.date_guess&&!r.date&&!coinflip(r));
  const n=rows.filter(show).filter(r=>!r.date&&coinflip(r)).length;
  const lo=sh.filter(r=>/date-order-learned:/.test(r.flags||'')); const o=lo.length?(lo[0].flags.match(/date-order-learned:([a-z]+)/)||[])[1]:'';
  const names={dmy:'day/month/year',mdy:'month/day/year',ymd:'year/month/day'};
  if(!confirm('Use the suggested date for '+sh.length+' photos?'
    +(lo.length?'\n\n'+lo.length+' of them read the stamp as '+(names[o]||o)+', the order this folder\'s camera uses on stamps that can only be read one way.':'')
    +(n?'\n\n'+n+' with two or more possible dates are left for you to pick.':'')))return; await bulk(sh,'date',r=>r.date_guess)};
render();
</script>"""


def is_date_flag(f):
    return f.startswith(DATE_FLAGS) or f in REVIEW_FREE


def cmd_review(a):
    """Local web page to check and fix photos.csv; every edit is written back immediately.
    Only this page can change the CSV: it must come from 127.0.0.1/localhost (no DNS rebinding),
    carry this session's token (other web pages can't read it) and be JSON."""
    import json
    import secrets
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlparse

    root, wd = work_dir_for(a.scans)
    csv_path = wd / "photos.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found: run `scan {a.scans}` first")

    # one review page per photos.csv: a second one would overwrite the first one's edits
    single = open(wd / "review.lock", "a+")
    try:
        fcntl.flock(single.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        single.seek(0)
        url = single.read().strip()
        print(f"The review page is already open: {url}")
        if url and not a.no_browser:
            webbrowser.open(url)
        return

    token = secrets.token_urlsafe(24)
    cookie = secrets.token_urlsafe(24)  # SameSite=Strict: other sites can't even load our images
    hosts = set()
    big_images = threading.BoundedSemaphore(2)
    ymin, ymax = (int(x) for x in YEARS.split("-"))

    def version(r):
        import hashlib
        return hashlib.sha1("|".join(r.get(k, "") for k in ("date", "date_guess", "flags", "status", "rotate",
                                                              "edited", "crop")).encode()).hexdigest()[:12]

    def public(r):
        return dict({k: r.get(k, "") for k in FIELDS if k != "evidence"}, id=row_id(r), key=thumb_key(r), v=version(r))

    class Stale(Exception):
        pass

    def find(rows, rid):
        for r in rows:
            if row_id(r) == rid:
                return r
        raise KeyError("photo not found (was photos.csv changed?)")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def host_ok(self):
            if self.headers.get("Host", "") in hosts:
                return True
            self.send(403, b"forbidden", "text/plain")
            return False

        def cookie_ok(self):
            # parsed by hand: another localhost app's odd cookie must not hide ours
            for part in self.headers.get("Cookie", "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == cname and secrets.compare_digest(v.strip().encode(), cookie.encode()):
                    return True
            return False

        def do_GET(self):
            if not self.host_ok():
                return
            path = urlparse(self.path).path
            try:
                if path == "/":
                    data = json.dumps([public(r) for r in read_csv(csv_path)]).replace("<", "\\u003c")
                    page = REVIEW_PAGE.replace("__TOKEN__", token).replace("ROWS", data).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Set-Cookie", f"{cname}={cookie}; Path=/; HttpOnly; SameSite=Strict")
                    self.send_header("X-Frame-Options", "DENY")  # no clickjacking from another page
                    self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
                    self.end_headers()
                    self.wfile.write(page)
                    return
                kind, _, rid = path.strip("/").partition("/")
                if kind in ("thumb", "large"):
                    if not self.cookie_ok():
                        return self.send(403, b"forbidden", "text/plain")
                    r = find(read_csv(csv_path), rid)
                    if kind == "thumb":
                        return self.send(200, make_thumb(r, wd, root).read_bytes(), "image/jpeg")
                    with big_images:  # at most 2 full-size decodes at a time
                        img, _ = load(src_of(root, r))
                        im = render(img, parse_crop(r["crop"]), int(r["rotate"] or 0))
                    im.thumbnail((1800, 1800))
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=90)
                    return self.send(200, buf.getvalue(), "image/jpeg")
            except KeyError:
                return self.send(404, b"not found", "text/plain")
            except Exception as e:
                return self.send(500, str(e).encode(), "text/plain")
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self.host_ok():
                return
            origin = self.headers.get("Origin")
            if (urlparse(self.path).path != "/api/update"
                    or (origin is not None and origin not in {f"http://{h}" for h in hosts})
                    or not self.headers.get("Content-Type", "").startswith("application/json")
                    or not secrets.compare_digest(self.headers.get("X-Token", ""), token)):
                return self.send(403, b"forbidden", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 10000:
                    raise ValueError("bad request size")
                req = json.loads(self.rfile.read(n))
                rid, field, value = str(req["id"]), str(req["field"]), str(req["value"]).strip()
                seen = str(req.get("v", ""))
                if field not in ("date", "rotate", "status"):
                    raise ValueError("field not editable")
                if field == "date" and value:
                    m = DATE_RE.match(value)
                    if not m:
                        raise ValueError(f"{value!r}: use YYYY-MM-DD, YYYY-MM or YYYY")
                    date(int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1))  # real calendar date?
                    if not ymin <= int(m.group(1)) <= ymax:
                        raise ValueError(f"{value}: outside {ymin}-{ymax}, the years these prints can be from")
                if field == "rotate" and value not in ("0", "90", "180", "270"):
                    raise ValueError("rotate must be 0/90/180/270")
                if field == "status" and value not in ("ok", "review", "skip"):
                    raise ValueError("bad status")

                if field == "rotate":  # render first: a failing photo must not get a half-applied edit
                    cur = find(read_csv(csv_path), rid)
                    if cur["status"] == "error":
                        raise ValueError("this photo failed analysis; skip it, or run scan again to retry")
                    make_thumb(dict(cur, rotate=value), wd, root)

                def edit(rows):
                    r = find(rows, rid)
                    if version(r) != seen:  # the scan (or another tab) changed it since the page showed it
                        raise Stale(dict(r))
                    failed = r["status"] == "error" or not r["crop"]
                    if failed and not (field == "status" and value in ("skip", "review")):
                        raise ValueError("this photo failed analysis; skip it, or run scan again to retry")
                    unskip = field == "status" and value == "review"
                    r[field] = value
                    # yours now: a running or later scan never overwrites what you set (un-skip: the scan's again)
                    mine = edited_fields(dict(r, status="")) | {field}
                    r["edited"] = ",".join(sorted(mine - {"status"} if unskip else mine))
                    if failed and field == "status" and value == "review":
                        r["status"] = "error"  # un-skipping a failed photo: it's still failed
                    # typing a date settles the date question only; other problems keep it flagged
                    if (field == "date" and value and r["status"] == "review"
                            and all(is_date_flag(f) for f in r["flags"].split())):
                        r["status"] = "ok"
                    return dict(r)
                r = update_csv(csv_path, edit)
                body = {"ok": True, "row": public(r)}
            except Stale as e:
                body = {"ok": False, "stale": True, "row": public(e.args[0]),
                        "error": "This photo was just updated by the scan; have another look."}
            except Exception as e:
                body = {"ok": False, "error": str(e)}
            self.send(200, json.dumps(body).encode(), "application/json")

    srv = None
    for port in range(8765, 8800):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), H)
            break
        except OSError:
            continue
    if srv is None:
        raise SystemExit("no free port between 8765 and 8799")
    hosts.update({f"127.0.0.1:{port}", f"localhost:{port}"})
    cname = f"sd{port}"  # cookies are shared across ports on 127.0.0.1
    url = f"http://127.0.0.1:{port}/"
    single.seek(0)
    single.truncate()
    single.write(url)
    single.flush()
    print(f"Review page: {url}  (edits are saved to {csv_path})\nPress Ctrl+C when done.")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        single.close()


# ======================================================================= commands
def work_dir_for(scans):
    """The review folder sits next to the scans folder (<name>-review), outside it so nothing is ever
    written among your scans. If that isn't possible (e.g. the scans folder is a share's root), it
    goes to ~/stampede/<name>-review (~/stamp_dates/... if an older version already put it there)."""
    root = Path(scans).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"not a folder: {root}")
    wd = root.parent / (root.name + "-review")
    if not wd.exists() and not os.access(root.parent, os.W_OK):
        # e.g. a NAS share's root; "Photos-1" is the same share remounted, so use the base name
        name = re.sub(r"-\d+$", "", root.name) + "-review"
        wd = Path.home() / "stampede" / name
        if not wd.exists() and (Path.home() / "stamp_dates" / name).exists():
            wd = Path.home() / "stamp_dates" / name  # made before the rename to stampede
    return root, wd


_SCAN_LOCK = []


def cmd_scan(a):
    a.ymin, a.ymax = (int(x) for x in YEARS.split("-"))
    try:
        a.allowed = [0, 90, 180, 270] if a.rotate == "auto" else [int(x) for x in a.rotate.split(",")]
        assert a.allowed and all(x in (0, 90, 180, 270) for x in a.allowed)
    except (ValueError, AssertionError):
        raise SystemExit("--rotate: auto, one angle (0/90/180/270), or a list to choose from like 0,270")
    if len(a.allowed) == 1:
        a.rotate = str(a.allowed[0])  # fixed
    else:
        a.rotate = "auto"
    a.faces = "vision" if sys.platform == "darwin" else "dnn"
    root, wd = work_dir_for(a.scans)
    wd.mkdir(parents=True, exist_ok=True)
    out = wd / "photos.csv"
    one_scan = open(wd / "scan.lock", "a")  # kept open (and locked) until the process ends
    try:
        fcntl.flock(one_scan.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"Another scan of {root} is still running.")
    _SCAN_LOCK.append(one_scan)  # not on `a`: that goes to the worker processes
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in EXTS
                   and not any(part.startswith((".", "@", "#")) for part in p.relative_to(root).parts))

    # Check every known row against the files (outside the lock: may hash files on a slow share)
    snapshot = read_csv(out) if out.exists() else []
    on_disk = {str(p.relative_to(root)): p for p in files}
    status, moved = {}, {}
    known = {r["rel"] for r in snapshot}
    new_files = [rel for rel in on_disk if rel not in known]
    new_sigs = None
    crashed = {r["rel"] for r in snapshot if "worker-crashed" in r["flags"]}
    outdated = 0
    for r in snapshot:
        if r["status"] == "error" and "crashes-the-analysis" in r["flags"]:
            status[r["rel"]] = ("same", r.get("src_sig", ""))  # crashed even on its own: skip it on the review page
        elif r["status"] == "error" or (not r["crop"] and r["status"] != "skip"):
            status[r["rel"]] = ("retry", "")
        elif (r["rel"] in on_disk and r["evidence"] and not frozen(r)
              and evidence_of(r).get("av", 1) < ANALYSIS_VERSION):
            status[r["rel"]] = ("retry", "")  # analysed by an older version: do it again
            outdated += 1
        elif r["rel"] in on_disk:
            status[r["rel"]] = scan_status(r, on_disk[r["rel"]])
        else:
            # gone from its old place: was it moved/sorted into another folder? match by content
            if new_sigs is None:  # hash only new files whose size matches a missing row's
                sizes = {x.get("src_sig", "").split("-")[0] for x in snapshot if x["rel"] not in on_disk}
                new_sigs = {src_sig(on_disk[rel]).partition("@")[0]: rel for rel in new_files
                            if str(on_disk[rel].stat().st_size) in sizes}
            content = r.get("src_sig", "").partition("@")[0]
            if content and ":" not in content and content in new_sigs:
                moved[r["rel"]] = new_sigs.pop(content)
                status[r["rel"]] = ("moved", "")
            else:
                status[r["rel"]] = ("missing", "")
    if len(snapshot) >= 5 and sum(v[0] == "missing" for v in status.values()) > len(snapshot) / 2:
        raise SystemExit(f"{out} doesn't match {root}: most of its photos can't be found there. "
                         f"Is this the right folder? (to start fresh, move or rename {wd.name})")

    import json
    carry_file = wd / "carry-over.json"  # your date/skip of replaced scans, until the new scan is analysed
    try:
        carry = json.loads(carry_file.read_text())
    except (OSError, ValueError):
        carry = {}

    def save_carry():
        if carry:
            tmp = carry_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(carry))
            os.replace(tmp, carry_file)
        else:
            carry_file.unlink(missing_ok=True)

    def prepare(rows):
        keep, changed, unsure = [], [], []
        for r in rows:
            st, sig = status.get(r["rel"], ("same", r.get("src_sig", "")))
            if st == "retry":
                continue
            if st == "changed":
                changed.append(r["rel"])
                if frozen(r) or "date" in edited_fields(r):  # your date / skip carries over to the new scan
                    carry[r["rel"]] = {"date": r["date"] if "date" in edited_fields(r) else "",
                                       "skip": r["status"] == "skip"}
                continue
            if st == "moved":
                r["rel"] = moved[r["rel"]]
                r["source"] = str(root / r["rel"])
            elif st in ("same", "unsure"):
                r["src_sig"] = sig or r.get("src_sig", "")
                r["source"] = str(root / r["rel"])
                if st == "unsure" and r["status"] != "skip":
                    r["status"] = "review"
                    r["flags"] = (r["flags"] + " scan-may-have-changed").strip()
                    unsure.append(r["rel"])
            keep.append(r)
        if changed:
            import time
            bak = out.with_name(f"photos.before-rescan-{time.strftime('%Y%m%d-%H%M%S')}.csv")
            write_csv(bak, rows)  # the replaced scans' old rows (and your edits) stay recoverable
            print(f"{len(changed)} scans were replaced since they were analysed; analysing them again "
                  f"(previous rows saved in {bak.name}): {', '.join(changed[:5])}{' ...' if len(changed) > 5 else ''}")
        if unsure:
            print(f"{len(unsure)} scans may have been replaced (same size, new timestamp): flagged for a look")
        if moved:
            print(f"{len(moved)} scans were moved to other folders: their rows and edits moved with them")
        rows[:] = keep
        save_carry()
        return {r["rel"] for r in rows}
    done = update_csv(out, prepare)
    if outdated:
        print(f"{outdated} photos were analysed by an older version of this tool: analysing them again "
              f"(photos you've OK'd, dated or skipped keep everything)")
    todo = [p for p in files if str(p.relative_to(root)) not in done][: a.limit or None]
    print(f"{len(files)} scans in {root}, {len(done)} already analysed, analysing {len(todo)}")

    import json
    import time
    t0 = time.time()
    if todo:
        AppleVision()  # fail fast if Vision isn't usable
        jobs = max(1, min(a.jobs, len(todo)))
        print(f"Pass 1/3: Apple Vision on {len(todo)} scans ({jobs} at a time) ...")
        if a.jobs < 0:
            raise SystemExit("--jobs: 0 or more")
        buf, last_flush = [], time.time()

        def flush():
            nonlocal buf, last_flush
            if buf:
                used = []
                for r in buf:
                    if r["rel"] in carry and r["status"] != "error":
                        ev = evidence_of(r)
                        ev["previous"] = carry[r["rel"]]
                        r["evidence"] = json.dumps(ev)
                        used.append(r["rel"])
                got = {r["rel"]: r for r in buf}
                update_csv(out, lambda rows: rows.__setitem__(slice(None), [r for r in rows if r["rel"] not in got]
                                                               + list(got.values())))
                if used:  # only now safely in photos.csv
                    for rel in used:
                        carry.pop(rel, None)
                    save_carry()
            buf, last_flush = [], time.time()
        for n, row in enumerate(pass1(todo, root, wd, a, a.jobs, crashed), 1):
            buf.append(row)
            if len(buf) >= 10 or time.time() - last_flush > 5:
                flush()
            eta = (time.time() - t0) / n * (len(todo) - n) / 60
            what = f"read {row['ocr_raw']!r}" if row["ocr_raw"] else "no stamp read"
            if row["status"] == "error":
                what = row["flags"]
            print(f"  [{n}/{len(todo)}] {row['rel']}  {what}" + (f"   (~{eta:.0f} min left)" if n % 10 == 0 else ""))
        flush()

    # Pass 2: decide from the whole batch. Rows you've edited are never touched.
    rows = read_csv(out)
    now = scan_settings(a)
    resettled = [r for r in rows if r["status"] not in ("error", "skip") and "date" not in edited_fields(r)
                 and r["evidence"]
                 and "unresolved" not in r["flags"].split() and evidence_of(r).get("settings", now) != now]
    again = {id(r) for r in resettled}
    if resettled:
        print(f"{len(resettled)} photos were decided with other --date-order/--rotate settings: deciding them again "
              f"(photos you've edited keep your edits)")
    # rows not decided yet are decided even if you've already edited them (your fields are kept)
    pending = [r for r in rows if r["status"] != "error"
               and ("unresolved" in r["flags"].split()
                    or (not frozen(r) and not a.fast and evidence_of(r).get("model_pending"))
                    or id(r) in again)]
    if pending:
        batch = Batch(rows, a)
        for f, o in sorted(batch.orders.items()):
            print(f"Pass 2/3: {f if f != '.' else 'top folder'}: the camera prints dates as {o} "
                  f"(used for guesses you can accept on the review page)")
        later, decided = [], {}
        for r in pending:
            old_date = r["date"]
            ask_date, ask_rot = batch.finish(r)
            ev = evidence_of(r)
            if id(r) in again and frozen(r) and r["status"] != "skip" and r["date"] != old_date:
                r["_unfreeze"] = True  # OK'd by you, but new settings changed its date: yours to look at again
            if "rotate" in edited_fields(r):
                ask_rot = None  # you've set the rotation
            if (frozen(r) or r["status"] == "skip") and not r.get("_unfreeze"):
                if ask_date or ask_rot:  # settled by you: no model time on it, unless you un-skip it later
                    ev["model_pending"] = True
            elif ask_date or ask_rot:
                if a.fast:
                    ev["model_pending"] = True  # a later run without --fast asks the model
                else:
                    later.append(r["rel"])
                    r["flags"] = (r["flags"] + " unresolved").strip()  # pass-2 decision saved; model still to come
                    r["status"] = "review"
            else:
                ev.pop("model_pending", None)
            r["evidence"] = json.dumps(ev)
            decided[r["rel"]] = r
        commit(out, decided)
        print(f"Pass 2/3: {len(pending) - len(later)} of {len(pending)} decided"
              f"{f'; {len(later)} get one look from the Qwen model' if later else ''}")

        # Pass 3: the model, for what's still open (one small stamp crop each, plus rotation if needed)
        if later:
            try:
                mlx = MLX(a.model)
            except (Exception, SystemExit) as e:
                print(f"Qwen model unavailable ({e}); those photos keep their guesses and stay flagged. "
                      f"Run scan again to retry.")
                later = []
            t1 = time.time()
            by_rel = {r["rel"]: r for r in rows}
            later.sort(key=lambda rel: (not evidence_of(by_rel[rel]).get("reads"), rel))
            for n, rel in enumerate(later, 1):
                r = by_rel[rel]
                before = dict(r)
                try:
                    ask_date, _ = batch.finish(dict(r))
                    work = model_check(r, root, a, mlx, ask_date)
                    _, ask = batch.finish(r)
                    if ask:
                        cands, settle = ask
                        pick, sure = vlm_pick(mlx, work, cands, settle)
                        ev = evidence_of(r)
                        ev["model_orient"] = {"pick": pick, "sure": sure}
                        r["evidence"] = json.dumps(ev)
                    batch.finish(r)
                    ev = evidence_of(r)
                    ev.pop("model_pending", None)
                    r["evidence"] = json.dumps(ev)
                except KeyboardInterrupt:
                    r.clear()
                    r.update(before)  # keeps its pass-2 decision and 'unresolved': retried next run
                    raise
                except Exception as e:
                    r.clear()
                    r.update(before)
                    r["flags"] = (r["flags"] + f" model-error:{type(e).__name__}").strip()
                commit(out, {rel: r})
                left = (time.time() - t1) / n * (len(later) - n) / 60
                print(f"Pass 3/3: [{n}/{len(later)}] {rel}: "
                      f"{r['date'] or ('guess ' + r['date_guess'] if r['date_guess'] else '-')}"
                      f" rotate {r['rotate']} {r['flags']}   (~{left:.0f} min left)")
    print(f"Done in {(time.time() - t0) / 60:.1f} min.")
    rows = read_csv(out)
    st = defaultdict(int)
    for r in rows:
        st[r["status"]] += 1
    print(f"\n{dict(st)}\nCSV: {out}")
    _SCAN_LOCK.pop().close()  # the scan is over: another one may start while the review page is open
    if rows and not a.no_review:
        a.no_browser = False
        cmd_review(a)  # opens the editable review page; Ctrl+C when done


DATE_RE = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")
YEAR_RE = re.compile(r"(?<![0-9A-Za-z-])((?:19|20)\d\d)(?![0-9A-Za-z-])")  # "Summer 1987", "1987_x"; not "1990s"/"2024-03"
def folder_year(rel, ymin, ymax):
    """Year from the innermost folder that names one plainly (1990s/1987/x.jpg -> 1987)."""
    for part in reversed(Path(rel).parent.parts):
        for m in YEAR_RE.finditer(part):
            if ymin <= int(m.group(1)) <= ymax:
                return m.group(1)
    return None


def inside(path, root):
    """True if path is root or inside it, comparing real files (case-insensitive APFS, symlinks)."""
    p = Path(path)
    for q in [p, *p.parents]:
        try:
            if q.exists() and os.path.samefile(q, root):
                return True
        except OSError:
            pass
    return False


TAG = "Scanned prints"  # Immich shows XMP TagsList as (nested) tags
MANIFEST = ".stampede-manifest.json"  # in --out: what each output was written from
OLD_MANIFEST = ".stamp_dates-manifest.json"  # its name before the rename to stampede


def load_manifest(out_dir):
    import json
    for name in (MANIFEST, OLD_MANIFEST):
        try:
            return json.loads((out_dir / name).read_text())
        except (OSError, ValueError):
            continue
    return {}


def folder_id(wd):
    """Random id of a scans folder, kept in its review folder; tells apart outputs of different
    scans folders written into the same --out."""
    import uuid
    f = wd / "folder-id"
    if not f.exists():
        wd.mkdir(parents=True, exist_ok=True)
        f.write_text(uuid.uuid4().hex)
    return f.read_text().strip()


def manifest_key(fid, rel):
    return f"{fid}/{rel}"


def save_manifest(out_dir, m):
    import json
    tmp = out_dir / (MANIFEST + ".tmp")
    with tmp.open("w") as f:
        f.write(json.dumps(m, indent=0, sort_keys=True))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_dir / MANIFEST)
    (out_dir / OLD_MANIFEST).unlink(missing_ok=True)  # migrated


def plan(a):
    """Validate every row of photos.csv; returns (jobs, problems). Used by dry-run and the real run."""
    root, wd = work_dir_for(a.scans)
    csv_path = wd / "photos.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found: run `scan {a.scans}` first")
    rows = read_csv(csv_path)
    problems, jobs, seen, flagged, existing, missing = [], [], {}, [], [], []
    manifest = load_manifest(Path(a.out).expanduser().resolve())
    fid = folder_id(wd)
    by_path = {}  # output path -> manifest key that wrote it
    for k in manifest:
        by_path[k.split("/", 1)[1] if "/" in k and len(k.split("/", 1)[0]) == 32 else k] = k
    ymin, ymax = (int(x) for x in YEARS.split("-"))
    out_dir = Path(a.out).expanduser().resolve()
    if inside(out_dir, root):
        problems.append(f"--out {out_dir} is inside the scans folder; use a separate folder")
    if not shutil.which("exiftool"):
        problems.append("exiftool not found (brew install exiftool)")
    if a.fallback_date and not DATE_RE.match(a.fallback_date):
        problems.append(f"--fallback-date {a.fallback_date!r}: use YYYY, YYYY-MM or YYYY-MM-DD")
    def date_text(r):
        ds = r["date"].strip()
        if not ds:
            ds = folder_year(r["rel"], ymin, ymax) or a.fallback_date or ""
        return ds
    same_day = defaultdict(list)  # photos sharing a date get noon + their rank (by file name) in seconds:
    for r in rows:                # keeps scan order in Immich and doesn't change when others are skipped
        same_day[date_text(r)].append(r["rel"])
    rank = {(ds, rel): k for ds, rels in same_day.items() for k, rel in enumerate(sorted(rels))}
    for i, r in enumerate(rows, 2):
        tag = f"row {i} ({r['rel']})"
        if r["status"] == "skip":
            continue
        if r["status"] == "error":
            problems.append(f"{tag}: analysis failed: {r['flags']} (set status to skip, or delete the row "
                            f"and run scan again)")
            continue
        src = src_of(root, r)
        if not src.is_file():
            missing.append(r["rel"])
            continue
        if not same_scan(r, src):
            problems.append(f"{tag}: the scan file changed after it was analysed; run scan again")
            continue
        if r["status"] == "review" and not a.include_flagged:
            flagged.append(r["rel"])
            continue
        try:
            x0, y0, x1, y1 = parse_crop(r["crop"])
            W, H = oriented_size(src)
            if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
                raise ValueError
        except Exception:
            problems.append(f"{tag}: bad crop {r['crop']!r} for image size")
            continue
        if r["rotate"].strip() not in ("0", "90", "180", "270"):
            problems.append(f"{tag}: rotate must be 0/90/180/270, got {r['rotate']!r}")
            continue
        ds, how = r["date"].strip(), "date"
        if not ds:  # fallbacks: a year in a folder name, then --fallback-date
            yr = folder_year(r["rel"], ymin, ymax)
            ds, how = (yr, "folder") if yr else ((a.fallback_date, "fallback") if a.fallback_date else ("", "none"))
        ts = None
        if ds:
            m = DATE_RE.match(ds)
            try:
                if not m:
                    raise ValueError
                dd = date(int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1))
            except ValueError:
                problems.append(f"{tag}: bad date {ds!r} (use YYYY-MM-DD, YYYY-MM or YYYY)")
                continue
            if not ymin <= dd.year <= ymax:
                problems.append(f"{tag}: date {ds} is outside {ymin}-{ymax}")
                continue
            if how == "date" and not m.group(3):
                how = "partial"  # you typed only a year or month: written as the 1st, tagged estimated
            n = rank[(date_text(r), r["rel"])]
            ts = f"{dd:%Y:%m:%d} {12 + n // 3600:02d}:{n // 60 % 60:02d}:{n % 60:02d}"
        dst = out_dir / r["rel"]
        if inside(dst, root) or (dst.exists() and os.path.samefile(dst, src)):
            problems.append(f"{tag}: output {dst} would overwrite a scan")
            continue
        if str(dst) in seen:
            problems.append(f"{tag}: same output file as row {seen[str(dst)]}: {dst}")
            continue
        seen[str(dst)] = i
        fp = f"{r.get('src_sig', '').partition('@')[0]}|{r['crop']}|{r['rotate']}|{ts}|{how}"  # everything the output depends on
        key = manifest_key(fid, r["rel"])
        prev, legacy = manifest.get(key), None
        if prev is None and by_path.get(r["rel"]) == r["rel"]:
            prev, legacy = manifest[r["rel"]], r["rel"]  # written by an older version of this tool (no folder id)
        if dst.exists() or os.path.islink(dst):
            if prev is None:  # never replace a file this tool didn't write from this folder
                other = by_path.get(r["rel"])
                problems.append(f"{tag}: {dst} exists and was "
                                f"{'written from a different scans folder' if other else 'not written by this tool'}"
                                f"; use another --out (or move that file away)")
                continue
            if prev == fp and not a.overwrite:
                existing.append(r["rel"])  # identical to what an earlier run wrote: skip (resume)
                continue
            # written by an earlier run from this folder but the row changed since: redo
        jobs.append(dict(row=i, rel=r["rel"], key=key, src=src, dst=dst, crop=(x0, y0, x1, y1),
                         rotate=int(r["rotate"]), ts=ts, how=how, fp=fp, flagged=r["status"] == "review",
                         legacy=legacy))
    active = {j["rel"] for j in jobs} | set(existing)
    own = [k.split("/", 1)[1] for k in manifest if k.startswith(fid + "/")]  # only this folder's outputs
    stale = sorted(rel for rel in own if rel not in active and (out_dir / rel).exists())
    if rows and len(missing) == len([r for r in rows if r["status"] != "skip"]):
        problems.append(f"none of the scans were found under {root}: is the folder (or NAS share) available?")
    return jobs, problems, flagged, existing, missing, stale


def save(img, icc, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    ext = dst.suffix.lower()
    kw = {"icc_profile": icc} if icc else {}
    if ext in (".jpg", ".jpeg"):
        img.save(dst, "JPEG", quality=95, subsampling=0, **kw)
    elif ext in (".tif", ".tiff"):
        img.save(dst, "TIFF", compression="tiff_lzw", **kw)
    else:
        img.save(dst, **kw)


def cmd_apply(a):
    jobs, problems, flagged, existing, missing, stale = plan(a)
    count = defaultdict(int)
    for j in jobs:
        count[j["how"]] += 1
    if a.dry_run:
        for j in jobs:
            print(f"row {j['row']}: {j['dst']}  rotate {j['rotate']}  "
                  f"{j['ts'][:10].replace(':', '-') if j['ts'] else 'no date'}"
                  f"{'' if j['how'] in ('date', 'none') else ' (' + j['how'] + ')'}"
                  f"{'  [flagged]' if j['flagged'] else ''}")
    print(f"\n{len(jobs)} photos to write: {count['date']} dated, {count['partial']} year/month only, "
          f"{count['folder']} with the year from their folder name, {count['fallback']} with --fallback-date, "
          f"{count['none']} without date"
          f"{' (Immich will put these on the day they are written)' if count['none'] else ''}.")
    if existing:
        print(f"{len(existing)} already written by an earlier run and unchanged since: left as they are.")
    if missing:
        print(f"{len(missing)} scans in photos.csv no longer exist and are left out: "
              f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
    if stale:
        print(f"NOTE: {len(stale)} photos in --out were written earlier but are now skipped, flagged or gone, so "
              f"they may be out of date (delete them if unwanted): {', '.join(stale[:10])}"
              f"{' ...' if len(stale) > 10 else ''}")
    if flagged:
        print(f"{len(flagged)} flagged photos are NOT written (check them on the review page and mark OK, "
              f"or pass --include-flagged).")
    for p in problems:
        print("  PROBLEM:", p)
    if problems:
        raise SystemExit(1)
    if a.dry_run:
        print("Dry run: nothing written.")
        return
    failed = 0
    out_dir = Path(a.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(out_dir)
    for k, j in enumerate(jobs, 1):
        dst = j["dst"]
        tmp = dst.with_name(f".{dst.stem}.partial{dst.suffix}")  # renamed into place only when complete
        try:
            img, icc = load(j["src"])
            save(render(img, j["crop"], j["rotate"]), icc, tmp)
            args = ["exiftool", "-q", "-overwrite_original", f"-XMP-digiKam:TagsList+={TAG}"]
            if j["how"] in ("folder", "fallback", "partial"):
                args.append(f"-XMP-digiKam:TagsList+={TAG}/Date estimated")
            elif not j["ts"]:
                args.append(f"-XMP-digiKam:TagsList+={TAG}/No date")
            if j["ts"]:
                args += [f"-DateTimeOriginal={j['ts']}", f"-CreateDate={j['ts']}", f"-ModifyDate={j['ts']}",
                         f"-FileModifyDate={j['ts']}"]
            res = subprocess.run(args + [str(tmp)], capture_output=True, text=True)
            if res.returncode:
                raise RuntimeError(res.stderr.strip())
            manifest[j["key"]] = "pending"  # a crash from here on: next run sees "pending" and redoes it
            save_manifest(out_dir, manifest)
            os.replace(tmp, dst)
            manifest[j["key"]] = j["fp"]
            if j["legacy"]:
                manifest.pop(j["legacy"], None)
            save_manifest(out_dir, manifest)
            print(f"[{k}/{len(jobs)}] {dst}")
        except Exception as e:
            failed += 1
            tmp.unlink(missing_ok=True)
            print(f"FAILED row {j['row']} {j['src']}: {e}", file=sys.stderr)
        except BaseException:  # Ctrl+C: no half-written file left in --out
            tmp.unlink(missing_ok=True)
            raise
    print(f"Done: {len(jobs) - failed} written, {failed} failed.")


def cmd_debug(a):
    """Explain every step for one scan (for troubleshooting). Saves stamp crops next to the scan."""
    a.ymin, a.ymax = (int(x) for x in YEARS.split("-"))
    a.order, a.rotate, a.allowed = a.date_order, "auto", [0, 90, 180, 270]
    a.faces = "vision" if sys.platform == "darwin" else "dnn"
    p = Path(a.file).expanduser()
    if p.is_dir():  # a folder: debug each scan in it
        for f in sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in EXTS
                        and not x.name.startswith(".")):
            print(f"\n===== {f.name}")
            a.file = str(f)
            cmd_debug(a)
        return
    out = Path.home() / "stampede-debug" / p.stem  # outside the scans folder
    out.mkdir(parents=True, exist_ok=True)
    img, _ = load(p)
    rgb = np.asarray(img)
    box, flags = find_crop(rgb, 18, 0.004)
    print(f"image {img.size}  crop {box}  flags {flags}")
    work, _ = shrink(np.ascontiguousarray(rgb[box[1]:box[3], box[0]:box[2]]), WORK)
    print(f"working copy {work.shape[1]}x{work.shape[0]}")
    eng = AppleVision()
    cands = stamp_candidates(work)
    print(f"stamp candidates (score, rotation, box): {[(round(float(c[0]), 1), c[1], tuple(map(int, c[2]))) for c in cands]}")
    for k, (_, r, bx) in enumerate(cands[:3]):
        c, variants, masks, h = line_crops(rot(work, r), bx)
        lays = layouts(masks, h)
        Image.fromarray(np.vstack([cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)] + masks)).save(out / f"stamp{k}_rot{r}.png")
        print(f"\ncandidate {k} rot {r}: line height {h}, groups {[(int(x), int(y)) for x, y in groups(masks[0], h)]}")
        print(f"  layouts (group widths/height, apostrophe): {[([round(x, 2) for x in w], ap) for w, ap in lays]}")
        for txt, sc in eng.line(variants, groups(masks[0], h), h):
            d, note = parse_date(txt, a.order, a.ymin, a.ymax)
            rg = None if d else regroup_any(txt, lays, a.order, a.ymin, a.ymax)
            print(f"  Vision read {txt!r} conf {sc:.2f} -> parse {d} {note} | regroup {rg!r}")
    print("\nVision full-photo reads (rotation, text):")
    top = []
    if cands:
        _, _, tm, th = line_crops(rot(work, cands[0][1]), cands[0][2])
        top = layouts(tm, th)
    for r, txt, sc in eng.full(work):
        print(f"  {r:>3}: {txt!r} {sc:.2f} -> {parse_date(txt, a.order, a.ymin, a.ymax)[0]} | regroup "
              f"{regroup_any(txt, top, a.order, a.ymin, a.ymax)!r}")
    print("\nfaces (Apple Vision): (confidence, roll degrees) on the photo as scanned, then turned 90:")
    for r in (0, 90):
        try:
            print(f"  rot {r:>3}: {[(round(c, 2), None if rl is None else round(rl)) for c, rl in _vision_faces(rot(work, r))]}")
        except Exception as e:
            print(f"  rot {r:>3}: ERROR {e}")
    print(f"  upright-face score per rotation: {face_scores(work, a.faces)}")
    print(f"\nstamp crops saved in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="analyse scans, then open the review page")
    s.add_argument("scans", help="folder with the scans (subfolders included)")
    s.add_argument("--rotate", default="auto",
                   help="auto (default), one angle for all (0/90/180/270 clockwise), "
                        "or the angles to choose from, e.g. 0,270 if all prints lay the same way")
    s.add_argument("--date-order", dest="order", choices=["auto", "ymd", "dmy", "mdy"], default="auto",
                   help="how your camera printed dates without an apostrophe year, e.g. ymd for 87 6 14 "
                        "(default auto: unclear ones like 05 06 07 are flagged)")
    s.add_argument("--fast", action="store_true", help="skip the Qwen model: open questions stay flagged for you")
    s.add_argument("--limit", type=int, help="only analyse this many new scans (for a trial)")
    s.add_argument("--jobs", type=int, default=4, help=argparse.SUPPRESS)
    s.add_argument("--model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit", help=argparse.SUPPRESS)
    s.add_argument("--no-review", action="store_true", help=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_scan)

    r = sub.add_parser("review", help="reopen the review page later without scanning")
    r.add_argument("scans", help="the same folder you scanned")
    r.add_argument("--no-browser", action="store_true", help=argparse.SUPPRESS)
    r.set_defaults(fn=cmd_review)

    d = sub.add_parser("debug", help=argparse.SUPPRESS)
    d.add_argument("file")
    d.add_argument("--date-order", choices=["auto", "ymd", "dmy", "mdy"], default="auto")
    d.set_defaults(fn=cmd_debug)

    p = sub.add_parser("apply", help="write cropped/rotated/dated copies as photos.csv says")
    p.add_argument("scans", help="the same folder you scanned")
    p.add_argument("--out", required=True, help="where to write the finished photos")
    p.add_argument("--dry-run", action="store_true", help="check everything and show the plan; write nothing")
    p.add_argument("--fallback-date", help="date for photos with no date and no year in their folder, e.g. 1990")
    p.add_argument("--overwrite", action="store_true",
                   help="rewrite every photo this tool wrote to --out (default: only new or changed ones)")
    p.add_argument("--include-flagged", action="store_true", help="also write photos still flagged for review")
    p.set_defaults(fn=cmd_apply)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
