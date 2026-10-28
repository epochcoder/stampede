"""Crop (print off the scanner bed) and stamp detection on synthetic scans."""
import os
import random
from pathlib import Path

import numpy as np
import pytest

import stampede as sd
import synth

REVIEW_CROP = ("edge-uncertain", "print-tilted", "maybe-cut", "odd-aspect", "multiple-prints")


def analyse_crop(path):
    img, _ = sd.load(path)
    rgb = np.asarray(img)
    box, flags = sd.find_crop(rgb, 18, 0.004)
    work, _ = sd.shrink(np.ascontiguousarray(rgb[box[1]:box[3], box[0]:box[2]]), sd.WORK)
    return rgb, box, flags, work


def frame_rotation(truth, portrait):
    """Rotation at which the camera's (landscape) frame is upright: the stamp's rotation."""
    return (truth["undo_rot"] + (90 if portrait else 0)) % 360


def test_crop_matches_the_print(standard):
    folder, truth = standard
    for name, t in truth.items():
        _, box, flags, _ = analyse_crop(folder / f"{name}.jpg")
        err = max(abs(a - b) for a, b in zip(box, t["box"]))
        if "maybe-cut-top" in flags:  # white sky into white bed: may cut, but must say so
            continue
        assert err <= 12, (name, box, t["box"], flags)  # inside the print by the ~0.7 mm margin, no more


def test_stamp_found_at_the_camera_frame(standard):
    folder, truth = standard
    for name, src, portrait, stamp, *_ in synth.STANDARD:
        if not stamp:
            continue
        *_, work = analyse_crop(folder / f"{name}.jpg")
        cands = sd.stamp_candidates(work)
        assert cands, name
        assert cands[0][1] == frame_rotation(truth[name], portrait), (name, cands)


@pytest.mark.parametrize("portrait, placed", [(True, 0), (False, 0), (False, 90), (True, 180)])
def test_orange_things_away_from_the_stamp_corner_are_not_stamps(tmp_path, samples, portrait, placed):
    # wooden armrests, tartan, shelf edges: thin bright orange lines in the middle of a photo
    pr = synth.make_print("apple.jpg", portrait=portrait, kind="orange", seed=5)
    synth.make_scan(tmp_path / "x.jpg", pr, placed, seed=5)
    *_, work = analyse_crop(tmp_path / "x.jpg")
    assert sd.stamp_candidates(work) == []


def test_stamp_still_found_next_to_orange_things(tmp_path, samples):
    pr = synth.make_print("apple.jpg", stamp="'91 7 4", kind="orange", seed=6)
    synth.make_scan(tmp_path / "x.jpg", pr, 90, seed=6)
    *_, work = analyse_crop(tmp_path / "x.jpg")
    cands = sd.stamp_candidates(work)
    assert cands and cands[0][1] == 270


def _edge_cases(n):
    rng = random.Random(7)
    srcs = [s for s in synth.SAMPLES]
    for i in range(n):
        yield dict(src=rng.choice(srcs), dpi=rng.choice([300, 600]),
                   tilt=rng.choice([0, 0, 0.1, -0.15, 0.25, -0.3, 0.4, -0.5, 0.7]),
                   kind=rng.choice(["plain", "plain", "sky", "dark", "whitewall"]), rim=rng.choice([0, 0, 1, 2]),
                   placed=rng.choice([0, 90, 180, 270]), seed=i)


def _visible_bed_line(rgb, inprint, box):
    """Fraction of the outermost two lines on each side that is scanner bed next to photo content
    that isn't itself near-white (i.e. a white line you'd see)."""
    x0, y0, x1, y1 = box
    g = rgb.astype(np.int16).mean(-1)
    d = 10
    worst = 0.0
    for i in range(2):
        for line, inner in ((inprint[y0 + i, x0:x1], g[y0 + d, x0:x1]), (inprint[y1 - 1 - i, x0:x1], g[y1 - 1 - d, x0:x1]),
                            (inprint[y0:y1, x0 + i], g[y0:y1, x0 + d]), (inprint[y0:y1, x1 - 1 - i], g[y0:y1, x1 - 1 - d])):
            worst = max(worst, float(((~line) & (inner < 225)).mean()))
    return worst


@pytest.mark.parametrize("n", [int(os.environ.get("STAMPEDE_TEST_EDGES", "0")) or (120 if os.environ.get("STAMPEDE_TEST_FULL") else 16)])
def test_no_white_edges_and_no_silent_overtrim(tmp_path, samples, n):
    """Tilted prints, lid shadow/highlight, paper rims, 300/600 dpi, white sky/walls: no bed line may
    remain on any edge, and a crop that cuts into the photo beyond what a tilt makes unavoidable
    must be flagged for review."""
    problems = []
    for c in _edge_cases(n):
        pr = synth.make_print(c["src"], kind=c["kind"], rim=c["rim"], dpi=c["dpi"], seed=c["seed"])
        t = synth.make_scan(tmp_path / "s.jpg", pr, c["placed"], tilt=c["tilt"], dpi=c["dpi"], seed=c["seed"])
        rgb, box, flags, _ = analyse_crop(tmp_path / "s.jpg")
        flagged = any(f.startswith(REVIEW_CROP) for f in flags)
        line = _visible_bed_line(rgb, t["inprint"], box)
        w, h = synth.ideal_trim(t["size"], c["tilt"])  # size is already as placed
        extra = max((w - (box[2] - box[0])) / 2, (h - (box[3] - box[1])) / 2)
        if line > 0.002 and not flagged:
            problems.append(("white line", c, round(line, 3), flags))
        if extra > max(8, 0.006 * max(t["size"])) and not flagged:
            problems.append(("over-trim", c, round(extra, 1), flags))
    assert not problems, problems
