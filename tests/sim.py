"""Scenario simulator: runs a full `scan` in this process with Apple Vision's reads, faces, stamp
location and the Qwen model replaced by a scenario's table, so the batch logic (passes 2 and 3,
CSV merging, edits during a scan, model errors) can be tested exactly.

    python tests/sim.py tests/scenarios/<name>.py <work dir> [extra scan args]

A scenario module defines FILES = {rel: dict(v=[Vision texts], q=Qwen text or None,
face=rotation or None, faces={rot: score}, stamprot=90, nocand=bool)} and optionally:
  ORIENT(rel, candidates) -> rotation      the model's rotation answer (default: first candidate)
  MLX_RAISE = {rel}                       the model raises for these
  on_mlx_line(rel, sd, root)              called while the model "reads" a photo
  BEFORE_PASS2(sd, root)                  called after pass 1 is in photos.csv, before pass 2
Prints MODEL CALLS {...} and writes the usual <root>-review/photos.csv."""
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS / "mocks"))
sys.path.insert(0, str(TESTS.parent))
sys.path.insert(0, str(TESTS))
import synth  # noqa: E402
import urllib.request  # noqa: E402,F401  (before faking macOS: on "darwin" it needs _scproxy)
sys.platform = "darwin"
os.environ.setdefault("STAMPEDE_MOCK_VISION", "1")

import stampede as sd  # noqa: E402

spec = importlib.util.spec_from_file_location("scenario", sys.argv[1])
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
FILES = S.FILES
work = Path(sys.argv[2])
root = work / "Scans"
CALLS = {"mlx_line": 0, "mlx_orient": 0}
cur = {}

_load = sd.load


def load(p):
    cur["rel"] = str(Path(p).relative_to(root))
    return _load(p)


def line(self, variants, grp=None, h=None):
    for t in FILES[cur["rel"]].get("v", []):
        yield t, 0.9


def faces(work_img, how):
    f, fs = FILES[cur["rel"]].get("face"), FILES[cur["rel"]].get("faces")
    if fs:
        return {r: fs.get(r, 0.0) for r in (0, 90, 180, 270)}
    return {r: (0.99 if f == r else 0.0) for r in (0, 90, 180, 270)}


def cands(work_img):
    spec_ = FILES[cur["rel"]]
    if not spec_.get("v") and spec_.get("nocand"):
        return []
    return [(5.0, spec_.get("stamprot", 90), (100, 900, 300, 940))]


class FakeMLX:
    name, strong = "mlx", True

    def __init__(self, *a, **k):
        pass

    def line(self, variants, grp=None, h=None):
        CALLS["mlx_line"] += 1
        if hasattr(S, "on_mlx_line"):
            S.on_mlx_line(cur["rel"], sd, root)
        if cur["rel"] in getattr(S, "MLX_RAISE", set()):
            raise RuntimeError("metal OOM")
        q = FILES[cur["rel"]].get("q")
        yield (q if q is not None else "NONE"), 1.0


def pick_once(eng, work_img, c):
    CALLS["mlx_orient"] += 1
    return S.ORIENT(cur["rel"], c) if hasattr(S, "ORIENT") else c[0]


sd.load = load
sd.AppleVision.line = line
sd.AppleVision.full = lambda self, w: iter(())
sd.face_scores = faces
sd.stamp_candidates = cands
sd.MLX = FakeMLX
sd._vlm_pick_once = pick_once

if hasattr(S, "BEFORE_PASS2"):  # pass 2 starts with a plain read of photos.csv holding every row
    _read, _update, state = sd.read_csv, sd.update_csv, {"in_update": False, "done": False}

    def update_csv(p, fn):
        state["in_update"] = True
        try:
            return _update(p, fn)
        finally:
            state["in_update"] = False

    def read_csv(p):
        rows = _read(p)
        if (not state["in_update"] and not state["done"] and str(p).endswith("photos.csv")
                and {r["rel"] for r in rows} >= set(FILES)):
            state["done"] = True
            S.BEFORE_PASS2(sd, root)
            rows = _read(p)
        return rows

    sd.read_csv, sd.update_csv = read_csv, update_csv

if not os.environ.get("KEEP"):
    shutil.rmtree(work, ignore_errors=True)
    base = work / "base.jpg"
    synth.make_scan(base, synth.make_print("fruits.jpg", stamp="'87 6 14"), 90, seed=1)
    for rel in FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(base.read_bytes() + rel.encode())  # distinct content per file
sys.argv = ["stampede.py", "scan", str(root), "--no-review", "--jobs", "0"] + sys.argv[3:]
try:
    sd.main()
finally:
    print("MODEL CALLS", json.dumps(CALLS))
