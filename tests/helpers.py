"""Shared helpers for the test suite."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
SCRIPT = REPO / "stampede.py"


def env(**extra):
    """Environment for a mocked run: no OCR/model answers leak in from the caller's shell."""
    e = {k: v for k, v in os.environ.items() if not k.startswith("MOCK_")}
    e.update({k: str(v) for k, v in extra.items()})
    return e


def run_mock(*args, timeout=900, **mock_env):
    """stampede.py <args> with Apple Vision and mlx-vlm mocked (in-process unless --jobs > 0)."""
    p = subprocess.run([sys.executable, str(TESTS / "run_mock.py"), *map(str, args)], capture_output=True,
                       text=True, timeout=timeout, env=env(**mock_env))
    assert "Traceback" not in p.stdout + p.stderr, p.stdout + p.stderr
    return p.stdout + p.stderr


def run_workers(*args, timeout=900, **mock_env):
    """stampede.py run directly with the mocks on PYTHONPATH (real worker processes)."""
    e = env(**mock_env, STAMPEDE_MOCK_VISION=1, PYTHONPATH=str(TESTS / "mocks"))
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True,
                          timeout=timeout, env=e)


def run_sim(scenario, work, *args, keep=False, timeout=600):
    """tests/sim.py: a whole scan with Vision/Qwen answers from a scenario table. Returns MODEL CALLS."""
    e = env(KEEP=1) if keep else env()
    p = subprocess.run([sys.executable, str(TESTS / "sim.py"), str(TESTS / "scenarios" / f"{scenario}.py"),
                        str(work), *map(str, args)], capture_output=True, text=True, timeout=timeout, env=e)
    out = p.stdout + p.stderr
    assert "Traceback" not in out, out
    line = next(x for x in out.splitlines() if x.startswith("MODEL CALLS"))
    return json.loads(line.split(" ", 2)[2]), out


def rows(scans):
    """photos.csv of a scans folder, as {rel: row}."""
    p = Path(scans).parent / (Path(scans).name + "-review") / "photos.csv"
    with open(p, newline="", encoding="utf-8-sig") as f:
        return {r["rel"]: r for r in csv.DictReader(f)}


def evidence(row):
    return json.loads(row["evidence"] or "{}")


def flags(row):
    return row["flags"].split()
