"""Run stampede.py as if on a Mac: Apple Vision and mlx-vlm replaced by tests/mocks.

    python tests/run_mock.py scan <folder> --no-review --jobs 0 [...]

Env: see tests/mocks/Vision.py and tests/mocks/mlx_vlm/__init__.py. STRIP_CV2=1 removes cv2.dnn
and CascadeClassifier (some OpenCV builds lack them)."""
import os
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS / "mocks"))
sys.path.insert(0, str(TESTS.parent))
import urllib.request  # noqa: E402,F401  (before faking macOS: on "darwin" it needs _scproxy)
sys.platform = "darwin"
os.environ.setdefault("STAMPEDE_MOCK_VISION", "1")

import cv2  # noqa: E402

if os.environ.get("STRIP_CV2"):
    del cv2.dnn
    del cv2.CascadeClassifier

import mlx_vlm  # noqa: E402
import stampede  # noqa: E402
import Vision  # noqa: E402

if __name__ == "__main__":  # not in worker processes (spawn re-imports this file)
    sys.argv = ["stampede.py"] + sys.argv[1:]
    try:
        stampede.main()
    finally:
        calls = {k: sum(1 for x in mlx_vlm.LOG if k in x) for k in ("load", "line", "full", "orient")}
        print("MOCK CALLS vision:", Vision.CALLS, "mlx:", calls)
