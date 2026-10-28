import importlib.util
import os
import shutil
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(TESTS.parent))

import synth  # noqa: E402


def has(module):
    return importlib.util.find_spec(module) is not None


needs_rapidocr = pytest.mark.skipif(not has("rapidocr_onnxruntime"),
                                    reason="pip install rapidocr-onnxruntime (stand-in for Apple Vision's OCR)")
needs_playwright = pytest.mark.skipif(not has("playwright"), reason="pip install playwright && playwright install chromium")
needs_exiftool = pytest.mark.skipif(not shutil.which("exiftool"), reason="exiftool not installed")
needs_node = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
full = pytest.mark.skipif(not os.environ.get("STAMPEDE_TEST_FULL"), reason="slow; set STAMPEDE_TEST_FULL=1")


@pytest.fixture(scope="session")
def samples():
    """The OpenCV sample images (downloaded once into tests/.cache)."""
    try:
        for name in synth.SAMPLES:
            synth.sample(name)
    except Exception as e:  # offline
        pytest.skip(f"sample images not available: {e}")


@pytest.fixture(scope="session")
def standard(tmp_path_factory, samples):
    """The standard synthetic scans (12 prints) in a fresh folder; returns (folder, truth)."""
    folder = tmp_path_factory.mktemp("standard") / "Scans"
    truth = synth.standard_set(folder)
    return folder, truth


@pytest.fixture
def one_scan(tmp_path, samples):
    """Factory: a folder with one synthetic stamped scan. one_scan(stamp="'87 6 14", **make_print args)."""
    def make(stamp="'87 6 14", src="fruits.jpg", placed=90, name="scan.jpg", **kw):
        folder = tmp_path / "Scans"
        synth.make_scan(folder / name, synth.make_print(src, stamp=stamp, **kw), placed, seed=3)
        return folder
    return make
