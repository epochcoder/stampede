# Tests

Runs on Linux or macOS without Apple Vision or the Qwen model: both are replaced by stand-ins in
`tests/mocks/`, and every image is synthetic (OpenCV's sample photos turned into flatbed scans
with drawn camera date stamps; see `synth.py`). No private photos are needed or included.

```bash
pip install -r tests/requirements.txt
playwright install chromium          # for the review-page tests
python -m pytest tests -q            # ~10 min
STAMPEDE_TEST_FULL=1 python -m pytest tests -q   # + 120-scan crop test
```

The first run downloads the OpenCV sample images into `tests/.cache/` and OpenCV's face detector
into `~/.cache/stampede/`. Offline, set `STAMPEDE_TEST_SAMPLES` to a folder holding the samples.
Tests that need something missing (RapidOCR, Playwright, node, exiftool) are skipped, not failed.

## What's covered

| file | what |
|---|---|
| `test_dates.py` | date parsing: field orders, apostrophe years, ambiguity, impossible dates, look-alike letters, folder years, the stamp zone |
| `test_crop.py` | crop accuracy; stamps found at the camera-frame rotation; orange things (wood, tartan) away from the stamp corner are never stamps; no visible bed line on any edge and no unflagged over-trim across tilts, lid shadows, paper rims, 300/600 dpi, white sky and walls |
| `test_scan.py` | full `scan`/`apply` runs with mocked Vision/Qwen: the date policy (agree → date; disagree, ambiguous, impossible, Qwen-only → no date), `--fast`, settings changes, an adversarial model on 12 scans, incremental runs and `--limit`, upgrades from older analyses, replaced and moved scans, crash isolation with real worker processes, `apply` output, idempotence and never overwriting foreign files |
| `test_scenarios.py` | batch logic on fixed readings (`scenarios/*.py`, run by `sim.py`): a real 16-scan batch as read on a Mac, misread sequences, cameras with different date orders, edits before pass 2 and during the model, model errors, `--fast` then full |
| `test_review_ui.py` | the review page in headless Chromium: filters, cards leaving the view, "use" buttons, bulk actions, stale-page refusal, validation, rotate/skip/un-skip, security headers/token/cookies; plus a JS syntax check |

Each fix made during development has a test that fails on the version before the fix (white crop
edges, false stamps from orange objects, Qwen-only suggestions, edits before pass 2, the
"No stamp" filter).

## Pieces

- `mocks/`: `Vision`/`objc` (PyObjC surface; text from env vars or RapidOCR, faces from OpenCV's
  DNN detector with a roll) and `mlx_vlm` (fixed answers from env vars; default is adversarial:
  it "reads" `'87 6 14` on everything). Env vars are documented in each module.
- `run_mock.py`: runs `stampede.py` as if on a Mac with the mocks.
- `sim.py` + `scenarios/`: a whole scan in-process with Vision's reads, faces, stamp location and
  the model's answers taken from a scenario table.
- `synth.py`: synthetic scans with exact truth (print box, rotation, mask, stamp).

## Limits

- The OCR and face stand-ins are not Apple Vision: RapidOCR reads synthetic stamps worse, and the
  OpenCV face detector is noisy (a 2 px crop change can flip it). Tests assert safety ("nothing
  wrong is marked ok or written"), not Vision's accuracy. Real accuracy and speed need a Mac.
- Re-spacing run-together digits ("195118" → `'95 1 18`) depends on a real stamp's geometry and is
  only covered indirectly.
