# AGENTS.md

Guide for coding agents (and humans) working on Stampede (`stampede.py`). Read README.md first for what
the tool does. This file covers the hard rules, architecture, data model, decisions already made
(don't re-litigate them without a reason), testing, and known limits.

## Status

Feature-complete v1, used end to end on a collection of ~1200 flatbed scans of family prints
(1980s–2007) on an M1 Pro (32 GB, Python 3.14): scan → review → apply → Immich CLI upload into
albums. Typical input: JPEG 2550×3500 (300 dpi, whole A4 bed, lots of white around the print),
prints laid on the scanner the same way (portrait and landscape mixed), no intentional skew but
real prints sit a fraction of a degree off; stamps in many formats (`'95 1 18`, `2 3 '92`,
`25/12/2006`, 7-segment and dot-matrix).

## Working style expected by the maintainer

- Short, direct answers; no explanations of basics. Verify, don't guess; say what was and
  wasn't tested. Metric units.
- Keep the CLI small and stable: `scan`, `review`, `apply` (+ hidden `debug`). Don't add visible
  options lightly; don't rename or reshuffle commands. `scan` opens the review page at the end.
- Changes must stay compatible with existing `photos.csv` files (users stop halfway through a
  collection and continue later). If pass 1 output must change, bump `ANALYSIS_VERSION`.
- Past complaints to avoid repeating: too many CLI options; slowness from many model calls per
  photo; dates or rotations taken from things that aren't stamps; any wrong date written silently.

## Hard rules (do not break)

1. **Never write a wrong date silently.** `date` is filled automatically only when Apple Vision
   and Qwen independently read the same single date, unambiguous under the field order, with at
   least one read taken as printed (not re-spaced). Batch reasoning (learned order, neighbours,
   re-spacing, rescans) only produces `date_guess` + flags + status `review`. Guesses reach EXIF
   only through an explicit action on the review page.
2. **Originals are never modified.** Nothing is written inside the scans folder; the review
   folder sits next to it (`<name>-review`, fallback `~/stampede/<name>-review` when the parent
   isn't writable). `apply` refuses an `--out` inside the scans folder and never overwrites a file
   it didn't write (manifest).
3. **User edits are never overwritten** by a later scan (per field, see `edited`).
4. **No data loss** on Ctrl+C, crashes or a scan file that crashes the analysis. CSV writes are
   atomic and fsync'd, and locked between `scan` and the review server.
5. **Fast enough for ~1200 photos**: Vision in parallel; at most one small Qwen read per photo
   with a stamp; a rotation question only when faces/stamp/placement don't settle it.

## Files

- `stampede.py`: everything (~2400 lines, single file by design).
- `<scans>-review/` (created by `scan`):
  - `photos.csv`: the state (columns below).
  - `previews/`: 1200 px JPEG of each crop (pass 1), key = sha1(rel|content-hash).
  - `thumbs/`: review thumbnails (crop + rotate).
  - `photos.csv.lock` (flock), `review.lock` (one review server; holds its URL), `scan.lock`
    (one scan at a time), `folder-id` (UUID for manifest keys), `carry-over.json` (edits of
    replaced scans until re-analysed), `photos.before-rescan-*.csv` (backups).
- `<out>/.stampede-manifest.json`: `{ "<folder-id>/<rel>": "<fingerprint>" | "pending" }`;
  fingerprint = `content-hash|crop|rotate|timestamp|how`. Old keys without folder id are adopted.
- Renamed from `stamp_dates.py`: the old manifest name (`.stamp_dates-manifest.json`, migrated on
  the next `apply`) and the old fallback review folder (`~/stamp_dates/...`) are still read.
- `ANALYSIS_VERSION` (currently 2, stored as `evidence.av`): bump when pass 1 changes in a way
  that makes old results wrong; the next `scan` re-analyses older rows automatically, except rows
  the user has settled (OK'd, dated, skipped).

## photos.csv columns

`rel, date, rotate, status, date_guess, flags, orient_src, ocr_raw, ocr_conf, ocr_backend, crop,
source, src_sig, edited, evidence`. Extra columns are preserved; rows sorted by `rel`; a BOM
(Excel/Numbers) is tolerated.

- `rel`: path inside the scans folder (row identity; review ids are sha1(rel)[:12]).
- `date`: `YYYY-MM-DD`, or `YYYY` / `YYYY-MM` typed by the user. `date_guess`: suggestion only.
- `status`: `ok` | `review` | `skip` | `error`; `review` iff any flag is not in
  `REVIEW_FREE = ("no-date", "landscape-assumed")`.
- `crop`: `x0,y0,x1,y1` in the EXIF-transposed full-resolution image. `rotate`: clockwise degrees.
- `src_sig`: `size-sha1(whole file)[:16]@mtime_ns`. The mtime lets unchanged files skip hashing;
  the content part detects replaced scans and re-attaches moved ones. Legacy `size:mtime` → unsure.
- `edited`: comma list of fields the user set (`date`, `rotate`, `status`); legacy `1` = all.
  `frozen(r)` = date or status set, or skipped → scans leave the row alone. Rotate-only edits keep
  the row open for a later confirmed date. Un-skip removes `status` from `edited`.
- `evidence` (JSON): `av`; `reads` (each `{t, raw, eng: vision|mlx, kind: line|full, rot, conf,
  rg (re-spaced), clean, compact, box}`); `raw`, `stamp_raw`, `model_raw`; `cand` (rotation of the
  best stamp candidate), `cand_box`; `faces` ({rot: score}); `crop_flags`; `model_date` (Qwen
  asked); `model_orient` ({pick, sure}); `model_pending` (deferred: `--fast`, skipped/settled);
  `stamp_frame` (confirmed stamp's rotation → placement votes); `settings` (`order|rotations`
  decided under); `previous` (carried date/skip of a replaced scan).

## Architecture of `scan`

0. **Prepare** (under lock): compare every row with the files on disk (`scan_status`): same /
   unsure / changed / moved (matched by content hash; only same-size new files are hashed) /
   missing / retry (errors except `crashes-the-analysis`, and rows older than
   `ANALYSIS_VERSION`). Refuses if more than half of ≥5 rows are missing (wrong folder).
   Replaced scans: rows backed up; the user's date/skip saved to `carry-over.json`.
1. **Pass 1, Apple Vision, parallel** (`pass1` → `analyse`; ProcessPoolExecutor, spawn, hidden
   `--jobs` default 4; `--jobs 0` = in-process, for tests). Per scan: `load` (EXIF transpose,
   16-bit → 8, ICC → sRGB), `find_crop`, 2000 px working copy, `vision_evidence`, 1200 px preview.
   Rows flushed every 10 rows / 5 s with flag `unresolved`. A worker crash marks in-flight files
   `error:worker-crashed`; the next run retries them one at a time in a 1-worker pool; a repeat
   crash → `error:crashes-the-analysis`, never retried.
   - **Stamp candidates** (`stamp_candidates`): orange top-hat mask on R−B, per rotation, but only
     in the camera's date position (`STAMP_ZONE`/`in_stamp_zone`): the rotated view must be
     landscape (W ≥ 0.95·H, the 35 mm frame) and the blob centred at x ≥ 0.6·W, y ≥ 0.82·H with
     height 2–9 % of H. Without this rule, wooden armrests, tartan and shelf edges became
     "stamps", Qwen invented dates for them, the fake stamp set the rotation, and nearly every
     photo cost a model call.
   - **Reads**: Vision (`VNRecognizeTextRequest`, accurate, en-US, language correction off) on
     line crops at native resolution, plus per digit group; run-together digits ("195118") are
     re-spaced from the stamp's group widths (`layouts`, `regroup_any`). Full-photo text reads
     are a fallback and also count only inside the stamp zone.
   - **Faces**: Vision `VNDetectFaceRectanglesRequest` rev 3 at 0°/90°, roll used by magnitude.
2. **Pass 2, batch reasoning, no model** (`Batch`). Pending = not error, not frozen, and
   `unresolved`, or `model_pending` (non-fast), or decided under other `settings`.
   - `learn_orders`: per folder, one vote per photo from reads that allow only one field order;
     ≥3 photos and ≥90 % agreement; no global fallback; guesses only.
   - `placement_votes`: per folder, rotation of confirmed stamps; ≥5 and ≥80 % → placement.
   - `resolve_date`: (1) confirmed per rule 1, plus `date-out-of-sequence` if >365 days from
     neighbours; (2) both read the same ambiguous digits → guess by learned order
     (`date-order-learned:`) or nearest neighbour (`ambiguous-date-order:`); (3) otherwise a guess
     with `readers-disagree:` / `ambiguous-date-order:` / `date-unconfirmed`, or `no-date`
     (+ `stamp-unread` if the stamp line had ≥3 digits). Qwen reads only count when Vision saw ≥2
     digits on the stamp line: on a crop without a stamp, Qwen can make a date up.
   - `resolve_rotation`: faces vs frame (confirmed stamp, else located stamp, else folder
     placement); `landscape-assumed` when the stamp was located there, placement agrees and no
     stray face; otherwise model pick / frame with `rotation-check`, or `orientation-unknown`.
   - Rows still `unresolved` are decided even if the user already edited them (e.g. typed a date
     on the review page during the scan); `commit` keeps the user's fields.
   - `finish` writes the row; `commit` merges per field without touching user edits (`_carried`
     marks a carried skip; `_unfreeze` releases a user's OK when new settings changed the date).
3. **Pass 3, Qwen via MLX** (`mlx-vlm` 0.7.x, `Qwen3-VL-8B-Instruct-4bit`; loaded lazily; if it
   fails, rows keep guesses and stay flagged). Stamped photos first. `model_check` = one read of
   the same line crop Vision used (`PROMPT_LINE`, neutral: answer NONE rather than guess).
   Rotation via `vlm_pick` (2×2 grid of centre-square 330 px tiles; asked a second time with
   shuffled order only if the first answer would settle it). Each row committed immediately;
   exception → row restored + `model-error:*`; Ctrl+C → row restored.
4. Summary, release `scan.lock`, open the review page (`cmd_review`).

### Crop (`find_crop` + `fit_edges`)

Coarse: 1500 px copy, threshold (bed − 18 or saturation > 40), morphology, largest contour →
bounding box (+ `multiple-prints`, `irregular-edge`). Fine (`fit_edges`, full resolution): per
side ~80 profiles walked from 5 % outside the coarse edge inward to the first non-bed pixel
(two in a row; |gray − bed| > 12 or chroma > 25); RANSAC line (|slope| ≤ 0.05, tolerance
max(1.5 px, 0.08 %)), accepted with ≥40 % inliers spanning ≥50 % of the side. Box = inside all
four lines + a margin of 0.45 % of the long side (~0.7 mm). An unseen side → coarse edge minus
the tilt wedge + `edge-uncertain-<side>`; `print-tilted-<n>deg` above 1°; `maybe-cut-<side>`
whenever the strip inside an edge is near-white; `odd-aspect-<n>` vs `STD_ASPECTS`.
(The earlier coarse-box crop left a thin white line on random edges: lid shadow/highlight plus
sub-degree tilt.)

### Review server (`cmd_review`)

`ThreadingHTTPServer` on 127.0.0.1:8765–8799. Security: Host allow-list, per-session `X-Token`
on POST, Origin check, JSON content type, 10 kB body cap, SameSite=Strict HttpOnly cookie
`sd<port>` for `/thumb` and `/large` (parsed by hand), `X-Frame-Options: DENY` + CSP
`frame-ancestors 'none'`, JSON `<` escaped, at most 2 concurrent `/large` decodes.
Optimistic concurrency: each row carries `v` (hash of date/guess/flags/status/rotate/edited/crop);
a stale edit is refused and the card refreshed; bulk actions report skipped rows.
Filters: Flagged (default), No stamp (`review`, no date, nothing stamp-related), Date to enter
(`review`, no date, a guess or stamp flag), Assumed landscape, Skipped, All. Cards leave the view
when they stop matching it. Typed dates are validated (calendar, `YEARS`). Error rows can only be
skipped. `REVIEW_PAGE` is a **raw string**: a non-raw one once broke the page's JS (`\n`, `\'`).

### apply (`plan` + `cmd_apply`)

Validates every row first (exiftool present, out not inside scans, crop within image, scan
unchanged, dates valid); flagged rows skipped unless `--include-flagged`; missing scans and stale
outputs reported. Per photo: render → `.<name>.partial.<ext>` → exiftool
(`DateTimeOriginal`, `CreateDate`, `ModifyDate`, `FileModifyDate`; `XMP-digiKam:TagsList` =
`Scanned prints`, `/Date estimated`, `/No date`) → manifest `pending` → `os.replace` → manifest
fingerprint. Ctrl+C removes the partial file. Immich reads DateTimeOriginal and TagsList.

## Decisions already made

- Local only; no cloud APIs. MLX instead of Ollama. Apple Vision for OCR; PaddleOCR, RapidOCR
  and a hand-written 7-segment decoder were tried and dropped.
- Strict date policy (Vision + Qwen agreement) instead of neighbour heuristics, which wrote
  wrong dates.
- Stamps only in the camera's date position (bottom right of the landscape frame).
- Folder year: innermost folder, plain 4-digit year; no ranges or decades. For an unknown range,
  rename the folder, e.g. `1998-2002 (2000)`.
- Albums are made in Immich (CLI `--album`/`--album-name`, or the `Scanned prints` tag → Add to
  album), not by this tool.
- The `apply` summary counts folder-year and `--fallback-date` dates separately.

## Testing

`tests/` (see tests/README.md): pytest, runs on Linux or macOS with Apple Vision and mlx-vlm
mocked (`tests/mocks/`) and synthetic scans only (`tests/synth.py`, OpenCV sample photos).
`python -m pytest tests -q` (~10 min; `STAMPEDE_TEST_FULL=1` adds the 120-scan crop test).

- `test_dates.py`: parsing, orders, folder years, stamp zone.
- `test_crop.py`: crop accuracy, stamp detection at the camera frame, orange objects are not
  stamps, no visible bed edge / no unflagged over-trim.
- `test_scan.py`: mocked end-to-end runs: date policy, `--fast`, settings changes, adversarial
  model, incremental runs, upgrades (`ANALYSIS_VERSION`), replaced/moved scans, crash isolation
  with real worker processes, `apply`.
- `test_scenarios.py`: batch logic on fixed readings (`tests/sim.py` + `tests/scenarios/`).
- `test_review_ui.py`: the review page in headless Chromium + JS syntax check.

Rules: every bug fix gets a test that fails on the version before the fix (this was checked for
the fixes so far). Run the suite before handing back a change. Mocks are not Apple Vision: tests
assert safety (nothing wrong marked ok or written), not OCR accuracy; real behaviour and speed
need a Mac. To run the tool itself with mocks: `python tests/run_mock.py scan <folder> --jobs 0`.

## Known limits / open items

- macOS + Apple Silicon only. A Linux/Windows path (other OCR + VLM backend) is the obvious
  feature request.
- Stamps outside the bottom-right of the camera frame are not detected → folder year.
- If Vision reads no digits on a real stamp, Qwen's read is ignored → no date, no flag.
- Blown-out white touching an edge can't be told from the bed; the crop may cut a little there
  (always flagged).
- Hand edits to `photos.csv` aren't marked `edited` and can be overwritten by a later scan.
- Adding a scan can shift the same-date seconds of later-named photos → `apply` rewrites them.
- 16-bit scans are written as 8-bit.
- A photo that deterministically segfaults MLX in pass 3 would stop each run at the same place.
- `debug <file|folder>` (hidden) prints every step and saves stamp crops to
  `~/stampede-debug/<stem>`.
