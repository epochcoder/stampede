# Stampede

Crop, rotate and date flatbed scans of old photo prints, entirely on your own Mac, then hand
them to Immich (or any photo library that reads EXIF).

Scanned prints carry the scan date, not the day they were taken. Many film cameras of the
1980s–2000s printed the date on the photo itself, in orange digits. Stampede reads those
stamps, cuts each print off the scanner bed, turns it upright and writes the date into the file.
**Your scans are never modified**, and **no date is written unless two independent readers
agree on it**; anything uncertain is shown to you on a local review page first.

Used on a collection of ~1200 family prints (1980s–2007) on an M1 Pro.

## What it does

- **Crop**: finds the print on the white scanner bed and cuts it out at full resolution. Edges
  are fitted per side, so no bed, lid shadow or paper edge is left, and prints lying slightly
  off-square are handled.
- **Rotate**: from the date stamp's position, faces (Apple Vision), how the prints in a folder lay
  on the scanner, and, only when still unsure, a local vision model.
- **Date**: reads the camera date stamp with Apple Vision and a local Qwen3-VL model. A date is
  filled in only when both read the same date. Everything else becomes a suggestion you accept,
  pick or type on the review page. No stamp: the year from the folder name, else
  `--fallback-date`, else no date.
- **Review page**: a local web page showing every photo exactly as it will be written. Fix the
  date or rotation, mark OK or skip; every change is saved immediately.
- **Apply**: writes cropped, rotated, dated copies with EXIF dates and Immich tags to a separate
  folder.

Everything runs locally: no cloud service, no account, no upload.

## Requirements

- Mac with Apple Silicon (Apple Vision for OCR and faces, MLX for the vision model).
  Tested on an M1 Pro with 32 GB and Python 3.14.
- ~5 GB disk for the model (`mlx-community/Qwen3-VL-8B-Instruct-4bit`, downloaded on first use).

```bash
brew install exiftool
pip install opencv-python-headless pillow numpy pyobjc-framework-Vision mlx-vlm
```

## Quick start

```bash
# 1. analyse (subfolders included); opens the review page when done (Ctrl+C to close it)
python stampede.py scan ~/Pictures/Scans

# reopen the review page later without scanning
python stampede.py review ~/Pictures/Scans

# 2. write the finished photos (dry run first)
python stampede.py apply ~/Pictures/Scans --out ~/Pictures/Dated --dry-run
python stampede.py apply ~/Pictures/Scans --out ~/Pictures/Dated
```

The working data lives next to the scans folder, never inside it: `~/Pictures/Scans-review/`
(`photos.csv`, previews, thumbnails, locks). Delete that folder to start over.

### scan options

| option | meaning |
|---|---|
| `--rotate auto` | default. Or one angle for all (`0/90/180/270`, clockwise), or a list to choose from (`0,270`) |
| `--date-order auto\|ymd\|dmy\|mdy` | how the camera prints dates without an apostrophe year. `auto` flags unclear ones (05 06 07) |
| `--fast` | skip the Qwen model; open questions stay flagged. A later run without `--fast` asks the model |
| `--limit N` | analyse only the next N new scans (work through a large folder in chunks) |

### apply options

| option | meaning |
|---|---|
| `--out DIR` | required; a separate folder for the finished photos |
| `--dry-run` | validate everything, print the plan, write nothing |
| `--fallback-date YYYY[-MM[-DD]]` | date for photos with no date and no year in their folder name |
| `--include-flagged` | also write photos still flagged for review (default: left out) |
| `--overwrite` | rewrite every photo this tool wrote to `--out` (never touches other files) |

## How dates are chosen

1. The `date` column: confirmed by Vision and Qwen, or set by you on the review page.
2. The year in the innermost folder name that contains one plainly: `Pink Folder 2001`,
   `Summer 1987`, `1998-2002 (2000)`. Ranges (`1998-2002`) and decades (`1990s`) give no year.
3. `--fallback-date`.
4. No date.

Years outside 1975–2008 are rejected everywhere (`YEARS` in the script; change it for your
collection). Folder-year and fallback dates are written as 1 January; a typed year or month as
the 1st. These are tagged `Scanned prints/Date estimated`. Photos sharing a date get noon plus a
few seconds each in file-name order, so your library keeps the scan order. Every photo is tagged
`Scanned prints`; undated ones also `Scanned prints/No date`.

Date stamps are only looked for where cameras print them: the bottom-right corner of the
(landscape) camera frame, checked in every orientation. Orange things elsewhere in a photo are
never taken for a stamp.

## Working through a large collection

- `scan` only analyses new scans (and failed or replaced ones). Photos you've OK'd, dated or
  skipped are never changed by a later scan.
- Stop any time (Ctrl+C, crash): progress is saved continuously; the next run continues.
- `apply` only writes new or changed photos; flagged ones are left out until you OK them.
- Moving or renaming scans and folders is fine: rows are matched by file content.
- Changing `--date-order` or `--rotate` re-decides earlier photos, except dates you set yourself.

## Review page

Filters:

- **Flagged** (default): everything that needs a look.
- **No stamp**: flagged, nothing to date from. Check the rotations, then "Mark shown as OK".
- **Date to enter**: flagged, a stamp was seen. Use a suggested date or type it.
- **Assumed landscape**, **Skipped**, **All**.

Per photo: date field, "use <date>" buttons, rotate left/right, OK, Skip; click to enlarge.
Bulk: "Use guesses shown" (leaves photos with two or more possible dates for you) and
"Mark shown as OK". A photo leaves the view as soon as it no longer matches the filter.

A typical pass: **No stamp** → check rotations → Mark shown as OK; then **Date to enter**;
then whatever is left under **Flagged** (crop or rotation checks).

## Into Immich

Either drop the `--out` folder into a folder Immich watches (external library), or upload with
the Immich CLI, which can create albums as it goes:

```bash
npm i -g @immich/cli
immich login http://<your-immich>:2283/api <API key>
immich upload --recursive --album-name "Scanned prints" ~/Pictures/Dated   # one album
immich upload --recursive --album ~/Pictures/Dated                         # one album per folder
```

The CLI may need an API key with all permissions (see immich-app/immich#21456); create one for
the upload and delete it afterwards. Use one route or the other, not both, to avoid duplicates.

## Flags on the review page

| flag | meaning |
|---|---|
| `date-unconfirmed` | only one reader read the date |
| `readers-disagree:a/b` | Vision and Qwen read different dates; buttons offer each |
| `ambiguous-date-order:a/b` | both read the same digits but day/month order is unclear |
| `date-order-learned:dmy` | suggestion using the order this folder's camera provably uses |
| `date-out-of-sequence` | confirmed, but more than a year from neighbouring scans |
| `stamp-unread` | digits on the stamp, but no valid date in them |
| `date-before-rescan:d` | the scan file was replaced; this is the date you had entered |
| `no-date` | no stamp found (needs no review on its own) |
| `rotation-check`, `orientation-unknown`, `stamp-orientation-conflict` | check the rotation |
| `landscape-assumed` | rotation from stamp + scanner placement (no review needed; own filter) |
| `maybe-cut-<side>`, `edge-uncertain-<side>` | near-white photo at that edge: check the crop |
| `print-tilted-<n>deg`, `odd-aspect-<n>`, `irregular-edge`, `multiple-prints` | check the crop |
| `error:worker-crashed`, `error:crashes-the-analysis` | the scan crashed the analysis (retried once) |
| `model-error:*` | Qwen failed on this photo; it keeps its suggestion and is retried next scan |

## Limitations

- macOS on Apple Silicon only.
- Stamps outside the bottom-right corner of the camera frame aren't detected; those photos fall
  back to the folder year.
- Blown-out white sky or walls touching a print's edge look like the scanner bed; the crop may
  cut a little there and the photo is flagged.
- One print per scan. Several prints on one scan are flagged (`multiple-prints`), not split.
- 16-bit scans are written as 8-bit.
- Edit dates on the review page, not in `photos.csv` directly: hand edits there aren't marked
  as yours and a later scan may overwrite them.

## Tests

`python -m pytest tests -q`: runs anywhere (Apple Vision and the model are mocked, all images are
synthetic). See [tests/README.md](tests/README.md).

## Troubleshooting

`python stampede.py debug <scan or folder>` prints every step for a scan (crop, stamp
candidates, every Vision read, faces per rotation) and saves the stamp crops to
`~/stampede-debug/<name>/`.
