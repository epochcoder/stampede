"""End-to-end `scan` / `apply` runs with Apple Vision and the Qwen model mocked (tests/mocks)."""
import csv
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

import stampede as sd
import synth
from conftest import needs_exiftool, needs_rapidocr
from helpers import evidence, flags, rows, run_mock, run_workers

# Vision's line read, Vision's full-photo read, Qwen's line read
QUIET = dict(MOCK_VISION_FULL="-")


def scan(folder, *args, **mock):
    return run_mock("scan", folder, "--no-review", "--jobs", "0", *args, **{**QUIET, **mock})


def only(folder):
    return next(iter(rows(folder).values()))


# ----------------------------------------------------------------- the date policy
def test_both_readers_agree_fills_the_date(one_scan):
    f = one_scan("'87 6 14")
    scan(f, MOCK_VISION_LINE="'87 6 14", MOCK_LINE="'87 6 14")
    r = only(f)
    assert (r["date"], r["status"], r["rotate"]) == ("1987-06-14", "ok", "270")


def test_readers_disagree_is_only_a_guess(one_scan):
    f = one_scan("'87 6 14")
    scan(f, MOCK_VISION_LINE="'87 6 11", MOCK_LINE="'87 6 14")
    r = only(f)
    assert r["date"] == "" and r["status"] == "review"
    assert "readers-disagree:1987-06-11/1987-06-14" in flags(r)


def test_fast_skips_the_model_and_leaves_a_guess(one_scan):
    f = one_scan("'87 6 14")
    out = scan(f, "--fast", MOCK_VISION_LINE="'87 6 14", MOCK_LINE="'87 6 14")
    r = only(f)
    assert "'load': 0" in out
    assert r["date"] == "" and r["date_guess"] == "1987-06-14" and "date-unconfirmed" in flags(r)
    assert evidence(r).get("model_pending")
    scan(f, MOCK_VISION_LINE="'87 6 14", MOCK_LINE="'87 6 14")  # a later run without --fast asks
    assert only(f)["date"] == "1987-06-14"


def test_ambiguous_order_is_never_decided(one_scan):
    f = one_scan("07 06 05")
    scan(f, MOCK_VISION_LINE="07 06 05", MOCK_LINE="07 06 05")
    r = only(f)
    assert r["date"] == "" and any(x.startswith("ambiguous-date-order:") for x in flags(r))


def test_impossible_under_the_cameras_order_is_unread(one_scan):
    f = one_scan("12 13 87")
    scan(f, "--date-order", "dmy", MOCK_VISION_LINE="12 13 87", MOCK_LINE="12 13 87")
    r = only(f)
    assert r["date"] == "" and r["date_guess"] == "" and "stamp-unread" in flags(r)


def test_qwen_alone_cannot_even_suggest_a_date(one_scan):
    # Vision sees no digits on the stamp line; a model can make dates up on such crops
    f = one_scan("'87 6 14")
    scan(f, MOCK_VISION_LINE="-", MOCK_LINE="'87 6 14")
    r = only(f)
    assert r["date"] == "" and r["date_guess"] == "" and flags(r).count("no-date") == 1


def test_changing_the_date_order_re_decides_even_photos_marked_ok(one_scan):
    f = one_scan("07 06 05")
    scan(f, "--date-order", "dmy", MOCK_VISION_LINE="07 06 05", MOCK_LINE="07 06 05")
    assert only(f)["date"] == "2005-06-07"
    sd.update_csv(f.parent / "Scans-review" / "photos.csv", lambda rs: [r.update(edited="status") for r in rs])
    scan(f, "--date-order", "mdy", MOCK_VISION_LINE="07 06 05", MOCK_LINE="07 06 05")
    assert only(f)["date"] == "2005-07-06"


def test_a_date_you_typed_survives_a_settings_change(one_scan):
    f = one_scan("07 06 05")
    scan(f, "--date-order", "dmy", MOCK_VISION_LINE="07 06 05", MOCK_LINE="07 06 05")
    sd.update_csv(f.parent / "Scans-review" / "photos.csv",
                  lambda rs: [r.update(date="2001-01-01", edited="date") for r in rs])
    scan(f, "--date-order", "mdy", MOCK_VISION_LINE="07 06 05", MOCK_LINE="07 06 05")
    assert only(f)["date"] == "2001-01-01"


@needs_rapidocr
def test_adversarial_model_on_the_standard_set(standard, tmp_path):
    """Real (stand-in) OCR on 12 synthetic scans, a model that 'reads' '87 6 14 everywhere and always
    answers tile 1: no wrong date may reach the date column, no wrong rotation may be marked ok,
    and orange things that aren't stamps must not produce a suggestion."""
    folder, truth = standard
    work = tmp_path / "Scans"
    shutil.copytree(folder, work)
    run_mock("scan", work, "--no-review", "--jobs", "0", MOCK_LINE="'87 6 14", MOCK_ORIENT="1")
    rs = rows(work)
    for name, t in truth.items():
        r = rs[f"{name}.jpg"]
        assert r["date"] in ("", t["date"]), (name, r["date"], r["flags"])
        if r["status"] == "ok":
            assert int(r["rotate"]) == t["undo_rot"], (name, r["rotate"], r["flags"])
    orange = rs["12_orange_nostamp.jpg"]
    assert orange["date_guess"] == "" and evidence(orange).get("cand") is None


# ----------------------------------------------------------------- incremental runs, upgrades
def test_older_analyses_are_redone_but_settled_photos_kept(tmp_path, samples):
    f = tmp_path / "Scans"
    for i, stamp in enumerate(["'87 6 14", "'88 7 15"]):
        synth.make_scan(f / f"s{i}.jpg", synth.make_print("fruits.jpg", stamp=stamp), 90, seed=i)
    scan(f, "--fast", MOCK_VISION_LINE="'87 6 14")
    csv_path = tmp_path / "Scans-review" / "photos.csv"

    def make_old(rs):
        for r in rs:
            ev = json.loads(r["evidence"])
            ev.pop("av")
            r["evidence"] = json.dumps(ev)
            if r["rel"] == "s0.jpg":
                r.update(status="ok", edited="status")  # one you've already OK'd
    sd.update_csv(csv_path, make_old)
    out = scan(f, "--fast", MOCK_VISION_LINE="'87 6 14")
    assert "1 photos were analysed by an older version" in out
    rs = rows(f)
    assert evidence(rs["s1.jpg"]).get("av") == sd.ANALYSIS_VERSION
    assert "av" not in evidence(rs["s0.jpg"]) and rs["s0.jpg"]["edited"] == "status"


def test_only_new_scans_are_analysed_and_limit_works_in_chunks(tmp_path, samples):
    f = tmp_path / "Scans"
    for i in range(3):
        synth.make_scan(f / f"s{i}.jpg", synth.make_print("fruits.jpg"), 0, seed=i)
    assert "analysing 2" in scan(f, "--fast", "--limit", "2")
    assert "2 already analysed, analysing 1" in scan(f, "--fast")
    assert "3 already analysed, analysing 0" in scan(f, "--fast")


def test_replaced_scan_keeps_your_date_and_skip_across_limited_runs(tmp_path, samples):
    f = tmp_path / "Scans"
    for i in range(3):
        synth.make_scan(f / f"s{i}.jpg", synth.make_print("fruits.jpg"), 0, seed=i)
    scan(f, "--fast")
    csv_path = tmp_path / "Scans-review" / "photos.csv"

    def edit(rs):
        for r in rs:
            if r["rel"] == "s0.jpg":
                r.update(date="1980-01-01", edited="date", status="ok")
            if r["rel"] == "s1.jpg":
                r.update(status="skip", edited="status")
    sd.update_csv(csv_path, edit)
    for n in ("s0.jpg", "s1.jpg"):  # re-scanned prints: different files, same names
        Image.open(f / n).save(f / n, quality=70)
    scan(f, "--fast", "--limit", "1")  # only s0 now; s1's skip must not get lost meanwhile
    assert json.loads((tmp_path / "Scans-review" / "carry-over.json").read_text()) == {
        "s1.jpg": {"date": "", "skip": True}}
    scan(f, "--fast")
    rs = rows(f)
    assert "date-before-rescan:1980-01-01" in flags(rs["s0.jpg"]) and rs["s0.jpg"]["date_guess"] == "1980-01-01"
    assert rs["s0.jpg"]["date"] == ""  # a suggestion now, not a date: it was for the old scan
    assert rs["s1.jpg"]["status"] == "skip"
    assert not (tmp_path / "Scans-review" / "carry-over.json").exists()
    assert list((tmp_path / "Scans-review").glob("photos.before-rescan-*.csv"))


def test_moved_scans_keep_their_rows(tmp_path, samples):
    f = tmp_path / "Scans"
    synth.make_scan(f / "a.jpg", synth.make_print("fruits.jpg"), 0, seed=1)
    scan(f, "--fast")
    sd.update_csv(tmp_path / "Scans-review" / "photos.csv",
                  lambda rs: [r.update(date="1999-09-09", edited="date") for r in rs])
    (f / "1999").mkdir()
    (f / "a.jpg").rename(f / "1999" / "a.jpg")
    out = scan(f, "--fast")
    assert "moved to other folders" in out and "analysing 0" in out
    assert rows(f)["1999/a.jpg"]["date"] == "1999-09-09"


# ----------------------------------------------------------------- crashes
def test_a_scan_that_crashes_the_analysis_is_isolated(tmp_path, samples):
    f = tmp_path / "Scans"
    for i in range(6):
        synth.make_scan(f / f"{i:02d}{'_CRASH' if i == 3 else ''}.jpg", synth.make_print("fruits.jpg"), 0, seed=i)
    mock = dict(MOCK_CRASH="CRASH", MOCK_VISION_LINE="-", MOCK_VISION_FULL="-")
    args = ("scan", f, "--fast", "--no-review", "--jobs", "4")

    def errors():
        return {rel: r["flags"] for rel, r in rows(f).items() if r["status"] == "error"}

    p = run_workers(*args, **mock)
    assert p.returncode == 0, p.stderr
    first = errors()
    assert "03_CRASH.jpg" in first and all(v == "error:worker-crashed" for v in first.values())
    p = run_workers(*args, **mock)  # suspects again, one per worker process: only the culprit fails
    assert p.returncode == 0, p.stderr
    assert errors() == {"03_CRASH.jpg": "error:crashes-the-analysis"}
    p = run_workers(*args, **mock)
    assert "analysing 0" in p.stdout  # never retried
    # even if it's the only scan left and --jobs 1, it never runs inside the main process
    sd.update_csv(tmp_path / "Scans-review" / "photos.csv",
                  lambda rs: [r.update(flags="error:worker-crashed") for r in rs if r["status"] == "error"])
    p = run_workers("scan", f, "--fast", "--no-review", "--jobs", "1", **mock)
    assert p.returncode == 0, p.stderr
    assert errors() == {"03_CRASH.jpg": "error:crashes-the-analysis"}


# ----------------------------------------------------------------- apply
@needs_exiftool
def test_apply_writes_dated_copies_and_never_touches_foreign_files(tmp_path, samples):
    f = tmp_path / "Scans"
    synth.make_scan(f / "1987" / "a.jpg", synth.make_print("fruits.jpg"), 0, seed=1)
    synth.make_scan(f / "b.jpg", synth.make_print("home.jpg"), 0, seed=2)
    scan(f, "--fast")
    sd.update_csv(tmp_path / "Scans-review" / "photos.csv", lambda rs: [r.update(status="ok") for r in rs])
    src_bytes = {p: p.read_bytes() for p in f.rglob("*.jpg")}
    out_dir = tmp_path / "Dated"
    p = run_workers("apply", f, "--out", out_dir, "--fallback-date", "1990", "--dry-run")
    assert "1 with the year from their folder name, 1 with --fallback-date" in p.stdout
    assert not out_dir.exists()
    p = run_workers("apply", f, "--out", out_dir, "--fallback-date", "1990")
    assert p.returncode == 0, p.stdout + p.stderr
    import subprocess
    dto = subprocess.run(["exiftool", "-s3", "-DateTimeOriginal", str(out_dir / "1987" / "a.jpg")],
                         capture_output=True, text=True).stdout.strip()
    assert dto.startswith("1987:01:01 12:00")
    assert all(p.read_bytes() == b for p, b in src_bytes.items())  # originals untouched
    assert "already written" in run_workers("apply", f, "--out", out_dir).stdout  # idempotent
    # a file this tool didn't write is never overwritten, not even with --overwrite
    foreign = tmp_path / "Other"
    (foreign / "1987").mkdir(parents=True)
    (foreign / "b.jpg").write_bytes(b"not ours")
    p = run_workers("apply", f, "--out", foreign, "--overwrite", "--fallback-date", "1990")
    assert p.returncode != 0 and "not written by this tool" in p.stdout
    assert (foreign / "b.jpg").read_bytes() == b"not ours"


@needs_exiftool
def test_apply_adopts_outputs_of_older_versions(tmp_path, samples):
    f = tmp_path / "Scans"
    synth.make_scan(f / "a.jpg", synth.make_print("fruits.jpg"), 0, seed=1)
    scan(f, "--fast")
    sd.update_csv(tmp_path / "Scans-review" / "photos.csv", lambda rs: [r.update(status="ok") for r in rs])
    out_dir = tmp_path / "Dated"
    assert run_workers("apply", f, "--out", out_dir).returncode == 0
    m = out_dir / sd.MANIFEST
    m.unlink()  # what versions before the rename wrote: old file name, keys without a folder id
    (out_dir / sd.OLD_MANIFEST).write_text(json.dumps({"a.jpg": "old-format"}))
    p = run_workers("apply", f, "--out", out_dir)
    assert p.returncode == 0 and "PROBLEM" not in p.stdout
    assert list(json.loads(m.read_text())) == [f"{(tmp_path / 'Scans-review' / 'folder-id').read_text().strip()}/a.jpg"]
    assert not (out_dir / sd.OLD_MANIFEST).exists()  # migrated
