"""Batch logic (passes 2 and 3) on scenario tables: what Vision and Qwen 'read' per photo is fixed
by tests/scenarios/*.py, so the decisions can be checked exactly. See tests/sim.py."""
from helpers import evidence, flags, rows, run_sim


def sim(name, tmp_path, *args, keep=False):
    calls, out = run_sim(name, tmp_path, *args, keep=keep)
    return calls, rows(tmp_path / "Scans"), out


def test_batch16_real_readings(tmp_path):
    """16 scans as read on a Mac (dd/mm/yyyy camera); the model misreads some."""
    calls, rs, _ = sim("batch16", tmp_path)
    r = {int(k[6:10]): v for k, v in rs.items()}
    confirmed = {3: "2006-12-25", 4: "2006-12-25", 5: "2006-11-19", 7: "2006-03-22", 9: "2006-11-19",
                 10: "2006-03-22", 12: "2006-12-25", 15: "2006-12-25"}
    assert {n: x["date"] for n, x in r.items() if x["date"]} == confirmed
    for n in (1, 2):  # 12/03/2006 by both: the folder's proven order makes 12 March the suggestion
        assert r[n]["date_guess"] == "2006-03-12" and "date-order-learned:dmy" in flags(r[n])
    for n, right in ((6, "2006-11-19"), (8, "2006-12-25"), (11, "2006-12-25")):  # model misread
        f = next(x for x in flags(r[n]) if x.startswith("readers-disagree:"))
        assert right in f.split(":")[1].split("/")
    for n in (13, 14, 16):  # nothing on the stamp for Vision: no suggestion, even where Qwen "read" one
        assert r[n]["date"] == "" and r[n]["date_guess"] == ""
    assert all(x["rotate"] == "90" for x in r.values())
    assert calls == {"mlx_line": 16, "mlx_orient": 4}  # one stamp read per photo, rotation only when needed


def test_misread_sequence_never_becomes_a_date(tmp_path):
    """A roll from 14-16 June 1987; Vision misreads two stamps cleanly; the model sees nothing."""
    _, rs, _ = sim("misread_sequence", tmp_path)
    assert all(r["date"] == "" and r["status"] == "review" for r in rs.values())
    assert rs["roll/roll_03.jpg"]["date_guess"] == "1987-06-11"  # Vision's read, flagged, not "fixed"


def test_one_cameras_order_is_not_applied_to_another_folder(tmp_path):
    _, rs, _ = sim("mixed_cameras", tmp_path)
    assert all(r["date"] == "" for r in rs.values())
    for rel in ("Oma/b1.jpg", "Oma/b2.jpg", "Oma/b3.jpg", "Oma/b4.jpg"):  # mdy camera next to a dmy folder
        assert any(f.startswith("ambiguous-date-order:") for f in flags(rs[rel])), rs[rel]["flags"]


def test_one_photo_does_not_teach_an_order(tmp_path):
    _, rs, _ = sim("one_photo_order", tmp_path)
    for rel in ("Own/b1.jpg", "Own/b2.jpg", "Own/b3.jpg"):
        assert any(f.startswith("ambiguous-date-order:") for f in flags(rs[rel]))
        assert not any(f.startswith("date-order-learned") for f in flags(rs[rel]))


def test_no_stamp_no_faces_is_flagged_for_rotation(tmp_path):
    calls, rs, _ = sim("landscape", tmp_path)
    assert "rotation-check" in flags(rs["F/n1.jpg"]) and rs["F/n1.jpg"]["status"] == "review"
    assert "rotation-check" in flags(rs["F/n2.jpg"])  # face below the threshold disagrees with the frame
    assert not any("rotation-check" in flags(rs[f"F/p{i}.jpg"]) for i in range(1, 7))


def test_edits_before_pass_2_are_kept_and_the_rest_still_decided(tmp_path):
    _, rs, out = sim("edit_unresolved", tmp_path)
    assert "USER EDITS DONE" in out
    assert rs["F/p1.jpg"]["status"] == "skip"
    p2 = rs["F/p2.jpg"]  # date typed while undecided: kept, and its rotation still worked out
    assert (p2["date"], p2["rotate"], p2["status"]) == ("1987-07-01", "90", "ok")
    assert (rs["F/p5.jpg"]["rotate"], rs["F/p5.jpg"]["status"]) == ("0", "ok")
    assert not any("unresolved" in flags(r) for r in rs.values())


def test_edits_during_the_model_and_model_errors(tmp_path):
    _, rs, out = sim("pass3", tmp_path)
    assert "USER EDIT during model call" in out
    u1 = rs["F/u1.jpg"]
    assert (u1["date"], u1["status"]) == ("1987-06-05", "skip")  # the model finishing doesn't undo it
    u2 = rs["F/u2.jpg"]  # model raised: guess kept, still open, retried next scan
    assert u2["date"] == "" and u2["date_guess"] and "unresolved" in flags(u2)
    assert any(f.startswith("model-error:") for f in flags(u2))
    calls, rs2, _ = sim("pass3", tmp_path, keep=True)
    assert calls["mlx_line"] == 1  # only the failed one is asked again


def test_fast_first_then_the_model_later(tmp_path):
    calls, rs, _ = sim("fast_then_full", tmp_path, "--fast")
    assert calls == {"mlx_line": 0, "mlx_orient": 0}
    assert all(evidence(r).get("model_pending") for r in rs.values())
    calls, rs, _ = sim("fast_then_full", tmp_path, keep=True)
    assert calls["mlx_line"] == 4
    assert "readers-disagree:1987-06-11/1987-06-14" in flags(rs["F/x.jpg"])
    assert not any(evidence(r).get("model_pending") for r in rs.values())
