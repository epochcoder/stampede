"""Date parsing and folder years (pure functions, no images)."""
from datetime import date

import pytest

import stampede as sd

Y = (1975, 2008)


def ds(*xs):
    return {date.fromisoformat(x) for x in xs}


@pytest.mark.parametrize("text, order, expect", [
    ("25/12/2006", None, ds("2006-12-25")),
    ("'87 6 14", None, ds("1987-06-14")),           # apostrophe marks the year: only ymd
    ("14 6 '87", None, ds("1987-06-14")),           # year last: dmy or mdy, only dmy is a real date
    ("07 06 05", None, ds("2005-06-07", "2005-07-06", "2007-06-05")),  # truly ambiguous
    ("2 3 '92", None, ds("1992-02-03", "1992-03-02")),
    ("12 03 06", "dmy", ds("2006-03-12")),
    ("12 13 87", "dmy", set()),                     # impossible under the camera's order: a misread
    ("12 13 87", "mdy", ds("1987-12-13")),
    ("'95 1 18", "dmy", ds("1995-01-18")),          # the marks rule the order out: marks decide
    ("'60 1 1", None, set()),                       # outside the plausible years
    ("NONE", None, set()),
    ("", None, set()),
])
def test_dates_for(text, order, expect):
    assert sd.dates_for(text, order, *Y) == expect


def test_parse_date_reports_ambiguity_instead_of_picking():
    d, note = sd.parse_date("07 06 05", "auto", *Y)
    assert d is None and note.startswith("ambiguous:")


def test_parse_date_other_order_is_only_reported():
    d, note = sd.parse_date("12 13 87", "dmy", *Y)
    assert d is None and "mdy=1987-12-13" in note


def test_lookalikes_only_inside_digit_chunks():
    assert sd._normalize("'O7 l2 25").split() == ["'07", "12", "25"]
    assert "AUG" not in sd._normalize("AUG 87 5") and "8" not in sd._normalize("AUG")  # a word stays a word


@pytest.mark.parametrize("rel, year", [
    ("Pink Folder 2001/x.jpg", "2001"),
    ("Summer 1987/a/b.jpg", "1987"),          # nearest folder that names a year
    ("a/1990/b/1995/x.jpg", "1995"),          # innermost wins
    ("1987_x/y.jpg", "1987"),
    ("1998-2002 (2000)/x.jpg", "2000"),       # a range is ignored, a plain year next to it counts
    ("~2000/x.jpg", "2000"),
    ("1998-2002/x.jpg", None),                # ranges give no year
    ("1990s/x.jpg", None),                    # decades give no year
    ("2024-03/x.jpg", None),
    ("1970/x.jpg", None),                     # outside the plausible years
    ("x.jpg", None),
])
def test_folder_year(rel, year):
    assert sd.folder_year(rel, *Y) == year


def test_stamp_zone_is_bottom_right_of_landscape_frame():
    W, H = 2000, 1333
    assert sd.in_stamp_zone(0.88 * W, 0.93 * H, W, H)
    assert not sd.in_stamp_zone(0.30 * W, 0.93 * H, W, H)   # bottom left
    assert not sd.in_stamp_zone(0.88 * W, 0.50 * H, W, H)   # mid right
    assert not sd.in_stamp_zone(0.88 * H, 0.93 * W, H, W)   # portrait view: never the camera frame
