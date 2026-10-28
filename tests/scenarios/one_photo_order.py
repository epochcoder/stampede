# Real Vision yields one read per OCR variant (3 masks + colour crop + per-group read): the same stamp
# read 4x counts as 4 votes. One dmy photo in the whole collection (a print from a relative's camera)
# sets the global order; the owner's own camera is mdy (stamps like 03 05 90 = 5 March 1990).
FILES = {
    'Misc/x1.jpg': dict(v=["25 12 89"] * 4),
    'Own/b1.jpg': dict(v=["03 05 90"] * 4),
    'Own/b2.jpg': dict(v=["03 06 90"] * 4),
    'Own/b3.jpg': dict(v=["03 07 90"] * 4),
}
