# Folder with a learned scanner placement (stamps read at 90). n1: no stamp at all, no faces
# (a print from a non-dating camera, could be portrait or upside down) -> rotation from prior only.
# n2: dated, but a face detector says 0.75 for rotation 0 (portrait shot) - below 0.8, ignored.
FILES = {f'F/p{i}.jpg': dict(v=[f"'87 6 1{i}"]) for i in range(1, 7)}
FILES['F/n1.jpg'] = dict(v=[], nocand=True)
FILES['F/n2.jpg'] = dict(v=["'87 6 17"], faces={0: 0.75})
