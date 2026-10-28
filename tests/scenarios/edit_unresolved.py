# Pass 1 done (rows 'unresolved'); the user edits on the review page (which marks `edited`) before pass 2.
FILES = {f'F/p{i}.jpg': dict(v=[f"'87 6 1{i}"]) for i in range(1, 5)}
FILES['F/p5.jpg'] = dict(v=[], nocand=True)
def BEFORE_PASS2(sd, root):
    out = root.parent / (root.name + '-review') / 'photos.csv'
    def edit(rows):
        for r in rows:
            if r['rel'] == 'F/p1.jpg': r.update(status='skip', edited='status')            # duplicate, skip it
            if r['rel'] == 'F/p2.jpg': r.update(date='1987-07-01', edited='date')          # typed the real date
            if r['rel'] == 'F/p5.jpg': r.update(rotate='0', status='ok', edited='rotate,status')  # rotated + OK
    sd.update_csv(out, edit)
    print("USER EDITS DONE:", [(r['rel'], r['status'], r['date'], r['rotate'], r['flags']) for r in sd.read_csv(out)][:5])
