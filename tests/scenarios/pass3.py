# Pass 3: (a) user types a date / skips a row while the model is working on it (each call ~10-30 s);
# (b) the model raises for one row (Metal OOM) -> what is left in the row, and is it retried?
FILES = {f'F/p{i}.jpg': dict(v=[f"'87 6 1{i}"]) for i in range(1, 6)}
FILES['F/u1.jpg'] = dict(v=["05 06 87"], q="05 06 87")   # ambiguous order -> model
FILES['F/u2.jpg'] = dict(v=["06 07 87"], q="06 07 87")   # ambiguous order -> model, model raises
MLX_RAISE = {'F/u2.jpg'}
def on_mlx_line(rel, sd, root):
    if rel == 'F/u1.jpg':
        out = root.parent / (root.name + '-review') / 'photos.csv'
        def edit(rows):
            for r in rows:
                if r['rel'] == rel: r.update(date='1987-06-05', status='skip', edited='date,status')
        sd.update_csv(out, edit)
        print("USER EDIT during model call:", [(r['date'], r['status']) for r in sd.read_csv(out) if r['rel'] == rel])
