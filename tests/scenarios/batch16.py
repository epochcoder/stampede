# A real 16-scan batch as Apple Vision read it on a Mac (stamps like 25/12/2006, camera prints
# day/month/year), with a Qwen stand-in that misreads some, and faces on five photos at rotation 90.
VISION = {1: '12/03/2006', 2: '12/03/2006', 3: '25/12/2006', 4: '25/12/2006', 5: '19/11/2006', 6: '19/11/2006',
          7: '22/03/2006', 8: '25 * 2/2006', 9: '19/11/2006', 10: '22/03/2006', 11: '25/12/2006', 12: '25/12/2006',
          13: '', 14: '', 15: '25/12/2006', 16: ''}
QWEN = {6: '15/11/2006', 8: '25/12/2006', 11: '12/02/2006', 13: 'NONE', 14: '19/11/2006', 16: 'NONE'}
FACES = {3: 90, 5: 90, 9: 90, 12: 90, 14: 90}
FILES = {f'SB450_{i:04d}.jpg': dict(v=[VISION[i]] if VISION[i] else [], q=QWEN.get(i, VISION[i] or 'NONE'),
                                    face=FACES.get(i), stamprot=90) for i in range(1, 17)}


def ORIENT(rel, cands):
    return 90 if 90 in cands else cands[0]
