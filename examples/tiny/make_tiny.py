#!/usr/bin/env python3
"""Write the two tiny example tasks used by the posbench self-test (standard library only, deterministic).

clean/  : 600 train, 100 dev, 200 test random 200-bp sequences; positives carry the motif TATAAA at a random position
          with probability MOTIF_P (a real, position-free signal that a model can learn); no leak.
leaky/  : the same task, but 15% of the test items are replaced by copies of training items (half of them reverse
          complemented) and 10% by near copies (3% substitutions), each with the label of its training source.

Files: <dir>/train.csv, dev.csv, test.csv with columns sequence,label (the GUE layout).
Usage: python make_tiny.py [OUTDIR]   (default: the folder of this script)"""
import csv
import os
import random
import sys

SEED, L, N = 20261022, 200, dict(train=600, dev=100, test=200)
MOTIF, MOTIF_P = 'TATAAA', 0.6
COPY, NEAR, NEAR_RATE = 0.15, 0.10, 0.03


def revcomp(s):
    return s.translate(str.maketrans('ACGT', 'TGCA'))[::-1]


def make_clean(r):
    data = {}
    for split, n in N.items():
        rows = []
        for i in range(n):
            y = i % 2
            s = ''.join(r.choice('ACGT') for _ in range(L))
            if y == 1 and r.random() < MOTIF_P:
                p = r.randrange(L - len(MOTIF))
                s = s[:p] + MOTIF + s[p + len(MOTIF):]
            rows.append((s, y))
        r.shuffle(rows)
        data[split] = rows
    return data


def make_leaky(clean, r):
    data = {k: list(v) for k, v in clean.items()}
    test, train = data['test'], data['train']
    idx = list(range(len(test)))
    r.shuffle(idx)
    n_copy, n_near = round(COPY * len(test)), round(NEAR * len(test))
    for k, i in enumerate(idx[:n_copy + n_near]):
        s, y = train[r.randrange(len(train))]
        if k < n_copy:
            s = revcomp(s) if k % 2 else s
        else:
            s = ''.join(r.choice([b for b in 'ACGT' if b != c]) if r.random() < NEAR_RATE else c for c in s)
        test[i] = (s, y)
    return data


def write(data, d):
    os.makedirs(d, exist_ok=True)
    for split, rows in data.items():
        with open(os.path.join(d, split + '.csv'), 'w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(['sequence', 'label'])
            w.writerows(rows)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    r = random.Random(SEED)
    clean = make_clean(r)
    write(clean, os.path.join(out, 'clean'))
    write(make_leaky(clean, r), os.path.join(out, 'leaky'))
    print('wrote', os.path.join(out, 'clean'), 'and', os.path.join(out, 'leaky'))


if __name__ == '__main__':
    main()
