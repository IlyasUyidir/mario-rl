import sys, glob, csv
import numpy as np

rows = []
for f in glob.glob(f"{sys.argv[1]}/*.monitor.csv"):
    with open(f) as fh:
        fh.readline()                                   # skip the '#' JSON comment line
        for r in csv.DictReader(fh):
            rows.append((float(r["t"]), float(r["max_x"]), r["flag_get"] == "True"))
t, x, fl = (np.array(c) for c in zip(*sorted(rows)))
idx = np.minimum((t / t.max() * 10).astype(int), 9)     # chunk number 0..9 by wall-clock time
for i in range(10):
    m = idx == i
    print(f"{i * 10:3d}-{i * 10 + 10:<3d}% of run: eps {m.sum():5d}  mean max_x {x[m].mean():5.0f}  flags {int(fl[m].sum())}")
