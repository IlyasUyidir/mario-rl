import sys, glob, csv, collections

rows = []
for f in glob.glob(f"{sys.argv[1]}/*.monitor.csv"):
    with open(f) as fh:
        fh.readline()                                   # skip the '#' JSON comment line
        for r in csv.DictReader(fh):
            rows.append((float(r["t"]), float(r["max_x"])))
rows.sort()                                             # by wall-clock time
last = [x for _, x in rows[-1000:]]                     # late training only
bins = collections.Counter(int(x // 100) * 100 for x in last)
peak = max(bins.values())
for b in sorted(bins):
    if bins[b] >= 10:                                   # hide bins with under 1% of episodes
        print(f"{b:5d}-{b + 99:<5d} {'#' * (bins[b] * 40 // peak)} {bins[b]}")
