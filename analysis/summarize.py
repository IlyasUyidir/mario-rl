import sys, glob, csv
import numpy as np

rows = []
for f in glob.glob(f"{sys.argv[1]}/*.monitor.csv"):
    with open(f) as fh:
        fh.readline()                              # first line is a '#' JSON comment
        for r in csv.DictReader(fh):               # columns: r (return), l, t, max_x, flag_get
            rows.append((float(r["t"]), float(r["r"]), float(r["max_x"]), r["flag_get"] == "True"))
rows.sort()                                        # order by wall-clock time across envs
rets = np.array([r[1] for r in rows]); xs = np.array([r[2] for r in rows])
q = max(1, len(rows) // 4)
print(f"episodes: {len(rows)}   flags: {sum(r[3] for r in rows)}")
print(f"last-100 mean return (SB3's ep_rew_mean): {rets[-100:].mean():.0f}")
print(f"first quarter: return {rets[:q].mean():.0f}  max_x {xs[:q].mean():.0f}")
print(f"last quarter:  return {rets[-q:].mean():.0f}  max_x {xs[-q:].mean():.0f}")
print(f"wall-clock (max t): {max(r[0] for r in rows):.0f}s")
