import glob, csv, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(folder):
    rows = []
    for f in glob.glob(f"{folder}/*.monitor.csv"):
        with open(f) as fh:
            fh.readline()                                   # skip the '#' JSON comment line
            for r in csv.DictReader(fh):
                rows.append((float(r["t"]), float(r["max_x"]), r["flag_get"] == "True"))
    return (np.array(c) for c in zip(*sorted(rows)))

def stats(t, x, fl):
    idx = np.minimum((t / t.max() * 10).astype(int), 9)      # 10 equal time chunks
    last = idx == 9
    return dict(final_x=x[last].mean(), final_pct=100 * fl[last].mean(), flags=int(fl.sum()),
                wall=t.max(), curve=[x[idx == i].mean() for i in range(10)])

runs = {}                                                    # variant -> list of per-seed stats
for folder in sorted(glob.glob("runs/*_seed*_1000k")):
    m = re.search(r"runs/(\w+?)_seed(\d+)_1000k", folder)
    t, x, fl = load(folder)
    runs.setdefault(m.group(1), []).append((int(m.group(2)), stats(t, x, fl)))

print("per run (last chunk = final 10% of the run)")
for v, lst in runs.items():
    for seed, s in lst:
        print(f"  {v:<5} seed {seed}: final max_x {s['final_x']:5.0f}  finish rate {s['final_pct']:4.1f}%  "
              f"flags {s['flags']:4d}  wall {s['wall']:.0f}s")

def ms(a):
    a = np.array(a, dtype=float)
    return f"{a.mean():.0f} +/- {a.std(ddof=1) if len(a) > 1 else 0:.0f}"

print("\nper variant, mean +/- sd across seeds")
for v, lst in runs.items():
    S = [s for _, s in lst]
    print(f"  {v:<5} n={len(S)}  final max_x {ms([s['final_x'] for s in S])}  "
          f"finish rate % {ms([s['final_pct'] for s in S])}  total flags {ms([s['flags'] for s in S])}  "
          f"wall {ms([s['wall'] for s in S])}s")

xs = np.arange(5, 100, 10)                                   # centre of each 10% chunk
colors = {"grid": "tab:blue", "kin": "tab:red"}
for v, lst in runs.items():
    for _, s in lst:
        plt.plot(xs, s["curve"], color=colors.get(v), alpha=0.25, lw=1)
    plt.plot(xs, np.mean([s["curve"] for _, s in lst], axis=0), color=colors.get(v), lw=3, label=f"{v} (mean of {len(lst)})")
plt.axhline(570, ls=":", color="gray", label="random (mean 570)")
plt.axhline(695, ls="--", color="gray", label="scripted (695)")
plt.xlabel("% of the 1M-step run"); plt.ylabel("mean max x reached"); plt.legend(); plt.grid(alpha=0.3)
plt.title("Super Mario Bros 1-1: PPO learning curves")
plt.savefig("curves.png", dpi=150, bbox_inches="tight")
