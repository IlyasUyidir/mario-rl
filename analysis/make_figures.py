import csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)
ORDER = ["pixel PPO, 1M steps", "grid PPO, 1M steps", "kin PPO, 1M steps",
         "CNN student, plain imitation", "CNN student + DAgger", "kin PPO, 3M steps (teacher)"]
SHORT = ["pixel PPO\n1M", "grid PPO\n1M", "kin PPO\n1M", "CNN student\nimitation", "CNN student\n+ DAgger", "kin PPO 3M\n(teacher)"]
COL = ["#999999", "#9ecae1", "#3182bd", "#fdae6b", "#e6550d", "#31a354"]
# approximate compute in minutes; the students include their 3M teacher (76.5) + data collection (7.4) + training (about 5, estimated)
COST = [53, 25, 25, 89, 114, 76.5]

by_model = {}
for r in csv.DictReader(open("results/final_eval.csv")):
    by_model.setdefault((r["group"], r["label"]), []).append((float(r["max_x"]), int(r["flag"])))
N = len(next(iter(by_model.values())))

def per_model(g, k):                                   # k = 0: mean max_x, k = 1: finish rate in %
    return [np.mean([e[k] for e in v]) * (100 if k == 1 else 1) for (gg, _), v in by_model.items() if gg == g]

def pooled_x(g):
    return [e[0] for (gg, _), v in by_model.items() if gg == g for e in v]

def sd(a):
    return float(np.std(a, ddof=1)) if len(a) > 1 else 0.0

# 1. finish rate per group
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, g in enumerate(ORDER):
    f = per_model(g, 1)
    ax.bar(i, np.mean(f), color=COL[i])
    if len(f) > 1:
        ax.errorbar(i, np.mean(f), yerr=sd(f), color="black", capsize=4)
    ax.scatter([i] * len(f), f, color="black", s=12, zorder=3)
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(SHORT, fontsize=8)
ax.set_ylabel("episodes reaching the flag (%)")
ax.set_title(f"Finish rate on World 1-1 ({N} sampled episodes per model; dots = seeds, bars = mean ± sd)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_finish_rate.png", dpi=150); plt.close(fig)

# 2. distance distribution
fig, ax = plt.subplots(figsize=(9, 4.5))
parts = ax.violinplot([pooled_x(g) for g in ORDER], showmedians=True)
for body, c in zip(parts["bodies"], COL):
    body.set_facecolor(c); body.set_alpha(0.85)
ax.axhline(3161, ls="--", color="gray"); ax.text(0.6, 3200, "flag (x = 3161)", fontsize=8, color="gray")
ax.set_xticks(range(1, len(ORDER) + 1)); ax.set_xticklabels(SHORT, fontsize=8)
ax.set_ylabel("furthest x reached in an episode"); ax.set_title("Distance reached (all episodes of all seeds pooled)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_distance.png", dpi=150); plt.close(fig)

# 3. DAgger by round
fig, ax = plt.subplots(figsize=(6.5, 4.5))
curves = []
for f in sorted(glob.glob("results/dagger_dagger_*.csv")):
    d = list(csv.DictReader(open(f)))
    curves.append([2 * int(r["success_of_50"]) for r in d])      # % of 50 episodes
    ax.plot(range(len(d)), curves[-1], marker="o", color="#e6550d", alpha=0.3)
ax.plot(range(len(curves[0])), np.mean(curves, axis=0), marker="o", lw=3, color="#e6550d", label=f"mean of {len(curves)} seeds")
ax.axhline(np.mean(per_model(ORDER[5], 1)), ls="--", color="#31a354", label="teacher")
ax.set_xticks(range(len(curves[0])))
ax.set_xlabel("DAgger round (0 = plain imitation)"); ax.set_ylabel("finish rate (%), 50 episodes per point")
ax.set_title("DAgger: finish rate by round", fontsize=10); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_dagger_rounds.png", dpi=150); plt.close(fig)

# 4. cost versus finish rate
fig, ax = plt.subplots(figsize=(7, 4.8))
for i, g in enumerate(ORDER):
    f = per_model(g, 1)
    ax.errorbar(COST[i], np.mean(f), yerr=sd(f), fmt="o", color=COL[i], ms=10, capsize=3)
    ax.annotate(SHORT[i].replace("\n", " "), (COST[i], np.mean(f)), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xlabel("approximate compute in minutes (students include their teacher's training)")
ax.set_ylabel("finish rate (%)"); ax.set_title("Cost versus finish rate", fontsize=10); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_cost_vs_finish.png", dpi=150); plt.close(fig)

# 5. student: confusion matrices and per-class precision / recall / F1 (held-out episodes, plain imitation, 3 seeds)
def prf(cm):
    tp, sup = np.diag(cm), cm.sum(1)
    p, r = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(sup, 1)
    return p, r, 2 * p * r / np.maximum(p + r, 1e-9)

A = ["noop", "right", "right+A", "right+B", "r+A+B", "A", "left"]
G = ["enemy", "empty", "solid", "mario"]
js = [json.load(open(f)) for f in sorted(glob.glob("results/student_student_*.json"))]
fig, axs = plt.subplots(2, 2, figsize=(11, 8))
for c, (key, names, idx, title) in enumerate([("confusion_buttons", A, [2, 3, 4, 5], "buttons"),
                                              ("confusion_grid", G, [0, 1, 2, 3], "tile-grid cells")]):
    cms = [np.array(j[key]) for j in js]
    sub = cms[0][np.ix_(idx, idx)] / np.maximum(cms[0][idx].sum(1, keepdims=True), 1)
    ax = axs[0, c]
    ax.imshow(sub, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(idx)):
        for j in range(len(idx)):
            ax.text(j, i, f"{sub[i, j]:.2f}", ha="center", va="center", color="white" if sub[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(idx))); ax.set_xticklabels([names[i] for i in idx], rotation=30)
    ax.set_yticks(range(len(idx))); ax.set_yticklabels([names[i] for i in idx])
    ax.set_xlabel("predicted"); ax.set_ylabel("true (teacher)")
    ax.set_title(f"{title}: confusion, seed 0 (rows sum to 1)", fontsize=10)
    P = np.array([[m[idx] for m in prf(cm)] for cm in cms])                  # (seeds, 3 metrics, classes)
    ax = axs[1, c]
    for k, (mname, col) in enumerate([("precision", "#6baed6"), ("recall", "#fd8d3c"), ("F1", "#31a354")]):
        ax.bar(np.arange(len(idx)) + 0.27 * (k - 1), P[:, k].mean(0), 0.27, yerr=P[:, k].std(0, ddof=1),
               color=col, label=mname, capsize=2)
    ax.set_xticks(range(len(idx))); ax.set_xticklabels([names[i] for i in idx], rotation=30)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    ax.set_title(f"{title}: per-class metrics (mean ± sd over {len(js)} seeds)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_student_metrics.png", dpi=150); plt.close(fig)
print("saved:", ", ".join(sorted(os.listdir(OUT))))
