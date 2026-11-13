import glob, os, re, time, csv
import numpy as np
import torch
from stable_baselines3 import PPO
from mario_env import MarioEnv
from mario_grid_env import MarioGridEnv
from mario_kin_env import MarioKinEnv

N_EPS = 50
ENVS = {"grid": MarioGridEnv, "kin": MarioKinEnv, "pix": MarioEnv}
os.makedirs("results", exist_ok=True)
rows = []
for path in sorted(glob.glob("models/*_seed?_1000k.zip")):
    variant, seed = re.search(r"models/(\w+?)_seed(\d)_1000k", path).groups()
    torch.manual_seed(0)                                   # same sampling seed for every model
    model = PPO.load(path, device="cpu")
    env = ENVS[variant]()
    xs, flags, lat = [], 0, []
    for _ in range(N_EPS):
        obs, _ = env.reset()
        while True:
            t = time.perf_counter()
            action, _ = model.predict(obs, deterministic=False)
            lat.append(time.perf_counter() - t)
            obs, r, done, trunc, info = env.step(int(action))
            if done or trunc:
                break
        xs.append(info["max_x"])
        flags += int(info["flag_get"])
    env.close()
    rows.append((variant, int(seed), flags, float(np.mean(xs)), float(np.median(xs)), max(xs),
                 float(np.mean(lat) * 1000)))
    print(f"{variant} seed {seed}: success {flags}/{N_EPS}  mean {np.mean(xs):.0f}  "
          f"median {np.median(xs):.0f}  best {max(xs)}  latency {np.mean(lat) * 1000:.2f} ms", flush=True)

with open("results/eval_1000k.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "seed", "successes", "mean_max_x", "median_max_x", "best_max_x", "latency_ms_cpu"])
    w.writerows(rows)

def ms(a):
    a = np.array(a, dtype=float)
    return f"{a.mean():.0f} +/- {a.std(ddof=1) if len(a) > 1 else 0:.0f}"

print(f"\nper variant: {N_EPS} sampled episodes per model, mean +/- sd across seeds")
for v in ENVS:
    R = [r for r in rows if r[0] == v]
    if R:
        print(f"  {v:<5} n={len(R)}  success {sum(r[2] for r in R)}/{N_EPS * len(R)}  "
              f"median max_x {ms([r[4] for r in R])}  mean max_x {ms([r[3] for r in R])}  "
              f"latency {np.mean([r[6] for r in R]):.2f} ms (cpu)")
