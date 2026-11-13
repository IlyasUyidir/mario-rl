import csv, os, sys, time
import numpy as np
import torch
from stable_baselines3 import PPO
from mario_env import MarioEnv
from mario_grid_env import MarioGridEnv
from mario_kin_env import MarioKinEnv
from student import Net

N_EPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
ENVS = {"pix": MarioEnv, "grid": MarioGridEnv, "kin": MarioKinEnv, "student": MarioEnv}

# (group, label, path): PPO models are SB3 zips, students are .pt state dicts
MODELS = ([("pixel PPO, 1M steps", "pix_seed0", "models/pix_seed0_1000k")]
          + [("grid PPO, 1M steps", f"grid_seed{s}", f"models/grid_seed{s}_1000k") for s in range(3)]
          + [("kin PPO, 1M steps", f"kin_seed{s}", f"models/kin_seed{s}_1000k") for s in range(3)]
          + [("kin PPO, 3M steps (teacher)", "kin_seed2_3M", "models/kin_seed2_3000k")]
          + [("CNN student, plain imitation", n, f"models/{n}.pt") for n in ["student_v1", "student_s1", "student_s2"]]
          + [("CNN student + DAgger", n, f"models/{n}_r3.pt") for n in ["dagger_v1", "dagger_s1", "dagger_s2"]])

def load(path):
    if path.endswith(".pt"):
        net = Net()
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        def act(obs):
            with torch.no_grad():
                logits, _ = net(torch.from_numpy(obs).permute(2, 0, 1)[None].float() / 255.0)
                return int(torch.multinomial(torch.softmax(logits[0], 0), 1))
        return act
    model = PPO.load(path, device="cpu")
    return lambda obs: int(model.predict(obs, deterministic=False)[0])

rows, summary = [], []
for group, label, path in MODELS:
    if not os.path.exists(path if path.endswith(".pt") else path + ".zip"):
        print("missing:", path)
        continue
    env = ENVS["student" if path.endswith(".pt") else label.split("_")[0]]()
    act = load(path)
    torch.manual_seed(0)                       # seeded AFTER loading: every model draws the same random stream
    lat, res = [], []
    for e in range(N_EPS):
        obs, _ = env.reset()
        steps = 0
        while True:
            t0 = time.perf_counter()
            a = act(obs)
            lat.append(time.perf_counter() - t0)
            obs, r, done, trunc, info = env.step(a)
            steps += 1
            if done or trunc:
                break
        res.append((info["max_x"], int(info["flag_get"]), steps))
        rows.append((group, label, e) + res[-1])
    env.close()
    fin, mx, lt = 100 * np.mean([r[1] for r in res]), np.mean([r[0] for r in res]), np.mean(lat) * 1000
    summary.append((group, label, fin, mx, lt))
    print(f"{label:<14} finish {fin:5.1f}%  max_x mean {mx:5.0f}  latency {lt:.2f} ms", flush=True)

os.makedirs("results", exist_ok=True)
with open("results/final_eval.csv", "w", newline="") as f:
    csv.writer(f).writerows([("group", "label", "episode", "max_x", "flag", "steps")] + rows)

groups = {}
for g, lab, fin, mx, lt in summary:
    groups.setdefault(g, []).append((fin, mx, lt))
sd = lambda a: float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
print(f"\n{N_EPS} sampled episodes per model; mean +/- sd across seeds")
for g, L in groups.items():
    f, m = [x[0] for x in L], [x[1] for x in L]
    print(f"  {g:<30} n={len(L)}  finish {np.mean(f):5.1f} +/- {sd(f):.1f} %   max_x {np.mean(m):5.0f} +/- {sd(m):.0f}   "
          f"latency {np.mean([x[2] for x in L]):.2f} ms")
