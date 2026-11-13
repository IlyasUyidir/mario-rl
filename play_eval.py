import sys, os, csv, time
import numpy as np
import torch
from mario_env import MarioEnv
from student import Net

PATH = sys.argv[1]
N_EPS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
MODE = sys.argv[3] if len(sys.argv) > 3 else "sample"        # "sample" or "greedy"
name = os.path.basename(PATH).replace(".pt", "")
torch.manual_seed(0)
net = Net()
net.load_state_dict(torch.load(PATH, map_location="cpu"))     # trained on the GPU, played on the CPU
net.eval()
env = MarioEnv()
rows, lat = [], []
for e in range(N_EPS):
    obs, _ = env.reset()
    while True:
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, _ = net(torch.from_numpy(obs).permute(2, 0, 1)[None].float() / 255.0)
            p = torch.softmax(logits[0], 0)
            a = int(torch.multinomial(p, 1)) if MODE == "sample" else int(p.argmax())
        lat.append(time.perf_counter() - t0)
        obs, r, done, trunc, info = env.step(a)
        if done or trunc:
            break
    rows.append((e, info["max_x"], int(info["flag_get"])))
os.makedirs("results", exist_ok=True)
with open(f"results/play_{name}_{MODE}.csv", "w", newline="") as f:
    csv.writer(f).writerows([("episode", "max_x", "flag")] + rows)
xs = [r[1] for r in rows]
print(f"{name} [{MODE}]: success {sum(r[2] for r in rows)}/{N_EPS}  max_x mean {np.mean(xs):.0f}  "
      f"median {np.median(xs):.0f}  best {max(xs)}  latency {np.mean(lat) * 1000:.2f} ms/decision (cpu)")
env.close()
