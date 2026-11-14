import sys, csv
import numpy as np
import torch
from mario_env import MarioEnv
from student import Net

N_EPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
HUD_ROWS = 11                                           # the top 11 of 84 rows hold score, coins and timer
PATHS = [("seed 0", "models/dagger_v1_r3.pt"), ("seed 1", "models/dagger_s1_r3.pt"), ("seed 2", "models/dagger_s2_r3.pt")]

def mask(obs):
    out = obs.copy()
    for c in range(out.shape[2]):                       # fill the HUD rows with the typical sky grey of each frame
        out[:HUD_ROWS, :, c] = int(np.median(out[HUD_ROWS:, :, c]))
    return out

def run(net, masked):
    torch.manual_seed(0)
    env = MarioEnv()
    flags, xs = 0, []
    for _ in range(N_EPS):
        obs, _ = env.reset()
        while True:
            o = mask(obs) if masked else obs
            with torch.no_grad():
                logits, _ = net(torch.from_numpy(o).permute(2, 0, 1)[None].float() / 255.0)
                a = int(torch.multinomial(torch.softmax(logits[0], 0), 1))
            obs, r, done, trunc, info = env.step(a)
            if done or trunc:
                break
        flags += int(info["flag_get"])
        xs.append(info["max_x"])
    env.close()
    return 100 * flags / N_EPS, float(np.mean(xs))

rows = []
for name, path in PATHS:
    net = Net()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    for masked in (False, True):
        fin, mx = run(net, masked)
        rows.append((name, int(masked), fin, mx))
        print(f"{name}  HUD {'masked ' if masked else 'visible'}  finish {fin:5.1f}%  max_x {mx:5.0f}", flush=True)
with open("results/hud_test.csv", "w", newline="") as f:
    csv.writer(f).writerows([("model", "hud_masked", "finish_pct", "mean_max_x")] + rows)
