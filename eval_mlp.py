import sys, time
import numpy as np
import torch
from stable_baselines3 import PPO
from mario_grid_env import MarioGridEnv
from mario_kin_env import MarioKinEnv

variant, path = sys.argv[1], sys.argv[2]
n_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
ENV = {"grid": MarioGridEnv, "kin": MarioKinEnv}[variant]
torch.manual_seed(0)                       # fixed seed so the eval itself is repeatable
model = PPO.load(path, device="cpu")
env = ENV()
xs, flags, lat = [], 0, []
for _ in range(n_eps):
    obs, _ = env.reset()
    while True:
        t = time.perf_counter()
        action, _ = model.predict(obs, deterministic=False)   # sample, like in training
        lat.append(time.perf_counter() - t)
        obs, r, done, trunc, info = env.step(int(action))
        if done or trunc:
            break
    xs.append(info["max_x"])
    flags += int(info["flag_get"])
print(f"{path}: success {flags}/{n_eps}  max_x mean {np.mean(xs):.0f}  "
      f"median {np.median(xs):.0f}  best {max(xs)}  latency {np.mean(lat) * 1000:.2f} ms")
env.close()
