import os, sys, time
import numpy as np
import torch
from stable_baselines3 import PPO
from mario_env import MarioEnv
from mario_kin_env import MarioKinEnv

class CollectEnv(MarioKinEnv):
    def _encode(self, frame):
        self.pixels = MarioEnv._encode(frame)      # what the pix variant would see
        return super()._encode(frame)              # tile grid, what grid/kin see

N = int(sys.argv[1])
TEACHER = sys.argv[2]
OUT = sys.argv[3]
EPS_RANDOM = 0.1                                   # 10% random buttons, for more varied states

env = CollectEnv()
model = PPO.load(TEACHER, device="cpu")
rng = np.random.default_rng(0)
pix = np.zeros((N, 84, 84), np.uint8)
grid = np.zeros((N, 13, 11), np.int8)
kin = np.zeros((N, 4), np.float32)
probs = np.zeros((N, 7), np.float16)
act = np.zeros(N, np.int8)
ep = np.zeros(N, np.int32)
tt = np.zeros(N, np.int32)
xs = np.zeros(N, np.int16)

i, e, t0 = 0, 0, time.time()
while i < N:
    obs, _ = env.reset()
    t = 0
    while i < N:
        with torch.no_grad():                      # the teacher's opinion of this exact state
            obs_t, _ = model.policy.obs_to_tensor(obs)
            p = model.policy.get_distribution(obs_t).distribution.probs[0].numpy().astype(np.float64)
        p /= p.sum()
        a = int(rng.integers(7)) if rng.random() < EPS_RANDOM else int(rng.choice(7, p=p))
        pix[i], grid[i], kin[i] = env.pixels, env.frames[-1], obs[-4:]   # record BEFORE acting
        probs[i], act[i], ep[i], tt[i], xs[i] = p, a, e, t, env.last_x
        i += 1
        t += 1
        if i % 10000 == 0:
            print(f"  {i}/{N} samples, {i / (time.time() - t0):.0f} steps/s", flush=True)
        obs, r, done, trunc, info = env.step(a)
        if done or trunc:
            break
    e += 1

np.savez(OUT, pix=pix, grid=grid, kin=kin, probs=probs, act=act, ep=ep, t=tt, x=xs)
print(f"saved {OUT}: {N} samples, {e} episodes (mean length {N / e:.0f} steps), {time.time() - t0:.0f} s")
print("x reached, percentiles 10/50/90:", np.percentile(xs, [10, 50, 90]).astype(int))
print("buttons played (noop,right,right+A,right+B,right+A+B,A,left):", np.bincount(act, minlength=7))
print(f"teacher's mean top probability: {probs.astype(np.float32).max(1).mean():.2f}")
env.close()
