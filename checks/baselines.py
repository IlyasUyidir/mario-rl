import time, resource
import numpy as np
from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

MAX_STEPS = 4000
# SIMPLE_MOVEMENT indices: 0 NOOP, 1 right, 2 right+A, 3 right+B, 4 right+A+B, 5 A, 6 left

def run_episode(env, policy):
    env.reset()
    best_x, flag, t = 0, False, 0
    while t < MAX_STEPS:
        res = env.step(policy(t))
        done = res[2] or (len(res) == 5 and res[3])  # old and new API
        info = res[-1]
        best_x = max(best_x, info["x_pos"])          # running max, safe against reset on death
        flag = flag or bool(info["flag_get"])
        t += 1
        if done:
            break
    return best_x, flag, t

def evaluate(env, name, policy, n_eps):
    t0, results = time.time(), []
    for _ in range(n_eps):
        results.append(run_episode(env, policy))
    xs = [r[0] for r in results]
    steps = sum(r[2] for r in results)
    print(f"{name}: episodes={n_eps}")
    print(f"  max x per episode: {xs}")
    print(f"  mean x={np.mean(xs):.0f}  best x={max(xs)}  flags={sum(r[1] for r in results)}/{n_eps}")
    print(f"  steps/sec={steps/(time.time()-t0):.0f}")

env = JoypadSpace(make("SuperMarioBros-1-1-v0"), SIMPLE_MOVEMENT)
rng = np.random.default_rng(0)

evaluate(env, "random", lambda t: int(rng.integers(7)), 10)
# 16 frames right+A+B (jump held), 8 frames right+B (release so the next jump works)
evaluate(env, "scripted", lambda t: 4 if t % 24 < 16 else 3, 3)

print(f"peak RAM: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f} MB")  # macOS reports bytes
env.close()
