import time
import cv2
import numpy as np
from stable_baselines3.common.env_checker import check_env
from mario_env import MarioEnv

env = MarioEnv()
check_env(env)                                  # raises if the Gymnasium interface is broken
obs, _ = env.reset()
print("obs:", obs.shape, obs.dtype, "min", obs.min(), "max", obs.max())
cv2.imwrite("obs_check.png", np.concatenate([obs[:, :, i] for i in range(4)], axis=1))
total, t0 = 0.0, time.time()
for i in range(300):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    total += r
    if term or trunc:
        print(f"episode ended at step {i+1}: max_x={info['max_x']} term={term} trunc={trunc}")
        env.reset()
print(f"total reward {total:.1f}; agent steps/sec {300 / (time.time() - t0):.0f}")
env.close()
