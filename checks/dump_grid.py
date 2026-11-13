import sys
import cv2
from stable_baselines3.common.env_checker import check_env
from mario_grid_env import MarioGridEnv

env = MarioGridEnv()
check_env(env)                                    # interface check, like before
env.reset()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
for t in range(n):                                # the Stage 0 "run right + jump" pattern
    _, _, done, trunc, _ = env.step(4 if t % 6 < 4 else 3)
    if done or trunc:
        break
obs, _, info = env._inner_step(0)                 # one NOOP frame: also grabs the raw screen
grid = env._encode(obs)
cv2.imwrite("frame.png", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
print("x_pos:", info["x_pos"], "y_pos:", info["y_pos"])
for row in grid:
    print("".join({0: ".", 1: "#", -1: "E", 2: "M"}[int(v)] for v in row))
env.close()
