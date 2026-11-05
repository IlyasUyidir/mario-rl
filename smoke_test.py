import time
from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = JoypadSpace(make("SuperMarioBros-1-1-v0"), SIMPLE_MOVEMENT)
print("n actions:", env.action_space.n)

out = env.reset()
obs = out[0] if isinstance(out, tuple) else out   # new API returns (obs, info)
print("obs shape:", obs.shape)

t0, steps = time.time(), 0
for _ in range(2000):
    res = env.step(env.action_space.sample())
    done = res[2] or (len(res) == 5 and res[3])   # handles old and new API
    info = res[-1]
    steps += 1
    if done:
        env.reset()
print("info keys:", sorted(info.keys()))
print("x_pos last:", info["x_pos"])
print(f"steps/sec: {steps / (time.time() - t0):.0f}")
env.close()
