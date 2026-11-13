from stable_baselines3.common.env_checker import check_env
from mario_kin_env import MarioKinEnv

env = MarioKinEnv()
check_env(env)
obs, _ = env.reset()
print("shape:", obs.shape, " standing :", obs[-4:].round(2))
for _ in range(15):
    obs, *_ = env.step(3)                       # right + B (run)
print("after running      :", obs[-4:].round(2))
for k in range(1, 7):
    obs, *_ = env.step(4)                       # right + A + B (jump held)
    print(f"jump step {k}        :", obs[-4:].round(2))
env.close()
