import os, sys, time, resource
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from mario_env import MarioEnv

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "mps"
N_ENVS, STEPS = 4, 20_000

def make_env(i):
    def thunk():
        # Monitor logs one CSV row per finished episode: return, length, time, max_x, flag_get
        return Monitor(MarioEnv(), filename=f"runs/smoke/env{i}",
                       info_keywords=("max_x", "flag_get"))
    return thunk

if __name__ == "__main__":
    os.makedirs("runs/smoke", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    venv = DummyVecEnv([make_env(i) for i in range(N_ENVS)])
    model = PPO("CnnPolicy", venv, device=DEVICE, n_steps=128, batch_size=128,
                n_epochs=4, learning_rate=2.5e-4, clip_range=0.1, ent_coef=0.01,
                seed=0, verbose=1)
    t0 = time.time()
    model.learn(total_timesteps=STEPS)
    print(f"training wall-clock: {time.time() - t0:.0f}s on {DEVICE}")
    model.save("models/smoke_pixels")

    # reload and play one deterministic episode: checks save/load/predict end to end
    model = PPO.load("models/smoke_pixels", device=DEVICE)
    env = MarioEnv()
    obs, _ = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(int(action))
        if done or trunc:
            break
    print(f"eval episode: max_x={info['max_x']} flag={info['flag_get']}")
    print(f"peak RAM: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f} MB")
