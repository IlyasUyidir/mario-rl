import os, sys, time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from mario_grid_env import MarioGridEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
N_ENVS = 4
RUN = f"runs/grid_seed{SEED}_{STEPS // 1000}k"

def make_env(i):
    def thunk():
        return Monitor(MarioGridEnv(), filename=f"{RUN}/env{i}",
                       info_keywords=("max_x", "flag_get"))
    return thunk

if __name__ == "__main__":
    os.makedirs(RUN, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    venv = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    model = PPO("MlpPolicy", venv, device="cpu", n_steps=128, batch_size=128,
                n_epochs=4, learning_rate=2.5e-4, clip_range=0.1, ent_coef=0.01,
                policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                seed=SEED, verbose=1)
    ckpt = CheckpointCallback(save_freq=50_000 // N_ENVS, save_path="models",
                              name_prefix=f"grid_seed{SEED}_{STEPS // 1000}k")
    t0 = time.time()
    model.learn(total_timesteps=STEPS, callback=ckpt, log_interval=50)
    print(f"training wall-clock: {time.time() - t0:.0f}s")
    model.save(f"models/grid_seed{SEED}_{STEPS // 1000}k")
    venv.close()

    # one deterministic episode + inference latency (one observation at a time, like a live demo)
    env = MarioGridEnv()
    obs, _ = env.reset()
    lat = []
    for _ in range(2000):
        t = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        lat.append(time.perf_counter() - t)
        obs, r, done, trunc, info = env.step(int(action))
        if done or trunc:
            break
    print(f"eval episode: max_x={info['max_x']} flag={info['flag_get']} steps={len(lat)}")
    print(f"inference latency: {np.mean(lat) * 1000:.2f} ms/step (cpu, batch 1)")
