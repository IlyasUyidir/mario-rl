import os, sys, time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from mario_grid_env import MarioGridEnv
from mario_kin_env import MarioKinEnv
from mario_env import MarioEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
VARIANT = sys.argv[3] if len(sys.argv) > 3 else "grid"      # "grid" or "kin"
ENV = {"grid": MarioGridEnv, "kin": MarioKinEnv, "pix": MarioEnv}[VARIANT]
PIX = VARIANT == "pix"
N_ENVS = 4
NAME = f"{VARIANT}_seed{SEED}_{STEPS // 1000}k"
RUN = f"runs/{NAME}"

def make_env(i):
    def thunk():
        return Monitor(ENV(), filename=f"{RUN}/env{i}", info_keywords=("max_x", "flag_get"))
    return thunk

if __name__ == "__main__":
    os.makedirs(RUN, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    venv = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    model = PPO("CnnPolicy" if PIX else "MlpPolicy", venv, device="mps" if PIX else "cpu", n_steps=128, batch_size=128,
                n_epochs=4, learning_rate=2.5e-4, clip_range=0.1, ent_coef=0.01,
                policy_kwargs=None if PIX else dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                seed=SEED, verbose=1)
    ckpt = CheckpointCallback(save_freq=50_000 // N_ENVS, save_path="models", name_prefix=NAME)
    t0 = time.time()
    model.learn(total_timesteps=STEPS, callback=ckpt, log_interval=50)
    print(f"training wall-clock: {time.time() - t0:.0f}s")
    model.save(f"models/{NAME}")
    venv.close()

    env = ENV()                                 # one greedy episode + inference latency
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
