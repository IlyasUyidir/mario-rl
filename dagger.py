import sys, os, csv, time
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from mario_env import MarioEnv
from mario_kin_env import MarioKinEnv
from student import Net

BASE, TEACHER, INIT, NAME = sys.argv[1:5]
ROUNDS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
N_NEW = int(sys.argv[6]) if len(sys.argv) > 6 else 40000
EPOCHS = int(sys.argv[7]) if len(sys.argv) > 7 else 8
SEED = int(sys.argv[8]) if len(sys.argv) > 8 else 0
DEV, BATCH = "mps", 128
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

class CollectEnv(MarioKinEnv):
    def _encode(self, frame):
        self.pixels = MarioEnv._encode(frame)          # what the student sees
        return super()._encode(frame)                  # tile grid + kin numbers, what the teacher sees

def cpu_copy(n):
    c = Net()
    c.load_state_dict({k: v.cpu() for k, v in n.state_dict().items()})
    return c.eval()

def collect(student, n, ep0):
    """The STUDENT plays; the TEACHER labels every state the student reaches."""
    env = CollectEnv()
    pix = np.zeros((n, 84, 84), np.uint8); grid = np.zeros((n, 13, 11), np.int8)
    probs = np.zeros((n, 7), np.float16); ep = np.zeros(n, np.int32); tt = np.zeros(n, np.int32)
    i, e = 0, ep0
    while i < n:
        obs, _ = env.reset()
        stack, t = [env.pixels] * 4, 0                 # reset fills the stack with 4 copies
        while i < n:
            with torch.no_grad():
                obs_t, _ = teacher.policy.obs_to_tensor(obs)
                p = teacher.policy.get_distribution(obs_t).distribution.probs[0].numpy().astype(np.float64)
                x = torch.from_numpy(np.stack(stack)).float()[None] / 255.0
                a = int(torch.multinomial(torch.softmax(student(x)[0][0], 0), 1))
            pix[i], grid[i], probs[i], ep[i], tt[i] = env.pixels, env.frames[-1], p / p.sum(), e, t
            i, t = i + 1, t + 1
            obs, r, done, trunc, info = env.step(a)    # execute the STUDENT's button
            stack = stack[1:] + [env.pixels]
            if done or trunc:
                break
        e += 1
    env.close()
    return [pix, grid, probs, ep, tt]

def fit(net, data, epochs):
    pix, grid, probs, ep, t = data
    N = len(pix)
    k = np.minimum(np.arange(3, -1, -1)[None, :], t[:, None])
    stack_idx = np.arange(N)[:, None] - k              # 4-frame stacks rebuilt from single frames
    pix_t = torch.from_numpy(pix)
    gy = torch.from_numpy((grid.astype(np.int64) + 1).reshape(N, -1))
    soft = torch.from_numpy(probs.astype(np.float32))
    freq = np.bincount(gy.numpy().ravel(), minlength=4)
    gw = torch.tensor(np.sqrt(freq.max() / np.maximum(freq, 1)), dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(net.parameters(), 3e-4)
    net.train()
    for epoch in range(epochs):
        perm = rng.permutation(N)
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            xb = pix_t[torch.from_numpy(stack_idx[idx])].to(DEV).float() / 255.0
            al, gl = net(xb)
            loss = F.cross_entropy(al, soft[idx].to(DEV)) + F.cross_entropy(gl, gy[idx].to(DEV), weight=gw)
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"  trained {epochs} epochs on {N} samples", flush=True)

def play(student, n):
    torch.manual_seed(0)
    env = MarioEnv()
    rows = []
    for e in range(n):
        obs, _ = env.reset()
        while True:
            with torch.no_grad():
                logits, _ = student(torch.from_numpy(obs).permute(2, 0, 1)[None].float() / 255.0)
                a = int(torch.multinomial(torch.softmax(logits[0], 0), 1))
            obs, r, done, trunc, info = env.step(a)
            if done or trunc:
                break
        rows.append((e, info["max_x"], int(info["flag_get"])))
    env.close()
    return rows

base = np.load(BASE)
data = [base["pix"], base["grid"], base["probs"], base["ep"], base["t"]]
ep_next = int(base["ep"].max()) + 1
teacher = PPO.load(TEACHER, device="cpu")
net = Net()
net.load_state_dict(torch.load(INIT, map_location="cpu"))
net.to(DEV)
os.makedirs("results", exist_ok=True)
log = []

def evaluate(rnd):
    rows = play(cpu_copy(net), 50)
    xs, ok = [r[1] for r in rows], sum(r[2] for r in rows)
    with open(f"results/play_{NAME}_r{rnd}_sample.csv", "w", newline="") as f:
        csv.writer(f).writerows([("episode", "max_x", "flag")] + rows)
    log.append((rnd, len(data[0]), ok, float(np.mean(xs)), float(np.median(xs))))
    with open(f"results/dagger_{NAME}.csv", "w", newline="") as f:
        csv.writer(f).writerows([("round", "samples", "success_of_50", "mean_max_x", "median_max_x")] + log)
    print(f"round {rnd}: {len(data[0])} samples, success {ok}/50, max_x mean {np.mean(xs):.0f} median {np.median(xs):.0f}", flush=True)

evaluate(0)                                            # starting point: the plain imitation student
for rnd in range(1, ROUNDS + 1):
    t0 = time.time()
    new = collect(cpu_copy(net), N_NEW, ep_next)
    ep_next = int(new[3].max()) + 1
    np.savez(f"data/dagger_{NAME}_r{rnd}.npz", pix=new[0], grid=new[1], probs=new[2], ep=new[3], t=new[4])
    data = [np.concatenate([a, b]) for a, b in zip(data, new)]
    fit(net, data, EPOCHS)
    torch.save(net.state_dict(), f"models/{NAME}_r{rnd}.pt")
    evaluate(rnd)
    print(f"  (round {rnd} took {time.time() - t0:.0f}s)", flush=True)
