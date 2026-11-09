import sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0)
d = np.load(sys.argv[1])
pix, grid, probs = d["pix"], d["grid"], d["probs"].astype(np.float32)
ep, t, x = d["ep"], d["t"], d["x"]
N, EPOCHS, BATCH, DEV = len(pix), 8, 128, "mps"

# split by EPISODE (random frame splits would leak: neighbouring frames are near-identical)
eps = np.unique(ep)
np.random.default_rng(0).shuffle(eps)
is_val = np.isin(ep, eps[: max(1, len(eps) // 5)])
train_idx, val_idx = np.where(~is_val)[0], np.where(is_val)[0]
print(f"{N} samples: {len(train_idx)} train / {len(val_idx)} validation ({len(np.unique(ep[is_val]))} validation episodes)")

# rebuild the 4-frame stack from single frames, clamped at the episode start (as reset() does)
k = np.minimum(np.arange(3, -1, -1)[None, :], t[:, None])
stack_idx = np.arange(N)[:, None] - k
pix_t = torch.from_numpy(pix)
gy = torch.from_numpy(grid.astype(np.int64) + 1).view(N, -1)     # classes: 0 enemy, 1 empty, 2 solid, 3 mario
soft = torch.from_numpy(probs)                                    # teacher probabilities (soft labels)
hard = probs.argmax(1)

def batch(idx):
    return pix_t[torch.from_numpy(stack_idx[idx])].to(DEV).float() / 255.0   # (B, 4, 84, 84)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(4, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(),
                                 nn.Linear(3136, 512), nn.ReLU())
        self.act = nn.Linear(512, 7)
        self.grid = nn.Linear(512, 4 * 13 * 11)
    def forward(self, xb):
        h = self.cnn(xb)
        return self.act(h), self.grid(h).view(-1, 4, 13 * 11)

net = Net().to(DEV)
opt = torch.optim.Adam(net.parameters(), 3e-4)

def predict(idx_all):
    net.eval()
    a_out, g_out = [], []
    with torch.no_grad():
        for s in range(0, len(idx_all), 512):
            al, gl = net(batch(idx_all[s:s + 512]))
            a_out.append(al.argmax(1).cpu().numpy())
            g_out.append(gl.argmax(1).cpu().numpy())
    net.train()
    return np.concatenate(a_out), np.concatenate(g_out)

for epoch in range(EPOCHS):
    t0, tot, nb = time.time(), 0.0, 0
    perm = np.random.default_rng(epoch).permutation(train_idx)
    for s in range(0, len(perm), BATCH):
        idx = perm[s:s + BATCH]
        al, gl = net(batch(idx))
        loss = F.cross_entropy(al, soft[idx].to(DEV)) + F.cross_entropy(gl, gy[idx].to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        tot, nb = tot + loss.item(), nb + 1
    a_pred, g_pred = predict(val_idx)
    print(f"epoch {epoch + 1}/{EPOCHS}  train loss {tot / nb:.3f}  val button acc {np.mean(a_pred == hard[val_idx]):.3f}  "
          f"val grid-cell acc {np.mean(g_pred == gy[val_idx].numpy()):.3f}  ({time.time() - t0:.0f}s)", flush=True)

def confusion(true, pred, n):
    return np.bincount(true * n + pred, minlength=n * n).reshape(n, n)

def report(name, cm, names):
    tp, sup = np.diag(cm), cm.sum(1)
    p, r = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(sup, 1)
    f = 2 * p * r / np.maximum(p + r, 1e-9)
    print(f"\n{name}: accuracy {tp.sum() / cm.sum():.3f}  (always-majority baseline {sup.max() / cm.sum():.3f})")
    print(f"  {'class':<10}{'precision':>10}{'recall':>8}{'F1':>7}{'support':>9}")
    for i, n in enumerate(names):
        print(f"  {n:<10}{p[i]:>10.2f}{r[i]:>8.2f}{f[i]:>7.2f}{sup[i]:>9d}")
    print("  confusion (rows = true, columns = predicted):")
    for i, n in enumerate(names):
        print(f"  {n:<10}" + " ".join(f"{v:6d}" for v in cm[i]))

A_NAMES = ["noop", "right", "right+A", "right+B", "r+A+B", "A", "left"]
G_NAMES = ["enemy", "empty", "solid", "mario"]
a_pred, g_pred = predict(val_idx)
cm_a = confusion(hard[val_idx], a_pred, 7)
cm_g = confusion(gy[val_idx].numpy().ravel(), g_pred.ravel(), 4)
report("BUTTON (teacher's most likely button)", cm_a, A_NAMES)
report("TILE GRID (all cells)", cm_g, G_NAMES)
near = (x[val_idx] >= 550) & (x[val_idx] <= 800)
print(f"\nbutton accuracy near the pipe wall (x 550-800): {np.mean(a_pred[near] == hard[val_idx][near]):.3f} on {near.sum()} samples")

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, cm, names, title in [(axs[0], cm_a, A_NAMES, "buttons"), (axs[1], cm_g, G_NAMES, "tile-grid cells")]:
    ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{title} (row-normalised)")
plt.tight_layout()
plt.savefig("results/cnn_diag.png", dpi=150)
