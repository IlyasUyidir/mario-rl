import sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from student import Net, A_NAMES, G_NAMES

DATA, NAME = sys.argv[1], sys.argv[2]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 15
AUX = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0      # weight of the tile-grid task (0 = off)
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
BATCH, DEV = 128, "mps"
torch.manual_seed(SEED)

d = np.load(DATA)
pix, probs = d["pix"], d["probs"].astype(np.float32)
ep, t = d["ep"], d["t"]
N = len(pix)
eps = np.unique(ep)
np.random.default_rng(0).shuffle(eps)                        # fixed split: same validation episodes for every variant
is_val = np.isin(ep, eps[: max(1, len(eps) // 5)])
train_idx, val_idx = np.where(~is_val)[0], np.where(is_val)[0]
print(f"{N} samples: {len(train_idx)} train / {len(val_idx)} validation ({len(np.unique(ep[is_val]))} validation episodes)")

k = np.minimum(np.arange(3, -1, -1)[None, :], t[:, None])    # rebuild stacks, clamped at the episode start
stack_idx = np.arange(N)[:, None] - k
pix_t = torch.from_numpy(pix)
gy = torch.from_numpy((d["grid"].astype(np.int64) + 1).reshape(N, -1))   # 0 enemy, 1 empty, 2 solid, 3 mario
soft, hard = torch.from_numpy(probs), probs.argmax(1)

freq = np.bincount(gy[torch.from_numpy(train_idx)].numpy().ravel(), minlength=4)
w = np.sqrt(freq.max() / np.maximum(freq, 1))                # rare classes count more (sqrt of inverse frequency)
print("grid class weights (enemy, empty, solid, mario):", w.round(1))
gw = torch.tensor(w, dtype=torch.float32, device=DEV)

def batch(idx):
    return pix_t[torch.from_numpy(stack_idx[idx])].to(DEV).float() / 255.0

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
    perm = np.random.default_rng(SEED * 1000 + epoch).permutation(train_idx)
    for s in range(0, len(perm), BATCH):
        idx = perm[s:s + BATCH]
        al, gl = net(batch(idx))
        loss = F.cross_entropy(al, soft[idx].to(DEV))
        if AUX > 0:
            loss = loss + AUX * F.cross_entropy(gl, gy[idx].to(DEV), weight=gw)
        opt.zero_grad(); loss.backward(); opt.step()
        tot, nb = tot + loss.item(), nb + 1
    a_pred, g_pred = predict(val_idx)
    print(f"epoch {epoch + 1}/{EPOCHS}  loss {tot / nb:.3f}  val button acc {np.mean(a_pred == hard[val_idx]):.3f}  "
          f"val grid acc {np.mean(g_pred == gy[val_idx].numpy()):.3f}  ({time.time() - t0:.0f}s)", flush=True)

torch.save(net.state_dict(), f"models/{NAME}.pt")             # the last epoch (no best-on-validation picking)

def confusion(true, pred, n):
    return np.bincount(true * n + pred, minlength=n * n).reshape(n, n)

def prf(cm):
    tp, sup = np.diag(cm), cm.sum(1)
    p, r = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(sup, 1)
    return p, r, 2 * p * r / np.maximum(p + r, 1e-9), sup

def report(title, cm, names):
    p, r, f, sup = prf(cm)
    print(f"\n{title}: accuracy {np.diag(cm).sum() / cm.sum():.3f}  (majority baseline {sup.max() / cm.sum():.3f})")
    print(f"  {'class':<10}{'precision':>10}{'recall':>8}{'F1':>7}{'support':>9}")
    for i, n in enumerate(names):
        if sup[i] > 0:
            print(f"  {n:<10}{p[i]:>10.2f}{r[i]:>8.2f}{f[i]:>7.2f}{sup[i]:>9d}")

a_pred, g_pred = predict(val_idx)
cm_a = confusion(hard[val_idx], a_pred, 7)
cm_g = confusion(gy[val_idx].numpy().ravel(), g_pred.ravel(), 4)
report("BUTTONS (teacher's most likely button)", cm_a, A_NAMES)
report("TILE GRID (cells)", cm_g, G_NAMES)
res = dict(name=NAME, data=DATA, epochs=EPOCHS, aux=AUX, seed=SEED, n_train=int(len(train_idx)), n_val=int(len(val_idx)),
           button_acc=float(np.diag(cm_a).sum() / cm_a.sum()), grid_acc=float(np.diag(cm_g).sum() / cm_g.sum()),
           confusion_buttons=cm_a.tolist(), confusion_grid=cm_g.tolist())
json.dump(res, open(f"results/student_{NAME}.json", "w"))
