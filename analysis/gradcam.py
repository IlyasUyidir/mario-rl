import sys
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mario_env import MarioEnv
from student import Net, A_NAMES

PATH = sys.argv[1] if len(sys.argv) > 1 else "models/dagger_v1_r3.pt"
X_TARGETS = [330, 560, 1100, 1380, 2350]               # where to look (world x); edit to taste
HUD_ROWS = 11                                           # the top 11 of 84 rows hold score, coins and timer
net = Net()
net.load_state_dict(torch.load(PATH, map_location="cpu"))
net.eval()

class RecEnv(MarioEnv):
    def __init__(self):
        super().__init__()
        self.raw = []
    def _inner_step(self, action):
        obs, done, info = super()._inner_step(action)
        self.raw.append(obs)
        return obs, done, info

def prep(obs):
    return torch.from_numpy(obs).permute(2, 0, 1)[None].float() / 255.0

def explain(obs):
    """Why this button rather than the runner-up? Returns (button, runner-up, signed Grad-CAM 7x7, saliency 84x84)."""
    x = prep(obs).requires_grad_(True)
    feats = net.cnn[:6](x)                              # last convolution block: (1, 64, 7, 7)
    feats.retain_grad()
    logits = net.act(net.cnn[6:](feats))[0]
    order = logits.argsort(descending=True)
    a, b = int(order[0]), int(order[1])
    net.zero_grad()
    (logits[a] - logits[b]).backward()                  # the margin that decides the choice
    w = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = (w * feats).sum(1)[0].detach().numpy()        # signed: red supports the choice, blue opposes it
    sal = x.grad[0].abs().max(0)[0].numpy()             # which input pixels the margin is sensitive to
    return a, b, cam, sal

# one greedy episode, remembering what the CNN saw at every decision
env = RecEnv()
obs, _ = env.reset()
log = []
while True:
    with torch.no_grad():
        p = torch.softmax(net(prep(obs))[0][0], 0).numpy()
    a = int(p.argmax())
    log.append((env.last_x, obs.copy(), env.raw[-1].copy(), p, a))
    obs, r, done, trunc, info = env.step(a)
    if done or trunc:
        break
print(f"greedy run: {len(log)} decisions, max_x {info['max_x']}, flag {info['flag_get']}")

exp = [explain(e[1]) for e in log]
hud = np.array([s[:HUD_ROWS].sum() / (s.sum() + 1e-12) for *_, s in exp])
print(f"share of input-gradient saliency on the HUD rows: mean {100 * hud.mean():.1f}%, median {100 * np.median(hud):.1f}% "
      f"(the HUD is {100 * HUD_ROWS / 84:.1f}% of the image rows)")

picks = [next((k for k, e in enumerate(log) if e[0] >= xt), None) for xt in X_TARGETS]
picks = [i for i in picks if i is not None]
print("looking at x positions:", [log[i][0] for i in picks])
fig, axs = plt.subplots(3, len(picks), figsize=(3.3 * len(picks), 9.2))
for c, i in enumerate(picks):
    xpos, ob, raw, p, a = log[i]
    a_, b_, cam, sal = exp[i]
    cam_up = np.clip(cv2.resize(cam / (np.abs(cam).max() + 1e-8), (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_CUBIC), -1, 1)
    rgba = plt.cm.bwr((cam_up + 1) / 2)
    rgba[..., 3] = np.abs(cam_up) ** 0.7 * 0.8
    axs[0, c].imshow(raw); axs[0, c].imshow(rgba); axs[0, c].axis("off")
    axs[0, c].set_title(f"x = {xpos}: {A_NAMES[a_]} over {A_NAMES[b_]}\n(p = {p[a_]:.2f}; HUD share {100 * hud[i]:.0f}%)", fontsize=9)
    s_up = cv2.resize(cv2.GaussianBlur(sal, (5, 5), 0), (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_CUBIC)
    s_up = np.clip(s_up / (np.percentile(s_up, 99.5) + 1e-8), 0, 1)
    rgba2 = plt.cm.hot(s_up)
    rgba2[..., 3] = s_up ** 0.7 * 0.9
    axs[1, c].imshow(raw); axs[1, c].imshow(rgba2); axs[1, c].axis("off")
    axs[2, c].barh(range(7), p, color=["#e6550d" if k == a else "#bbbbbb" for k in range(7)])
    axs[2, c].set_yticks(range(7)); axs[2, c].set_yticklabels(A_NAMES, fontsize=8)
    axs[2, c].invert_yaxis(); axs[2, c].set_xlim(0, 1); axs[2, c].set_xlabel("probability", fontsize=8)
fig.suptitle("Top: Grad-CAM of 'chosen button minus runner-up' (red supports, blue opposes). Middle: input-gradient saliency. "
             "Bottom: button probabilities", fontsize=10)
fig.tight_layout(); fig.savefig("docs/figures/fig_gradcam.png", dpi=150); plt.close(fig)

w = net.cnn[0].weight.detach().numpy()                  # (32 filters, 4 stacked frames, 8, 8)
fig, axs = plt.subplots(4, 8, figsize=(10, 5.4))
for k, ax in enumerate(axs.ravel()):
    f = w[k, 3]
    ax.imshow(f, cmap="gray", vmin=-abs(f).max(), vmax=abs(f).max()); ax.axis("off")
fig.suptitle("First convolution layer: the 32 learned 8x8 filters (newest frame)", fontsize=10)
fig.tight_layout(); fig.savefig("docs/figures/fig_filters.png", dpi=150); plt.close(fig)
print("saved docs/figures/fig_gradcam.png and docs/figures/fig_filters.png")
