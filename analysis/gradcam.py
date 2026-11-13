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
net = Net()
net.load_state_dict(torch.load(PATH, map_location="cpu"))
net.eval()

class RecEnv(MarioEnv):
    def __init__(self):
        super().__init__()
        self.raw = []
    def _inner_step(self, action):
        obs, done, info = super()._inner_step(action)
        self.raw.append(obs)                            # keep the full-resolution frames for the overlay
        return obs, done, info

def prep(obs):
    return torch.from_numpy(obs).permute(2, 0, 1)[None].float() / 255.0

def grad_cam(x, target):
    feats = net.cnn[:6](x)                              # last convolution block: (1, 64, 7, 7)
    feats.retain_grad()
    logit = net.act(net.cnn[6:](feats))[0, target]      # score of the button that was pressed
    net.zero_grad()
    logit.backward()
    w = feats.grad.mean(dim=(2, 3), keepdim=True)       # how much each channel matters for that button
    cam = torch.relu((w * feats).sum(1))[0].detach().numpy()
    return cam / (cam.max() + 1e-8)

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

picks = []
for xt in X_TARGETS:
    i = next((k for k, e in enumerate(log) if e[0] >= xt), None)
    if i is not None:
        picks.append(i)
print("looking at x positions:", [log[i][0] for i in picks])

fig, axs = plt.subplots(2, len(picks), figsize=(3.3 * len(picks), 6.4))
for c, i in enumerate(picks):
    xpos, ob, raw, p, a = log[i]
    cam = grad_cam(prep(ob), a)
    heat = np.clip(cv2.resize(cam, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_CUBIC), 0, 1)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]   # BGR -> RGB
    axs[0, c].imshow((0.55 * raw + 0.45 * color).astype(np.uint8)); axs[0, c].axis("off")
    axs[0, c].set_title(f"x = {xpos}\npressed {A_NAMES[a]} (p = {p[a]:.2f})", fontsize=9)
    axs[1, c].barh(range(7), p, color=["#e6550d" if k == a else "#bbbbbb" for k in range(7)])
    axs[1, c].set_yticks(range(7)); axs[1, c].set_yticklabels(A_NAMES, fontsize=8)
    axs[1, c].invert_yaxis(); axs[1, c].set_xlim(0, 1); axs[1, c].set_xlabel("probability", fontsize=8)
fig.suptitle("Grad-CAM of the CNN student: where it looks when it picks a button (red = most influential)", fontsize=10)
fig.tight_layout(); fig.savefig("docs/figures/fig_gradcam.png", dpi=150); plt.close(fig)

w = net.cnn[0].weight.detach().numpy()                  # (32 filters, 4 stacked frames, 8, 8)
fig, axs = plt.subplots(4, 8, figsize=(10, 5.4))
for k, ax in enumerate(axs.ravel()):
    f = w[k, 3]                                          # the newest of the 4 frames
    ax.imshow(f, cmap="gray", vmin=-abs(f).max(), vmax=abs(f).max()); ax.axis("off")
fig.suptitle("First convolution layer: the 32 learned 8x8 filters (newest frame)", fontsize=10)
fig.tight_layout(); fig.savefig("docs/figures/fig_filters.png", dpi=150); plt.close(fig)
print("saved docs/figures/fig_gradcam.png and docs/figures/fig_filters.png")
