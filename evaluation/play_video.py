import sys, time
import numpy as np
import torch
import cv2
from mario_env import MarioEnv
from student import Net, A_NAMES

PATH, MODE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]       # model, "greedy" or "sample", output .mp4
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
torch.manual_seed(SEED)
SCALE, PANEL = 2, 300

class RecEnv(MarioEnv):
    def __init__(self):
        super().__init__()
        self.raw = []
    def _inner_step(self, action):
        obs, done, info = super()._inner_step(action)
        self.raw.append(obs)                                    # keep every game frame (60 per second)
        return obs, done, info

def draw(frame, probs, a, x, step, ms):
    img = cv2.resize(np.ascontiguousarray(frame[:, :, ::-1]), None, fx=SCALE, fy=SCALE,
                     interpolation=cv2.INTER_NEAREST)           # RGB -> BGR, enlarged
    panel = np.full((img.shape[0], PANEL, 3), 30, np.uint8)
    cv2.putText(panel, "CNN student (pixels only)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    for i, name in enumerate(A_NAMES):
        y = 60 + i * 34
        color = (80, 200, 80) if i == a else (150, 150, 150)
        cv2.rectangle(panel, (110, y), (110 + int(170 * probs[i]), y + 18), color, -1)
        cv2.putText(panel, name, (8, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    for j, line in enumerate([f"x = {x}", f"decision {step}", f"latency {ms:.2f} ms (cpu)"]):
        cv2.putText(panel, line, (10, 340 + j * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack([img, panel])

net = Net()
net.load_state_dict(torch.load(PATH, map_location="cpu"))
net.eval()
env = RecEnv()
obs, _ = env.reset()
writer, step = None, 0
while True:
    n0 = len(env.raw)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, _ = net(torch.from_numpy(obs).permute(2, 0, 1)[None].float() / 255.0)
        p = torch.softmax(logits[0], 0)
        a = int(torch.multinomial(p, 1)) if MODE == "sample" else int(p.argmax())
    ms = (time.perf_counter() - t0) * 1000
    obs, r, done, trunc, info = env.step(a)
    for fr in env.raw[n0:]:                                     # the 4 frames this decision lasted
        img = draw(fr, p.numpy(), a, env.last_x, step, ms)
        if writer is None:
            writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 60, (img.shape[1], img.shape[0]))
        writer.write(img)
    step += 1
    if done or trunc:
        break
writer.release()
print(f"{OUT}: {step} decisions, max_x {info['max_x']}, flag {info['flag_get']}")
