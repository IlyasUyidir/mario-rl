import collections
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

SKIP, STACK = 4, 4
STUCK_LIMIT = 250        # agent steps without a new max x -> cut the episode
STEP_PENALTY = 0.1       # small cost per step, so dawdling isn't free
DEATH_PENALTY = 15.0
FLAG_BONUS = 50.0

class MarioEnv(gym.Env):
    def __init__(self, level="SuperMarioBros-1-1-v0"):
        super().__init__()
        self.inner = JoypadSpace(make(level), SIMPLE_MOVEMENT)
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(0, 255, (84, 84, STACK), np.uint8)
        self.frames = collections.deque(maxlen=STACK)

    def _inner_step(self, action):
        res = self.inner.step(action)          # handle the 4- and 5-value step APIs
        if len(res) == 5:
            obs, _, term, trunc, info = res
            return obs.copy(), term or trunc, info   # copy: the emulator reuses its buffer
        obs, _, done, info = res
        return obs.copy(), done, info

    @staticmethod
    def _encode(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def _stacked(self):
        return np.stack(self.frames, axis=-1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.inner.reset()
        obs, _, info = self._inner_step(0)     # NOOP: gives us the starting info
        first = self._encode(obs)
        self.frames.clear()
        for _ in range(STACK):
            self.frames.append(first)          # at reset, all 4 slots hold the same frame
        self.last_x = self.best_x = info["x_pos"]
        self.stuck = 0
        return self._stacked(), {}

    def step(self, action):
        recent, done = [], False
        for _ in range(SKIP):                  # hold the button for SKIP frames
            obs, done, info = self._inner_step(action)
            recent = (recent + [obs])[-2:]
            if done:
                break
        frame = np.max(np.stack(recent), axis=0)   # max of last 2 frames kills flicker
        self.frames.append(self._encode(frame))

        x = info["x_pos"]
        dx = float(np.clip(x - self.last_x, -15, 15))
        self.last_x = x
        flag = bool(info.get("flag_get", False))
        reward = dx - STEP_PENALTY
        if done and not flag:
            reward -= DEATH_PENALTY
        if flag:
            reward += FLAG_BONUS

        if x > self.best_x:
            self.best_x, self.stuck = x, 0
        else:
            self.stuck += 1
        truncated = (not done) and self.stuck >= STUCK_LIMIT
        return self._stacked(), reward, done, truncated, {"max_x": self.best_x, "flag_get": flag}

    def close(self):
        self.inner.close()
