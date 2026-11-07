import numpy as np
from gymnasium import spaces
from mario_grid_env import MarioGridEnv, ROWS, COLS, STACK

def signed(b):                      # RAM bytes are 0..255; speeds are signed (e.g. 255 = -1)
    b = int(b)
    return b - 256 if b > 127 else b

class MarioKinEnv(MarioGridEnv):
    def __init__(self, level="SuperMarioBros-1-1-v0"):
        super().__init__(level)
        self.observation_space = spaces.Box(-2.0, 2.0, (ROWS * COLS * STACK + 4,), np.float32)

    def _stacked(self):
        ram = self.inner.unwrapped.ram
        mario_x = int(ram[0x6D]) * 256 + int(ram[0x86])
        extras = np.array([
            ((mario_x + 8) % 16) / 16.0,                  # x position inside the tile
            ((int(ram[0x3B8]) - 16) % 16) / 16.0,         # y position inside the tile
            np.clip(signed(ram[0x57]) / 40.0, -2, 2),     # horizontal speed
            np.clip(signed(ram[0x9F]) / 5.0, -2, 2),      # vertical speed (negative = going up)
        ], dtype=np.float32)
        return np.concatenate([super()._stacked(), extras])
