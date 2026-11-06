import numpy as np
from gymnasium import spaces
from mario_env import MarioEnv, STACK

ROWS, BEHIND, AHEAD = 13, 2, 8
COLS = BEHIND + AHEAD + 1

class MarioGridEnv(MarioEnv):
    def __init__(self, level="SuperMarioBros-1-1-v0"):
        super().__init__(level)
        self.observation_space = spaces.Box(-1.0, 2.0, (ROWS * COLS * STACK,), np.float32)

    def _gray84(self, frame):
        # override: ignore the pixels, read the NES RAM and build a tile grid instead
        ram = self.inner.unwrapped.ram
        tiles = ram[0x500:0x6A0].reshape(2, 13, 16)       # 2 pages x 13 rows x 16 columns
        mario_x = int(ram[0x6D]) * 256 + int(ram[0x86])    # page * 256 + x inside the page
        base = (mario_x + 8) // 16                          # Mario's tile column in the level
        grid = np.zeros((ROWS, COLS), np.float32)
        for j in range(COLS):
            col = base - BEHIND + j
            if col < 0:
                grid[:, j] = 1.0                            # before the level start = wall
            else:
                grid[:, j] = tiles[(col // 16) % 2, :, col % 16] != 0
        for i in range(5):                                  # 5 enemy slots
            if ram[0x0F + i] == 0:                          # slot not active
                continue
            ex = int(ram[0x6E + i]) * 256 + int(ram[0x87 + i])
            c = (ex + 8) // 16 - base + BEHIND
            r = (int(ram[0xCF + i]) - 24) // 16
            if 0 <= c < COLS and 0 <= r < ROWS:
                grid[r, c] = -1.0
        r = (int(ram[0x3B8]) - 16) // 16                    # Mario's row (top of his sprite)
        if 0 <= r < ROWS:
            grid[r, BEHIND] = 2.0
        return grid

    def _stacked(self):
        # flat vector for MlpPolicy (also silences SB3's "this looks like an image" warnings)
        return np.stack(self.frames, axis=-1).ravel()
