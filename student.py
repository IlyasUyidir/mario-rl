import torch.nn as nn

A_NAMES = ["noop", "right", "right+A", "right+B", "r+A+B", "A", "left"]
G_NAMES = ["enemy", "empty", "solid", "mario"]

class Net(nn.Module):
    """Same convolutional trunk as the default pixel network, with two heads."""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(), nn.Linear(3136, 512), nn.ReLU())
        self.act = nn.Linear(512, 7)                 # which button to press
        self.grid = nn.Linear(512, 4 * 13 * 11)      # auxiliary task: which tile is where

    def forward(self, x):
        h = self.cnn(x)
        return self.act(h), self.grid(h).view(-1, 4, 13 * 11)
