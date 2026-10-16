"""CompactEuroSAT from simple_model; historical notebook remains unchanged.

Registering this model uses the shared CLI's saved splits, FP32 training,
augmentation and minimum-validation-loss selection rather than the notebook's
different optimization protocol. No historical result is a controlled CLI run.
"""

import torch
from torch import nn


class DSBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.skip = stride == 1 and cin == cout
        self.layers = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
            nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.layers(x)
        if self.skip:
            y = y + x
        return self.relu(y)


class CompactEuroSAT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            DSBlock(16, 32, 2), DSBlock(32, 48, 2),
            DSBlock(48, 48), DSBlock(48, 80, 2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.10), nn.Linear(80, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
