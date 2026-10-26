"""The accepted mid20 32x32 RGB generator.

Layer shapes and parameter names match the accepted checkpoint and the
onchain engine.

mid20 is the deployed/target variant:
    z (32,) -> FC (32 -> c0*16) -> reshape (c0, 4, 4)
    -> TConv (c0 -> c1, 4, 2, 1) -> (c1, 8, 8)
    -> TConv (c1 -> c2, 4, 2, 1) -> (c2, 16, 16)
    -> TConv (c2 -> c3, 4, 2, 1) -> (c3, 32, 32)
    -> Conv1x1 (c3 -> 3)          -> (3, 32, 32), then tanh

Biases are separate nn.Parameters (not layer bias=) so they quantize
independently; the integer engine carries them in Q(ACT_SHIFT).
NO batchnorm, NO dropout, NO skip connections (the engine cannot express them).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


Z_DIM = 32
C0, C1, C2, C3 = 19, 19, 10, 5


class Generator32(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(Z_DIM, C0 * 4 * 4, bias=False)
        self.b_fc = nn.Parameter(torch.zeros(C0 * 4 * 4))

        self.t0 = nn.ConvTranspose2d(C0, C1, 4, 2, 1, bias=False)
        self.b0 = nn.Parameter(torch.zeros(C1))

        self.t1 = nn.ConvTranspose2d(C1, C2, 4, 2, 1, bias=False)
        self.b1 = nn.Parameter(torch.zeros(C2))

        self.t2 = nn.ConvTranspose2d(C2, C3, 4, 2, 1, bias=False)
        self.b2 = nn.Parameter(torch.zeros(C3))

        self.out = nn.Conv2d(C3, 3, 1, bias=False)
        self.b_out = nn.Parameter(torch.zeros(3))

    def features(self, z: torch.Tensor) -> torch.Tensor:
        """Inference without the final tanh. Mirrors the onchain engine path."""
        h = F.relu(self.fc(z) + self.b_fc).view(-1, C0, 4, 4)
        h = F.relu(self.t0(h) + self.b0.view(1, -1, 1, 1))
        h = F.relu(self.t1(h) + self.b1.view(1, -1, 1, 1))
        h = F.relu(self.t2(h) + self.b2.view(1, -1, 1, 1))
        return self.out(h) + self.b_out.view(1, -1, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Float reference: features -> tanh; the integer engine uses a LUT."""
        return torch.tanh(self.features(z))
