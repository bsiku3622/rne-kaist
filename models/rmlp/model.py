"""(P, t, z, y, x) -> T. The other way to spend the same three hidden layers.

``models/fmlp`` hands its network a 2-dimensional input and asks for 73,402
numbers at once, because the Fourier basis is already carrying the space. Here the
network takes all five coordinates and returns a single temperature, so it has to
carry the space itself. The trade is stark:

    fmlp   2 -> 128 -> 128 -> 128 -> 73402     9.5M params,  217 samples
    rmlp   5 -> 128 -> 128 -> 128 ->     1      34k params,  276M samples

The network is asked for ``dT = T - T_amb``, scaled to O(1). Predicting ``T``
directly would spend its whole output range on the 298 K offset every point in the
plate shares, so the ambient is subtracted going in and added back coming out --
which is what :meth:`temperature` is for.
"""

from __future__ import annotations

import torch
from torch import nn

T_AMB = 298.0


class RealMLP(nn.Module):
    def __init__(self, width: int = 128, depth: int = 3, scale: float = 1.0):
        super().__init__()
        layers, d = [], 5  # (P, t, z, y, x)
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, x):
        """Normalised dT, which is what the loss is taken against."""
        return self.net(x).squeeze(-1)

    def dT(self, x):
        return self.forward(x) * self.scale

    def temperature(self, x):
        return self.dT(x) + T_AMB
