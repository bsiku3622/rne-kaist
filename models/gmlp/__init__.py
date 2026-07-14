"""GMLP -- the dense-stack baseline, gated by a Gaussian on the moving beam.

:mod:`models.mlp` with the gate :mod:`models.gdon` applies to its operator: the
output of the dense stack is multiplied by ``G = g + p`` -- a unit-peak Gaussian
riding on the beam, lifted by a single learnable scalar -- before the bias is
added. Same inputs, same objective, same training loop, so what it isolates is
what the gate alone is worth; and ``p``, which the fit is free to raise, says how
much of that gate the data actually wanted.
"""

from .agent import GMLPAgent, build_agent
from .model import GatedMLP

__all__ = ["GatedMLP", "GMLPAgent", "build_agent"]
