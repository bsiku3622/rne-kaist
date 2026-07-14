"""CMLP -- the dense-stack baseline with the process parameter taken away.

:mod:`models.mlp` on ``(t, z, y, x) -> T`` instead of ``(P, t, z, y, x) -> T``.
Same stack, same objective, same training loop, so what it isolates is what the
laser power is worth as an input: the corpus holds seven of them and this network
cannot tell them apart, so it can only learn the power-averaged field. It is the
floor ``mlp`` has to clear.
"""

from .agent import CMLPAgent, build_agent
from .model import ControlMLP

__all__ = ["ControlMLP", "CMLPAgent", "build_agent"]
