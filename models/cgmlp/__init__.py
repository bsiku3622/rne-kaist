"""CGMLP -- the gated dense stack with the process parameter taken away.

:mod:`models.gmlp` on ``(t, z, y, x) -> T`` instead of ``(P, t, z, y, x) -> T``.
The gate is built from where the beam *is*, which is a function of time alone, so
it crosses over from ``gmlp`` untouched: this model keeps the whole Gaussian prior
and loses only the power that sets its amplitude. Against ``cmlp`` it isolates the
gate exactly as ``gmlp`` does against ``mlp``.
"""

from .agent import CGMLPAgent, build_agent
from .model import ControlGatedMLP

__all__ = ["ControlGatedMLP", "CGMLPAgent", "build_agent"]
