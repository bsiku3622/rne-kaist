"""PiMLP -- the dense-stack baseline, trained against the physics.

:mod:`models.mlp` fitted with the PINN objective of :mod:`models.pidon`: the same
five inputs and the same dense stack, but the heat equation, the boundary
conditions and the initial condition are enforced alongside the labels. It is the
controlled comparison for both -- against ``mlp`` it isolates the physics, and
against ``pidon`` it isolates the operator architecture.
"""

from .agent import PiMLPAgent, build_agent
from .model import PhysicsMLP

__all__ = ["PhysicsMLP", "PiMLPAgent", "build_agent"]
