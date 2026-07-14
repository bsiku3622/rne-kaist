"""CPiMLP -- the physics-informed dense stack with the process parameter taken away.

:mod:`models.pimlp` on ``(x, y, z, t) -> T`` instead of ``(x, y, z, t; P) -> T``.
The heat equation, the boundary conditions and the initial condition are still
enforced alongside the labels; the network simply cannot see which power it is
being asked about. Against ``pimlp`` it isolates the process parameter, and
against ``cmlp`` it isolates the physics -- the same two comparisons ``pimlp``
supports, run one input short.
"""

from .agent import CPiMLPAgent, build_agent
from .model import ControlPhysicsMLP

__all__ = ["ControlPhysicsMLP", "CPiMLPAgent", "build_agent"]
