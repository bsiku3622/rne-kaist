"""PiDoN -- Physics-informed Deep Operator Network.

The predecessor of :mod:`models.gpidon`, kept so the checkpoints trained before
the Gaussian gate was added to the operator output can still be loaded, plotted
and compared. Here the only Gaussian in the problem is the laser source term of
the top boundary condition; the network output itself is unconstrained.
"""

from .agent import PiDoNAgent, build_agent
from .model import PiDoN

__all__ = ["PiDoN", "PiDoNAgent", "build_agent"]
