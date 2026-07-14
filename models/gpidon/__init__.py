"""GPiDoN -- Gaussian-source Physics-informed Deep Operator Network."""

from .agent import GPiDoNAgent, build_agent
from .model import GPiDoN

__all__ = ["GPiDoN", "GPiDoNAgent", "build_agent"]
