"""GDoN -- Gaussian-gated Deep Operator Network, fitted to data alone.

:mod:`models.gpidon` without the physics-informed residuals: same DeepONet, same
Gaussian gate on the inner product, but the objective is a plain scaled MSE
against the labelled temperatures. It isolates what the PINN terms contribute.
"""

from .agent import GDoNAgent, build_agent
from .model import GDoN

__all__ = ["GDoN", "GDoNAgent", "build_agent"]
