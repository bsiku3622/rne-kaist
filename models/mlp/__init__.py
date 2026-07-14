"""A dense-stack regression baseline: ``(P, t, z, y, x) -> T``."""

from .agent import MLPAgent, build_agent
from .model import SimpleMLP

__all__ = ["SimpleMLP", "MLPAgent", "build_agent"]
