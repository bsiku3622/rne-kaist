"""The MLP baseline behind the shared :class:`~agent.BaseAgent` inference contract.

The only work ``predict_at`` does beyond a forward pass is reordering the columns:
the contract is ``(x, y, z, t, P)`` and the network takes ``(P, t, z, y, x)``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from share.agent import ArrayLike, BaseAgent
from share.corpus import DEFAULT_FIELD_SHAPE
from share.checkpoints import load_checkpoint, resolve_device

from .dataset import from_contract
from .model import SimpleMLP


class MLPAgent(BaseAgent):
    """Wraps a trained :class:`~models.mlp.model.SimpleMLP` for inference."""

    def __init__(
        self,
        model: SimpleMLP,
        bounds: Tensor,
        shape: tuple[int, int, int] = DEFAULT_FIELD_SHAPE,
        device: torch.device | str = "cpu",
        chunk: int = 65536,
    ) -> None:
        dtype = next(model.parameters()).dtype
        super().__init__(bounds, shape=shape, device=device, dtype=dtype, chunk=chunk)
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict_at(self, inputs: ArrayLike) -> Tensor:
        """``[B, 5]`` of ``(x, y, z, t, P)`` to ``[B, 1]`` of Kelvin."""
        inputs = self._as_tensor(inputs, columns=5, name="predict_at")

        outputs = []
        for start in range(0, inputs.size(0), self.chunk):
            block = inputs[start : start + self.chunk]
            outputs.append(self.model(from_contract(block)))
        return torch.cat(outputs)


def build_agent(
    checkpoint: Path,
    shape: tuple[int, int, int] = DEFAULT_FIELD_SHAPE,
    device: torch.device | str | None = None,
) -> MLPAgent:
    """Rebuild the network from the architecture stored alongside the weights."""
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    payload = load_checkpoint(checkpoint, map_location=device)

    if "bounds" not in payload:
        raise KeyError(f"{checkpoint} predates the `bounds` key; retrain or add it by hand")

    model = SimpleMLP(**payload["architecture"])
    model.load_state_dict(payload["model"])  # normalisation buffers ride along
    return MLPAgent(model, payload["bounds"], shape=shape, device=device)
