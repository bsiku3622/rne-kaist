"""GPiDoN behind the shared :class:`~agent.BaseAgent` inference contract.

``predict_at`` is a direct forward pass -- the network is already a pointwise
map ``(x, y, z, t; P) -> T``, which is what makes the operator formulation
attractive here: no interpolation is involved at any query point, and the
volumetric ``predict_of`` inherited from the base class is just the same map
evaluated on a regular grid.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from share.agent import ArrayLike, BaseAgent
from share.corpus import DEFAULT_FIELD_SHAPE
from share.checkpoints import load_checkpoint, resolve_device

from .model import GPiDoN


class GPiDoNAgent(BaseAgent):
    """Wraps a trained :class:`~models.gpidon.model.GPiDoN` for inference."""

    def __init__(
        self,
        model: GPiDoN,
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
            outputs.append(self.model(block[:, 4:5], block[:, 0:4]))
        return torch.cat(outputs)


def build_agent(
    checkpoint: Path,
    shape: tuple[int, int, int] = DEFAULT_FIELD_SHAPE,
    device: torch.device | str | None = None,
) -> GPiDoNAgent:
    """Rebuild the network from the architecture stored alongside the weights.

    ``bounds`` also rides along in the checkpoint, so the reconstruction grid of
    :meth:`~agent.BaseAgent.predict_of` matches the box the model was trained on
    without re-reading the dataset.
    """
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    payload = load_checkpoint(checkpoint, map_location=device)

    if "bounds" not in payload:
        raise KeyError(
            f"{checkpoint} predates the `bounds` key; retrain or add it by hand"
        )

    model = GPiDoN(**payload["architecture"])
    model.load_state_dict(payload["model"])  # normalisation buffers ride along
    return GPiDoNAgent(model, payload["bounds"], shape=shape, device=device)
