"""CPiMLP behind the shared :class:`~agent.BaseAgent` inference contract.

``predict_at`` still takes the ``(x, y, z, t, P)`` of the contract -- every
plotting and benchmarking script is written against it -- but the ``P`` column is
sliced off and thrown away, and the remaining four columns are already the
``(x, y, z, t)`` the network takes. Asking this agent for 100 W and for 250 W at
the same point therefore returns the same number, by construction.

Beyond that it is a direct forward pass: the network is a pointwise map, so no
interpolation is involved at any query point and the volumetric ``predict_of``
inherited from the base class is the same map evaluated on a regular grid.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from share.agent import ArrayLike, BaseAgent
from share.corpus import DEFAULT_FIELD_SHAPE
from share.checkpoints import load_checkpoint, resolve_device

from .model import ControlPhysicsMLP


class CPiMLPAgent(BaseAgent):
    """Wraps a trained :class:`~models.cpimlp.model.ControlPhysicsMLP` for inference."""

    def __init__(
        self,
        model: ControlPhysicsMLP,
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
        """``[B, 5]`` of ``(x, y, z, t, P)`` to ``[B, 1]`` of Kelvin; ``P`` is ignored."""
        inputs = self._as_tensor(inputs, columns=5, name="predict_at")

        outputs = []
        for start in range(0, inputs.size(0), self.chunk):
            block = inputs[start : start + self.chunk]
            outputs.append(self.model(block[:, 0:4]))
        return torch.cat(outputs)


def build_agent(
    checkpoint: Path,
    shape: tuple[int, int, int] = DEFAULT_FIELD_SHAPE,
    device: torch.device | str | None = None,
) -> CPiMLPAgent:
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

    model = ControlPhysicsMLP(**payload["architecture"])
    model.load_state_dict(payload["model"])  # normalisation buffers ride along
    return CPiMLPAgent(model, payload["bounds"], shape=shape, device=device)
