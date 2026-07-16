"""``rmlp`` behind the shared :class:`~share.agent.BaseAgent` inference contract.

``rmlp`` is pointwise -- one ``(x, y, z, t; P)`` to one ``T`` -- so it fits the
contract the ordinary way, implementing ``predict_at`` and getting the volume
derived. The only work beyond a forward pass is normalisation: the network was
trained on each coordinate divided by that axis's maximum, so the agent divides the
physical query by the same maxima before handing it over, and reorders the columns
from the contract's ``(x, y, z, t, P)`` to the network's ``(P, t, z, y, x)``.

Those maxima are not in the checkpoint -- ``rmlp`` predates the ``bounds`` key the
harness models carry -- so they are read back from the run the checkpoint names,
preferring the copy hard-linked into the archive entry beside it (so a plotted model
still never needs the original ``data/``) and falling back to ``run_dir``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from share.agent import ArrayLike, BaseAgent
from share.checkpoints import resolve_device
from share.corpus import DEFAULT_FIELD_SHAPE
from share.grid import load_run

from .model import T_AMB, RealMLP

MM = 1.0e-3


class RealMLPAgent(BaseAgent):
    """Wraps a trained :class:`~models.rmlp.model.RealMLP` for inference."""

    def __init__(
        self,
        model: RealMLP,
        bounds: Tensor,
        max_power: float,
        shape: tuple[int, int, int] = DEFAULT_FIELD_SHAPE,
        device: torch.device | str = "cpu",
        chunk: int = 65536,
    ) -> None:
        dtype = next(model.parameters()).dtype
        super().__init__(bounds, shape=shape, device=device, dtype=dtype, chunk=chunk)
        self.model = model.to(self.device).eval()
        self.max_power = float(max_power)

    @torch.no_grad()
    def predict_at(self, inputs: ArrayLike) -> Tensor:
        """``[B, 5]`` of ``(x, y, z, t, P)`` to ``[B, 1]`` of Kelvin."""
        inputs = self._as_tensor(inputs, columns=5, name="predict_at")
        upper = self.bounds[:, 1]  # (x, y, z, t) maxima, in SI -- the values it was normalised by

        outputs = []
        for start in range(0, inputs.size(0), self.chunk):
            block = inputs[start : start + self.chunk]
            x, y, z, t, power = (block[:, i] for i in range(5))
            query = torch.stack(
                (power / self.max_power, t / upper[3], z / upper[2], y / upper[1], x / upper[0]),
                dim=-1,
            )
            outputs.append(self.model.temperature(query).unsqueeze(-1))
        return torch.cat(outputs)


def _run_beside(checkpoint: Path, payload: dict):
    """The run this checkpoint was fitted to: the archived copy first, then ``run_dir``."""
    local = Path(checkpoint).parent / "data"
    if (local / "manifest.json").is_file() or any(local.glob("data_*W.npy")):
        return load_run(local)
    return load_run(Path(payload["run_dir"]))


def build_agent(
    checkpoint: Path,
    shape: tuple[int, int, int] = DEFAULT_FIELD_SHAPE,
    device: torch.device | str | None = None,
) -> RealMLPAgent:
    """Rebuild the network and read its normalisation back off the run it was fitted to."""
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    payload = torch.load(checkpoint, map_location=device, weights_only=False)

    model = RealMLP(**payload["architecture"])
    model.load_state_dict(payload["state"])

    run = _run_beside(checkpoint, payload)
    bounds = torch.tensor(
        [
            [0.0, float(run.x[-1]) * MM],
            [0.0, float(run.y[-1]) * MM],
            [0.0, float(run.z[-1]) * MM],
            [0.0, float(run.t[-1])],
        ]
    )
    return RealMLPAgent(model, bounds, float(run.powers.max()), shape=shape, device=device)
