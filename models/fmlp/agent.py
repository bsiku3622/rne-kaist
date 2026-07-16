"""``fmlp`` behind the shared contract, with the two questions in the opposite order.

Every other model is pointwise, so :class:`~share.agent.BaseAgent` makes ``predict_at``
the primitive and derives the volume from it. ``fmlp`` runs the other way, exactly as
``share/agent.py`` describes: its native operation is the whole volume -- ``(P, t)`` in,
one inverse transform out -- so :meth:`predict_of` is overridden to do that directly, and
:meth:`predict_at` is the derived one, reading the reconstructed volume at the query nodes.

The transform that turns coefficients back into Kelvin lives in :mod:`share.spectral`, not
in the network, so the agent carries the :class:`~share.spectral.SpectralDataset` the model
was fitted to and reconstructs through it -- the same path ``visualize.render`` takes, so the
field an agent returns is the field the figures were drawn from. That box is derived, not
copied into the archive, so it is read from the run the checkpoint names, which must still
hold its ``spectral_fft2`` file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from share import spectral
from share.agent import ArrayLike, BaseAgent
from share.checkpoints import resolve_device
from share.corpus import DEFAULT_FIELD_SHAPE
from share.grid import load_run

from .model import FourierMLP

AMBIENT = 298.0
MM = 1.0e-3


def _nearest(axis: Tensor, values: Tensor) -> Tensor:
    """Index of the node in sorted ``axis`` closest to each of ``values``."""
    upper = torch.searchsorted(axis, values.contiguous()).clamp(1, axis.numel() - 1)
    take_lower = (values - axis[upper - 1]) <= (axis[upper] - values)
    return torch.where(take_lower, upper - 1, upper)


class FourierMLPAgent(BaseAgent):
    """Wraps a trained :class:`~models.fmlp.model.FourierMLP` for inference."""

    def __init__(
        self, model: FourierMLP, ds: spectral.SpectralDataset, bounds: Tensor, *,
        derotate: bool, vel: float, n_coef: int, ramp_shape, max_power: float,
        max_time: float, device: torch.device | str = "cpu",
    ) -> None:
        nx, ny, nz = (int(v) for v in ds.grid)
        super().__init__(bounds, shape=(nz, ny, nx), device=device, dtype=torch.float32)
        self.model = model.to(self.device).eval()
        self.ds = ds
        self.derotate, self.vel = derotate, float(vel)
        self.n_coef, self.ramp_shape = n_coef, ramp_shape
        self.max_power, self.max_time = float(max_power), float(max_time)

    def _volume(self, time: float, power: float) -> np.ndarray:
        """``dT`` on the grid for one ``(t, P)``, ``(nx, ny, nz)`` -- the native op."""
        query = np.array([[power / self.max_power, time / self.max_time]], dtype=np.float32)
        with torch.no_grad():
            flat = self.model.denormalise(
                self.model(torch.tensor(query, device=self.device)).cpu().numpy()
            ).reshape(1, -1)
        packed = flat[:, : self.n_coef].reshape(1, *self.ds.shape, 2)
        coef = packed[..., 0] + 1j * packed[..., 1]  # (1, mx, my, nz)
        if self.derotate:
            coef = coef / self.ds.spin(np.array([time]), self.vel)[:, :, None, None]
        ramp = (
            flat[:, self.n_coef :].reshape(1, *self.ramp_shape)
            if self.ramp_shape is not None else None
        )
        return self.ds.reconstruct(coef, ramp)[0]  # (nx, ny, nz)

    @torch.no_grad()
    def predict_of(self, inputs: ArrayLike) -> Tensor:
        """``[B, 2]`` of ``(t, P)`` to ``[B, 1, D, H, W]`` of Kelvin -- one iFFT per row."""
        inputs = self._as_tensor(inputs, columns=2, name="predict_of")
        volumes = []
        for row in inputs:
            rise = self._volume(float(row[0]), float(row[1]))  # (nx, ny, nz)
            kelvin = torch.tensor(rise + AMBIENT, dtype=self.dtype, device=self.device)
            volumes.append(kelvin.permute(2, 1, 0))  # (nz, ny, nx) = (D, H, W)
        return torch.stack(volumes).unsqueeze(1)

    @torch.no_grad()
    def predict_at(self, inputs: ArrayLike) -> Tensor:
        """``[B, 5]`` of ``(x, y, z, t, P)`` to ``[B, 1]`` of Kelvin, off the reconstruction.

        Grouped by ``(t, P)`` so a slice at one instant costs a single transform, then read
        at the nearest node -- exact on the grid, which is where every figure samples.
        """
        inputs = self._as_tensor(inputs, columns=5, name="predict_at")
        z, y, x = self.axes
        instants, inverse = torch.unique(inputs[:, 3:5], dim=0, return_inverse=True)

        outputs = torch.empty((inputs.size(0), 1), dtype=self.dtype, device=self.device)
        for index, instant in enumerate(instants):
            rows = inverse == index
            volume = self.predict_of(instant.reshape(1, 2))[0, 0]  # (nz, ny, nx)
            query = inputs[rows]
            iz = _nearest(z, query[:, 2])
            iy = _nearest(y, query[:, 1])
            ix = _nearest(x, query[:, 0])
            outputs[rows] = volume[iz, iy, ix].unsqueeze(-1)
        return outputs


def build_agent(
    checkpoint: Path,
    shape: tuple[int, int, int] = DEFAULT_FIELD_SHAPE,
    device: torch.device | str | None = None,
) -> FourierMLPAgent:
    """Rebuild the network and the coefficient box it reconstructs through."""
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload["config"]

    run = load_run(Path(payload["run_dir"]))  # the box lives here, not in the archive
    ds = spectral.load(run.dir, config.get("energy") or 0.9999, bool(config.get("detrend")))

    model = FourierMLP(**payload["architecture"])
    model.load_state_dict(payload["state"])
    model.set_normalisation(payload["mu"], payload["sd"])

    n_coef = 2 * int(np.prod(ds.shape))
    ramp_shape = ds.ramp.shape[2:] if ds.ramp is not None else None
    bounds = torch.tensor(
        [
            [0.0, float(run.x[-1]) * MM],
            [0.0, float(run.y[-1]) * MM],
            [0.0, float(run.z[-1]) * MM],
            [0.0, float(run.t[-1])],
        ]
    )
    return FourierMLPAgent(
        model, ds, bounds, derotate=bool(config.get("derotate")),
        vel=config.get("vel", 10.0), n_coef=n_coef, ramp_shape=ramp_shape,
        max_power=float(run.powers.max()), max_time=float(run.t.max()), device=device,
    )
