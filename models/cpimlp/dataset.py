"""Turn the shared point cloud into the six batches CPiMLP's PINN loss consumes.

A copy of :mod:`models.pimlp.dataset` with the laser power taken out of every
batch the network sees, which -- since the network is the only thing that took it
-- means every batch. The base :class:`~dataset.SimulationDataset` only knows
about labelled samples, while the PDE, boundary and initial-condition terms need
points that no simulation row supplies: interior collocation points, points pinned
to a face with the outward normal attached, points at ``t = 0``. They are sampled
here from the same domain the labels span.

The one place the power does not disappear is :meth:`CPiMLPDataset.top`, and it
cannot: the top-surface source term is the laser, and the laser's strength *is*
its power. PiMLP samples a power per point there, the same way it samples one for
every other physics batch, because its network is told which power it is answering
for. This one is not, so a sampled power would ask the boundary condition to
demand a different heat flux at the same point from one step to the next while the
network has no way to tell the two apart -- noise, not physics. Instead the flux is
pinned to a single :attr:`physics_power`, so the residuals describe one definite
problem. The data term still spans all seven powers, and that tension is the
subject of the docstring of :mod:`models.cpimlp.train`.

:func:`normalisation` returns a ``[4]`` mean and scale rather than PiMLP's ``[5]``
-- the same one :meth:`models.cmlp.dataset.CMLPDataset.normalisation` derives.
"""

from __future__ import annotations

import torch
from torch import Tensor

from share.corpus import Domain, SimulationDataset

from .loss import PointSet, laser_flux


def normalisation(domain: Domain) -> tuple[list[float], list[float]]:
    """``(mean, scale)`` per column of ``(t, z, y, x)``, taken from the corpus bounds.

    The box centre and half-width, not the sample mean and standard deviation:
    the grid is uniform, so they agree up to a constant, and the bounds are
    already known from the domain.
    """
    centre, half_width = domain.center.tolist(), domain.half_width.tolist()

    # (t, z, y, x). PiMLP normalises a fifth column, P, from zero; there is no
    # such column here, so `max_power` is not an argument either.
    mean = [centre[3], centre[2], centre[1], centre[0]]
    scale = [half_width[3], half_width[2], half_width[1], half_width[0]]
    return mean, [value if value > 0.0 else 1.0 for value in scale]


class CPiMLPDataset:
    """Samplers for every term of :class:`~models.cpimlp.loss.PINNLoss`.

    ``labelled`` supplies the data loss and defines the domain the unlabelled
    batches are drawn from. Unlike PiMLP's sampler it never draws a power for
    them: the network has no input to receive one. ``physics_power`` is the single
    power the top-surface laser flux is evaluated at.
    """

    def __init__(
        self,
        labelled: SimulationDataset,
        generator: torch.Generator,
        physics_power: float,
        domain: Domain | None = None,
        gaussian_exponent_scale: float = 1.0,
    ) -> None:
        self.labelled = labelled
        self.generator = generator
        # A validation split's own bounding box is slightly smaller than the
        # corpus's, so the caller can pin the domain to the full one.
        self.domain = domain if domain is not None else labelled.domain
        self.physics_power = float(physics_power)
        # Matches the model's own `gaussian_exponent_scale` buffer, so the top
        # BC target seen during training is consistent with the checkpoint.
        self.gaussian_exponent_scale = gaussian_exponent_scale

    # -- helpers -------------------------------------------------------------

    def _normal(self, count: int, coords: Tensor) -> Tensor:
        return torch.zeros(count, 3, device=coords.device, dtype=coords.dtype)

    # -- one sampler per loss term -------------------------------------------

    def data(self, count: int) -> PointSet:
        """Labelled ``coords -> T`` rows drawn from the simulation.

        The rows are drawn exactly as PiMLP draws them, from every power in the
        split; only the ``P`` column of each row is dropped on the way out. So the
        same ``coords`` arrives carrying seven different temperatures over the
        course of training, and the data term can do no better than their mean.
        """
        coords, _power, temperature = self.labelled.sample(count, self.generator)
        return PointSet(coords=coords, temperature=temperature)

    def collocation(self, count: int) -> PointSet:
        """Interior space-time points where the conduction residual is enforced."""
        coords = self.domain.uniform(count, self.generator)
        return PointSet(coords=coords.requires_grad_(True))

    def bottom(self, count: int) -> PointSet:
        """The ``z = 0`` base plate, held at ambient (Dirichlet)."""
        coords = self.domain.face(axis=2, upper=False, count=count, generator=self.generator)
        return PointSet(coords=coords)

    def top(self, count: int) -> PointSet:
        """The ``z = z_max`` surface the laser scans, with its Gaussian source term.

        The flux is evaluated at :attr:`physics_power`, not at a sampled one -- see
        the module docstring.
        """
        coords = self.domain.face(axis=2, upper=True, count=count, generator=self.generator)
        normal = self._normal(count, coords)
        normal[:, 2] = 1.0
        flux = laser_flux(
            coords, self.physics_power, exponent_scale=self.gaussian_exponent_scale
        )
        return PointSet(
            coords=coords.requires_grad_(True),
            normal=normal,
            q_laser=flux,
        )

    def surrounding(self, count: int) -> PointSet:
        """The four lateral faces (x = 0, x = Lx, y = 0, y = Ly) with outward normals."""
        coords = self.domain.uniform(count, self.generator)
        normal = self._normal(count, coords)
        face = torch.randint(0, 4, (count,), generator=self.generator, device=coords.device)

        for index, (axis, is_upper) in enumerate(((0, False), (0, True), (1, False), (1, True))):
            selected = face == index
            if not selected.any():
                continue
            coords[selected, axis] = (
                self.domain.upper[axis] if is_upper else self.domain.lower[axis]
            )
            normal[selected, axis] = 1.0 if is_upper else -1.0

        return PointSet(coords=coords.requires_grad_(True), normal=normal)

    def initial(self, count: int) -> PointSet:
        """The ``t = 0`` slab, held at ambient."""
        coords = self.domain.face(axis=3, upper=False, count=count, generator=self.generator)
        return PointSet(coords=coords)

    def batches(
        self, batch_data: int, batch_physics: int, batch_boundary: int
    ) -> dict[str, PointSet]:
        """One batch per loss term, keyed by :meth:`PINNLoss.forward`'s argument names."""
        return {
            "data": self.data(batch_data),
            "collocation": self.collocation(batch_physics),
            "bottom": self.bottom(batch_boundary),
            "top": self.top(batch_boundary),
            "surrounding": self.surrounding(batch_boundary),
            "initial": self.initial(batch_boundary),
        }
