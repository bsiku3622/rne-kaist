"""Turn the shared point cloud into the six batches GPiDoN's PINN loss consumes.

The base :class:`~dataset.SimulationDataset` only knows about labelled samples.
The PDE, boundary and initial-condition terms need points that no simulation row
supplies -- interior collocation points, points pinned to a face with the outward
normal attached, points at ``t = 0`` -- so they are sampled here from the same
domain the labels span.

Only the data batch carries temperatures; the rest are unlabelled and get their
targets from the physics.
"""

from __future__ import annotations

import torch
from torch import Tensor

from share.corpus import Domain, SimulationDataset

from .loss import PointSet, laser_flux


class GPiDoNDataset:
    """Samplers for every term of :class:`~models.gpidon.loss.PINNLoss`.

    ``labelled`` supplies the data loss and defines the domain and the set of
    laser powers the unlabelled batches are drawn from -- so a training split
    never samples physics points at a power the model will not be asked about.
    """

    def __init__(
        self,
        labelled: SimulationDataset,
        generator: torch.Generator,
        domain: Domain | None = None,
    ) -> None:
        self.labelled = labelled
        self.generator = generator
        # A validation split's own bounding box is slightly smaller than the
        # corpus's, so the caller can pin the domain to the full one.
        self.domain = domain if domain is not None else labelled.domain
        self.powers = labelled.powers

    # -- helpers -------------------------------------------------------------

    def _power(self, count: int) -> Tensor:
        return self.labelled.sample_power(count, self.generator)

    def _normal(self, count: int, coords: Tensor) -> Tensor:
        return torch.zeros(count, 3, device=coords.device, dtype=coords.dtype)

    # -- one sampler per loss term -------------------------------------------

    def data(self, count: int) -> PointSet:
        """Labelled ``(coords, P) -> T`` rows drawn from the simulation."""
        coords, power, temperature = self.labelled.sample(count, self.generator)
        return PointSet(laser_power=power, coords=coords, temperature=temperature)

    def collocation(self, count: int) -> PointSet:
        """Interior space-time points where the conduction residual is enforced."""
        coords = self.domain.uniform(count, self.generator)
        return PointSet(
            laser_power=self._power(count), coords=coords.requires_grad_(True)
        )

    def bottom(self, count: int) -> PointSet:
        """The ``z = 0`` base plate, held at ambient (Dirichlet)."""
        coords = self.domain.face(axis=2, upper=False, count=count, generator=self.generator)
        return PointSet(laser_power=self._power(count), coords=coords)

    def top(self, count: int) -> PointSet:
        """The ``z = z_max`` surface the laser scans, with its Gaussian source term."""
        coords = self.domain.face(axis=2, upper=True, count=count, generator=self.generator)
        power = self._power(count)
        normal = self._normal(count, coords)
        normal[:, 2] = 1.0
        flux = laser_flux(coords, power)
        return PointSet(
            laser_power=power,
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

        return PointSet(
            laser_power=self._power(count),
            coords=coords.requires_grad_(True),
            normal=normal,
        )

    def initial(self, count: int) -> PointSet:
        """The ``t = 0`` slab, held at ambient."""
        coords = self.domain.face(axis=3, upper=False, count=count, generator=self.generator)
        return PointSet(laser_power=self._power(count), coords=coords)

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
