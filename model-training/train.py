"""Train DeepONeuralNet on laser-scan temperature fields with PINN losses.

Run with ``python train.py``. Every quantity below is SI (metres, seconds,
Kelvin); the raw ``.npy`` files store millimetres and are converted on load.

Each file under ``--data-dir`` is a structured grid of
``(x, y, z, t, P, T)`` rows -- ``321 x 81 x 49 x 31 = 39495519`` for the shipped
``data_100W.npy``. ``P`` is the branch input and is constant within a file, so
the branch network only has something to learn once several files at different
powers are present; all of them are globbed and concatenated automatically.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loss import LossWeights, PINNLoss, PointSet, ResidualScales, ThermalProperties
from model import DeepONeuralNet, PlainNeuralNet

MM = 1.0e-3

# ---------------------------------------------------------------------------
# Fitted jointly across all seven powers by `python calibrate.py`. The
# top-surface energy balance closes to 0.79% of peak flux. Re-run calibrate.py
# whenever the contents of the data directory change.
# ---------------------------------------------------------------------------
ABSORPTIVITY = 0.4656  # A [-]
BEAM_RADIUS = 1.5266 * MM  # r_b [m]
LASER_START_X = 4.9924 * MM
LASER_Y = 5.0000 * MM
SCAN_SPEED = 10.0011 * MM  # [m s^-1]

# `conductivity` follows from the fitted diffusivity alpha = 2.3685e-6 m^2/s
# (corr 0.978 against the interior Laplacian) and the assumed rho and c_p; only
# the ratio k/(rho*c_p) is identifiable from the data. `convection_coeff` and
# `emissivity` are NOT identifiable -- on the lateral faces the normal
# derivative sits at the noise floor of the export grid -- so they stay as
# inputs. They account for ~2.5% of the top-surface balance.
PROPERTIES = ThermalProperties(
    density=7990.0,  # rho [kg m^-3]      (assumed)
    specific_heat=500.0,  # c_p [J kg^-1 K^-1] (assumed)
    conductivity=9.4620,  # k [W m^-1 K^-1]    (= alpha * rho * c_p)
    convection_coeff=20.0,  # h [W m^-2 K^-1]    (assumed)
    emissivity=0.35,  # epsilon [-]        (assumed)
    ambient_temperature=298.0,  # T_amb [K]  (matches t=0 and z=0 in the data)
)

# The 0.125 mm grids bottom out at exactly T_amb, so nothing is clipped. Kept
# for coarser exports, whose sub-ambient rows were a solver artefact: with
# heating only, T can never fall below T_amb.
CLIP_SUBAMBIENT = False


@dataclass
class Domain:
    """Axis-aligned space-time bounds, ``[4, 2]`` as ``(lower, upper)`` per axis."""

    bounds: Tensor

    @property
    def lower(self) -> Tensor:
        return self.bounds[:, 0]

    @property
    def upper(self) -> Tensor:
        return self.bounds[:, 1]

    @property
    def center(self) -> Tensor:
        return 0.5 * (self.lower + self.upper)

    @property
    def half_width(self) -> Tensor:
        return 0.5 * (self.upper - self.lower)

    def uniform(self, count: int, generator: torch.Generator) -> Tensor:
        unit = torch.rand(
            (count, self.bounds.size(0)),
            generator=generator,
            device=self.bounds.device,
            dtype=self.bounds.dtype,
        )
        return self.lower + unit * (self.upper - self.lower)


def load_dataset(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``(x, y, z, t, P, T)`` files into coords [N,4], power [N,1], temperature [N,1].

    Spatial columns are converted from millimetres to metres; time is already
    in seconds and power in watts.
    """
    if not paths:
        raise FileNotFoundError("no .npy files found")

    coord_blocks, power_blocks, temperature_blocks = [], [], []
    for path in paths:
        raw = np.load(path).astype(np.float64)
        if raw.ndim != 2 or raw.shape[1] != 6:
            raise ValueError(
                f"{path}: expected [N, 6] of (x, y, z, t, P, T), got {raw.shape}"
            )

        coords = raw[:, :4].copy()
        coords[:, :3] *= MM
        power = raw[:, 4:5].copy()
        temperature = raw[:, 5:6].copy()

        below = temperature < PROPERTIES.ambient_temperature
        if below.any():
            message = (
                f"{path.name}: {below.sum()} of {below.size} rows below T_amb="
                f"{PROPERTIES.ambient_temperature} K (min {temperature.min():.2f} K)"
            )
            if CLIP_SUBAMBIENT:
                temperature = np.maximum(temperature, PROPERTIES.ambient_temperature)
                print(f"[data] {message} -- clipped")
            else:
                print(f"[data] {message} -- kept as-is")

        print(f"[data] {path.name}: {raw.shape[0]} rows, P={np.unique(power).tolist()} W")
        coord_blocks.append(coords)
        power_blocks.append(power)
        temperature_blocks.append(temperature)

    return (
        np.concatenate(coord_blocks),
        np.concatenate(power_blocks),
        np.concatenate(temperature_blocks),
    )


def peak_laser_flux(power: Tensor | float) -> Tensor | float:
    """Centreline intensity of the Gaussian beam, ``2*A*P / (pi*r_b^2)`` in W m^-2."""
    return 2.0 * ABSORPTIVITY * power / (math.pi * BEAM_RADIUS**2)


def laser_flux(coords: Tensor, power: Tensor) -> Tensor:
    """Gaussian moving heat source on the top surface, ``[batch, 1]`` in W m^-2.

    ``q(x, y, t) = (2*A*P / (pi*r_b^2)) * exp(-2*((x - x_l(t))^2 + (y - y_l)^2) / r_b^2)``
    with the beam centre ``x_l(t) = LASER_START_X + SCAN_SPEED * t``.
    """
    x, y, t = coords[:, 0:1], coords[:, 1:2], coords[:, 3:4]
    centre_x = LASER_START_X + SCAN_SPEED * t
    squared_distance = (x - centre_x) ** 2 + (y - LASER_Y) ** 2
    return peak_laser_flux(power) * torch.exp(-2.0 * squared_distance / BEAM_RADIUS**2)


def sample_power(count: int, values: Tensor, generator: torch.Generator) -> Tensor:
    """Draw ``[count, 1]`` branch inputs from the powers present in the dataset."""
    index = torch.randint(0, values.numel(), (count,), generator=generator, device=values.device)
    return values[index].reshape(count, 1)


def sample_face(
    domain: Domain, axis: int, upper: bool, count: int, generator: torch.Generator
) -> Tensor:
    """Uniform points on one axis-aligned face, with that axis pinned to a bound."""
    coords = domain.uniform(count, generator)
    coords[:, axis] = domain.upper[axis] if upper else domain.lower[axis]
    return coords


def sample_collocation(
    domain: Domain, powers: Tensor, count: int, generator: torch.Generator
) -> PointSet:
    coords = domain.uniform(count, generator)
    return PointSet(
        laser_power=sample_power(count, powers, generator),
        coords=coords.requires_grad_(True),
    )


def sample_bottom(
    domain: Domain, powers: Tensor, count: int, generator: torch.Generator
) -> PointSet:
    coords = sample_face(domain, axis=2, upper=False, count=count, generator=generator)
    return PointSet(laser_power=sample_power(count, powers, generator), coords=coords)


def sample_top(
    domain: Domain, powers: Tensor, count: int, generator: torch.Generator
) -> PointSet:
    coords = sample_face(domain, axis=2, upper=True, count=count, generator=generator)
    power = sample_power(count, powers, generator)
    normal = torch.zeros(count, 3, device=coords.device, dtype=coords.dtype)
    normal[:, 2] = 1.0
    flux = laser_flux(coords, power)
    return PointSet(
        laser_power=power,
        coords=coords.requires_grad_(True),
        normal=normal,
        q_laser=flux,
    )


def sample_surrounding(
    domain: Domain, powers: Tensor, count: int, generator: torch.Generator
) -> PointSet:
    """The four lateral faces (x = 0, x = Lx, y = 0, y = Ly) with outward normals."""
    coords = domain.uniform(count, generator)
    normal = torch.zeros(count, 3, device=coords.device, dtype=coords.dtype)
    face = torch.randint(0, 4, (count,), generator=generator, device=coords.device)

    for index, (axis, is_upper) in enumerate(((0, False), (0, True), (1, False), (1, True))):
        selected = face == index
        if not selected.any():
            continue
        coords[selected, axis] = domain.upper[axis] if is_upper else domain.lower[axis]
        normal[selected, axis] = 1.0 if is_upper else -1.0

    return PointSet(
        laser_power=sample_power(count, powers, generator),
        coords=coords.requires_grad_(True),
        normal=normal,
    )


def sample_initial(
    domain: Domain, powers: Tensor, count: int, generator: torch.Generator
) -> PointSet:
    coords = sample_face(domain, axis=3, upper=False, count=count, generator=generator)
    return PointSet(laser_power=sample_power(count, powers, generator), coords=coords)


def sample_data(
    coords: Tensor,
    power: Tensor,
    temperature: Tensor,
    train_index: Tensor,
    count: int,
    generator: torch.Generator,
    device: torch.device,
) -> PointSet:
    """Draw ``count`` training rows from the CPU-resident dataset onto ``device``.

    The dataset itself never leaves the CPU: at 276M points the coordinates
    alone are 4.4 GB in float32, and gathering the training split would double
    that again, leaving no room for the autograd graph of the PDE residual.
    """
    where = torch.randint(0, train_index.numel(), (count,), generator=generator)
    index = train_index[where]
    return PointSet(
        laser_power=power[index].to(device),
        coords=coords[index].to(device),
        temperature=temperature[index].to(device),
    )


@torch.no_grad()
def evaluate(
    model: DeepONeuralNet,
    coords: Tensor,
    power: Tensor,
    temperature: Tensor,
    chunk: int = 65536,
) -> tuple[float, float]:
    """Return ``(rmse, max_abs_error)`` in Kelvin over the given points."""
    squared_error = 0.0
    worst = 0.0
    for start in range(0, coords.size(0), chunk):
        stop = start + chunk
        error = model(power[start:stop], coords[start:stop]) - temperature[start:stop]
        squared_error += float(error.pow(2).sum())
        worst = max(worst, float(error.abs().max()))
    return math.sqrt(squared_error / coords.size(0)), worst


# Width of PlainNeuralNet's four hidden layers, chosen so its parameter count
# (133134) lands within 0.01% of DeepONeuralNet's (133121) -- the two
# architectures are meant to be compared at matched capacity, not just
# matched depth.
PLAIN_MLP_WIDTH = 209


def build_model(
    domain: Domain, temperature_rise: float, max_power: float, architecture_name: str
) -> tuple[DeepONeuralNet | PlainNeuralNet, dict]:
    """Instantiate the network with normalisation baked in from the data statistics.

    The keyword dict is returned alongside so it can be stored in the checkpoint;
    `visualize.py` rebuilds the network from it without re-reading the dataset.
    """
    common = dict(
        branch_input_dim=1,
        coord_mean=domain.center.tolist(),
        coord_scale=domain.half_width.tolist(),
        branch_mean=[0.0],
        branch_scale=[max_power],
        temperature_offset=PROPERTIES.ambient_temperature,
        temperature_scale=temperature_rise,
    )
    if architecture_name == "mlp":
        architecture = dict(**common, hidden_layers=(PLAIN_MLP_WIDTH,) * 4)
        return PlainNeuralNet(**architecture), architecture

    architecture = dict(
        **common,
        hidden_layers=(128, 128, 128, 128),
        latent_dim=128,
        laser_beam_radius=BEAM_RADIUS,
        laser_start_x=LASER_START_X,
        laser_y=LASER_Y,
        laser_scan_speed=SCAN_SPEED,
    )
    return DeepONeuralNet(**architecture), architecture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-data", type=int, default=4096)
    parser.add_argument("--batch-physics", type=int, default=2048)
    parser.add_argument("--batch-boundary", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=250, help="validation and console cadence")
    parser.add_argument("--scalar-every", type=int, default=25, help="TensorBoard loss cadence")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--double", action="store_true", help="run in float64")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint.pt"))
    parser.add_argument("--logdir", type=Path, default=Path("runs"), help="TensorBoard output")
    parser.add_argument("--run-name", type=str, default=None, help="subdirectory under --logdir")
    parser.add_argument("--no-progress", action="store_true", help="disable the tqdm bar")
    parser.add_argument(
        "--architecture",
        type=str,
        choices=("deeponet", "mlp"),
        default="deeponet",
        help="deeponet: branch/trunk DeepONet plus the gaussian trunk feature. "
        "mlp: a single MLP over (x, y, z, t, P), no branch/trunk split, matched parameter count.",
    )
    parser.add_argument("--w-data", type=float, default=1.0)
    parser.add_argument("--w-pde", type=float, default=1.0)
    parser.add_argument("--w-bc", type=float, default=1.0)
    parser.add_argument("--w-ic", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.double else torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device(args.device)

    coords_np, power_np, temperature_np = load_dataset(sorted(args.data_dir.glob("*.npy")))

    coords = torch.as_tensor(coords_np, dtype=dtype)
    power = torch.as_tensor(power_np, dtype=dtype)
    temperature = torch.as_tensor(temperature_np, dtype=dtype)

    bounds = torch.stack((coords.min(dim=0).values, coords.max(dim=0).values), dim=1)
    domain = Domain(bounds=bounds.to(device))
    powers = torch.unique(power).to(device)
    max_power = float(powers.max())
    temperature_rise = float(temperature.max() - PROPERTIES.ambient_temperature)

    # The dataset stays on the CPU and `sample_data` moves one batch at a time;
    # only the collocation and boundary points are drawn on the device.
    generator = torch.Generator().manual_seed(args.seed)
    device_generator = torch.Generator(device=device).manual_seed(args.seed)

    permutation = torch.randperm(coords.size(0), generator=generator)
    validation_size = int(args.val_fraction * coords.size(0))
    val_index, train_index = permutation[:validation_size], permutation[validation_size:]

    val_coords = coords[val_index].to(device)
    val_power = power[val_index].to(device)
    val_temperature = temperature[val_index].to(device)

    model, architecture = build_model(domain, temperature_rise, max_power, args.architecture)
    model = model.to(device=device, dtype=dtype)
    scales = ResidualScales.characteristic(
        properties=PROPERTIES,
        temperature_rise=temperature_rise,
        time_scale=float(domain.upper[3]),
        peak_flux=float(peak_laser_flux(max_power)),
    )
    weights = LossWeights(data=args.w_data, pde=args.w_pde, bc=args.w_bc, ic=args.w_ic)
    criterion = PINNLoss(PROPERTIES, weights=weights, scales=scales)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations)

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(args.logdir / run_name))

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"[setup] tensorboard run -> {args.logdir / run_name}")
    print(f"[setup] device={device} dtype={dtype} params={parameter_count}")
    print(f"[setup] lower={domain.lower.tolist()} upper={domain.upper.tolist()}")
    print(f"[setup] powers={powers.tolist()} W  T_rise={temperature_rise:.1f} K")
    print(
        f"[setup] scales temperature={scales.temperature:.4g} K "
        f"pde={scales.pde:.4g} W/m^3 flux={scales.flux:.4g} W/m^2"
    )
    print(f"[setup] train={train_index.numel()} val={val_index.numel()}")

    best_rmse = math.inf
    progress = tqdm(
        range(1, args.iterations + 1),
        desc="train",
        unit="it",
        disable=args.no_progress,
        dynamic_ncols=True,
    )
    for iteration in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total, components = criterion(
            model,
            data=sample_data(
                coords, power, temperature, train_index, args.batch_data, generator, device
            ),
            collocation=sample_collocation(domain, powers, args.batch_physics, device_generator),
            bottom=sample_bottom(domain, powers, args.batch_boundary, device_generator),
            top=sample_top(domain, powers, args.batch_boundary, device_generator),
            surrounding=sample_surrounding(domain, powers, args.batch_boundary, device_generator),
            initial=sample_initial(domain, powers, args.batch_boundary, device_generator),
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # .item() synchronises with the GPU, so only pull the scalars we log.
        if iteration % args.scalar_every == 0 or iteration == 1:
            total_value = total.detach().item()
            writer.add_scalar("loss/total", total_value, iteration)
            for name, value in components.items():
                writer.add_scalar(f"loss/{name}", value.detach().item(), iteration)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], iteration)
            progress.set_postfix(loss=f"{total_value:.3e}", best=f"{best_rmse:.1f}K")

        if iteration % args.log_every == 0 or iteration == 1:
            model.eval()
            rmse, worst = evaluate(model, val_coords, val_power, val_temperature)
            writer.add_scalar("val/rmse", rmse, iteration)
            writer.add_scalar("val/max_error", worst, iteration)

            parts = " ".join(
                f"{name}={value.detach().item():.3e}" for name, value in components.items()
            )
            progress.write(
                f"[{iteration:6d}] total={total.detach().item():.4e} {parts} "
                f"| val_rmse={rmse:7.3f}K val_max={worst:8.3f}K lr={scheduler.get_last_lr()[0]:.2e}"
            )
            if rmse < best_rmse:
                best_rmse = rmse
                progress.set_postfix(loss=f"{total.detach().item():.3e}", best=f"{best_rmse:.1f}K")
                torch.save(
                    {
                        "iteration": iteration,
                        "model": model.state_dict(),
                        "model_class": type(model).__name__,
                        "architecture": architecture,
                        "val_rmse": rmse,
                        "properties": PROPERTIES,
                        "scales": scales,
                        "weights": weights,
                    },
                    args.checkpoint,
                )

    progress.close()
    writer.add_hparams(
        {
            "lr": args.lr,
            "iterations": args.iterations,
            "batch_data": args.batch_data,
            "batch_physics": args.batch_physics,
            "batch_boundary": args.batch_boundary,
            "w_data": weights.data,
            "w_pde": weights.pde,
            "w_bc": weights.bc,
            "w_ic": weights.ic,
        },
        {"hparam/val_rmse": best_rmse},
    )
    writer.close()
    print(f"[done] best val RMSE {best_rmse:.3f} K -> {args.checkpoint}")
    print(f"[done] tensorboard --logdir {args.logdir}")


if __name__ == "__main__":
    main()
