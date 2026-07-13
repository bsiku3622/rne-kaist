"""Compare a trained DeepONeuralNet against the simulation it was fitted to.

Three planes are available. ``--plane top`` shows the ``z = z_max`` surface the
laser scans, over ``(x, y)``, three panels wide: the finite-element field, the
network prediction, and their signed difference. ``--plane track`` cuts along
the scan line at ``y = y_c`` and shows ``(x, z)``, which is where the melt pool
depth lives and where the model is hardest to fit. Both share a colour scale
between truth and prediction; the error panel uses a symmetric diverging scale
centred on zero, so blue is under-prediction and red is over-prediction.

``--plane scanline`` instead cuts a single line out of the ``top``/``track``
image -- the one running down the middle of the laser track -- so the melt
pool's peak height and width can be read off directly rather than inferred
from a colour. One row per requested time: the profile on the left, the
signed error on the right.

Examples::

    python visualize.py --checkpoint checkpoint.pt --power 200
    python visualize.py --power 200 --plane track --times 0.5 1.5 2.5
    python visualize.py --power 200 --plane scanline --gaussian
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from calibrate import load_grid, Grid
from model import DeepONeuralNet, PlainNeuralNet
from train import BEAM_RADIUS, PROPERTIES

MODEL_CLASSES = {"DeepONeuralNet": DeepONeuralNet, "PlainNeuralNet": PlainNeuralNet}

MM = 1.0e-3

# Categorical slots 1 and 8 of the reference palette; worst-case CVD dE 96.7.
# Line style carries the same distinction, so colour is never the only cue.
TRUTH_COLOUR = "#2a78d6"
PREDICTION_COLOUR = "#eb6834"
INK_SECONDARY = "#52514e"


def find_grid(data_dir: Path, power: float) -> Grid:
    """Load the simulation file whose laser power matches ``power``."""
    paths = sorted(data_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"no .npy files under {data_dir}")

    available = []
    for path in paths:
        grid = load_grid(path)
        if abs(grid.power - power) < 1e-9:
            return grid
        available.append(grid.power)
    raise ValueError(f"no file at P = {power} W; available: {sorted(available)}")


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[DeepONeuralNet | PlainNeuralNet, dict]:
    """Rebuild the network from the architecture stored alongside the weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture is None:
        raise KeyError(
            f"{checkpoint_path} predates the `architecture` key; retrain or add it by hand"
        )

    model_class = MODEL_CLASSES[checkpoint.get("model_class", "DeepONeuralNet")]
    model = model_class(**architecture)
    model.load_state_dict(checkpoint["model"])  # normalisation buffers ride along
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def predict(
    model: DeepONeuralNet,
    coords: np.ndarray,
    power: float,
    device: torch.device,
    chunk: int = 65536,
) -> np.ndarray:
    """Evaluate the model on ``[N, 4]`` physical coordinates, returning ``[N]`` Kelvin."""
    dtype = next(model.parameters()).dtype
    tensor = torch.as_tensor(coords, dtype=dtype, device=device)
    outputs = []
    for start in range(0, tensor.size(0), chunk):
        block = tensor[start : start + chunk]
        laser_power = torch.full((block.size(0), 1), power, dtype=dtype, device=device)
        outputs.append(model(laser_power, block).squeeze(-1).cpu())
    return torch.cat(outputs).numpy()


def slice_coords(
    grid: Grid, plane: str, time: float, track_y: float
) -> tuple[np.ndarray, np.ndarray, tuple, tuple]:
    """Return ``(coords[N,4], truth[a,b], extent_mm, axis_labels)`` for one slice."""
    time_index = int(np.argmin(np.abs(grid.t - time)))
    actual_time = float(grid.t[time_index])

    if plane == "top":
        first, second = grid.x, grid.y
        a, b = np.meshgrid(first, second, indexing="ij")
        c = np.full(a.shape, grid.z[-1])
        truth = grid.temperature[:, :, -1, time_index]
        coords = np.stack([a, b, c, np.full(a.shape, actual_time)], axis=-1)
        labels = ("x [mm]", "y [mm]")
    elif plane == "track":
        row = int(np.argmin(np.abs(grid.y - track_y)))
        first, second = grid.x, grid.z
        a, c = np.meshgrid(first, second, indexing="ij")
        b = np.full(a.shape, grid.y[row])
        truth = grid.temperature[:, row, :, time_index]
        coords = np.stack([a, b, c, np.full(a.shape, actual_time)], axis=-1)
        labels = ("x [mm]", "z [mm]")
    else:
        raise ValueError(f"unknown plane {plane!r}; expected 'top' or 'track'")

    extent = (
        float(first[0]) / MM,
        float(first[-1]) / MM,
        float(second[0]) / MM,
        float(second[-1]) / MM,
    )
    return coords.reshape(-1, 4), truth, extent, labels


def draw(
    grid: Grid,
    model: DeepONeuralNet,
    times: list[float],
    plane: str,
    track_y: float,
    device: torch.device,
) -> Figure:
    """Render the truth / prediction / error grid and print per-slice metrics."""
    # Panels use an equal aspect, so let the slice's own shape set the row height
    # instead of leaving a band of whitespace above and below each strip.
    _, _, probe_extent, _ = slice_coords(grid, plane, times[0], track_y)
    span_x = probe_extent[1] - probe_extent[0]
    span_y = probe_extent[3] - probe_extent[2]
    panel_width = 4.3
    row_height = panel_width * (span_y / span_x) + 1.5

    figure, axes = plt.subplots(
        len(times),
        3,
        figsize=(15, row_height * len(times)),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(f"P = {grid.power:.0f} W, plane = {plane}", fontsize=13)

    for row, time in enumerate(times):
        coords, truth, extent, labels = slice_coords(grid, plane, time, track_y)
        prediction = predict(model, coords, grid.power, device).reshape(truth.shape)
        error = prediction - truth

        rmse = float(np.sqrt((error**2).mean()))
        worst = float(np.abs(error).max())
        print(
            f"  t = {time:4.2f}s  RMSE = {rmse:8.3f} K   max |error| = {worst:9.3f} K"
        )

        # Truth and prediction share limits; the error scale is symmetric about zero.
        low, high = float(truth.min()), float(truth.max())
        bound = max(float(np.abs(error).max()), 1e-9)

        style = dict(origin="lower", extent=extent, aspect="equal")
        field_style = dict(vmin=low, vmax=high, cmap="inferno")

        truth_image = axes[row][0].imshow(truth.T, **style, **field_style)
        axes[row][1].imshow(prediction.T, **style, **field_style)
        error_image = axes[row][2].imshow(
            error.T, **style, vmin=-bound, vmax=bound, cmap="RdBu_r"
        )

        axes[row][0].set_title(f"simulation\nt = {time:.2f} s", fontsize=10)
        axes[row][1].set_title(f"DeepONeuralNet\nRMSE {rmse:.1f} K", fontsize=10)
        axes[row][2].set_title(
            f"prediction - simulation\nmax |error| {worst:.0f} K", fontsize=10
        )

        for column in range(3):
            axes[row][column].set_xlabel(labels[0])
        axes[row][0].set_ylabel(labels[1])

        # Truth and prediction share a scale, so one bar serves both.
        figure.colorbar(truth_image, ax=axes[row][:2].tolist(), label="K", shrink=0.9)
        figure.colorbar(error_image, ax=axes[row][2], label="K", shrink=0.9)

    return figure


@dataclass
class GaussianFit:
    """A ``T_amb + amplitude * exp(-2 (x - centre)^2 / width^2)`` fit of one profile.

    The ``1/e^2`` convention matches :data:`train.BEAM_RADIUS`, so ``width`` is
    directly comparable to the beam radius that deposited the energy.
    """

    amplitude: float  # [K] above ambient
    centre: float  # [m]
    width: float  # [m], 1/e^2 radius
    r_squared: float

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        exponent = -2.0 * (x - self.centre) ** 2 / self.width**2
        return PROPERTIES.ambient_temperature + self.amplitude * np.exp(exponent)


def fit_gaussian(x: np.ndarray, temperature: np.ndarray, window: float) -> GaussianFit | None:
    """Least-squares fit around the profile's own peak, or None if it does not converge.

    Only points within ``window`` of the peak are used. The trailing thermal wake
    is not Gaussian, so fitting the whole line would let the tail set the width;
    restricting to a symmetric window measures the peak itself. Truth and
    prediction get the same treatment, so the comparison between them is fair.
    """
    from scipy.optimize import curve_fit

    def model(coord: np.ndarray, amplitude: float, centre: float, width: float) -> np.ndarray:
        return PROPERTIES.ambient_temperature + amplitude * np.exp(
            -2.0 * (coord - centre) ** 2 / width**2
        )

    peak = int(np.argmax(temperature))
    selected = np.abs(x - x[peak]) <= window
    if selected.sum() < 4:  # three parameters need at least four points
        return None

    guess = (temperature[peak] - PROPERTIES.ambient_temperature, x[peak], BEAM_RADIUS)
    try:
        parameters, _ = curve_fit(
            model, x[selected], temperature[selected], p0=guess, maxfev=10000
        )
    except RuntimeError:
        return None

    residual = temperature[selected] - model(x[selected], *parameters)
    total = temperature[selected] - temperature[selected].mean()
    r_squared = 1.0 - float((residual**2).sum() / (total**2).sum())
    amplitude, centre, width = parameters
    return GaussianFit(float(amplitude), float(centre), abs(float(width)), r_squared)


def line_coords(
    grid: Grid, time: float, track_y: float, depth: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Return ``(x[nx], coords[nx,4], truth[nx], actual)`` along the scan line."""
    time_index = int(np.argmin(np.abs(grid.t - time)))
    row = int(np.argmin(np.abs(grid.y - track_y)))
    layer = len(grid.z) - 1 if depth is None else int(np.argmin(np.abs(grid.z - depth)))

    coords = np.stack(
        [
            grid.x,
            np.full(grid.x.shape, grid.y[row]),
            np.full(grid.x.shape, grid.z[layer]),
            np.full(grid.x.shape, grid.t[time_index]),
        ],
        axis=-1,
    )
    actual = {
        "t": float(grid.t[time_index]),
        "y": float(grid.y[row]),
        "z": float(grid.z[layer]),
    }
    return grid.x, coords, grid.temperature[:, row, layer, time_index], actual


def draw_scanline(
    grid: Grid,
    model: DeepONeuralNet,
    times: list[float],
    track_y: float,
    depth: float | None,
    device: torch.device,
    gaussian_window: float | None = None,
) -> Figure:
    """Render one profile/error row per time and print per-row metrics."""
    figure, axes = plt.subplots(
        len(times),
        2,
        figsize=(12, 3.1 * len(times)),
        squeeze=False,
        sharex=True,
        constrained_layout=True,
    )

    # The cut is the same for every row, so name it once from the first probe.
    *_, placement = line_coords(grid, times[0], track_y, depth)

    for row, time in enumerate(times):
        x, coords, truth, actual = line_coords(grid, time, track_y, depth)
        prediction = predict(model, coords, grid.power, device)
        error = prediction - truth

        rmse = float(np.sqrt((error**2).mean()))
        worst = float(np.abs(error).max())
        peak_gap = float(prediction.max() - truth.max())
        print(
            f"  t = {actual['t']:4.2f}s  RMSE = {rmse:8.3f} K   max |error| = {worst:9.3f} K"
            f"   peak {truth.max():7.1f} -> {prediction.max():7.1f} K ({peak_gap:+.1f})"
        )

        profile_axis, error_axis = axes[row]
        profile_axis.plot(
            x / MM, truth, color=TRUTH_COLOUR, linewidth=1.8, label="simulation"
        )
        profile_axis.plot(
            x / MM,
            prediction,
            color=PREDICTION_COLOUR,
            linewidth=1.8,
            linestyle="--",
            label="DeepONeuralNet",
        )
        if gaussian_window is not None:
            dense = np.linspace(x[0], x[-1], 800)
            for series, colour, temperatures in (
                ("simulation", TRUTH_COLOUR, truth),
                ("DeepONeuralNet", PREDICTION_COLOUR, prediction),
            ):
                fit = fit_gaussian(x, temperatures, gaussian_window)
                if fit is None:
                    print(f"    {series:14s} gaussian fit did not converge")
                    continue
                inside = np.abs(dense - fit.centre) <= gaussian_window
                profile_axis.plot(
                    dense[inside] / MM,
                    fit.evaluate(dense[inside]),
                    color=colour,
                    linewidth=1.0,
                    linestyle=":",
                    label=f"{series}, gaussian fit" if row == 0 else None,
                )
                print(
                    f"    {series:14s} gaussian: amplitude {fit.amplitude:7.1f} K"
                    f"   centre {fit.centre / MM:6.3f} mm"
                    f"   width {fit.width / MM:5.3f} mm"
                    f"   R^2 {fit.r_squared:.4f}"
                )

        profile_axis.set_ylabel("T [K]")
        profile_axis.set_title(
            f"t = {actual['t']:.2f} s   RMSE {rmse:.1f} K", fontsize=10
        )
        if row == 0:
            profile_axis.legend(frameon=False, fontsize=8)

        error_axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
        error_axis.plot(x / MM, error, color=INK_SECONDARY, linewidth=1.5)
        error_axis.fill_between(
            x / MM, error, 0.0, color=INK_SECONDARY, alpha=0.12, linewidth=0
        )
        error_axis.set_ylabel("prediction - simulation [K]")
        error_axis.set_title(f"max |error| {worst:.0f} K", fontsize=10)

        for axis in (profile_axis, error_axis):
            axis.grid(alpha=0.25, linewidth=0.6)
            axis.spines[["top", "right"]].set_visible(False)

    for axis in axes[-1]:
        axis.set_xlabel("x [mm]")

    depth_label = (
        "top surface" if depth is None else f"z = {placement['z'] / MM:.2f} mm"
    )
    figure.suptitle(
        f"P = {grid.power:.0f} W,  y = {placement['y'] / MM:.2f} mm ({depth_label})",
        fontsize=13,
    )
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--power", type=float, required=True, help="which simulation to compare against"
    )
    parser.add_argument("--plane", choices=("top", "track", "scanline"), default="top")
    parser.add_argument("--times", type=float, nargs="+", default=[0.5, 1.5, 2.5])
    parser.add_argument(
        "--track-y",
        type=float,
        default=5.0,
        help="scan-line y in mm, for --plane track and --plane scanline",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=None,
        help="depth in mm for --plane scanline; defaults to the top surface",
    )
    parser.add_argument(
        "--gaussian",
        action="store_true",
        help="--plane scanline only: overlay a 1/e^2 gaussian fitted to each peak",
    )
    parser.add_argument(
        "--fit-window",
        type=float,
        default=2.5 * BEAM_RADIUS / MM,
        help="--plane scanline only: half-width in mm of the fit window around the peak",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="save here instead of showing"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    grid = find_grid(args.data_dir, args.power)
    model, checkpoint = load_model(args.checkpoint, device)
    print(
        f"[load] {args.checkpoint.name}: iteration {checkpoint['iteration']}, "
        f"val RMSE {checkpoint['val_rmse']:.3f} K"
    )
    print(f"[load] {grid.name}: {grid.temperature.shape} at P = {grid.power} W")

    if args.plane == "scanline":
        figure = draw_scanline(
            grid,
            model,
            args.times,
            args.track_y * MM,
            None if args.z is None else args.z * MM,
            device,
            args.fit_window * MM if args.gaussian else None,
        )
    else:
        figure = draw(grid, model, args.times, args.plane, args.track_y * MM, device)

    if args.out is None:
        plt.show()
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.out, dpi=140)
        print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
