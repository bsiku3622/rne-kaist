"""Archive a finished training run as a complete, read-only historical record.

Renders the top/track/scanline figures with ``visualize.py``, then copies the
checkpoint, the TensorBoard run, the data it was trained on, those figures,
and the exact code that produced it into
``archive/<run>_<minP>_<maxP>_<spacing_mm>[-<tag>]/`` (power range and grid
spacing parsed from the ``data_<power>W.npy`` filenames and the x-axis of the
first file; ``--tag`` appends a suffix identifying the code variant, e.g.
``-gaussfeature``), locks every file read-only, and clears the root (every subdirectory
under ``runs/`` -- not just ``<run>`` -- plus ``checkpoint.pt``, ``train.log``,
``train.err``, ``.train.pid``, ``figures/*.png``). ``runs/`` itself is kept
(TensorBoard needs the directory to exist) and ``data/`` is left alone -- it
is the live working dataset, not part of any one run.

The code files (``calibrate.py``, ``loss.py``, ``model.py``, ``train.py``,
``visualize.py``) land directly in the entry, alongside ``checkpoint.pt`` and
``train.log`` -- not nested in their own subfolder.

``--power`` picks which simulation to render against; it defaults to the
median of the powers found in ``--data-dir`` (parsed from the
``data_<power>W.npy`` filenames). Pass ``--skip-visualize`` to archive
without rendering, e.g. if the checkpoint's architecture cannot be loaded.

Example::

    python archive.py --run 20260710-124432
    python archive.py --run 20260710-131011 --data-dir archive/.../data --power 300
"""

from __future__ import annotations

import argparse
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np

CODE_FILES = ("calibrate.py", "loss.py", "model.py", "train.py", "visualize.py")
ROOT_CLEANUP = ("checkpoint.pt", "train.log", "train.err", ".train.pid")
PLANES = (("top", ()), ("track", ()), ("scanline", ("--gaussian",)))


def lock(path: Path) -> None:
    """Recursively mark every file under ``path`` (or ``path`` itself) read-only."""
    if path.is_dir():
        for file in path.rglob("*"):
            if file.is_file():
                file.chmod(stat.S_IREAD)
    else:
        path.chmod(stat.S_IREAD)


def _force_remove(func, path, exc) -> None:
    """``shutil.rmtree`` error handler: clear the read-only bit and retry.

    ``runs/<run>`` is normally writable (TensorBoard's own files), but a run
    directory repopulated from an archived, read-only copy -- as when
    reproducing an old run -- would otherwise abort the cleanup.
    """
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def unlink_writable(path: Path) -> None:
    """Clear the read-only bit before removing, for the same reason as ``_force_remove``."""
    path.chmod(stat.S_IWRITE)
    path.unlink()


def power_of(path: Path) -> float | None:
    match = re.search(r"data_([\d.]+)W\.npy$", path.name)
    return float(match.group(1)) if match else None


def grid_spacing_mm(path: Path) -> float:
    """Spacing between adjacent x-grid points, read straight off the raw file (in mm)."""
    x = np.unique(np.load(path, mmap_mode="r")[:, 0])
    return round(float(x[1] - x[0]), 6)


def render_figures(
    figures_dir: Path, checkpoint: Path, data_dir: Path, power: float
) -> None:
    for plane, extra_args in PLANES:
        out = figures_dir / f"P{power:g}_{plane}.png"
        command = [
            sys.executable,
            "visualize.py",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--power",
            str(power),
            "--plane",
            plane,
            "--out",
            str(out),
            *extra_args,
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            print(f"[warn] visualize.py --plane {plane} failed ({error}); skipped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", required=True, help="run name under --logdir, e.g. 20260710-124432")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint.pt"))
    parser.add_argument("--train-log", type=Path, default=Path("train.log"))
    parser.add_argument("--logdir", type=Path, default=Path("runs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--archive-dir", type=Path, default=Path("archive"))
    parser.add_argument(
        "--power", type=float, default=None, help="which simulation to render; defaults to the median power"
    )
    parser.add_argument(
        "--skip-visualize", action="store_true", help="archive without rendering figures"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="suffix describing the experiment variant, e.g. 'gaussfeature'; appended as -<tag>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = args.logdir / args.run
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no TensorBoard run at {run_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"no checkpoint at {args.checkpoint}")

    data_files = sorted(args.data_dir.glob("*.npy"))
    if not data_files:
        raise FileNotFoundError(f"no .npy files under {args.data_dir}")

    powers = sorted(p for p in (power_of(path) for path in data_files) if p is not None)
    if not powers:
        raise ValueError(
            f"could not parse a power from any filename under {args.data_dir}; "
            "expected data_<power>W.npy"
        )
    spacing = grid_spacing_mm(data_files[0])

    name = f"{args.run}_{powers[0]:g}_{powers[-1]:g}_{spacing:g}"
    if args.tag:
        name += f"-{args.tag}"
    entry = args.archive_dir / name
    if entry.exists():
        raise FileExistsError(f"{entry} already exists")

    (entry / "data").mkdir(parents=True)
    (entry / "figures").mkdir()

    shutil.copy2(args.checkpoint, entry / args.checkpoint.name)
    shutil.copy2(args.train_log, entry / args.train_log.name)
    shutil.copytree(run_dir, entry / "tensorboard_run", dirs_exist_ok=True)
    for path in data_files:
        shutil.copy2(path, entry / "data" / path.name)

    if args.skip_visualize:
        print("[skip] visualize.py not run (--skip-visualize)")
    else:
        power = args.power if args.power is not None else powers[len(powers) // 2]
        render_figures(entry / "figures", args.checkpoint, args.data_dir, power)

    for name in CODE_FILES:
        source = Path(name)
        if source.is_file():
            shutil.copy2(source, entry / name)
        else:
            print(f"[warn] {name} not found, skipped")

    lock(entry)
    print(f"[archive] {entry}")

    for child in args.logdir.iterdir():
        if child.is_dir():
            shutil.rmtree(child, onexc=_force_remove)
    for name in ROOT_CLEANUP:
        path = Path(name)
        if path.is_file():
            unlink_writable(path)
    for path in args.figures_dir.glob("*.png"):
        unlink_writable(path)
    print("[clean] root cleared (data/ left in place)")


if __name__ == "__main__":
    main()
