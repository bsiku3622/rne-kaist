"""One-off: bring the pre-restructure data and archives onto the current conventions.

Three things, in the order they have to happen.

**Rename the data runs.** They were named ``<stamp>_<whatever-it-felt-like>``; the
convention now is ``<solver>_<stamp>_<minP>_<maxP>_<spacing>[-<tag>]``, the same shape
the archive entries use, so a listing of either reads without a decoder ring.

**Backfill config.json.** Only some of it is recoverable. The powers, the grid, the
element size, the snapshot interval and the domain are all readable off the ``.npy``
itself. The material properties and the solver's internal timestep are *not* -- they
were only ever in a log file, or nowhere. So the reconstructed configs are marked as
such and say exactly which fields were read and which were assumed, rather than
quietly presenting the solver defaults as fact.

**Prefix the archive entries and dedup them.** Every existing entry is a DeepONet run,
so they take the ``deeponet_`` prefix. Each one also holds a full byte-for-byte copy of
the dataset it was fitted to -- 13 GB apiece, 78 GB in total, of a dataset the solver
regenerates in 20 s. Where the source still exists and the bytes match (checked by
sha256, not by name), the copy is replaced by a hard link to it.

Nothing here is inferred from a filename. The solver is identified by the column count
(``heat_torch`` writes six, ``heat_fenics`` five) and the duplicates by their hash.

    python share/migrate.py --dry-run     # say what would happen
    python share/migrate.py               # do it
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA, ARCHIVE = REPO / "data", REPO / "archive"

# heat_torch.py's module constants at the time these were run; not in the files
SOLVER_DEFAULTS = {
    "domain_mm": [40.0, 10.0, 6.0],
    "beam": {"radius_mm": 1.5, "scan_speed_mm_s": 10.0, "start_x_mm": 5.0, "y_mm": 5.0},
    "material": {
        "rho_g_mm3": 4.43e-3, "Cp_J_gK": 0.526, "k_W_mmK": 0.0067,
        "h_W_mm2K": 2e-5, "eta": 0.35, "emissivity": 0.40,
    },
    "T_ambient_K": 298.0,
}

# tags worth keeping: the ones that say something the name cannot
KEEP_TAG = ("bareplate", "ypsweep", "ysweep")


_SEEN: dict[tuple, str] = {}


def sha(path: Path, buf: int = 1 << 24) -> str:
    """sha256, cached by (inode, size, mtime) -- these files are 1.9 GB apiece."""
    st = os.stat(path)
    key = (st.st_ino, st.st_size, st.st_mtime_ns)
    if key not in _SEEN:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(buf):
                h.update(chunk)
        _SEEN[key] = h.hexdigest()
    return _SEEN[key]


def describe(run: Path) -> dict:
    """Everything about a run that can actually be read off its files."""
    npys = sorted(run.glob("data_*W*.npy"))
    cols = np.load(npys[0], mmap_mode="r").shape[1]
    solver = {6: "heat_torch.py", 5: "heat_fenics.py"}.get(cols, "unknown")

    a = np.load(npys[0], mmap_mode="r")
    x, y, z, t = (np.unique(np.asarray(a[:, c])) for c in range(4))
    powers = sorted({int(re.search(r"data_(\d+)W", f.name).group(1)) for f in npys})
    return {
        "solver": solver,
        "columns": cols,
        "powers": powers,
        "ele": round(float(x[1] - x[0]), 6),
        "grid": {"nx": len(x), "ny": len(y), "nz": len(z), "nt": len(t)},
        "snap_dt": round(float(t[1] - t[0]), 6) if len(t) > 1 else None,
        "total_t": round(float(t[-1]), 6),
        "domain": [round(float(v[-1]), 6) for v in (x, y, z)],
        "y0_sweep": any(re.search(r"_y[\d.]+\.npy$", f.name) for f in npys),
    }


def new_name(run: Path, d: dict) -> str:
    m = re.match(r"(\d{8})_(\d{6})_(.+)", run.name)
    stamp, old_tag = f"{m.group(1)}-{m.group(2)}", m.group(3)
    who = {"heat_torch.py": "torch", "heat_fenics.py": "fenics"}.get(d["solver"], "unknown")
    tag = next((k for k in KEEP_TAG if k in old_tag), None)
    base = f"{who}_{stamp}_{d['powers'][0]}_{d['powers'][-1]}_{d['ele']:g}"
    return f"{base}-{tag}" if tag else base


def config_for(run: Path, d: dict) -> dict:
    fenics = list(run.glob("*.pvd")) or list(run.glob("*.vtu"))
    cfg = {
        "solver": d["solver"],
        "scheme": "7-point finite difference, Heun in time",
        "reconstructed": True,
        "reconstructed_note": (
            "written after the fact, by reading the .npy files. The material "
            "properties and the solver's internal dt were never stored anywhere and "
            "are heat_torch.py's defaults, not measurements -- treat them as a "
            "guess. Everything under 'read_from_files' is exact."
        ),
        "read_from_files": {
            "powers_W": [float(p) for p in d["powers"]],
            "grid": d["grid"],
            "ele_size_mm": d["ele"],
            "snap_dt_s": d["snap_dt"],
            "total_t_s": d["total_t"],
            "domain_mm": d["domain"],
        },
        "assumed_solver_defaults": SOLVER_DEFAULTS,
        "dt_s": None,
    }
    if d["y0_sweep"]:
        cfg["note"] = (
            "the laser's y position is swept too, and lives in the file name "
            "(data_<P>W_y<Y>.npy). share/grid.py cannot load this run."
        )
    if fenics:
        cfg["note"] = (
            "this directory also holds a heat_fenics.py run of the same case "
            "(data.npy, 5 columns, plus the ParaView files) -- it is the "
            "cross-validation pair, not a stray."
        )
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    act = not a.dry_run

    print("=" * 78)
    print("1. data/ -- rename and backfill config.json")
    print("=" * 78)
    renames = {}
    for run in sorted(DATA.iterdir()):
        if not run.is_dir() or run.name.startswith("_"):
            continue
        d = describe(run)
        name = new_name(run, d)
        renames[run.name] = name
        print(f"  {run.name:<42} -> {name}")
        if act:
            (run / "config.json").write_text(json.dumps(config_for(run, d), indent=2))
            run.rename(DATA / name)

    print()
    print("=" * 78)
    print("2. archive/ -- prefix with the model that made it")
    print("=" * 78)
    for e in sorted(ARCHIVE.iterdir()):
        if not e.is_dir() or e.name.startswith("deeponet_"):
            continue
        print(f"  {e.name:<50} -> deeponet_{e.name}")
        if act:
            e.rename(ARCHIVE / f"deeponet_{e.name}")

    print()
    print("=" * 78)
    print("3. archive/*/data -- replace verified copies with hard links")
    print("=" * 78)
    index: dict[int, list[Path]] = {}
    for run in DATA.iterdir():
        if run.is_dir():
            for f in run.glob("*.npy"):
                index.setdefault(f.stat().st_size, []).append(f)

    freed = 0
    for e in sorted(ARCHIVE.iterdir()):
        d = e / "data"
        if not d.is_dir():
            continue
        hit = miss = linked = 0
        saved = 0
        for f in sorted(d.glob("*.npy")):
            cands = index.get(f.stat().st_size, [])
            src = None
            for c in cands:
                if os.stat(c).st_ino == os.stat(f).st_ino:
                    linked += 1
                    src = "already"
                    break
                if sha(c) == sha(f):
                    src = c
                    break
            if src is None:
                miss += 1
                continue
            if src == "already":
                continue
            hit += 1
            saved += f.stat().st_size
            if act:
                f.chmod(stat.S_IWRITE)  # archive.py's old lock()
                f.unlink()
                os.link(src, f)
        freed += saved
        note = f"{hit} linked, {linked} already, {miss} unique"
        print(f"  {e.name:<50} {saved / 1e9:>6.1f}G  {note}")

    print("-" * 78)
    print(f"  {'TOTAL':<50} {freed / 1e9:>6.1f}G")
    if not act:
        print("\n(dry run -- nothing was changed)")


if __name__ == "__main__":
    main()
