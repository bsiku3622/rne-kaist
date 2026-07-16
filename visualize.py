"""Render any trained model against the run it was fitted to -- one command for all.

Mirrors ``train.py``'s dispatcher: the first argument names the model and ``--entry``
points at its ``archive/`` directory. Every pointwise model is reached through the
same shared inference contract (``agent.predict_of``) and draws the same two figures,
so a checkpoint can be re-rendered long after training -- against a finer ``--times``
grid, or with the gaussian peak fit -- without going back to the model's own script.
This is what the nine harness models never had on their own: figures were written once,
at training time, into the archive, and there was no way back to them.

The two full-field models, ``fmlp`` and ``rmlp``, own a third figure the others cannot
draw -- the Fourier ``signal`` view, and for ``fmlp`` the truncation floor -- so the CLI
delegates to their own renderer rather than flatten them onto the common path and lose it.
The command is uniform either way: same invocation, same ``figures/`` gallery afterwards.

Examples::

    python visualize.py mlp   --entry archive/mlp_..._lbfgs-wide
    python visualize.py gmlp  --entry archive/gmlp_...  --gaussian 4.2
    python visualize.py fmlp  --entry archive/fmlp_...  --times 0.5 1.5 3.0
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.append(str(REPO))

from share import figures, plotting
from share.checkpoints import resolve_device
from share.grid import load_run
from share.registry import available, build_agent

AMBIENT = 298.0

# The full-field models keep their own renderer: it draws the Fourier signal view (and,
# for fmlp, the truncation floor) that the pointwise contract has no way to express.
DELEGATED = {"fmlp", "rmlp"}

# deeponet predates the whole share/ + archive refactor: its checkpoints pickle a
# standalone `loss` module that no longer resolves, its archive entries use the old
# layout, and it never joined the agent contract. It is kept only so its old runs still
# load in its own tree, and is rendered there -- not through this uniform path.
LEGACY = {"deeponet"}


def _field(agent, run, power: float):
    """The model's field on the grid, ``(nt, nx, ny, nz)`` of ``dT`` -- harness's own order.

    The same reshape :class:`share.harness.Predictor` does at training time, so a
    checkpoint re-rendered here is drawn exactly as it was when it was fitted.
    """
    query = [[float(t), float(power)] for t in run.t]
    volume = agent.predict_of(query)[:, 0]  # (nt, nz, ny, nx)
    return volume.permute(0, 3, 2, 1).cpu().numpy() - AMBIENT


def _render_contract(model: str, entry: Path, times, gaussian, power, device) -> None:
    """Draw ``field`` and ``scanline`` for any model that answers ``predict_of``."""
    agent = build_agent(model, entry / "checkpoint.pt", device=resolve_device(device))
    run = load_run(entry / "data")
    config = {}
    if (entry / "config.json").is_file():
        config = json.loads((entry / "config.json").read_text())

    holdout = power if power is not None else float(
        config.get("holdout", run.powers[len(run.powers) // 2])
    )
    truth = run.dT(run.index_of(holdout))
    mine = _field(agent, run, holdout)
    want = times or [
        float(run.t[len(run.t) // 4]), float(run.t[len(run.t) // 2]), float(run.t[-1])
    ]

    entry.joinpath("figures").mkdir(exist_ok=True)
    plotting.planes(truth, mine, run, int(holdout), entry / "figures" / "field.png", model)
    plotting.scanline(truth, mine, run, int(holdout), want,
                      entry / "figures" / "scanline.png", model, gaussian=gaussian)
    print(f"[figures] {entry / 'figures'}")


def _render_delegated(model: str, entry: Path, times) -> None:
    """Hand off to a full-field model's own renderer, which draws the signal view too."""
    module = importlib.import_module(f"models.{model}.visualize")
    argv = ["--entry", str(entry)]
    if times:
        argv += ["--times", *[str(t) for t in times]]
    module.main(argv)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python visualize.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("model", choices=available(), help="which model the entry holds")
    ap.add_argument("--entry", type=Path, required=True, help="an archive/ directory")
    ap.add_argument("--times", type=float, nargs="+", default=None,
                    help="snapshots to cut through; default: three spread over the run")
    ap.add_argument("--gaussian", type=float, default=None, metavar="MM",
                    help="overlay a 1/e^2 gaussian fitted to each profile peak, within "
                         "this half-window in mm (pointwise models only)")
    ap.add_argument("--power", type=float, default=None,
                    help="which power to render against; default: the held-out one")
    ap.add_argument("--device", type=str, default=None)
    a = ap.parse_args(argv)

    if a.model in LEGACY:
        raise SystemExit(
            f"{a.model} is a legacy standalone model, from before the agent contract; "
            f"render it with its own script, e.g.\n"
            f"    python models/{a.model}/visualize.py "
            f"--checkpoint {a.entry / 'checkpoint.pt'} --power <P>"
        )

    if not (a.entry / "checkpoint.pt").is_file():
        raise SystemExit(f"no checkpoint.pt in {a.entry}")

    if a.model in DELEGATED:
        _render_delegated(a.model, a.entry, a.times)
    else:
        _render_contract(a.model, a.entry, a.times, a.gaussian, a.power, a.device)

    figures.publish(a.entry)  # keep the flat gallery current, same as a training run does


if __name__ == "__main__":
    main()
