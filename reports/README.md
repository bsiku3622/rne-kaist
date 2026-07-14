# reports/

Figures that answer a question rather than describe a run.

An archive entry belongs to one training and carries its own figures; a report cuts
*across* entries — nine models on one axis, two optimisers on one dataset, a mode
budget argued from four transforms. It belongs to no single run, so it lives here.

## Naming

```
reports/<YYYYMMDD-HHMM>_<topic>/
```

`data/` and `archive/` lead with *who made it*, because that is what you sort them by.
A report is sorted by *when it was asked*, so the stamp comes first and the newest
question is at the bottom of the listing.

`<topic>` is kebab-case and names the question, not the output. `optimiser-adam-vs-lbfgs`,
not `figures`. If a stranger cannot guess what the folder argues from its name, rename it.

```
reports/20260714-1330_optimiser-adam-vs-lbfgs/
reports/20260713-1746_spectral-mode-budget-fine/
reports/20260710-1612_deeponet-architectures/
```

## Every report carries a README.md

This is the rule that makes the folder worth keeping. A figure with no provenance is
an assertion, and this repository has already caught itself making assertions that did
not survive being checked.

```markdown
# <the question, as a question>

<the answer, in a paragraph. What the figures show, and what it means.>

## Sources

| Figure | Archive entry |
|---|---|
| `mlp.png` | `archive/mlp_20260714-132130_100_250_0.25-ported` |
| ...       | ... |

## How to reproduce

    python train.py mlp --run data/torch_20260710-122446_100_250_0.25 --holdout 175
    ...
```

The **Sources** table is not optional. Every figure names the archive entry it came from,
and every entry carries its own `config.json`, `metrics.json`, `git.txt` and code snapshot
— so a number in a report can always be walked back to the run that produced it, the code
that ran, and the data it saw. Without the table that chain is broken at the first step.

If the figures came from a script rather than a training run (a dataset analysis, say),
name the script and its arguments instead. The point is the same: say what would have to
be re-run to get this again.

## Not tracked, and why the rest is

`data/` and `archive/` are ignored — they are gigabytes, and every byte of them is
regenerable from a `config.json`. `reports/` is tracked. It is a few megabytes of PNGs,
it is what anyone reading the repository actually wants to see, and unlike the runs, a
report is a *conclusion* — losing it loses the thinking, not just the bytes.
