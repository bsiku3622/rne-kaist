# Do the DeepONet variants beat a plain MLP?

Four architectures on the same sweep, rendered at 160 W and 200 W: `mlp` (a plain dense
stack), `pidon` (branch/trunk, physics-informed), `gdon` (a DeepONet handed the laser's
gaussian), and `gpidon` (both). The `_top` figures are the scanned face; the `_line`
figures cut along the scan track, where the melt pool's height can be read off directly
rather than inferred from a colour.

These predate the harness this repository now uses. They were selected on validation RMSE
alone and validated on a random tenth of the *points*, drawn from powers the models also
trained on -- not on a held-out power. **The numbers in them are not comparable with
anything under `archive/` produced since**, and they are kept as a record of where the
project was, not as a result.

## Sources

Produced by the pre-restructure `model/visualize.py` against archive entries
`deeponet_20260710-131353`, `-142936`, `-151923`, `-154445`, which no longer exist under
those names. The four architectures survive as `models/pidon`, `models/mlp`,
`models/gpidon` and `models/gdon`.
