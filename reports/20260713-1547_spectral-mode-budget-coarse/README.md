# How many Fourier modes does the field actually need?

The temperature field is transformed onto a spatial Fourier basis and truncated, and the
question is how many coefficients can be thrown away before the answer stops being the
answer. Three transforms are compared -- a full 3-D FFT, an FFT over the scan plane with
`z` left on the grid, and a DCT-II in depth -- and the budget is set by the *worst* power,
not the pooled energy: energy scales like `P^2`, so 100 W carries under 1% of the sweep and
a pooled criterion would discard it outright, and the low powers are the harder ones.

**The 3-D FFT never drops a single `kz` mode at any accuracy target.** Its optimum *is* the
2-D one, coefficient for coefficient. The plate is 6 mm deep with a 0.288 mm thermal
boundary layer under the beam, and wrapping the 4717 K top face onto the cold substrate
manufactures a cliff whose coefficients decay like `1/m`. Depth stays on the grid.

## Figures

| Figure | What it shows |
|---|---|
| `marginal_spectrum.png` | Energy against wavenumber, per axis. The `kz` curve is the one that never decays. |
| `pareto.png` | Stored coefficients against the truncation error they leave. |
| `kspace.png` | The scan plane's spectrum, with the retained box drawn on it. The horizontal ridge is the x-wrap artefact. |
| `reconstruction.png` | The scanned face, reconstructed from the kept modes, and what the truncation cost. |
| `profiles.png` | The scan line, and the depth profile that shows why `z` cannot be transformed. |

## Sources

Not a training run -- a dataset analysis. Reproduce with:

    python models/spectral/dataset.py --run torch_20260710-122446_100_250_0.25

Dataset: `data/torch_20260710-122446_100_250_0.25`  (100-250 W, 0.25 mm)
