# Adam or L-BFGS -- and does the physics loss earn its place?

Nine models, two optimisers, one dataset, one held-out power. Every network is 64 wide and
four deep; the only things that move are the architecture, the loss, and the optimiser. The
two questions the sweep was built to answer turned out to have one answer each, and neither
is the answer the models were designed around.

**The best model in the sweep is `gdon` under L-BFGS: 1.388 K RMSE, a peak error of
-2.35%.** It wins on every metric, and it wins with the optimiser that loses everywhere
else. **Every physics-informed model is beaten by a network that cannot see the laser
power at all.**

## The numbers

Held-out power: 175 W, never trained on -- not a random tenth of the points from powers the
model has already seen. Adam gets 20,000 iterations; L-BFGS gets 1,000 steps of 20 inner
updates each, so the two spend the same gradient budget. Selection is on `rmse + |peak|`.

| | loss | Adam | L-BFGS | |
|---|---|---|---|---|
| **`gdon`** | data | 2.770 K  +4.82% | **1.388 K  −2.35%** | **L-BFGS 2.0×** |
| `mlp` | data | **1.445 K  −4.88%** | 3.262 K  −12.65% | Adam 2.3× |
| `gmlp` | data | **1.453 K  +3.66%** | 3.182 K  −6.84% | Adam 2.2× |
| `cmlp` | *control* | **2.179 K  −9.39%** | 9.956 K  +63.76% | Adam 4.6× |
| `cgmlp` | *control* | **2.325 K  +2.09%** | 6.373 K  −27.70% | Adam 2.7× |
| `pidon` | + PDE | 11.396 K  −40.82% | **7.454 K  −12.81%** | L-BFGS 1.5× |
| `cpimlp` | *control* + PDE | 8.575 K  −26.36% | **7.885 K  −24.12%** | L-BFGS 1.1× |
| `pimlp` | + PDE | 8.777 K  −26.20% | **8.734 K  −29.62%** | a tie |
| `gpidon` | + PDE | **12.389 K  −39.53%** | 44.396 K  −98.67% | L-BFGS diverged |

The **controls** (`c*`) are the same networks with the laser power removed from their input.
They are the floor: the same `(t, z, y, x)` carries a different temperature at every power,
so a network that cannot see `P` can do no better than the mean over the sweep. Anything
that does not beat them has learned nothing about the laser.

## The physics loss is a net negative here, and the controls prove it

Read the table by loss instead of by optimiser and it falls apart cleanly. The best
physics-informed model in the sweep is `pidon` under L-BFGS at **7.454 K**. The blind
control `cmlp` -- a plain MLP that is not told the laser power, under plain Adam, in 30
seconds -- scores **2.179 K**. The best PINN is **3.4× worse than a model that does not
know the laser exists.**

It is worse than that. Compare a physics-informed model against *its own* blind control:

| | sees `P` | blind control | |
|---|---|---|---|
| Adam | `pimlp` 8.777 K | `cpimlp` **8.575 K** | the blind one wins |
| L-BFGS | `pimlp` 8.734 K | `cpimlp` **7.885 K** | the blind one wins |

Being told the laser power makes a physics-informed network *worse*, under both optimisers.
That is not a small effect to explain away: it means the PDE residual is drowning the data
term so completely that the `P` input carries no usable signal. The measured loss breakdown
says the same thing -- the data term is about 6% of the total, and the top-surface flux
boundary condition, with its `T⁴` radiation term, is about 74%.

The physics is not wrong. It is simply not what these networks are short of, and it is
consuming the gradient budget that the data was going to use.

## The optimiser has no universal answer -- the architecture decides

| | Adam wins | L-BFGS wins |
|---|---|---|
| MLP + data loss | `mlp` 2.3×, `gmlp` 2.2×, `cmlp` 4.6×, `cgmlp` 2.7× | |
| DeepONet + data loss | | `gdon` 2.0× |
| PDE loss | `gpidon` (L-BFGS diverged) | `pidon` 1.5×, `cpimlp` 1.1×, `pimlp` tie |

A plain MLP fitting labelled points is a well-conditioned problem and Adam eats it. A
DeepONet is not: branch and trunk meet in an inner product, so the loss is bilinear in two
sets of weights and its curvature is far worse behaved. That is exactly the situation
second-order information is for, and `gdon` gains a factor of two from it -- enough to take
the whole sweep. The PDE residuals are ill-conditioned in the same way and lean the same
direction, for whatever that is worth in a family that loses to the controls anyway.

L-BFGS costs 2-4× Adam's wall time for the same gradient budget, because the strong-Wolfe
line search evaluates the objective several times per step. `gdon` is 146 s against 48 s,
and it is still the cheapest way to the best model here.

**`gpidon` diverges under L-BFGS.** Its best checkpoint is at step 100 of 1000, and by the
end it has flattened the melt pool by 98.67% -- the peak is essentially gone. The gaussian
gate multiplies the branch-trunk product, adding a third factor to an already bilinear
objective, and a line search on that appears to walk off a cliff it cannot see. Stated as
an observation, not a mechanism.

## Two things that were checked, and one that did not survive

**The batch-coverage hypothesis is dead.** L-BFGS needs a fixed objective, so it holds a
batch still for 25 steps -- 40 batches over a run, 327,680 points, about 4% of the 7.9M
training set, where Adam's 20,000 minibatches cover essentially all of it. That looked like
the obvious reason L-BFGS loses on the MLPs. It is not. Giving L-BFGS a 32× batch (262,144,
so 10.5M draws exceed the training set) moves nothing in a consistent direction:

| | L-BFGS, 8k batch | L-BFGS, 262k batch | |
|---|---|---|---|
| `mlp` | 3.262 K | 3.340 K | no change |
| `gmlp` | 3.182 K | **1.569 K** | 2.0× better |
| `gdon` | **1.388 K** | 3.059 K | 2.2× worse |

Three models, three different signs. Whatever separates Adam from L-BFGS here, it is not
how much of the corpus each one saw. The hypothesis was ours and its own control killed it.

**The first L-BFGS sweep measured a stalled optimiser, and every number in it was wrong.**
torch's `LBFGS` ends its inner loop on *absolute* thresholds -- `|grad|_inf <= 1e-7`, or a
loss change below `1e-9` -- which assume an objective of order one. `ScaledMSELoss` lives
around `1e-6`, so a step that improves it by a part in a thousand moves it by less than
`1e-9`, and L-BFGS calls that convergence and stops nowhere near a minimum. Measured on
`mlp`, 300 steps in: with the defaults a step spends **one** gradient evaluation and the
loss sits at `1.311e-05`; with the tolerances off it spends **twenty-five** and reaches
`5.898e-06`.

The wall clock had been saying so quietly the whole time -- 61 ms/step against the 134 ms a
step costs when it actually runs its twenty inner iterations -- and that signal was first
misread as line-search overhead. With the tolerances off, `gdon` went from 2.676 K to
1.388 K and became the best model in the sweep. **A stalled optimiser had been hiding the
winner.** The PINN runs were untouched by the fix, and reproduced bit-for-bit: their total
loss is around `1e-2`, so the thresholds never fired. That is the diagnosis confirming
itself.

## What this sweep cannot tell you

**175 W is the midpoint of the 100–250 W sweep**, and a power-blind control predicts
something close to the sweep mean. So the controls are sitting almost exactly where the
held-out answer is, and they are an unusually *strong* floor here. Two consequences, in
opposite directions: the data models' 1.5× margin over the control understates how much
they know about `P`, and the PINNs' loss to the control is, if anything, flattered. Holding
out 200 W -- or 250 W, which asks for extrapolation rather than interpolation -- would say
more. That is the next run.

The nine models here are the ones that share `share/harness.py`. `coord`, `deeponet` and
`spectral` each still carry their own loop with Adam hard-coded, so they are not in this
comparison; putting them under the harness is what would let them in.

## Sources

Every figure is the scanline through the melt pool at the held-out power, `scanline.png`
from the entry named beside it. Each entry carries its own `config.json`, `metrics.json`,
`git.txt` and a snapshot of the code that ran.

| Figure | Archive entry |
|---|---|
| `mlp_adam.png` | `archive/mlp_20260714-135017_100_250_0.25-adam` |
| `mlp_lbfgs.png` | `archive/mlp_20260714-152009_100_250_0.25-lbfgs` |
| `mlp_lbfgs-wide.png` | `archive/mlp_20260714-161308_100_250_0.25-lbfgs-wide` |
| `gmlp_adam.png` | `archive/gmlp_20260714-135052_100_250_0.25-adam` |
| `gmlp_lbfgs.png` | `archive/gmlp_20260714-152212_100_250_0.25-lbfgs` |
| `gmlp_lbfgs-wide.png` | `archive/gmlp_20260714-161807_100_250_0.25-lbfgs-wide` |
| `gdon_adam.png` | `archive/gdon_20260714-151147_100_250_0.25-adam` |
| `gdon_lbfgs.png` | `archive/gdon_20260714-152422_100_250_0.25-lbfgs` |
| `gdon_lbfgs-wide.png` | `archive/gdon_20260714-162239_100_250_0.25-lbfgs-wide` |
| `pimlp_adam.png` | `archive/pimlp_20260714-135133_100_250_0.25-adam` |
| `pimlp_lbfgs.png` | `archive/pimlp_20260714-152654_100_250_0.25-lbfgs` |
| `pidon_adam.png` | `archive/pidon_20260714-135935_100_250_0.25-adam` |
| `pidon_lbfgs.png` | `archive/pidon_20260714-153746_100_250_0.25-lbfgs` |
| `gpidon_adam.png` | `archive/gpidon_20260714-140652_100_250_0.25-adam` |
| `gpidon_lbfgs.png` | `archive/gpidon_20260714-154737_100_250_0.25-lbfgs` |
| `cmlp_adam.png` | `archive/cmlp_20260714-141648_100_250_0.25-adam` |
| `cmlp_lbfgs.png` | `archive/cmlp_20260714-155925_100_250_0.25-lbfgs` |
| `cgmlp_adam.png` | `archive/cgmlp_20260714-141722_100_250_0.25-adam` |
| `cgmlp_lbfgs.png` | `archive/cgmlp_20260714-160137_100_250_0.25-lbfgs` |
| `cpimlp_adam.png` | `archive/cpimlp_20260714-141803_100_250_0.25-adam` |
| `cpimlp_lbfgs.png` | `archive/cpimlp_20260714-160353_100_250_0.25-lbfgs` |

`_data.json` holds every number quoted above, read back out of those entries.

## How to reproduce

Dataset: `data/torch_20260710-122446_100_250_0.25` -- seven powers, 100 to 250 W, 0.25 mm.

```powershell
$RUN = "data/torch_20260710-122446_100_250_0.25"

foreach ($m in "mlp","gmlp","gdon","pimlp","pidon","gpidon","cmlp","cgmlp","cpimlp") {
    python train.py $m --run $RUN --holdout 175 --optimizer adam  --tag adam
    python train.py $m --run $RUN --holdout 175 --optimizer lbfgs --tag lbfgs
}

# the coverage control
foreach ($m in "mlp","gmlp","gdon") {
    python train.py $m --run $RUN --holdout 175 --optimizer lbfgs --batch-size 262144 --tag lbfgs-wide
}
```

Defaults do the rest: 64 wide and four deep, Adam at 20,000 iterations and `lr` 1e-3,
L-BFGS at 1,000 steps of 20 inner updates and `lr` 1.0 -- one gradient budget, two ways of
spending it. See `share/harness.DEFAULTS` and `share/training.optimiser_for`.
