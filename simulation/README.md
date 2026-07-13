# rne-am-simulation

Transient thermal simulation of a laser scanning a bare metal plate, as in laser
powder-bed fusion. Two independent solvers for the same problem, so results can
be cross-checked.

## The problem

A Gaussian laser sweeps the top surface of a 40 x 10 x 6 mm plate along `x` at
constant speed. Heat diffuses through the solid; the free surfaces lose heat by
convection and radiation; the substrate underneath is held at ambient.

```
rho*Cp*dT/dt = k * laplacian(T)                    in the volume
k * grad(T).n = q_laser - q_conv - q_rad           on the top face
k * grad(T).n =        - q_conv - q_rad            on the four sides
T = T_ambient                                       on the bottom face
```

with

```
q_laser = 2*P*eta/(pi*r^2) * exp(-2*((x - x0 - v*t)^2 + (y - y0)^2) / r^2)
q_conv  = h * (T - T_ambient)
q_rad   = sigma * emiss * (T^4 - T_ambient^4)
```

Units are mm-g-s throughout, so `k` is W/(mm.K), `rho` is g/mm^3, and the
Stefan-Boltzmann constant is `5.6704e-14` W/(mm^2.K^4).

## Two solvers

| | `src/heat_fenics.py` | `src/heat_torch.py` |
|---|---|---|
| method | FEM, CG1 | finite difference, 7-point |
| time | Crank-Nicolson (implicit) | Heun / RK2 (explicit) |
| linear solve | Newton + MUMPS | none |
| hardware | CPU | GPU (batched over laser power) |
| dependency | legacy FEniCS 2019 | PyTorch |

The mesh is a uniform box, so the CG1 stencil collapses to a 7-point difference.
That is what makes the GPU version possible: every time step is one stencil pass,
with no assembly and no factorisation. A seven-power sweep that takes 5.5 hours
across seven CPU processes finishes in about 20 seconds on one GPU.

## Install

`heat_torch.py` needs only PyTorch with CUDA.

```bash
pip install torch numpy
```

`heat_fenics.py` needs **legacy** FEniCS 2019 (`import fenics`), which is not the
same as the current FEniCSx. It is Linux-only, so on Windows use WSL. The
easiest route needs no root:

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj bin/micromamba
./bin/micromamba create -y -n fenics -c conda-forge python=3.10 fenics=2019.1.0 numpy
./bin/micromamba run -n fenics python -c "import fenics; print('ok')"
```

Alternatively `apt install python3-dolfin` from `ppa:fenics-packages/fenics`.
Install `python3-dolfin` specifically — the `fenics` metapackage in that PPA now
points at FEniCSx, which has a different API.

## Usage

### GPU (recommended)

```bash
python src/heat_torch.py \
    --powers 100 125 150 175 200 225 250 \
    --ele_size 0.125 --dt 6.25e-4 \
    --outdir data/sweep
```

All powers advance together as one batched tensor. Defaults are Ti-6Al-4V
(`rho=4.43e-3`, `Cp=0.526`, `k=0.0067`, `eta=0.35`, `emiss=0.40`).

Explicit stepping is stable only while

```
dt < ele_size^2 / (6 * alpha),    alpha = k / (rho * Cp)
```

The solver asserts this on entry. Halving `ele_size` therefore means quartering
`dt`. Note that `x0` and `y0` are module constants here, not CLI flags.

### CPU

```bash
python src/heat_fenics.py --P 250 --y0 5.0 --ele_size 0.25 --dt 2.5e-3 \
    --rho 4.43e-3 --Cp 0.526 --k 0.0067 --eta 0.35 --emiss 0.40 \
    --outdir data/fem --tag _250W --no_vtk
```

Run it from the repository root; output paths are relative. `--no_vtk` skips the
ParaView files, which are large. Set `OMP_NUM_THREADS=1` when launching several
runs at once — MUMPS otherwise grabs every core in each process and the runs
spend their time contending rather than solving.

## Output

Both write `data{tag}.npy` under `--outdir`, one row per node per snapshot.

```
heat_torch.py    [x, y, z, t, P, T]     (6 columns)
heat_fenics.py   [x, y, z, t, T]        (5 columns)
```

Snapshots are taken every `--snap_dt` seconds. `heat_fenics.py` also writes
`u.pvd` / `mesh.pvd` for ParaView unless `--no_vtk` is given.

## Verification

The two solvers agree to 0.14% RMSE on a common `ele_size=0.5` mesh. The residual
is confined to the beam peak: FEniCS projects the Gaussian onto the CG1 basis,
which averages it over the surrounding elements, while the finite-difference
scheme samples it pointwise. Replacing the point sample with the exact cell
average closes 84% of the peak gap. Away from the beam the two agree to 0.03%.

Independent of that cross-check:

- the discrete surface integral of `q_laser` matches `eta*P` to machine precision;
- global energy balance, `dU/dt` against the closed-surface flux integral, closes
  to 0.007% and converges at second order under refinement;
- Heun converges at second order in time (measured 2.01–2.04).

Peak temperature is not mesh-converged on coarse grids. The thermal penetration
depth of the moving source is `alpha/v = 0.288` mm, so `ele_size=0.5` does not
resolve the boundary layer at all and the peak is 4% low. At `ele_size=0.125` the
peak error is about 0.35% of the temperature rise, judged against a Richardson
extrapolation of the solver's own refinement sequence.

## A note on the flux sign

The top-surface flux here is `q_laser - q_conv - q_rad`. Convection and radiation
cool the body. Reference implementations of this problem sometimes add all three,
which makes the `T^4` term a positive feedback: above a runaway threshold of
roughly `(k / (4 * sigma * emiss * dz))^(1/3)` the temperature diverges. For
Ti-6Al-4V that threshold is near 5290 K, and a 700 W scan fails to converge.
