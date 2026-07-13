import fenics as fe
import numpy as np
import os
import argparse

try:
    from tqdm import trange
except ImportError:
    def trange(n, **kwargs):
        class _Bar:
            def __iter__(self):
                return iter(range(n))
            def set_postfix(self, **kw):
                pass
        return _Bar()


def simulation(P=500., vel=10., total_t=3., x0=5., y0=5., outdir='data/bareplate', tag='',
               save_vtk=True, rho=8.e-3, Cp=0.5, k=0.01, h=2e-5, eta=0.4, emiss=0.3,
               ele_size=0.5, dt=1e-2, snap_dt=0.1, verbose=False, pos=0):
    # dolfin's Newton log would shred the progress bar
    fe.set_log_level(fe.LogLevel.INFO if verbose else fe.LogLevel.WARNING)
    domain_x = 40.
    domain_y = 10.
    domain_z = 6.
    ambient_T = 298.
    r = 1.5
    Rboltz = 5.6704e-14

    EPS = 1e-8
    ts = np.arange(0., total_t + dt, dt)

    # snapshots every snap_dt seconds, regardless of the time-step size
    snap_every = max(1, int(round(snap_dt/dt)))

    # Building mesh, see https://fenicsproject.org/olddocs/dolfin/1.5.0/python/programmers-reference/cpp/mesh/BoxMesh.html
    mesh = fe.BoxMesh(fe.Point(0., 0., 0.), fe.Point(domain_x, domain_y, domain_z), 
                      round(domain_x/ele_size), round(domain_y/ele_size), round(domain_z/ele_size))

    # Save mesh to local file, optional, just for inspection
    os.makedirs(outdir, exist_ok=True)
    if save_vtk:
        mesh_file = fe.File(f'{outdir}/mesh{tag}.pvd')
        mesh_file << mesh

    # Define bottom surface 
    class Bottom(fe.SubDomain):
        def inside(self, x, on_boundary):
            # The condition for a point x to be on bottom side is that x[2] < EPS
            return on_boundary and x[2] < EPS

    # Define top surface
    class Top(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[2] > domain_z - EPS

    # Define the other four surfaces
    class SurroundingSurfaces(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (x[0] < EPS or x[0] > domain_x - EPS or x[1] < EPS or x[1] > domain_y - EPS)

    # The following few lines mark different boundaries with different numbers
    # For example, the top surface is marked with the integer number 2
    bottom = Bottom()
    top = Top()
    surrounding_surfaces = SurroundingSurfaces()
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    bottom.mark(boundaries, 1)
    top.mark(boundaries, 2)
    surrounding_surfaces.mark(boundaries, 3)
    ds = fe.Measure('ds')(subdomain_data=boundaries)

    # Define FEM function space to be first order continuous Galerkin (the most commonly used)
    V = fe.FunctionSpace(mesh, 'CG', 1)

    # u_crt is the temperature field we want to solve 
    u_crt = fe.Function(V)

    # u_pre is the temperature from the previous step
    # We initialize u_pre to be a constant field = ambient_T (assign initial values)
    u_pre = fe.interpolate(fe.Constant(ambient_T), V)

    # v is the test function in FEM
    v = fe.TestFunction(V)

    # If theta = 0., we recover implicit Eulear; if theta = 1., we recover explicit Euler; theta = 0.5 seems to be a good choice.
    theta = 0.5
    u_rhs = theta*u_pre + (1 - theta)*u_crt

    # Define Dirichlet boundary conditions for the bottom surface to be always at ambient temperature
    bcs = [fe.DirichletBC(V, fe.Constant(ambient_T), bottom)]

    # Define the laser heat source, note that t is a changeble parameter
    class LaserExpression(fe.UserExpression):
        def __init__(self, t):
            # Construction method of base class has to be called first
            super(LaserExpression, self).__init__()
            self.t = t

        def eval(self, values, x):
            t = self.t
            values[0] = 2*P*eta/(np.pi*r**2) * np.exp(-2*((x[0] - x0 - vel*t)**2 + (x[1] - y0)**2)/r**2)
    
        def value_shape(self):
            return ()

    q_laser = LaserExpression(None)
    q_convection = h * (u_rhs - ambient_T)
    q_radiation = Rboltz * emiss * (u_rhs**4 - ambient_T**4)

    # These are k*grad(u).n, i.e. the flux *into* the body: the laser deposits heat
    # while convection and radiation carry it away.
    # For the top surface, we will consider both convection and laser heating
    q_top = q_laser - q_convection - q_radiation
    # For the four side surfaces, we will only consider convection
    q_surr = -q_convection - q_radiation

    # Deine the weak form residual
    # For the terms with fe.dx, they are volume integrals
    # Note that ds(2) means that it is a surface integral only computed on surface number 2 (the top surface), which we defined previously!
    residual = rho*Cp/dt*(u_crt - u_pre) * v * fe.dx + k * fe.dot(fe.grad(u_rhs), fe.grad(v)) * fe.dx \
                - q_top * v * ds(2) - q_surr * v * ds(3)

    # Open a pvd file to store results
    if save_vtk:
        u_vtk_file = fe.File(f'{outdir}/u{tag}.pvd')

        # Store solution at the 0th step
        u_vtk_file << u_pre

    # CG1 dof ordering matches vector().get_local(), so coordinates can be tabulated once
    dof_coords = V.tabulate_dof_coordinates().reshape((-1, 3))

    def snapshot(t):
        n = dof_coords.shape[0]
        return np.hstack((dof_coords,
                          np.full((n, 1), t),
                          u_pre.vector().get_local().reshape((n, 1))))

    snapshots = [snapshot(ts[0])]

    steps = trange(len(ts) - 1, desc=f'P={P:.0f}W', unit='step',
                   mininterval=2.0, dynamic_ncols=True, ascii=True,
                   position=pos, leave=True)

    for i in steps:
        # Update the time parameter in laser
        q_laser.t = theta*ts[i] + (1 - theta)*ts[i + 1]

        # Solve the problem at this time step
        solver_parameters = {'newton_solver': {'maximum_iterations': 20, 'linear_solver': 'mumps'}}
        fe.solve(residual == 0, u_crt, bcs, solver_parameters=solver_parameters)

        # After solving, update u_pre so that it is equal to the newly solved u_crt
        u_pre.assign(u_crt)
        if (i+1) % snap_every == 0:
            # Store solution at this step
            if save_vtk:
                u_vtk_file << u_pre
            snapshots.append(snapshot(ts[i + 1]))
            steps.set_postfix(t=f'{ts[i+1]:.2f}s', Tmax=f'{u_pre.vector().max():.0f}K')

    data = np.vstack(snapshots)
    out = f'{outdir}/data{tag}.npy'
    np.save(out, data)
    print(f'saved {out}  shape={data.shape}  P={P} vel={vel} '
          f'T range=[{data[:,4].min():.1f}, {data[:,4].max():.1f}] K')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--P', type=float, default=500.)
    parser.add_argument('--vel', type=float, default=10.)
    parser.add_argument('--total_t', type=float, default=3.)
    parser.add_argument('--x0', type=float, default=5.)
    parser.add_argument('--y0', type=float, default=5.)
    parser.add_argument('--outdir', type=str, default='data/bareplate')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--no_vtk', action='store_true', help='skip .pvd/.vtu output')
    # material properties, mm-g-s units; defaults are the original steel values
    parser.add_argument('--rho', type=float, default=8.e-3, help='density g/mm^3')
    parser.add_argument('--Cp', type=float, default=0.5, help='specific heat J/(g.K)')
    parser.add_argument('--k', type=float, default=0.01, help='conductivity W/(mm.K)')
    parser.add_argument('--h', type=float, default=2e-5, help='convection W/(mm^2.K)')
    parser.add_argument('--eta', type=float, default=0.4, help='laser absorptivity')
    parser.add_argument('--emiss', type=float, default=0.3, help='emissivity')
    # discretisation
    parser.add_argument('--ele_size', type=float, default=0.5, help='element size mm')
    parser.add_argument('--dt', type=float, default=1e-2, help='time step s')
    parser.add_argument('--snap_dt', type=float, default=0.1, help='snapshot interval s')
    parser.add_argument('--verbose', action='store_true', help='show dolfin Newton log')
    parser.add_argument('--pos', type=int, default=0, help='tqdm bar row, for parallel runs')
    a = parser.parse_args()
    simulation(P=a.P, vel=a.vel, total_t=a.total_t, x0=a.x0, y0=a.y0,
               outdir=a.outdir, tag=a.tag, save_vtk=not a.no_vtk,
               rho=a.rho, Cp=a.Cp, k=a.k, h=a.h, eta=a.eta, emiss=a.emiss,
               ele_size=a.ele_size, dt=a.dt, snap_dt=a.snap_dt, verbose=a.verbose,
               pos=a.pos)
