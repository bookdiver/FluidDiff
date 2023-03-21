from phi.flow import *

res = vis.control(7, (6, 8))
pressure_solver = vis.control('auto', ('auto', 'CG', 'CG-adaptive', 'CG-native', 'direct', 'GMres', 'lGMres', 'biCG', 'CGS', 'QMR', 'GCrotMK'))

BOUNDS = Box(x=16, y=16)

velocity = StaggeredGrid((0, 0), extrapolation.ZERO, x=2**res, y=2**res, bounds=BOUNDS)
smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=2**res, y=2**res, bounds=BOUNDS)
DT = 0.05

viewer = view(smoke, velocity, namespace=globals(), play=False)
viewer.info(f"Position source at ({source_x}, {source_y}), radius {source_radius}")
for _ in viewer.range(warmup=1):
    # Resize grids if needed
    INFLOW = Sphere(x=source_x, y=source_y, radius=source_radius)
    inflow = SoftGeometryMask(INFLOW) @ CenteredGrid(0, smoke.extrapolation, x=2**res, y=2**res, bounds=BOUNDS)
    smoke = smoke @ (inflow * DT)
    velocity = velocity @ StaggeredGrid(0, velocity.extrapolation, x=2**res, y=2**res, bounds=BOUNDS)
    # Physics step
    smoke = advect.mac_cormack(smoke, velocity, DT) + inflow
    buoyancy_force = smoke * (0, 0.1) @ velocity  # resamples smoke to velocity sample points
    velocity = advect.semi_lagrangian(velocity, velocity, DT) + buoyancy_force * DT
    try:
        with math.SolveTape() as solves:
            velocity, pressure = fluid.make_incompressible(velocity, (), Solve(pressure_solver, 1e-5, 0))
        viewer.log_scalars(solve_time=solves[0].solve_time)
        viewer.info(f"Presure solve {2**res}x{2**res} with {solves[0].method}: {solves[0].solve_time * 1000:.0f} ms ({solves[0].iterations} iterations)")
    except ConvergenceException as err:
        viewer.info(f"Presure solve {2**res}x{2**res} with {err.result.method}: {err}\nMax residual: {math.max(abs(err.result.residual.values))}")
        velocity -= field.spatial_gradient(err.result.x, velocity.extrapolation, type=type(velocity))