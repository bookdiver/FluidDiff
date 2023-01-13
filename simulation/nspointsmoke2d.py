import logging
import os

import h5py
import numpy as np
from phi.torch.flow import (  # SoftGeometryMask,; 
    Sphere,
    batch,
    tensor,
    channel,
    Box,
    CenteredGrid,
    StaggeredGrid,
    advect,
    diffuse,
    extrapolation,
    fluid,
    TORCH
)
from phi.math import reshaped_native
from tqdm import tqdm

from pde import PDEConfig

logger = logging.getLogger(__name__)

def generate_trajectories_pointsmoke(
    pde: PDEConfig,
    mode: str,
    dirname: str = "../data"
) -> None:
    """
    Generate data trajectories for smoke inflow in bounded domain
    Args:
        pde (PDE): pde at hand [NS2D]
        pde.samples (int): how many trajectories do we create
    Returns:
        None
    """

    TORCH.set_default_device('GPU')

    pde_string = str(pde)
    logger.info(f"Equation: {pde_string}")
    logger.info("Experiment: 2D smoke simulation with different source locations")
    logger.info(f"Mode: {mode}")
    logger.info(f"Number of samples: {pde.n_samples}")

    save_name = os.path.join(dirname, "_".join(["NSPointSmoke2D", "resolution", f"{pde.nx}x{pde.ny}", mode]))
    h5f = h5py.File("".join([save_name, ".h5"]), "a")

    tcoord = {}
    h5f_u, h5f_vx, h5f_vy, h5f_src = {}, {}, {}, {}

    nt, nx, ny = pde.grid_size['t'], pde.grid_size['x'], pde.grid_size['y']
    # The scalar field u, the components of the vector field vx, vy,
    # the coordinations (tcoord, xcoord, ycoord) and dt, dx, dt are saved
    h5f_u = h5f.create_dataset("u", (pde.n_samples, nt, nx, ny), dtype=float)
    h5f_vx = h5f.create_dataset("vx", (pde.n_samples, nt, nx, ny), dtype=float)
    h5f_vy = h5f.create_dataset("vy", (pde.n_samples, nt, nx, ny), dtype=float)
    h5f_src = h5f.create_dataset("src", (pde.n_samples, nx, ny), dtype=float)
    tcoord = h5f.create_dataset("t", (pde.n_samples, nt), dtype=float)
    dt = h5f.create_dataset("dt", (pde.n_samples,), dtype=float)

    def genfunc():
        src_locs = np.asarray(pde.source_coord)
        inflow_locs = tensor(src_locs, batch('inflow_loc'), channel(vector='x, y'))
        inflow = CenteredGrid(
            Sphere(center=inflow_locs, radius=pde.source_radius),
            extrapolation.BOUNDARY,
            x=pde.nx,
            y=pde.ny,
            bounds=Box(x=pde.Lx, y=pde.Ly)
        )
        smoke = CenteredGrid(
            0,
            extrapolation.BOUNDARY,
            x=pde.nx,
            y=pde.ny,
            bounds=Box(x=pde.Lx, y=pde.Ly),
        )  # sampled at cell centers
        velocity = StaggeredGrid(
            0, 
            extrapolation.ZERO, 
            x=pde.nx, 
            y=pde.ny, 
            bounds=Box(x=pde.Lx, y=pde.Ly)
        )  # sampled in staggered form at face centers
        fluid_field_ = []
        velocity_ = []
        for i in tqdm(range(0, pde.nt)):
            smoke = advect.semi_lagrangian(smoke, velocity, pde.dt) + pde.source_strength * inflow * pde.dt
            buoyancy_force = (smoke * (0.0, pde.buoyancy)).at(velocity)  # resamples smoke to velocity sample points
            velocity = advect.semi_lagrangian(velocity, velocity, pde.dt) + pde.dt * buoyancy_force
            velocity = diffuse.explicit(velocity, pde.nu, pde.dt)
            velocity, _ = fluid.make_incompressible(velocity)
            fluid_field_.append(
                reshaped_native(
                    smoke.values, 
                    groups=("inflow_loc", "y", "x", "vector"), 
                    to_numpy=True)
                )
            velocity_.append(
                reshaped_native(
                    velocity.staggered_tensor(),
                    groups=("inflow_loc", "y", "x", "vector"),
                    to_numpy=True
                )
            )
        init_field_ = reshaped_native(
            inflow.values,
            groups=("inflow_loc", "y", "x", "vector"),
            to_numpy=True
        )

        fluid_field_ = np.asarray(fluid_field_[pde.skip_nt :]).transpose((1, 0, 2, 3, 4)).squeeze()
        velocity_corrected_ = np.asarray(velocity_[pde.skip_nt :]).transpose((1, 0, 2, 3, 4)).squeeze()[:, :, :-1, :-1, :]
        init_field_ = np.asarray(init_field_).squeeze()
        return (fluid_field_[:, :: pde.t_sample_rate, ...], 
                velocity_corrected_[:, :: pde.t_sample_rate, ...], 
                init_field_)

    fluid_field, velocity_corrected, init_field = genfunc()

    for idx in range(pde.n_samples):
        # fmt: off
        # Saving the trajectories
        h5f_u[idx, ...] = fluid_field[idx, ...]
        h5f_vx[idx, ...] = velocity_corrected[idx][..., 0]
        h5f_vy[idx, ...] = velocity_corrected[idx][..., 1]
        h5f_src[idx, ...] = init_field[idx, ...]
        # fmt:on
        tcoord[idx, ...] = np.asarray([np.linspace(pde.skip_t, pde.tmax, pde.trajlen)])
        dt[idx] = pde.dt * pde.sample_rate


    print()
    print("Data saved")
    print()
    print()
    h5f.close()