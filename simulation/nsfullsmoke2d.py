import logging
import os

import h5py
import numpy as np
from phi.torch.flow import (  # SoftGeometryMask,; 
    Noise,
    batch,
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

def generate_trajectories_fullsmoke(
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
    logger.info("Experiment: 2D smoke simulation with different initial scens")
    logger.info(f"Mode: {mode}")
    logger.info(f"Number of samples: {pde.n_samples}")

    save_name = os.path.join(dirname, "_".join(["NSFullSmoke2D", "resolution", f"{pde.nx}x{pde.ny}", mode]))
    h5f = h5py.File("".join([save_name, ".h5"]), "a")
    dataset = h5f.create_group(pde_string)

    tcoord = {}
    h5f_u_init, h5f_u, h5f_vx, h5f_vy= {}, {}, {}, {}

    nt, nx, ny = pde.grid_size['t'], pde.grid_size['x'], pde.grid_size['y']
    # The scalar field u, the components of the vector field vx, vy,
    # the coordinations (tcoord, xcoord, ycoord) and dt, dx, dt are saved
    h5f_u = dataset.create_dataset("u", (pde.n_samples, nt, nx, ny), dtype=float)
    h5f_u_init = dataset.create_dataset("u_init", (pde.n_samples, nx, ny), dtype=float)
    h5f_vx = dataset.create_dataset("vx", (pde.n_samples, nt, nx, ny), dtype=float)
    h5f_vy = dataset.create_dataset("vy", (pde.n_samples, nt, nx, ny), dtype=float)
    tcoord = dataset.create_dataset("t", (pde.n_samples, nt), dtype=float)

    def genfunc():
        smoke = CenteredGrid(
            Noise((batch(n_exp=pde.n_samples)), scale=4.0, smoothness=1.5),
            extrapolation.BOUNDARY,
            x=pde.nx,
            y=pde.ny,
            bounds=Box(x=pde.Lx, y=pde.Ly),
        )  # sampled at cell centers
        init_rho_ = reshaped_native(
            smoke.values,
            groups=("n_exp", "y", "x"),
            to_numpy=True
        )
        velocity = StaggeredGrid(
            0, 
            extrapolation.ZERO, 
            x=pde.nx, 
            y=pde.ny, 
            bounds=Box(x=pde.Lx, y=pde.Ly)
        )  # sampled in staggered form at face centers
        rho_ = []
        velocity_ = []
        for i in tqdm(range(0, pde.nt)):
            smoke = advect.semi_lagrangian(smoke, velocity, pde.dt)
            buoyancy_force = (smoke * (0.0, pde.buoyancy)).at(velocity)  # resamples smoke to velocity sample points
            velocity = advect.semi_lagrangian(velocity, velocity, pde.dt) + pde.dt * buoyancy_force
            velocity = diffuse.explicit(velocity, pde.nu, pde.dt)
            velocity, _ = fluid.make_incompressible(velocity)
            rho_.append(
                reshaped_native(
                    smoke.values, 
                    groups=("n_exp", "y", "x", "vector"), 
                    to_numpy=True)
                )
            velocity_.append(
                reshaped_native(
                    velocity.staggered_tensor(),
                    groups=("n_exp", "y", "x", "vector"),
                    to_numpy=True
                )
            )

        rho_ = np.asarray(rho_[pde.skip_nt :]).transpose((1, 0, 2, 3, 4)).squeeze()
        velocity_corrected_ = np.asarray(velocity_[pde.skip_nt :]).transpose((1, 0, 2, 3, 4))[:, :, :-1, :-1, :]
        init_rho_ = np.asarray(init_rho_)
        return (rho_[:, :: pde.t_sample_rate, ...], 
                velocity_corrected_[:, :: pde.t_sample_rate, ...], 
                init_rho_)

    rho, velocity_corrected, init_rho = genfunc()

    for idx in range(pde.n_samples):
        # fmt: off
        # Saving the trajectories
        h5f_u[idx, ...] = rho[idx, ...]
        h5f_u_init[idx, ...] = init_rho[idx, ...]
        h5f_vx[idx, ...] = velocity_corrected[idx][..., 0]
        h5f_vy[idx, ...] = velocity_corrected[idx][..., 1]
        # fmt:on
        tcoord[idx, ...] = np.linspace(pde.skip_t, pde.tmax, pde.trajlen+1)[1:]


    print()
    print("Data saved")
    print()
    h5f.close()