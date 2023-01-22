import logging
import yaml
import numpy as np
from phi.torch.flow import (  # SoftGeometryMask,; 
    Noise,
    Box,
    CenteredGrid,
    StaggeredGrid,
    Solve,
    advect,
    diffuse,
    extrapolation,
    fluid,
    batch
)
from phi.torch import TORCH
from tqdm import tqdm

import data_io


logger = logging.getLogger(__name__)

def call_many(fns, *args, **kwargs):
    for fn in fns:
        fn(*args, **kwargs)

def incomp_ns_sim(
            sim_name: str='incomp_2d_ns',
            domain_size: list=[1.0, 1.0],
            nu: float=0.01,
            bouyancy: float=0.5,
            scale: float=10.0,
            smoothness: float=3.0,
            grid_size: list=[64, 64],
            dt: float=0.01,
            n_steps: int=1000,
            t_sample_interval: int=1,
            n_batch: int=1,
            config: dict={}):
        
    def cauchy_momentum_step(velocity, density, pressure, nu, dt, obstacles=None) -> tuple:
        if obstacles is None:
            obstacles = ()

        density = advect.semi_lagrangian(density, velocity, dt)

        bouyancy_force = (density * (0.0, bouyancy)).at(velocity)
        
        velocity = advect.semi_lagrangian(velocity, velocity, dt) + bouyancy_force * dt
        velocity = diffuse.explicit(velocity, nu, dt)

        velocity, pressure = fluid.make_incompressible(velocity, solve=Solve('CG-adaptive', 1e-3, 1e-5, max_iterations=2000, x0=None))

        return (velocity, density, pressure)

    TORCH.set_default_device('GPU')

    callbacks = []
    cleanups = []

    data_store = data_io.h5_for(config)
    h5_path = data_store.filename
    def _store(frame_i, t, density, velocity, pressure, **kwargs):
        data_store['density'][:, frame_i, ...] = data_io.to_ndarray(density)
        data_store['velocity'][:, frame_i, ...] = data_io.to_ndarray(velocity)
        data_store['pressure'][:, frame_i, ...] = data_io.to_ndarray(pressure)
        data_store['t'][:, frame_i] = t
        data_store.attrs['latestIndex'] = frame_i
    
    callbacks.append(_store)
    cleanups.append(lambda *args, **kwargs: data_store.close())
    
    # Initialization of density, velocity, force
    density = CenteredGrid(
        Noise(batch(batch=n_batch),
              scale=scale,
              smoothness=smoothness),
        extrapolation=extrapolation.BOUNDARY,
        x=grid_size[0],
        y=grid_size[1],
        bounds=Box(x=domain_size[0], y=domain_size[1])
    )

    velocity = StaggeredGrid(
        0,
        extrapolation=extrapolation.ZERO,
        x=grid_size[0],
        y=grid_size[1],
        bounds=Box(x=domain_size[0], y=domain_size[1])
    )

    pressure = None
    
    data_store['initial_density'][:, ...] = data_io.to_ndarray(density)

    # Simulation
    ts = np.linspace(dt, n_steps*dt, n_steps, endpoint=True)

    def sim_step(density, velocity, pressure):
        return cauchy_momentum_step(velocity, density, pressure, nu, dt)

    for step, t in enumerate(tqdm(ts), start=1):
        velocity, density, pressure = sim_step(density, velocity, pressure)

        if step % t_sample_interval == 0:
            frame_i = step // t_sample_interval
            logger.info(f'Saving frame {frame_i} at step {step}, t={t}')
            call_many(callbacks, 
                    frame_i=frame_i-1, 
                    t=t, 
                    density=density, 
                    velocity=velocity, 
                    pressure=pressure)
    
    call_many(cleanups)


if __name__ == '__main__':
    TORCH.set_default_device('GPU')
    config = yaml.safe_load(open('./configs/ns_incomp.yaml'))
    incomp_ns_sim(**config, config=config)

    




    



