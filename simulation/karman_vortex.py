import numpy as np
import h5py
import os
from phi.torch.flow import *
from phi.field import Field
from phi.math import Shape
from phi.torch import TORCH
from tqdm import tqdm

reynolds = np.arange(100, 10000, 200)
inlet_velocities = [5.0]
total_time = 250.0
dt = 0.5
skip_time = 150.0
total_steps = int(total_time // dt)
skip_steps = int(skip_time // dt)
n_frames = 20
n_steps_per_frame = (total_steps - skip_steps) // n_frames

DATA_ROOT = '../data/karman_vortex/'

os.makedirs(DATA_ROOT, exist_ok=True)

def to_centered_grid(field: Field):
    if isinstance(field, CenteredGrid):
        return field
    
    return CenteredGrid(field, resolution=field.shape.spatial, bounds=field.bounds)

def get_dim_order(shape: Shape):
    batch_names = shape.batch.names if (shape.batch_rank > 0) else ('batch', )
    channel_names = shape.channel.names if (shape.channel_rank > 0) else ('vector', )

    return batch_names + ('y', 'x') + channel_names

def to_numpy(field: Field):
    centered_field = to_centered_grid(field)
    shape_order = get_dim_order(centered_field.shape)

    return centered_field.values.numpy(order=shape_order).squeeze()


def kv_sim(
    re: float,
    vel_inlet: float,
    grid_size: tuple=(128, 128),
    domain_size: tuple=(128, 128),
    cylinder_radius: float=5,
    cylinder_center: tuple=(15, 64),
    ):
    filename = f"KarmannVortex_Ufree{inlet_v:.0f}_Re{re:.0f}.h5"
    data_store = h5py.File(DATA_ROOT + filename, 'a')
    data_store.require_dataset('velocity_x', shape=(n_frames, *grid_size), dtype=np.float64, compression='lzf')
    data_store.require_dataset('velocity_y', shape=(n_frames, *grid_size), dtype=np.float64, compression='lzf')
    data_store.require_dataset('pressure', shape=(n_frames, *grid_size), dtype=np.float64, compression='lzf')

    # Create domain
    velocity = StaggeredGrid((vel_inlet, 0.0), extrapolation=extrapolation.BOUNDARY,
                                 x=grid_size[0], y=grid_size[1], bounds=Box(x=domain_size[0], y=domain_size[1]))
    CYLINDER = Obstacle(geom.infinite_cylinder(x=cylinder_center[0], y=cylinder_center[1], radius=cylinder_radius, inf_dim=None))
    BOUNDARY_MASK = StaggeredGrid(Box(x=(-INF, 0.005), y=None), extrapolation=velocity.extrapolation, bounds=velocity.bounds, resolution=velocity.resolution)
    pressure = None

    @jit_compile
    def step(v, p, dt):
        v = advect.semi_lagrangian(v, v, dt)
        v = v * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * (vel_inlet, 0.0)
        v = diffuse.explicit(v, 1/re, dt)

        return fluid.make_incompressible(v, [CYLINDER], solve=Solve('auto', 1e-5, 0, x0=p))

    for i in tqdm(range(1, total_steps+1), desc=f"Re={re:.0f}, inlet_vel={vel_inlet:.1f}"):
        velocity, pressure = step(velocity, pressure, dt)
        if i % n_steps_per_frame == 0 and i > skip_steps:
            print(f"Saving frame {i} of {total_steps} for Re={re:.0f}, inlet_vel={vel_inlet:.1f}")
            data_store['velocity_x'][(i-skip_steps)//n_steps_per_frame - 1] = to_numpy(velocity['x'])[:, :-1]
            data_store['velocity_y'][(i-skip_steps)//n_steps_per_frame - 1] = to_numpy(velocity['y'])[:-1, :]
            data_store['pressure'][(i-skip_steps)//n_steps_per_frame - 1] = to_numpy(pressure)

    data_store.close()

if __name__ == '__main__':
    TORCH.set_default_device('GPU')
    for re in reynolds:
        for inlet_v in inlet_velocities:
            kv_sim(re, inlet_v)
