import os

import numpy as np
import h5py
import os
from phi.torch.flow import *
from phi.field import Field
from phi.math import Shape
from phi.torch import TORCH
from tqdm import tqdm

bouyancy = 0.1
source_x_range = np.arange(2.0, 14.0+0.25, 0.25)
source_y_range = np.arange(2.0, 14.0+0.25, 0.25)
xx, yy = np.meshgrid(source_x_range, source_y_range)
points = [[x, y] for x, y in zip(xx.flatten(), yy.flatten())]
n_scenes = 500
n_chunk = 10
n_scenes_per_chunk = n_scenes // n_chunk

np.random.seed(215)
selected_scenes = np.random.choice(len(points), n_scenes, replace=False)
points = [points[i] for i in selected_scenes]
chunked_points = [points[i:i+n_scenes_per_chunk] for i in range(0, len(points), n_scenes_per_chunk)]

DOMAIN = Box(x=16, y=16)
dt = 0.2
total_time = 40.0
n_steps = int(total_time / dt)

DATA_ROOT = '../data/smoke_plume_64x64/'

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


def sp_sim(
    source_radius: float,
    grid_size: tuple=(64, 64),
    n_experiment: int=0
    ):
    filename = f"SmokePlume_R{source_radius:.2f}_{n_experiment}.h5"
    data_store = h5py.File(DATA_ROOT + filename, 'a')
    data_store.require_dataset('source', shape=(n_scenes_per_chunk, *grid_size), dtype=np.float64, compression='lzf')
    data_store.require_dataset('density', shape=(n_steps, n_scenes_per_chunk, *grid_size), dtype=np.float64, compression='lzf')
    data_store.require_dataset('vorticity', shape=(n_steps, n_scenes_per_chunk, *grid_size), dtype=np.float64, compression='lzf')
    data_store.require_dataset('pressure', shape=(n_steps, n_scenes_per_chunk, *grid_size), dtype=np.float64, compression='lzf')

    INFLOW_LOCS = tensor(chunked_points[n_experiment], batch('scene'), channel(vector='x, y'))
    INFLOWS = Sphere(center=INFLOW_LOCS, radius=source_radius)

    velocity = StaggeredGrid((0, 0), extrapolation=extrapolation.ZERO, x=grid_size[0], y=grid_size[1], bounds=DOMAIN)
    density = CenteredGrid(0, extrapolation=extrapolation.BOUNDARY, x=grid_size[0], y=grid_size[1], bounds=DOMAIN)
    inflow_mask = SoftGeometryMask(INFLOWS) @ CenteredGrid(0, density.extrapolation, x=grid_size[0], y=grid_size[1], bounds=DOMAIN)
    pressure = None

    data_store['source'][:] = to_numpy(inflow_mask)

    def step(rho, v, p, dt):
        rho = advect.mac_cormack(rho, v, dt) + inflow_mask * dt
        bouyancy_force = rho * (0, 0.1) @ v
        v = advect.semi_lagrangian(v, v, dt) + bouyancy_force * dt
        try:
            v, p = fluid.make_incompressible(v, (), Solve('auto', 1e-5, 0, x0=p))
        except ConvergenceException as err:
            v -= field.spatial_gradient(err.result.x, v.extrapolation, type=type(velocity)) * dt

        vortex = field.curl(v)
        return rho, v, p, vortex

    for i in tqdm(range(1, n_steps+1), desc=f"Radius={source_radius:.2f}"):
        density, velocity, pressure, vortex = step(density, velocity, pressure, dt)
        data_store['density'][i-1] = to_numpy(density)
        data_store['vorticity'][i-1] = to_numpy(vortex)[..., :-1, :-1]
        data_store['pressure'][i-1] = to_numpy(pressure)

    data_store.close()

if __name__ == '__main__':
    TORCH.set_default_device('GPU')
    # for r in [0.5, 1.0, 1.5, 2.0]:
    for i in range(n_chunk):
        sp_sim(0.5, n_experiment=i)
        print(f"Chunk {i+1}/{n_chunk} done.")