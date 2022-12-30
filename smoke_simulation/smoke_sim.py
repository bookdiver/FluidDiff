import argparse
import logging

import numpy as np
from tqdm import tqdm
import phi.torch.flow as flow

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--save-name', type=str, help='Name of the experiment, should follow the format of \
res(resoluation)_dt(time_step)_t(total_time)_nsrc(number of sources)')

args = parser.parse_args()

def interpolate2d(x: np.ndarray) -> np.ndarray:
    """Interpolate a 2d array for each batch and vector component"""
    x_0 = ((x[:, :-1, :-1, 0] + x[:, :-1, 1:, 0] + x[:, 1:, :-1, 0] + x[:, 1:, 1:, 0]) / 4)[..., np.newaxis]
    x_1 = ((x[:, :-1, :-1, 1] + x[:, :-1, 1:, 1] + x[:, 1:, :-1, 1] + x[:, 1:, 1:, 1]) / 4)[..., np.newaxis]
    return np.concatenate((x_1, x_0), axis=-1)

RESOLUTION = (64, 64)
TIME_STEP = 1
TOTAL_TIME = 100
SRC_RADIUS = 2
SRC_STRENGTH = 1

NUM_SRC = 16 * 16
NUM_FRAMES = TOTAL_TIME // TIME_STEP

SAVE_ROOT = f"/media/bamf-big/gefan/DiffFluids/data/smoke/{args.save_name}"


source_positions = [(i, j)
                    for i in np.linspace(SRC_RADIUS, RESOLUTION[0]-SRC_RADIUS, 16, dtype=int)
                    for j in np.linspace(SRC_RADIUS, RESOLUTION[1]-SRC_RADIUS, 16, dtype=int)]
source_positions = np.asarray(source_positions)

def create_source() -> flow.CenteredGrid:
    inflow_loc = flow.tensor(source_positions, flow.batch('inflow_loc'), flow.channel(vector='x, y'))
    inflow = SRC_STRENGTH * flow.CenteredGrid(flow.Sphere(center=inflow_loc, radius=SRC_RADIUS), 
                                                flow.extrapolation.BOUNDARY, x=RESOLUTION[0], y=RESOLUTION[1])
    return (inflow_loc, inflow)

def init_fields() -> None:
    src_loc, src = create_source()
    density = flow.CenteredGrid(0, extrapolation=flow.extrapolation.BOUNDARY, x=RESOLUTION[0], y=RESOLUTION[1],
                                bounds=flow.Box(x=RESOLUTION[0], y=RESOLUTION[1]))
    velocity = flow.StaggeredGrid(0, extrapolation=flow.extrapolation.ZERO, x=RESOLUTION[0], y=RESOLUTION[1],
                                    bounds=flow.Box(x=RESOLUTION[0], y=RESOLUTION[1]))
    return density, velocity, src_loc, src

def advect_fields(density: flow.CenteredGrid, velocity: flow.StaggeredGrid, src: flow.CenteredGrid) -> tuple:
    density = flow.advect.semi_lagrangian(density, velocity, dt=TIME_STEP) + src
    buoyancy = (density *(0, 1)).at(velocity)
    velocity = flow.advect.semi_lagrangian(velocity, velocity, dt=TIME_STEP) + buoyancy * TIME_STEP
    velocity, _ = flow.fluid.make_incompressible(velocity)
    return (density, velocity)

def extract_data(density: flow.CenteredGrid, velocity: flow.StaggeredGrid, current_time: float, src_loc: flow.tensor) -> tuple:
    density_data = density.values.numpy('inflow_loc, y, x')
    # shape: (NUM_SRC, RESOLUTION[0], RESOLUTION[1])
    velocity_data = interpolate2d(velocity.staggered_tensor().numpy('inflow_loc, y, x, vector'))
    # shape: (NUM_SRC, RESOLUTION[0], RESOLUTION[1], 2)
    src_data = src_loc.numpy('inflow_loc, vector')
    time_data = np.full((NUM_SRC, 2), current_time, dtype=np.float32)
    time_data[:, 1] = time_data[:, 1] / TOTAL_TIME # normalize time
    param_data = np.concatenate((time_data, src_data), axis=-1)
    return (density_data, velocity_data, param_data)
    # the output shape of each component is:
    # density_data: (NUM_SRC, RESOLUTION[0], RESOLUTION[1])
    # velocity_data: (NUM_SRC, RESOLUTION[0], RESOLUTION[1], 2)
    # param_data: (NUM_SRC, 4) (normalized time, time, x, y)

def main():
    density, velocity, src_loc, src = init_fields()
    for frame in tqdm(range(NUM_FRAMES+1)):
        density, velocity = advect_fields(density, velocity, src)
        density_data, velocity_data, param_data = extract_data(density, velocity, frame*TIME_STEP, src_loc)
        if frame == 0:
            density_data_all = density_data
            velocity_data_all = velocity_data
            param_data_all = param_data
        else:
            density_data_all = np.concatenate((density_data_all, density_data), axis=0)
            velocity_data_all = np.concatenate((velocity_data_all, velocity_data), axis=0)
            param_data_all = np.concatenate((param_data_all, param_data), axis=0)
    
    density_data_all = density_data_all.reshape(((NUM_FRAMES+1)*NUM_SRC, *RESOLUTION))
    velocity_data_all = velocity_data_all.reshape(((NUM_FRAMES+1)*NUM_SRC, *RESOLUTION, 2))
    param_data_all = param_data_all.reshape(((NUM_FRAMES+1)*NUM_SRC, 4))
    
    logging.info(f"Density data shape: {density_data_all.shape}")
    logging.info(f"Velocity data shape: {velocity_data_all.shape}")
    logging.info(f"Param data shape: {param_data_all.shape}")
    np.savez(
        SAVE_ROOT,
        density=density_data_all,
        velocity=velocity_data_all,
        param=param_data_all
    )
    logging.info(f"Data saved to {SAVE_ROOT}")
        
if __name__ == '__main__':
    flow.TORCH.set_default_device('GPU')
    main()
        


