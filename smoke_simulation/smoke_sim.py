import argparse
import yaml
import logging

import numpy as np
from tqdm import tqdm
import phi.torch.flow as flow
import phi.math as pmath

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="The scale of dataset, can be chosen from 'small', 'medium', 'large'", default='small')

args = parser.parse_args()

def interpolate2d(x: np.ndarray) -> np.ndarray:
    """Interpolate a 2d array for each batch and vector component"""
    x_0 = ((x[:, :-1, :-1, 0] + x[:, :-1, 1:, 0] + x[:, 1:, :-1, 0] + x[:, 1:, 1:, 0]) / 4)[..., np.newaxis]
    x_1 = ((x[:, :-1, :-1, 1] + x[:, :-1, 1:, 1] + x[:, 1:, :-1, 1] + x[:, 1:, 1:, 1]) / 4)[..., np.newaxis]
    return np.concatenate((x_1, x_0), axis=-1)

class Smoke:
    def __init__(self, params: dict):
        self.resolution_x = params['resolution_x']
        self.resolution_y = params['resolution_y']
        self.time_step = params['time_step']
        self.num_frames = params['num_frames']

        self.num_src_pos_x = params['num_src_pos_x']
        self.num_src_pos_y = params['num_src_pos_y']
        self.batch_size = self.num_src_pos_x * self.num_src_pos_y

        self.src_radius = params['src_radius']
        self.src_strength = params['src_strength']
        self.src_pos_list = [(i, j) for i in np.linspace(3*self.src_radius, self.resolution_x-3*self.src_radius, self.num_src_pos_x, dtype=int)
                        for j in np.linspace(3*self.src_radius, self.resolution_y//2-3*self.src_radius, self.num_src_pos_y, dtype=int)]
        self.src_strength = params['src_strength']
        
        self.log_density = np.empty((self.num_frames, self.batch_size, self.resolution_y, self.resolution_x))
        self.log_velocity = np.empty((self.num_frames, self.batch_size, self.resolution_y, self.resolution_x, 2))
        # condition includes three components: time, src_x_pos, src_y_pos
        self.log_condition = np.empty((self.num_frames, self.batch_size, 3))
        self.data_save_path = params['data_save_path']
        self.params = params
    
    def _create_source(self) -> flow.CenteredGrid:

        inflow_loc = flow.tensor(self.src_pos_list, flow.batch('inflow_loc'), flow.channel(vector='x, y'))
        inflow = self.src_strength * flow.CenteredGrid(flow.Sphere(center=inflow_loc, radius=self.src_radius), 
                                                        flow.extrapolation.BOUNDARY, x=self.resolution_x, y=self.resolution_y)
        return inflow

    def _init_fields(self) -> None:
        self.src = self._create_source()
        self.density = flow.CenteredGrid(0, extrapolation=flow.extrapolation.BOUNDARY, x=self.resolution_x, y=self.resolution_y,
                                        bounds=flow.Box(x=self.resolution_x, y=self.resolution_y))
        self.velocity = flow.StaggeredGrid(0, extrapolation=flow.extrapolation.ZERO, x=self.resolution_x, y=self.resolution_y,
                                            bounds=flow.Box(x=self.resolution_x, y=self.resolution_y))
    def step(self) -> None:
        self.density = flow.advect.semi_lagrangian(self.density, self.velocity, dt=self.time_step) + self.src
        buoyancy = (self.density * (0, 1)).at(self.velocity)
        self.velocity = flow.advect.semi_lagrangian(self.velocity, self.velocity, dt=self.time_step) + buoyancy * self.time_step
        self.velocity, _ = flow.fluid.make_incompressible(self.velocity, (), flow.Solve('auto', 1e-4, 1e-4, x0=None, max_iterations=1500))
    
    def simulate(self) -> None:
        pbar = tqdm(range(self.num_frames))
        self._init_fields()
        for i_frame in pbar:
            self.step()
            current_time = i_frame * self.time_step
            self.log_density[i_frame] = self.density.values.numpy('inflow_loc, y, x')
            self.log_velocity[i_frame] = interpolate2d(self.velocity.staggered_tensor().numpy('inflow_loc, y, x, vector'))
            self.log_condition[i_frame] = np.array([[current_time, src_x_pos, src_y_pos] for src_x_pos, src_y_pos in self.src_pos_list])
            pbar.set_description(f"Frame {i_frame}")
    
    def save(self) -> None:
        np.savez(self.data_save_path, 
                log_params=self.params,
                log_density=self.log_density,
                log_velocity=self.log_velocity,
                log_condition=self.log_condition)
        logging.info(f"Data saved to {self.data_save_path}")
        logging.info(f"Data size: {self.log_density.shape[0]} frames, {self.log_density.shape[1]} batches, {self.log_density.shape[2]} x {self.log_density.shape[3]} grid")
        
if __name__ == '__main__':
    flow.TORCH.set_default_device('GPU')
    with open('./config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.config == 'small':
        params = config['small']
    elif args.config == 'medium':
        params = config['medium']
    elif args.config == 'large':
        params = config['large']
    else:
        raise ValueError('Invalid config')
    smoke = Smoke(params)
    smoke.simulate()
    smoke.save()
        


