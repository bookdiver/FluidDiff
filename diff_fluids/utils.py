import torch
from torch.utils.data import Dataset
# from torchvision.utils import make_grid
import numpy as np
# import matplotlib.pyplot as plt


class MyDataSet(Dataset):
    """ 
    Customized dataset for smoke simulation, the data npz file should contain the following keys:
    'log_params': dict of simulation parameters
    'log_density': (N, B, H, W) the density, N is the number of frames, B is the batch size, H and W are the height and width of the frame
    'log_velocity': (N, B, H, W, 2) the velocity, N is the number of frames, B is the batch size, H and W are the height and width of the frame, the last dimension is the x and y velocity
    'log_time': (N, B) the time of each frame, N is the number of frames, B is the batch size, which is consistent along the batch dimension
    (NOTE: the batch dimension here is just for multiple simulations, which differs from the batch dimension in the training process)
    """
    def __init__(self, loadpath: str) -> None:
        """
        The individual dataset should be the following format:
        self.density: (N*B, 1, H, W)
        self.velocity: (N*B, 2, H, W) (NOTE: for the 2nd dimension, the 1st one is the y conponent, the 2nd one is the x component)
        self.conditions: (N*B, 3)
        """
        self.data = np.load(loadpath, allow_pickle=True)
        self.density = torch.from_numpy(self.data['log_density']).float().flatten(start_dim=0, end_dim=1).unsqueeze(1)
        self.velocity = torch.from_numpy(self.data['log_velocity']).float().flatten(start_dim=0, end_dim=1).permute(0, 3, 1, 2)
        self.length = self.density.shape[0]
        self.params = self.data['log_params'].item()
        self.conditions = torch.from_numpy(self.data['log_condition']).float().flatten(start_dim=0, end_dim=1)
    
    def __getitem__(self, index: int) -> tuple:
        frame_rho = self.density[index]
        frame_vel = self.velocity[index]
        frame_con = self.conditions[index]
        return (frame_rho, frame_vel, frame_con)
    
    def __len__(self) -> int:
        return self.length
    