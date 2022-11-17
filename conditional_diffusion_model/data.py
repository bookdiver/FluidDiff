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
    
    # def _get_params(self) -> None:
    #     for key, value in self.params.items():
    #         print(f'{key}: {value}')
    
    # def _show_idv_sample(self, index: int) -> None:
    #     assert index < self.length
    #     _rho = self.density[index].numpy().squeeze()
    #     _vel = self.velocity[index].numpy()
    #     _time = self.conditions[index].numpy()[0]
    #     _pos = self.conditions[index].numpy()[1:]
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #     p1 = axes[0].imshow(_rho, cmap='gray', origin='lower')
    #     axes[0].axis('off')
    #     fig.colorbar(p1, ax=axes[0])
    #     axes[1].quiver(_vel[1, ::5, ::5], _vel[0, ::5, ::5])
    #     axes[1].axis('off')
    #     fig.suptitle(f"Sample {index}, time {_time:.2f} s, position ({_pos[0]:.2f}, {_pos[1]:.2f})")
    #     plt.show()
    
    # def _show_frame(self, frame: int) -> None:
    #     assert frame < self.params['num_frames']
    #     _rhos = self.density[(frame-1)*self.params['batch_size']:frame*self.params['batch_size']]
    #     _time = self.params['time_step'] * frame
    #     grid = make_grid(_rhos, nrow=14, padding=2, normalize=True, pad_value=1)
    #     plt.figure(figsize=(28, 18))
    #     plt.imshow(grid.detach().numpy()[0, ...], cmap='gray', origin='lower')
    #     plt.axis('off')
    #     plt.title(fr"Frame {frame}, time {_time:.2f} s")   