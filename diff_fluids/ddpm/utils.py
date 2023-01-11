import h5py
import torch
from torch.utils.data import Dataset

class FluidDataset(Dataset):
    def __init__(self, 
                name: str,
                root: str):
        with h5py.File(root+name, 'r') as f:
            b, trange, h, w = f['NavierStokes2D']['u'].shape
            self.n_samples = b * trange
            self.u = torch.from_numpy(f['NavierStokes2D']['u'][:].reshape(-1, h, w)).float()
            self.vx = torch.from_numpy(f['NavierStokes2D']['vx'][:].reshape(-1, h, w)).float()
            self.vy = torch.from_numpy(f['NavierStokes2D']['vy'][:].reshape(-1, h, w)).float()
            self.src = torch.from_numpy(f['NavierStokes2D']['src'][:]).unsqueeze(1).repeat(1, trange, 1, 1)
            self.src = self.src.reshape(-1, h, w).float()
            self.t = torch.from_numpy(f['NavierStokes2D']['t'][:]).float()
            t_max = self.t[0].max().item()
            self.t = self.t / t_max
            self.t = self.t[..., None, None].repeat(1, 1, h, w)
            self.t = self.t.reshape(-1, h, w)
        print(f"Loaded {self.n_samples} samples from {name} in {root}.")
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        u = self.u[idx].unsqueeze(0)
        vx = self.vx[idx].unsqueeze(0)
        vy = self.vy[idx].unsqueeze(0)
        src = self.src[idx].unsqueeze(0)
        t = self.t[idx].unsqueeze(0)
        return {"u": u,
                "v": torch.cat([vx, vy], dim=0), 
                "y": torch.cat([src, t], dim=0)}
    



    