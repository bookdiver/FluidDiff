import h5py
import torch
from torch.utils.data import Dataset
import glob

class FluidDataset(Dataset):
    def __init__(self, 
                root: str,
                name: str,
                mode: str = "train"):
        assert mode in ['train', 'test'] 
        "mode must be either 'train' or 'test'"

        self.name = name
        if name == '2DNSPointSmoke':
            with h5py.File(glob.glob(root+name+'/*'+mode+'.h5')[0], 'r') as f:
                b, trange, h, w = f[name]['u'].shape
                self.n_samples = b * trange
                self.u = torch.from_numpy(f[name]['u'][:].reshape(-1, h, w)).float()
                self.vx = torch.from_numpy(f[name]['vx'][:].reshape(-1, h, w)).float()
                self.vy = torch.from_numpy(f[name]['vy'][:].reshape(-1, h, w)).float()
                self.src = torch.from_numpy(f[name]['src'][:]).unsqueeze(1).repeat(1, trange, 1, 1)
                self.src = self.src.reshape(-1, h, w).float()
                self.t = torch.from_numpy(f[name]['t'][:]).float()
                t_max = self.t[0].max().item()
                self.t = self.t / t_max
                self.t = self.t[..., None, None].repeat(1, 1, h, w)
                self.t = self.t.reshape(-1, h, w)
        elif name == '2DNSFullSmoke':
            with h5py.File(glob.glob(root+name+'/*'+mode+'.h5')[0], 'r') as f:
                b, trange, h, w = f[name]['u'].shape
                self.n_samples = b * trange
                self.u = torch.from_numpy(f[name]['u'][:].reshape(-1, h, w)).float()
                self.vx = torch.from_numpy(f[name]['vx'][:].reshape(-1, h, w)).float()
                self.vy = torch.from_numpy(f[name]['vy'][:].reshape(-1, h, w)).float()
                self.u0 = torch.from_numpy(f[name]['u_init'][:]).unsqueeze(1).repeat(1, trange, 1, 1)
                self.u0 = self.u0.reshape(-1, h, w).float()
                self.t = torch.from_numpy(f[name]['t'][:]).float()
                t_max = self.t[0].max().item()
                self.t = self.t / t_max
                self.t = self.t[..., None, None].repeat(1, 1, h, w)
                self.t = self.t.reshape(-1, h, w)
        else:
            raise NotImplementedError(f"Dataset {name} not implemented.")
        print(f"Loaded {self.n_samples} samples from {name} in {root}.")
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.name == '2DNSPointSmoke':
            u = self.u[idx].unsqueeze(0)
            vx = self.vx[idx].unsqueeze(0)
            vy = self.vy[idx].unsqueeze(0)
            src = self.src[idx].unsqueeze(0)
            t = self.t[idx].unsqueeze(0)
            return {"u": u,
                    "v": torch.cat([vx, vy], dim=0), 
                    "y": torch.cat([src, t], dim=0)}
        elif self.name == '2DNSFullSmoke':
            u = self.u[idx].unsqueeze(0)
            vx = self.vx[idx].unsqueeze(0)
            vy = self.vy[idx].unsqueeze(0)
            u0 = self.u0[idx].unsqueeze(0)
            t = self.t[idx].unsqueeze(0)
            return {"u": u,
                    "v": torch.cat([vx, vy], dim=0), 
                    "y": torch.cat([u0, t], dim=0)}
        else:
            raise NotImplementedError(f"Dataset {self.name} not implemented.")
    



    