import scipy.io as sio
import torch
from torch.utils.data import Dataset

class Burgers_Dataset(Dataset):
    def __init__(self, data_dir):
        data = sio.loadmat(data_dir)
        to_tensor = lambda x: torch.from_numpy(x).unsqueeze(1).to(torch.float32)
        self.a = to_tensor(data['a'])
        self.u = to_tensor(data['u'])
        print(f"Loaded {len(self.a)} samples from {data_dir}")
        print(f"Shape of x: {self.u.shape}")
    
    def __len__(self):
        return len(self.a)
    
    def __getitem__(self, idx):
        return {'x': self.u[idx], 'y': self.a[idx]}

class NaiverStokes_Dataset(Dataset):
    def __init__(self, data_dir):
        data = sio.loadmat(data_dir)
        self.a = torch.from_numpy(data['a'][:]).unsqueeze(1).to(torch.float32)
        to_tensor = lambda x: torch.from_numpy(x).permute(0, 3, 1, 2).unsqueeze(1).to(torch.float32)
        self.w_now = to_tensor(data['w'])
        self.w_prev = to_tensor(data['w_prev'])
        self.w_next = to_tensor(data['w_next'])
        print(f"Loaded {len(self.a)} samples from {data_dir}")
        print(f"Shape of x: {self.w_now.shape}")
    
    def __len__(self):
        return len(self.a)
    
    def __getitem__(self, idx):
        return {'x': self.w_now[idx], 
                'x_prev': self.w_prev[idx],
                'x_next': self.w_next[idx],
                'y': self.a[idx]}
    
class Darcys_Dataset(Dataset):
    def __init__(self, data_dir):
        data = sio.loadmat(data_dir)
        to_tensor = lambda x: torch.from_numpy(x).unsqueeze(1).to(torch.float32)
        self.a = to_tensor(data['coeff'])
        self.u = to_tensor(data['sol'])

        print(f"Loaded {len(self.a)} samples from {data_dir}")
        print(f"Shape of x: {self.u.shape}")

    def __len__(self):
        return len(self.a)
    
    def __getitem__(self, idx):
        return {'x': self.a[idx], 'y': self.u[idx]}