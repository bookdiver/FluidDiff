import scipy.io as sio
import torch
from torch.utils.data import Dataset

class Burgers_Dataset(Dataset):
    def __init__(self, data_dir, normalize=True):
        data = sio.loadmat(data_dir)
        to_tensor = lambda x: torch.from_numpy(x).unsqueeze(1).to(torch.float32)
        self.a = to_tensor(data['a'])
        self.u = to_tensor(data['u'])
        print(f"Loaded {len(self.a)} samples from {data_dir}")
        print(f"Shape of x: {self.u.shape}")
        if normalize:
            self.u = (self.u - self.u.min()) / (self.u.max() - self.u.min())
            self.a = (self.a - self.a.min()) / (self.a.max() - self.a.min())
    
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

if __name__ == "__main__":
    data_dir = "../data/burgers_data_v1e-02_N200.mat"
    # dataset = NaiverStokes_Dataset(data_dir)
    dataset = Burgers_Dataset(data_dir)
    print(dataset[0]['x'].shape)
    print(dataset[0]['y'].shape)
    # print(dataset[0]['x_prev'].shape)
    # print(dataset[0]['x_next'].shape)