import h5py
import torch
from torch.utils.data import Dataset

class NaiverStokes_Dataset(Dataset):
    def __init__(self, data_dir):
        with h5py.File(data_dir, 'r') as f:
            self.a = torch.from_numpy(f['a'][:]).unsqueeze(1).to(torch.float32)
            self.u = torch.from_numpy(f['u'][:]).permute(0, 3, 1, 2).unsqueeze(1).to(torch.float32)
        print(f"Loaded {len(self)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.a)
    
    def __getitem__(self, idx):
        return {'x': self.u[idx], 'y': self.a[idx]}

if __name__ == "__main__":
    data_dir = "../../data/ns_V1e-5_T20_test.h5"
    dataset = NaiverStokes_Dataset(data_dir)
    print(dataset[0]['x'].shape, dataset[0]['y'].shape)