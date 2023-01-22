import h5py
import torch
from torch.utils.data import Dataset
import glob

class NavierStokesDataset(Dataset):
    def __init__(self, 
                 fileroot: str,
                 filename: str,
                 is_test: bool=False):
        
        with h5py.File(fileroot+filename, 'r') as f:
            b, trange, h, w, _ = f['density'].shape
            if is_test:
                b_actual = int(b * 0.2)
                b_start = int(b * 0.8)
                b_end = b
            else:
                b_actual = int(b * 0.8)
                b_start = 0
                b_end = int(b * 0.8)
            self.data = torch.zeros([b_actual * trange, h, w, 6])

            # density
            _data = torch.from_numpy(f['density'][:]).float()
            _data = _data[b_start:b_end, :, :, :, :]
            self.data[:, :, :, 0:1] = _data.reshape(b_actual*trange, h, w, 1)

            # velocity 
            _data = torch.from_numpy(f['velocity'][:]).float()
            _data = _data[b_start:b_end, :, :, :, :]
            self.data[:, :, :, 1:3] = _data.reshape(b_actual*trange, h, w, 2)

            # pressure
            _data = torch.from_numpy(f['pressure'][:]).float()
            _data = _data[b_start:b_end, :, :, :, :]
            self.data[:, :, :, 3:4] = _data.reshape(b_actual*trange, h, w, 1)

            # initial density
            _data = torch.from_numpy(f['initial_density'][:]).float()
            _data = _data.unsqueeze(1).repeat(1, trange, 1, 1, 1)
            _data = _data[b_start:b_end, :, :, :, :]
            self.data[:, :, :, 4:5] = _data.reshape(b_actual*trange, h, w, 1)

            # time
            _data = torch.from_numpy(f['t'][:]).float()
            t_max = _data[0].max()
            _data = _data / t_max
            _data = _data[..., None, None, None].repeat(1, 1, h, w, 1)
            _data = _data[b_start:b_end, :, :, :, :]
            self.data[:, :, :, 5:6] = _data.reshape(b_actual*trange, h, w, 1)
        
        print(f"The total number of samples is {len(self.data)}")
        print(f"The number of batches is {b_actual}")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return {
            'x': data[..., 1:3].permute(2, 1, 0),
            'y': data[..., 4:6].permute(2, 1, 0)
        }