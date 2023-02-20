import h5py
import torch
from torch.utils.data import Dataset
import glob
import re

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


class KVSDataSet(Dataset):
    def __init__(self, fileroot: str):
        all_files = glob.glob(fileroot + '/*.h5')
        RE_list, u_list, v_list, p_list = [], [], [], []
        for file in all_files:
            RE = float(re.findall(r'\d+', file)[-2])
            RE_list.append(torch.tensor((RE, )))
            with h5py.File(file, 'r') as f:
                u_list.append(torch.from_numpy(f['velocity_x'][:]).float())
                v_list.append(torch.from_numpy(f['velocity_y'][:]).float())
                p_list.append(torch.from_numpy(f['pressure'][:]).float())
        self.Re = torch.stack(RE_list, dim=0)
        self.data = torch.stack([torch.cat(u_list, dim=0), torch.cat(v_list, dim=0), torch.cat(p_list, dim=0)], dim=1)
    
    def __len__(self):
        return len(self.Re)
    
    def __getitem__(self, idx):
        return {
            'x': self.data[idx],
            'y': self.Re[idx]
        }

def _test():
    dataset = KVSDataSet(fileroot='/media/bamf-big/gefan/FluidDiff/data/karman_vortex')
    print(dataset[0]['x'].shape)
    print(dataset[0]['y'].shape)

    
if __name__ == '__main__':
    _test()





    