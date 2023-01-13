import h5py
import torch
from torch.utils.data import Dataset
import glob

class UNetDataset(Dataset):
    def __init__(self, 
                 fileroot: str,
                 filename: str,
                 is_test: bool=False,
                 reduced_batch_factor: int=1,
                 reduced_time_factor: int=1,
                 reduced_resolution_factor: int=1):
        
        file = glob.glob(fileroot + filename + '/*' + ('test' if is_test else 'train') + '.h5')[0]
        with h5py.File(file, 'r') as f:
            if 'PointSmoke' in filename:
                b, trange, h, w = f['2DNSPointSmoke']['u'].shape
                self.data = torch.zeros([(b // reduced_batch_factor) * (trange // reduced_time_factor),
                                          h // reduced_resolution_factor,
                                          w // reduced_resolution_factor,
                                          5])
                # density
                _data = torch.from_numpy(f['2DNSPointSmoke']['u'][:]).float()
                _data = _data[::reduced_batch_factor, ::reduced_time_factor, ::reduced_resolution_factor, ::reduced_resolution_factor]
                self.data[:, :, :, 0] = _data.reshape(-1, h // reduced_resolution_factor, w // reduced_resolution_factor)

                # velocity x
                _data = torch.from_numpy(f['2DNSPointSmoke']['vx'][:]).float()
                _data = _data[::reduced_batch_factor, ::reduced_time_factor, ::reduced_resolution_factor, ::reduced_resolution_factor]
                self.data[:, :, :, 1] = _data.reshape(-1, h // reduced_resolution_factor, w // reduced_resolution_factor)

                # velocity y
                _data = torch.from_numpy(f['2DNSPointSmoke']['vy'][:]).float()
                _data = _data[::reduced_batch_factor, ::reduced_time_factor, ::reduced_resolution_factor, ::reduced_resolution_factor]
                self.data[:, :, :, 2] = _data.reshape(-1, h // reduced_resolution_factor, w // reduced_resolution_factor)

                # source position
                _data = torch.from_numpy(f['2DNSPointSmoke']['src'][:]).float()
                _data = _data.unsqueeze(1).repeat(1, trange // reduced_time_factor, 1, 1)
                _data = _data[::reduced_batch_factor, :, ::reduced_resolution_factor, ::reduced_resolution_factor]
                self.data[:, :, :, 3] = _data.reshape(-1, h // reduced_resolution_factor, w // reduced_resolution_factor)

                # time
                _data = torch.from_numpy(f['2DNSPointSmoke']['t'][:]).float()
                t_max = _data[0].max()
                _data = _data / t_max
                _data = _data[..., None, None].repeat(1, 1, h // reduced_resolution_factor, w // reduced_resolution_factor)
                _data = _data[::reduced_batch_factor, ::reduced_time_factor, :, :]
                self.data[:, :, :, 4] = _data.reshape(-1, h // reduced_resolution_factor, w // reduced_resolution_factor)

            elif 'FullSmoke' in filename:
                b, trange, h, w = f['2DNSFullSmoke']['u'].shape
                self.data = torch.zeros([(b // reduced_batch_factor) * (trange // reduced_time_factor),
                                          h // reduced_resolution_factor,
                                          w // reduced_resolution_factor,
                                          5])
                # density
                _data = torch.from_numpy(f['2DNSFullSmoke']['u'][:]).float()
                _data = _data[::reduced_batch_factor, ::reduced_time_factor, ::reduced_resolution_factor, ::reduced_resolution_factor]
                self.data[:, :, :, 0] = _data.reshape(-1, h // reduced_resolution_factor, w // reduced_resolution_factor)

                # velocity x
                _data = torch.from_numpy(f['2DNSFullSmoke']['vx'][:]).float()
                _data = _data[::reduced_batch_factor, ::reduced_time_factor, ::reduced_resolution_factor, ::reduced_resolution_factor]
                self.data[:, :, :, 1] = _data.reshape(-1, h // reduced_resolution_factor, w // reduced_resolution_factor)

                # velocity y
                _data = torch.from_numpy(f['2DNSFullSmoke']['vy'][:]).float()
                _data = _data[::reduced_batch_factor, ::reduced_time_factor, ::reduced_resolution_factor, ::reduced_resolution_factor]
                self.data[:, :, :, 2] = _data.reshape(-1, h // reduced_resolution_factor, w // reduced_resolution_factor)

                # initial density
                _data = torch.from_numpy(f['2DNSFullSmoke']['u_init'][:]).float()
                _data = _data.unsqueeze(1).repeat(1, trange // reduced_time_factor, 1, 1)
                _data = _data[::reduced_batch_factor, ::reduced_resolution_factor, ::reduced_resolution_factor]
                self.data[:, :, :, 3] = _data.reshape(-1, h // reduced_resolution_factor, w // reduced_resolution_factor)

                # time
                _data = torch.from_numpy(f['2DNSFullSmoke']['t'][:]).float()
                t_max = _data[0].max()
                _data = _data / t_max
                _data = _data[..., None, None].repeat(1, 1, h // reduced_resolution_factor, w // reduced_resolution_factor)
                _data = _data[::reduced_batch_factor, ::reduced_time_factor, :, :]
                self.data[:, :, :, 4] = _data.reshape(-1, h // reduced_resolution_factor, w // reduced_resolution_factor)
            else:
                raise NotImplementedError(f"Dataset {filename} not implemented.")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return {
            'x': data[..., 0:3].permute(2, 0, 1),
            'y': data[..., 3:5].permute(2, 0, 1)
        }