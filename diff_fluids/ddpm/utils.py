import h5py
import torch
from torch.utils.data import Dataset

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


class DiffusionReactionDataset(Dataset):
    def __init__(self,
                 fileroot: str,
                 filename: str,
                 is_test: bool=False,
                 reduced_time_factor: int=10,
                 reduced_resolution_factor: int=2):
        with h5py.File(fileroot + filename + '.h5', 'r') as f:
            xlen = len(f['0001']['grid']['x']) // reduced_resolution_factor 
            ylen = len(f['0001']['grid']['y']) // reduced_resolution_factor 
            tlen = (len(f['0001']['grid']['t']) - 1) // reduced_time_factor
            t_max = f['0001']['grid']['t'][-1]
            if not is_test:
                n_sample_idxs = [i for i in range(0, 800)]
            else:
                n_sample_idxs = [i for i in range(800, 1000)]
            sample_names = [str(i).zfill(4) for i in n_sample_idxs]

            all_vars = []
            all_ts = []
            all_inits = []
            for sample_name in sample_names:

                # activator and inhibitor
                _data = torch.from_numpy(f[sample_name]['data'][:]).float()  # [tlen, xlen, ylen, 2]
                all_inits.append(_data[0, ::reduced_resolution_factor, ::reduced_resolution_factor, :].repeat(tlen, 1, 1, 1))
                all_vars.append(_data[1::reduced_time_factor, ::reduced_resolution_factor, ::reduced_resolution_factor, :])

                # time
                t = torch.from_numpy(f[sample_name]['grid']['t'][1::reduced_time_factor]).float()
                all_ts.append(t[:, None, None, None].repeat(1, xlen, ylen, 1))
            
            all_vars = torch.cat(all_vars, dim=0)
            all_ts = torch.cat(all_ts, dim=0)
            all_inits = torch.cat(all_inits, dim=0)

            self.data = torch.cat([all_vars, all_inits, all_ts / t_max], dim=-1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        return {
            'x': data[..., 0:2].permute(2, 0, 1),
            'y': data[..., 2:5].permute(2, 0, 1)
        }

def _test():
    dataset = DiffusionReactionDataset("/media/bamf-big/gefan/DiffFluids/data/", "2d_diffusion_reaction", is_test=True)
    print(len(dataset))
    print(dataset[0]['x'].shape)
    print(dataset[0]['y'].shape)

if __name__ == '__main__':
    _test()





    