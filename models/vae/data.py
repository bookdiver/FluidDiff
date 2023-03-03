import torch
import h5py
from torch.utils.data import Dataset
import glob

class FluidDataset(Dataset):
    def __init__(self, 
                 fileroot: str, 
                 physics_variables: list=['density', 'vorticity', 'pressure'],
                 read_frames: bool=False,
                 read_every_frames: int=1,
                 normalize_type: str='pm1'):
        all_files = glob.glob(fileroot + '/*.h5')
        data_dict = {}
        self.statistic_dict = {}
        for phi in physics_variables:
            data_dict[phi] = []
            self.statistic_dict[phi] = {}

        for i, file in enumerate(all_files):
            # each file contains 200 frames in 40.0s (5fps)
            with h5py.File(file, 'r') as data_store:
                for phi in physics_variables:
                    data_dict[phi].append(torch.from_numpy(data_store[phi][::read_every_frames]))
            print(f"Loaded {file} ({i+1}/{len(all_files)})")
        
        for phi in physics_variables:
            data_dict[phi] = torch.cat(data_dict[phi], dim=1)
            if not read_frames:
                data_dict[phi] = data_dict[phi].flatten(0, 1)
            if normalize_type == 'pm1':
                self.statistic_dict[phi]['min'] = data_dict[phi].min()
                self.statistic_dict[phi]['max'] = data_dict[phi].max()
                data_dict[phi] = (data_dict[phi] - self.statistic_dict[phi]['min'])  \
                        / (self.statistic_dict[phi]['max'] - self.statistic_dict[phi]['min']) * 2 - 1
            elif normalize_type == '01':
                self.statistic_dict[phi]['min'] = data_dict[phi].min()
                self.statistic_dict[phi]['max'] = data_dict[phi].max()
                data_dict[phi] = (data_dict[phi] - self.statistic_dict[phi]['min'])  \
                        / (self.statistic_dict[phi]['max'] - self.statistic_dict[phi]['min'])
            elif normalize_type == 'zscore':
                self.statistic_dict[phi]['mean'] = data_dict[phi].mean()
                self.statistic_dict[phi]['std'] = data_dict[phi].std()
                data_dict[phi] = (data_dict[phi] - self.statistic_dict[phi]['mean'])  \
                        / self.statistic_dict[phi]['std']
            elif normalize_type == 'none':
                pass
            else:
                raise NotImplementedError

        self.data = torch.stack([data_dict[phi] for phi in physics_variables], dim=0).float()
        if read_frames:
            self.data = self.data.permute(2, 0, 1, 3, 4)
        else:
            self.data = self.data.permute(1, 0, 2, 3)

        print(f"Loaded {len(self.data)} samples, data shape: {self.data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    dataset = FluidDataset(fileroot='/media/bamf-big/gefan/FluidDiff/data/smoke_plume_64x64',
                           physics_variables=['density', 'vorticity', 'pressure'],
                           read_frames=True,
                           normalize_data=True)
    print(dataset[0].shape)
    print(dataset.statistic_dict)