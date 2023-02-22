import torch
import h5py
from torch.utils.data import Dataset
import glob

class FluidDataset(Dataset):
    def __init__(self, fileroot: str, n_frames: int=16, normalize: bool=True):
        all_files = glob.glob(fileroot + '/*.h5')
        all_density, all_vorticity, all_pressure = [], [], []
        for i, file in enumerate(all_files):
            with h5py.File(file, 'r') as data_store:
                all_density.append(torch.from_numpy(data_store['density'][9::10, ...])[:n_frames])
                all_vorticity.append(torch.from_numpy(data_store['vorticity'][9::10, ...])[:n_frames])
                all_pressure.append(torch.from_numpy(data_store['pressure'][9::10, ...])[:n_frames])
            print(f"Loaded {file} ({i+1}/{len(all_files)})")
        self.density = torch.cat(all_density, dim=1)
        self.vorticity = torch.cat(all_vorticity, dim=1)
        self.pressure = torch.cat(all_pressure, dim=1)
        del all_density, all_vorticity, all_pressure

        if normalize:
            self.density, self.density_min, self.density_max = self.normalize(self.density, norm2pm1=False)
            self.vorticity, self.vorticity_min, self.vorticity_max = self.normalize(self.vorticity, norm2pm1=False)
            self.pressure, self.pressure_min, self.pressure_max = self.normalize(self.pressure, norm2pm1=False)

        self.data = torch.stack([self.density, self.vorticity, self.pressure], dim=0).float()
        self.data = self.data.permute(2, 0, 1, 3, 4)

        print(f"Loaded {len(self.data)} samples")
    
    def normalize(self, original_data: torch.Tensor, norm2pm1: bool=True):
        loc_min = original_data.min()
        loc_max = original_data.max()
        norm_data = (original_data - loc_min) / (loc_max - loc_min)
        if norm2pm1:
            norm_data = 2 * norm_data - 1
        
        return norm_data, loc_min, loc_max

    def unnormalize(self, norm_data: torch.Tensor, loc_min: torch.Tensor, loc_max: torch.Tensor, pm1to01: bool=True):
        if pm1to01:
            norm_data = (norm_data + 1) / 2
        original_data = norm_data * (loc_max - loc_min) + loc_min
        
        return original_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    dataset = FluidDataset('/media/bamf-big/gefan/FluidDiff/data/smoke_plume_64x64')
    print(dataset[0].shape)
    print(dataset[0].min(), dataset[0].max())