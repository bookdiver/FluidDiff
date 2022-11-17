import torch

class MyDataSet(object):
    def __init__(self, num_samples: int, orientation: torch.Tensor=None, noise: float=0.1):
        self.num_samples = num_samples
        self.orientation = orientation
        self.dir = dir
        self.noise = noise
        self.data = self.generate_dataset()


    def generate_sample(self, height: int=28, width: int=28) -> torch.Tensor:
        if self.orientation == None:
            sample_x = torch.randn(size=(height, width))
            sample_y = torch.randn(size=(height, width))
        else:
            sample_x = torch.randn(size=(height, width))
            sample_y = torch.randn(size=(height, width))
            sample_x = self.noise * sample_x + self.orientation[0]
            sample_y = self.noise * sample_y + self.orientation[1]
        sample = torch.stack([sample_x, sample_y], dim=0)
        return sample

    def generate_dataset(self)-> list:
        data = []
        for _ in range(self.num_samples):
            sample = self.generate_sample()
            data.append(sample)
        return data
    
    def __getitem__(self, index: int):
        return self.data[index]
    
    def __len__(self):
        return self.num_samples