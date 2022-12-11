import math
from typing import Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce

from transformer4unet import SpatialTransformer

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def num_to_groups(num: int, divisor: int) -> list:
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder != 0:
        arr.append(remainder)
    return arr

class Residual(nn.Module):
    def __init__(self, fn: callable):
        super().__init__()
        self.fn = fn
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.fn(x, *args, **kwargs)

class WeightStandardizedConv2d(nn.Conv2d):
    """ Weight Standardized Conv2d
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var= reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weigth = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, normalized_weigth, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 groups: int):
        """ Block for Resnet block, contains 1 weighted convolution layer and 1 normalization layer, followed by a SiLU activation
        """
        super().__init__()
        self.conv = WeightStandardizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor, scale_shift: tuple=None) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        
        return self.act(x)

class ResBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 d_emb: int,
                 groups: int=8,
                 *,
                 out_channels:Optional[int]=None):
        """ Residual block for the UNet

        Args:
            in_channels (int): the input channel size
            d_emb (int): the size of the embedding, can be either d_t_emb or d_cond_emb
            groups (int, optional): the number of groups for the normalization layer. Defaults to 8.
            out_channels (Optional[int], optional): the output channel size. Defaults to be same as in_channels.
        """
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_emb, 2 * out_channels)
        )

        self.block1 = Block(in_channels, out_channels, groups)
        self.block2 = Block(out_channels, out_channels, groups)
        
        # inner residual connection
        if in_channels == out_channels:
            self.res_connnection = nn.Identity()
        else:
            self.res_connnection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """ foward pass

        Args:
            x (torch.Tensor): input feature map with shape (batch_size, channels, height, width)
            emb (torch.Tensor): embedding with shape (batch_size, d_emb)

        Returns:
            torch.Tensor: output feature map with shape (batch_size, channels, height, width)
        """
        scale_shift = None
        if exists(self.emb_mlp) and exists(emb):
            emb = self.emb_mlp(emb)
            emb = rearrange(emb, 'b c -> b c 1 1')
            scale_shift = emb.chunk(2, dim=1)
        
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        
        return h + self.res_connnection(x)
        
class DownSample(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: Optional[int]=None
                 ):
        """ Downsample the feature map
        """
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ forward pass

        Args:
            x (torch.Tensor): input feature map with shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: output feature map with shape (batch_size, channels, height/2, width/2)
        """
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: Optional[int]=None
                 ):
        """ Upsample the feature map
        """
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ forward pass

        Args:
            x (torch.Tensor): input feature map with shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: output feature map with shape (batch_size, channels, height*2, width*2)
        """
        return self.conv(x)

class TimeStepEmbedSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond_emb: Optional[torch.Tensor]=None) -> torch.Tensor:
        """ forward pass

        Args:
            x (torch.Tensor): input feature map with shape (batch_size, channels, height, width)
            t_emb (torch.Tensor): diffusion time embedding with shape (batch_size, d_t_emb)
            cond_emb (Optional[torch.Tensor], optional): conditional feature map with shape (batch_size, channels, d_cond). Defaults to None.

        Returns:
            torch.Tensor: output feature map with shape (batch_size, channels, height, width)
        """
        for module in self:
            if isinstance(module, ResBlock):
                x = module(x, t_emb)
            elif isinstance(module, SpatialTransformer):
                x = module(x, cond_emb)
            else:
                x = module(x)
        return x

class UniversialEmbedSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """ forward pass, takes a universal embedding emb, containing both time and conditional embedding

        Args:
            x (torch.Tensor): input feature map with shape (batch_size, channels, height, width)
            emb (torch.Tensor): embedding with shape (batch_size, d_emb)

        Returns:
            torch.Tensor: output feature map with shape (batch_size, channels, height, width)
        """
        for module in self:
            if isinstance(module, ResBlock):
                x = module(x, emb)
            else:
                x = module(x)
        return x

class TimeEmbeddingBlock(nn.Module):
    def __init__(self, d_emb: int, max_period: int=10000):
        super().__init__()
        self.d_emb = d_emb
        self.max_period = max_period
    
    def forward(self, time_steps: torch.Tensor) -> torch.Tensor:
        """ Using GaussianFourierFeatures to generate time step embedding

        Args:
            time_steps (torch.Tensor): time steps with shape (batch_size)
            max_period (int, optional): minimum frequency of the embedding. Defaults to 10000.

        Returns:
            torch.Tensor: time step embedding with shape (batch_size, d_emb)
        """
        half = self.d_emb // 2
        freqs = torch.exp(
            math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)

        args = time_steps[:, None].float() * freqs[None]
        te = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return te

class ConditionEmbeddingBlock(nn.Module):
    def __init__(self, d_emb: int, max_period: int=10000):
        super().__init__()
        self.d_emb = d_emb
        self.max_period = max_period
    
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """ Using GaussianFourierFeatures to generate position embedding

        Args:
            condition (torch.Tensor): condtion with shape (batch_size, 3)
            max_period (int, optional): minimum frequency of the embedding, can be smaller since we don't need so precise as time right now. Defaults to 1000.

        Returns:
            torch.Tensor: position embedding with shape (batch_size, 3, d_emb)
        """
        half = self.d_emb // 2
        freqs = torch.exp(
            math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=condition.device)

        real_t, pos_x, pos_y = condition.chunk(3, dim=1)
        args = real_t[:, None].float() * freqs[None]
        pe_t = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        args = pos_x[:, None].float() * freqs[None]
        pe_x = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        args = pos_y[:, None].float() * freqs[None]
        pe_y = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        pe = torch.cat([pe_t, pe_x, pe_y], dim=1)
        return pe

class BasicUNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
def _test_time_step_embedding():
    import matplotlib.pyplot as plt
    block = TimeEmbeddingBlock(d_emb=128, max_period=10000)
    time_steps1 = torch.arange(0, 1000)
    time_steps2 = torch.arange(0, 1000) / 1000.0
    t_emb1 = block(time_steps1)
    t_emb2 = block(time_steps2)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(t_emb1.detach().numpy(), cmap='jet', aspect='auto', origin='lower')
    plt.xlabel('Dimension')
    plt.ylabel('Diffusion Time')
    plt.title('Fouriour Features for Diffusion Time (Unnormalized)')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(t_emb2.detach().numpy(), cmap='jet', aspect='auto', origin='lower')
    plt.xlabel('Dimension')
    plt.ylabel('Diffusion Time')
    plt.title('Fouriour Features for Diffusion Time (Normalized)')
    plt.colorbar()

    plt.show()

if __name__ == '__main__':
    _test_time_step_embedding()