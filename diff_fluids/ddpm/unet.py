import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_transformer import SpatialTransformer

class GroupNorm32(nn.GroupNorm):
    """ Group normalization with float32 casting
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)

def normalization(channels: int) -> nn.GroupNorm:
    """ get the normalization layer
    """
    return GroupNorm32(32, channels)

class ResBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 d_t_emb: int,
                 *,
                 out_channels:Optional[int]=None):
        """ Residual block for the UNet

        Args:
            in_channels (int): the input channel size
            d_t_emb (int): the size of the temporal embedding
            out_channels (Optional[int], optional): the output channel size. Defaults to be same as in_channels.
        """
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.in_layers = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.embedding_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t_emb, out_channels)
        )

        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # inner residual connection
        if in_channels == out_channels:
            self.res_connnection = nn.Identity()
        else:
            self.res_connnection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """ foward pass

        Args:
            x (torch.Tensor): input feature map with shape (batch_size, channels, height, width)
            t_emb (torch.Tensor): diffusion time embedding with shape (batch_size, d_t_emb)

        Returns:
            torch.Tensor: output feature map with shape (batch_size, channels, height, width)
        """ 
        h = self.in_layers(x)
        t_emb = self.embedding_layers(t_emb).type(h.dtype)
        h = h + t_emb[:, :, None, None]
        h = self.out_layers(h)
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
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond: Optional[torch.Tensor]=None) -> torch.Tensor:
        """ forward pass

        Args:
            x (torch.Tensor): input feature map with shape (batch_size, channels, height, width)
            t_emb (torch.Tensor): diffusion time embedding with shape (batch_size, d_t_emb)
            cond (Optional[torch.Tensor], optional): conditional feature map with shape (batch_size, channels, d_cond). Defaults to None.

        Returns:
            torch.Tensor: output feature map with shape (batch_size, channels, height, width)
        """
        for module in self:
            if isinstance(module, ResBlock):
                x = module(x, t_emb)
            elif isinstance(module, SpatialTransformer):
                x = module(x, cond)
            else:
                x = module(x)
        return x

class UNetModel(nn.Module):
    def __init__(self, 
                 *,
                 in_channels: int,
                 out_channels: int,
                 init_channels: int,
                 channel_multpliers: List[int],
                 n_res_blocks: int, 
                 attention_levels: List[int],
                 n_heads: int,
                 transformer_layers: int=1,
                 d_cond: int=3):
        """ UNet model for approximating noise

        Args:
            in_channels (int): the input channel size
            out_channels (int): the output channel size
            init_channels (int): the initial channel size for the first convolution layer
            channel_multpliers (List[int]): the channel multiplier for each level
            n_res_blocks (int): the number of residual blocks in each level
            attention_levels (List[int]): the levels where to apply attention at
            n_heads (int): the number of heads for the attention
            transformer_layers (int, optional): the number of transformer layers. Defaults to 1.
            d_cond (int, optional): the size of the conditional embedding. Defaults to 3.
        """ 
        super().__init__()
        self.init_channels = init_channels

        n_levels = len(channel_multpliers)

        d_time_emb = init_channels * 4
        self.time_embedding = nn.Sequential(
            nn.Linear(init_channels, )
        )