import math
from typing import Optional, List
import logging

import torch
import torch.nn as nn

from unet_transformer import SpatialTransformer

logging.basicConfig(level=logging.INFO)

class GroupNorm32(nn.GroupNorm):
    """ Group normalization with float32 casting
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)

def normalization(channels: int) -> nn.GroupNorm:
    """ get the normalization layer
    """
    return GroupNorm32(8, channels)

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
                 channels: int,
                 channel_multpliers: List[int],
                 n_res_blocks: int, 
                 attention_levels: List[int],
                 n_heads: int,
                 transformer_layers: int=1,
                 d_cond: int=3):
        """ UNet model for approximating noise

        Args:
            in_channels (int): the input channel size, should be same as the latent space channel in VAE
            out_channels (int): the output channel size, should be same as the latent space channel in VAE
            channels (int): the initial channel size for the first convolution layer, also for the dimension of time embedding
            channel_multpliers (List[int]): the channel multiplier for each level
            n_res_blocks (int): the number of residual blocks in each level
            attention_levels (List[int]): the levels where to apply attention at
            n_heads (int): the number of heads for the attention
            transformer_layers (int, optional): the number of transformer layers. Defaults to 1.
            d_cond (int, optional): the size of the conditional embedding. Defaults to 3.
        """ 
        super().__init__()
        self.channels = channels

        n_levels = len(channel_multpliers)

        d_time_emb = channels * 4
        self.time_embedding = nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb)
        )

        self.input_blocks = nn.ModuleList([])

        self.input_blocks.append(TimeStepEmbedSequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)))
        
        input_block_channels = [channels]
        channels_list = [channels * m for m in channel_multpliers]

        for i in range(n_levels):
            for _ in range(n_res_blocks):
                layers = [ResBlock(channels, d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]

                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, d_cond, n_heads, transformer_layers))

                self.input_blocks.append(TimeStepEmbedSequential(*layers))
                input_block_channels.append(channels)
            
            if i != n_levels - 1:
                self.input_blocks.append(TimeStepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)

        logging.debug(f"the number of input blocks: {len(self.input_blocks)}")
        
        self.middle_block = TimeStepEmbedSequential(
            ResBlock(channels, d_time_emb),
            SpatialTransformer(channels, d_cond, n_heads, transformer_layers),
            ResBlock(channels, d_time_emb)
        )

        self.output_blocks = nn.ModuleList([])

        for i in reversed(range(n_levels)):

            for j in range(n_res_blocks+1):
                layers = [ResBlock(channels+input_block_channels.pop(), d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]

                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, d_cond, n_heads, transformer_layers))
                
                if i != 0 and j == n_res_blocks:
                    layers.append(UpSample(channels))
                
                self.output_blocks.append(TimeStepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
        )
    
    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int=10000) -> torch.Tensor:
        """ Using GaussianFourierFeatures to generate time step embedding

        Args:
            time_steps (torch.Tensor): time steps with shape (batch_size)
            max_period (int, optional): minimum frequency of the embedding. Defaults to 10000.

        Returns:
            torch.Tensor: time step embedding with shape (batch_size, d_t_emb)
        """
        half = self.channels // 2
        freqs = torch.exp(
            math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)

        args = time_steps[:, None].float() * freqs[None]

        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: Optional[torch.Tensor]=None) -> torch.Tensor:
        """ forward pass

        Args:
            x (torch.Tensor): input feature map with shape (batch_size, channels, height, width)
            time_steps (torch.Tensor): time steps with shape (batch_size)
            cond (Optional[torch.Tensor], optional): conditional feature map with shape (batch_size, channels, d_cond). Defaults to None.

        Returns:
            torch.Tensor: output feature map with shape (batch_size, channels, height, width)
        """
        x_input_block = []

        t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_embedding(t_emb)

        for module in self.input_blocks:
            logging.debug(f"module name: {module.__class__.__name__}")
            x = module(x, t_emb, cond)
            x_input_block.append(x)
            logging.debug("pass")
        logging.debug("downsample passed, the output shape is {}".format(x.shape))
        
        x = self.middle_block(x, t_emb, cond)
        logging.debug("middle block passed, the output shape is {}".format(x.shape))

        for module in self.output_blocks:
            x = module(torch.cat([x, x_input_block.pop()], dim=1), t_emb, cond)
        logging.debug("upsample passed, the output shape is {}".format(x.shape))
        
        return self.out(x)

def _test_time_step_embedding():
    unet = UNetModel(in_channels=32,
                    out_channels=32,
                    channels=64,
                    channel_multpliers=[1, 2, 4, 8],
                    n_res_blocks=2,
                    attention_levels=[1, 2, 3],
                    n_heads=4,
                    transformer_layers=1,
                    d_cond=32)
    time_steps = torch.randint(0, 100, (16,))
    t_emb = unet.time_step_embedding(time_steps)
    print(t_emb.shape)

def _test_unet():
    unet = UNetModel(in_channels=4,
                     out_channels=4,
                     channels=16,
                     channel_multpliers=[1, 2, 4],
                     n_res_blocks=1,
                     attention_levels=[1, 2],
                     n_heads=4,
                     transformer_layers=1,
                     d_cond=16)
    # print(unet)
    # print(f"total params: {sum(p.numel() for p in unet.parameters())}")
    input = torch.randn((2, 4, 4, 4))
    time_steps = torch.randn(2)
    cond = torch.randn((2, 16, 3))
    output = unet(input, time_steps, cond=None)
    print(output.shape)

if __name__ == "__main__":
    # _test_time_step_embedding()
    _test_unet()