from typing import Optional, List

import logging

import torch
import torch.nn as nn

from transformer4unet import *
from blocks4unet import *

logging.basicConfig(level=logging.INFO)

class UNet(BasicUNet):
    def __init__(self,
                 *,
                 in_channels: int,
                 out_channels: int,
                 channels: int,
                 channel_multpliers: List[int],
                 n_res_blocks: int,
                 attention_levels: List[int],
                 n_heads: int,
                 cond_channels: int=3):
        """ UNet model for approximating noise

        Args:
            in_channels (int): the input channel size
            out_channels (int): the output channel size
            channels (int): the initial channel size for the first convolution layer, also for the dimension of time embedding
            channel_multpliers (List[int]): the channel multiplier for each level
            n_res_blocks (int): the number of residual blocks in each level
            attention_levels (List[int]): the levels where to apply attention at
            n_heads (int): the number of heads for the attention
            transformer_layers (int, optional): the number of transformer layers. Defaults to 1.
            cond_channels (int, optional): the channel of the conditional embedding. Defaults to 3.
        """
        super().__init__()
        self.channels = channels

        n_levels = len(channel_multpliers)
        channels_list = [channels * m for m in channel_multpliers]

        emb_dim = channels * 4

        d_time_emb = emb_dim
        self.time_embedding_mlp = nn.Sequential(
            TimeEmbeddingBlock(channels),
            nn.Linear(channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb)
        )

        if cond_channels > 0:
            d_cond_emb = cond_channels * emb_dim
            self.cond_embedding_mlp = nn.Sequential(
                ConditionEmbeddingBlock(channels),
                nn.Flatten(),
                nn.Linear(cond_channels * channels, d_cond_emb),
                nn.SiLU(),
                nn.Linear(d_cond_emb, d_cond_emb)
            )
        else:
            self.cond_embedding_mlp = None

        self.input_blocks = nn.ModuleList([])
        self.input_blocks.append(UniversialEmbedSequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        ))

        input_block_channels = [channels]
        
        for i in range(n_levels):
            for _ in range(n_res_blocks):
                layers = [ResBlock(channels, (cond_channels+1)*emb_dim, out_channels=channels_list[i])]
                channels = channels_list[i]

                input_block_channels.append(channels)
            
                if i in attention_levels:
                    layers.append(Residual(PreNorm(channels, LinearAttnBlock(channels, n_heads=n_heads, d_head=32))))
                
                self.input_blocks.append(UniversialEmbedSequential(*layers))
            
            if i != n_levels - 1:
                self.input_blocks.append(UniversialEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)
            
        
        self.middle_block = UniversialEmbedSequential(
            ResBlock(channels, (cond_channels+1)*emb_dim),
            Residual(PreNorm(channels, AttnBlock(channels, n_heads=8, d_head=32))),
            ResBlock(channels, (cond_channels+1)*emb_dim)
        )

        self.output_blocks = nn.ModuleList([])
        for i in reversed(range(n_levels)):
            
            for j in range(n_res_blocks+1):
                layers = [ResBlock(channels+input_block_channels.pop(), (cond_channels+1)*emb_dim, out_channels=channels_list[i])]
                channels = channels_list[i]
                
                if i in attention_levels:
                    layers.append(Residual(PreNorm(channels, LinearAttnBlock(channels, n_heads=8, d_head=32))))
            
                if i != 0 and j == n_res_blocks:
                    layers.append(UpSample(channels))
        
                self.output_blocks.append(UniversialEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: Optional[torch.Tensor]=None) -> torch.Tensor:
        """ Forward pass of the UNet

        Args:
            x (torch.Tensor): the input tensor, (batch_size, channels, height, width)
            time_steps (torch.Tensor): the time steps with shape (batch_size)
            cond (torch.Tensor): the conditional tensor with shape (batch_size, channels, d_cond)
        
        Returns:
            torch.Tensor: the output tensor with shape (batch_size, channels, height, width)
        """
        x_input_block = []

        t_emb = self.time_embedding_mlp(time_steps).unsqueeze(1)
        if cond is not None:
            c_emb = self.cond_embedding_mlp(cond).reshape(x.shape[0], cond.shape[-1], -1)
            emb = torch.cat([t_emb, c_emb], dim=1).flatten(start_dim=1)
        else:
            emb = t_emb.flatten(start_dim=1)

        for module in self.input_blocks:
            x = module(x, emb)
            x_input_block.append(x)
        
        x = self.middle_block(x, emb)

        for module in self.output_blocks:
            x = module(torch.cat([x, x_input_block.pop()], dim=1), emb)
        
        return self.out(x)


class UNetXAttn(BasicUNet):
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
                 cond_channels: int=3):
        """ UNet model with cross attention for approximating noise

        Args:
            in_channels (int): the input channel size
            out_channels (int): the output channel size
            channels (int): the initial channel size for the first convolution layer, also for the dimension of time embedding
            channel_multpliers (List[int]): the channel multiplier for each level
            n_res_blocks (int): the number of residual blocks in each level
            attention_levels (List[int]): the levels where to apply attention at
            n_heads (int): the number of heads for the attention
            transformer_layers (int, optional): the number of transformer layers. Defaults to 1.
            cond_channels (int, optional): the channel of condition, up to how many types of condition we have. Defaults to 3.
        """ 
        super().__init__()
        self.channels = channels

        n_levels = len(channel_multpliers)
        channels_list = [channels * m for m in channel_multpliers]

        d_time_emb = channels * 4
        self.time_embedding_mlp = nn.Sequential(
            TimeEmbeddingBlock(channels),
            nn.Linear(channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb)
        )

        d_cond_emb = channels * 4 * cond_channels
        self.cond_embedding_mlp = nn.Sequential(
            ConditionEmbeddingBlock(channels),
            nn.Flatten(),
            nn.Linear(cond_channels * channels, d_cond_emb),
            nn.SiLU(),
            nn.Linear(d_cond_emb, d_cond_emb)
        )

        self.input_blocks = nn.ModuleList([])

        self.input_blocks.append(TimeStepEmbedSequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)))
        
        input_block_channels = [channels]

        for i in range(n_levels):
            for _ in range(n_res_blocks):
                layers = [ResBlock(channels, d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]

                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, cond_channels, n_heads, transformer_layers))

                self.input_blocks.append(TimeStepEmbedSequential(*layers))
                input_block_channels.append(channels)
            
            if i != n_levels - 1:
                self.input_blocks.append(TimeStepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)
        
        self.middle_block = TimeStepEmbedSequential(
            ResBlock(channels, d_time_emb),
            SpatialTransformer(channels, cond_channels, n_heads, transformer_layers),
            ResBlock(channels, d_time_emb)
        )

        self.output_blocks = nn.ModuleList([])

        for i in reversed(range(n_levels)):

            for j in range(n_res_blocks+1):
                layers = [ResBlock(channels+input_block_channels.pop(), d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]

                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, cond_channels, n_heads, transformer_layers))
                
                if i != 0 and j == n_res_blocks:
                    layers.append(UpSample(channels))
                
                self.output_blocks.append(TimeStepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
        )

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

        t_emb = self.time_embedding_mlp(time_steps)
        cond_emb = self.cond_embedding_mlp(cond).view(x.shape[0], cond.shape[-1], -1)

        for module in self.input_blocks:
            x = module(x, t_emb, cond_emb)
            x_input_block.append(x)
        
        x = self.middle_block(x, t_emb, cond_emb)

        for module in self.output_blocks:
            x = module(torch.cat([x, x_input_block.pop()], dim=1), t_emb, cond_emb)
        
        return self.out(x)

def _test_time_step_embedding():
    import matplotlib.pyplot as plt
    unet = UNetXAttn(in_channels=1,
                    out_channels=1,
                    channels=64, 
                    channel_multpliers=[],
                    n_res_blocks=1,
                    attention_levels=[],
                    n_heads=1,
                    transformer_layers=1,
                    d_cond=1)
    time_steps = torch.arange(0, 1000)
    t_emb = unet.time_step_embedding(time_steps)
    plt.figure()
    plt.imshow(t_emb.detach().numpy(), cmap='jet', aspect='auto', origin='lower')
    plt.xlabel('Dimension')
    plt.ylabel('Diffusion Time')
    plt.title('Fouriour Features for Diffusion Time')
    plt.colorbar()
    plt.show()

def _test_pos_embedding():
    import matplotlib.pyplot as plt
    unet = UNetXAttn(in_channels=1,
                    out_channels=1,
                    channels=64, 
                    channel_multpliers=[],
                    n_res_blocks=1,
                    attention_levels=[],
                    n_heads=1,
                    transformer_layers=1,
                    d_cond=1)
    x_ranges = torch.linspace(0, 64, 641)
    y_ranges = torch.linspace(0, 64, 641)
    pos_emb = unet.pos_embedding(x_ranges, y_ranges)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(pos_emb.detach().numpy()[:, 0, :], cmap='jet', aspect='auto', origin='lower')
    plt.xlabel('Dimension')
    plt.ylabel('Position x')
    plt.title('Embedding for Position x')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(pos_emb.detach().numpy()[:, 1, :], cmap='jet', aspect='auto', origin='lower')
    plt.xlabel('Dimension')
    plt.ylabel('Position y')
    plt.title('Embedding for Position y')
    plt.colorbar()

    plt.show()

def _test_unet():
    unet = UNet(in_channels=1,
                out_channels=1,
                channels=64,
                channel_multpliers=[1, 2, 4, 8],
                n_res_blocks=2,
                attention_levels=[0, 1, 2],
                n_heads=8,
                cond_channels=0)
    # print(unet)
    print(f"total params: {sum(p.numel() for p in unet.parameters())}")
    input = torch.randn((2, 1, 64, 64))
    time_steps = torch.randn(2)
    # cond = torch.randn((2, 3))
    output = unet(input, time_steps, cond=None)
    print(output.shape)

def _test_xunet():
    unet = UNetXAttn(in_channels=1,
                     out_channels=1,
                     channels=64,
                     channel_multpliers=[1, 2, 4, 8],
                     n_res_blocks=2,
                     attention_levels=[0, 1, 2],
                     n_heads=8,
                     transformer_layers=1,
                     cond_channels=3)
    # print(unet)
    print(f"total params: {sum(p.numel() for p in unet.parameters())}")
    input = torch.randn((2, 1, 64, 64))
    time_steps = torch.randn(2)
    cond = torch.randn((2, 3))
    output = unet(input, time_steps, cond)
    print(output.shape)

if __name__ == "__main__":
    # _test_time_step_embedding()
    # _test_pos_embedding()
    _test_unet()
    # _test_xunet()