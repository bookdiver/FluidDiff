from typing import Optional, List

import logging

import torch
import torch.nn as nn

from transformer4unet import *
from blocks4unet import *

logging.basicConfig(level=logging.INFO)

class UNet(nn.Module):
    def __init__(self,
                 *,
                 in_channels: int,
                 out_channels: int,
                 cascade_channels: list,
                 condition_channels: int,
                 embedding_size: int,
                 down_blocks: list=["ResDownBlock", "AttnDownBlock", "AttnDownBlock", "AttnDownBlock"],
                 middle_block_type: str = "AttnMidBlock",
                 up_blocks: list=["AttnUpBlock", "AttnUpBlock", "AttnUpBlock", "ResUpBlock"],
                 n_res_blocks: int=2,
                 n_attn_heads: int=8,
                 n_res_groups: int=8,
                 n_attn_groups: int=16,
                 n_transformer_layers: int=1,
                 ) -> None:
        """ A UNet model to predict the noise 

        Args:
            in_channels (int): input sample channels
            out_channels (int): output sample channels
            cascade_channels (list): the channels of each block
            condition_channels (int): the channels of the condition, used when doing the cross attention
            embedding_size (int): the size of the embedding, for both time and condition
            down_blocks (list, optional): the down block types, can be chosen from ["ResDownBlock", "AttnDownBlock", "XAttnDownBlock"].
            Defaults to ["ResDownBlock", "AttnDownBlock", "AttnDownBlock", "AttnDownBlock"].

            middle_block_type (str, optional): the type of middle bottleneck block, can be chosen from ["ResMidBlock", "AttnMidBlock"]. Defaults to "AttnMidBlock".

            up_blocks (list, optional): the up block types, can be chosen from ["ResUpBlock", "AttnUpBlock", "XAttnUpBlock"]. 
            Defaults to ["AttnUpBlock", "AttnUpBlock", "AttnUpBlock", "ResUpBlock"].

            n_res_blocks (int, optional): the number of residual block in each level, also the number of transformers when using attention. Defaults to 2.
            n_attn_heads (int, optional): the number of attention heads. Defaults to 8.
            n_res_groups (int, optional): the number of groups when do the group normalization in residual blocks. Defaults to 8.
            n_attn_groups (int, optional): the number of groups when do the group normalization in attention blocks. Defaults to 16.
            n_transformer_layers (int, optional): the number of transformer layer in attention blocks. Defaults to 1.
        """        
        super().__init__()
        n_levels = len(cascade_channels) - 1
        assert n_levels == len(down_blocks) == len(up_blocks), "The number of levels should be equal to the number of down_blocks and up_blocks"

        # 1. embedding mlp for time and condition
        self.time_embedding_mlp = get_embedding_block(
            embedding_object="DiffusionStep",
            embedding_size=cascade_channels[0],
            embedding_channels=1,
            out_size=embedding_size
        )

        self.condition_embedding_mlp = get_embedding_block(
            embedding_object="Condition",
            embedding_size=cascade_channels[0],
            embedding_channels=condition_channels,
            out_size=embedding_size
        )

        # 2. initial convolution
        self.in_conv = nn.Conv2d(in_channels, cascade_channels[0], kernel_size=3, padding=1)

        # 3. down blocks
        down_block_channels = []
        self.down_blocks = nn.ModuleList([])

        for i in range(n_levels):
            is_last = (i == n_levels-1)
            block_type = down_blocks[i]
            if block_type == "ResDownBlock":
                down_block = get_down_block(
                    block_type='ResBlock',
                    in_channels=cascade_channels[i],
                    out_channels=cascade_channels[i+1],
                    res_embedding_channels=1,
                    res_embedding_size=embedding_size,
                    xattn_channels=condition_channels,
                    is_last=is_last,
                    n_res_groups=n_res_groups,
                    n_attn_groups=n_attn_groups,
                    n_attn_heads=n_attn_heads,
                    n_res_layers=n_res_blocks,
                    n_transformer_layers=n_transformer_layers
                )
            elif block_type == "AttnDownBlock":
                down_block = get_down_block(
                    block_type='AttnBlock',
                    in_channels=cascade_channels[i],
                    out_channels=cascade_channels[i+1],
                    res_embedding_channels=1,
                    res_embedding_size=embedding_size,
                    xattn_channels=condition_channels,
                    is_last=is_last,
                    n_res_groups=n_res_groups,
                    n_attn_groups=n_attn_groups,
                    n_attn_heads=n_attn_heads,
                    n_res_layers=n_res_blocks,
                    n_transformer_layers=n_transformer_layers
                )
            elif block_type == "XAttnDownBlock":
                down_block = get_down_block(
                    block_type='CrossAttnBlock',
                    in_channels=cascade_channels[i],
                    out_channels=cascade_channels[i+1],
                    res_embedding_channels=1,
                    res_embedding_size=embedding_size,
                    xattn_channels=condition_channels,
                    is_last=is_last,
                    n_res_groups=n_res_groups,
                    n_attn_groups=n_attn_groups,
                    n_attn_heads=n_attn_heads,
                    n_res_layers=n_res_blocks,
                    n_transformer_layers=n_transformer_layers
                )
            else:
                raise ValueError(f"Unknown block type {block_type}")
            self.down_blocks.append(down_block)

        # 4. middle block
        if middle_block_type == "ResMidBlock":
            self.middle_block = get_mid_block(
                block_type='ResBlock',
                channels=cascade_channels[-1],
                res_embedding_channels=1,
                res_embedding_size=embedding_size,
                xattn_channels=None,
                n_res_groups=n_res_groups,
                n_attn_groups=n_attn_groups,
                n_attn_heads=n_attn_heads,
                n_res_layers=n_res_blocks,
                n_transformer_layers=n_transformer_layers
            )
        elif middle_block_type == "AttnMidBlock":
            self.middle_block = get_mid_block(
                block_type='AttnBlock',
                channels=cascade_channels[-1],
                res_embedding_channels=1,
                res_embedding_size=embedding_size,
                xattn_channels=None,
                n_res_groups=n_res_groups,
                n_attn_groups=n_attn_groups,
                n_attn_heads=n_attn_heads,
                n_res_layers=n_res_blocks,
                n_transformer_layers=n_transformer_layers
            )
        else:
            raise ValueError(f"Unknown block type {middle_block_type}")

        # 5. up blocks
        self.up_blocks = nn.ModuleList([])

        for i in range(n_levels):
            is_last = (i == 0)
            block_type = up_blocks[i]
            if block_type == "ResUpBlock":
                up_block = get_up_block(
                    block_type='ResBlock',
                    in_channels=2*cascade_channels[-(i+1)] ,
                    out_channels=cascade_channels[-(i+2)],
                    res_embedding_channels=1,
                    res_embedding_size=embedding_size,
                    xattn_channels=condition_channels,
                    is_last=is_last,
                    n_res_groups=n_res_groups,
                    n_attn_groups=n_attn_groups,
                    n_attn_heads=n_attn_heads,
                    n_res_layers=n_res_blocks,
                    n_transformer_layers=n_transformer_layers
                )
            elif block_type == "AttnUpBlock":
                up_block = get_up_block(
                    block_type='AttnBlock',
                    in_channels=2*cascade_channels[-(i+1)],
                    out_channels=cascade_channels[-(i+2)],
                    res_embedding_channels=1,
                    res_embedding_size=embedding_size,
                    xattn_channels=condition_channels,
                    is_last=is_last,
                    n_res_groups=n_res_groups,
                    n_attn_groups=n_attn_groups,
                    n_attn_heads=n_attn_heads,
                    n_res_layers=n_res_blocks,
                    n_transformer_layers=n_transformer_layers
                )
            elif block_type == "XAttnUpBlock":
                up_block = get_up_block(
                    block_type='CrossAttnBlock',
                    in_channels=2*cascade_channels[-(i+1)],
                    out_channels=cascade_channels[-(i+2)],
                    res_embedding_channels=1,
                    res_embedding_size=embedding_size,
                    xattn_channels=condition_channels,
                    is_last=is_last,
                    n_res_groups=n_res_groups,
                    n_attn_groups=n_attn_groups,
                    n_attn_heads=n_attn_heads,
                    n_res_layers=n_res_blocks,
                    n_transformer_layers=n_transformer_layers
                )
            else:
                raise ValueError(f"Unknown block type {block_type}")
            self.up_blocks.append(up_block)
        
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, cascade_channels[0]),
            nn.SiLU(),
            nn.Conv2d(cascade_channels[0], out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """ Forward pass

        Args:
            x (torch.Tensor): input sample, with shape (B, C, H, W)
            time_steps (torch.Tensor): diffusion time steps, with shape (B, )
            cond (torch.Tensor): conditioning tensor, with shape (B, C_cond)

        Returns:
            torch.Tensor: noise prediction, with shape (B, C, H, W)
        """ 
        x_input_block = []

        t_emb = self.time_embedding_mlp(time_steps)
        c_emb = self.condition_embedding_mlp(cond)

        x = self.in_conv(x)

        for module in self.down_blocks:
            x = module(x, t_emb, c_emb)
            x_input_block.append(x)
        
        x = self.middle_block(x, t_emb, c_emb)

        for module in self.up_blocks:
            x = module(torch.cat([x, x_input_block.pop()], dim=1), t_emb, c_emb)
        
        return self.out_conv(x)

##############################################################################################################################################################
def _test_unet():
    unet = UNet(
        in_channels=1,
        out_channels=1,
        cascade_channels=[32, 64, 128, 256],
        condition_channels=3,
        embedding_size=128,
        down_blocks=["ResDownBlock", "XAttnDownBlock", "XAttnDownBlock"],
        middle_block_type="AttnMidBlock",
        up_blocks=["XAttnUpBlock", "XAttnUpBlock", "ResUpBlock"],
    )
    # print(unet)
    input = torch.randn(2, 1, 64, 64)
    time_steps = torch.randn(2)
    cond = torch.randn(2, 3)
    output = unet(input, time_steps, cond)
    print(output.shape)

if __name__ == "__main__":
    _test_unet()