from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, 
                 channels: int, 
                 num_channels_per_head: Optional[int]=None, 
                 norm_num_groups: int=32,
                 rescale_output_factor: float=1.0,
                 eps: float=1e-5,
                 ):
        """ Muti-head attention block.

        Args:
            channels (int): Number of channels.
            num_channels_per_head (Optional[int], optional): Number of channels in one head. Defaults to None.
            norm_num_groups (int, optional): Number of groups in group normalization. Defaults to 32.
            rescale_output_factor (float, optional): The factor to rescale the output of attention block. Defaults to 1.0.
            eps (float, optional): The epsilon used in group normalization. Defaults to 1e-5.
        """
        super().__init__()
        self.channels = channels

        self.num_heads = channels // num_channels_per_head if num_channels_per_head else 1
        self.head_size = num_channels_per_head if num_channels_per_head else channels
        self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=norm_num_groups, eps=eps, affine=True)

        self.wq = nn.Linear(self.channels, self.channels)
        self.wk = nn.Linear(self.channels, self.channels)
        self.wv = nn.Linear(self.channels, self.channels)

        self.rescale_output_factor = rescale_output_factor
        self.proj_atten = nn.Linear(self.channels, self.channels, 1)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """ Uncouple the channels dimension into two dimensions: head and head_size, the switch the head and query dimension.

        Args:
            x (torch.Tensor): input tensor with shape (batch_size, num_queries, channels=num_heads*head_size)

        Returns:
            torch.Tensor: tensor with shape (batch_size, num_heads, num_queries, head_size)
        """        
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch, channel, height, width = x.shape

        x = self.group_norm(x)
        x = x.view(batch, channel, -1).permute(0, 2, 1)

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        scale = 1 / math.sqrt(self.head_size)

        if self.num_heads > 1:
            q = self.transpose_for_scores(q)
            k = self.transpose_for_scores(k)
            v = self.transpose_for_scores(v)

            attention_scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        else:
            attention_scores = torch.baddbmm(
                torch.empty(q.shape[0], q.shape[1], k.shape[1], device=q.device, dtype=q.dtype),
                q,
                k.transpose(-1, -2),
                beta=0.0,
                alpha=scale,
            )
        
        attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(attention_scores.dtype)

        if self.num_heads > 1:
            x = torch.matmul(attention_probs, v)
            x = x.permute(0, 2, 1, 3).contiguous()
            new_x_shape = x.size()[:-2] + (self.num_heads * self.head_size,)
            x = x.view(*new_x_shape)
        else:
            x = torch.bmm(attention_probs, v)
        
        x = self.proj_atten(x)
        x = x.permute(0, 2, 1).view(batch, channel, height, width)

        x = (x + residual) / self.rescale_output_factor
        return x
