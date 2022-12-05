from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class GeGLU(nn.Module):
    """ GeGLU activation
    GeGLU(x) = (Wx+b) * GELU(Vx+c)
    """
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.fc = nn.Linear(dim_in, 2 * dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x, gate = x.chunk(2, dim=-1)
        output = x * F.gelu(gate)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(d_model, d_model * d_mult),
            nn.Dropout(dropout),
            nn.Linear(d_model * d_mult, d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int, is_inplace: bool=True):
        """ Cross Attention Block

        Args:
            d_model (int): the input embedding size, i.e. the channel of the input tensor
            d_cond (int): the condition embedding size, i.e. the channel of the conditioning tensor
            n_heads (int): the number of heads
            d_head (int): the size of each head
            is_inplace (bool, optional): Whether to perform the attention softmax inplace to save memory. Defaults to True.
        """        
        super().__init__()
        self.d_model = d_model
        self.d_cond = d_cond
        self.n_heads = n_heads
        self.is_inplace = is_inplace

        self.scale = d_head ** -0.5

        d_attn = n_heads * d_head
        self.q = nn.Linear(d_model, d_attn, bias=False)
        self.k_cond = nn.Linear(d_cond, d_attn, bias=False)
        self.k_x = nn.Linear(d_model, d_attn, bias=False)
        self.v_cond = nn.Linear(d_cond, d_attn, bias=False)
        self.v_x = nn.Linear(d_model, d_attn, bias=False)

        self.to_out = nn.Linear(d_attn, d_model)

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = q.reshape(*q.shape[:2], self.n_heads, -1)
        k = k.reshape(*k.shape[:2], self.n_heads, -1)
        v = v.reshape(*v.shape[:2], self.n_heads, -1)

        attn = torch.einsum('bihd, bjhd -> bhij', q, k) * self.scale

        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bhij, bjhd -> bihd', attn, v)
        return self.to_out(out.reshape(*out.shape[:2], -1))
    
    def forward(self, x: torch.Tensor, cond: Union[torch.Tensor, None]) -> torch.Tensor:
        """ do the cross attention between x and cond, when cond is None, do the self attention

        Args:
            x (torch.Tensor): input tensor, with size of (batch_size, width*height, n_channels)
            cond (Union[torch.Tensor, None]): the conditioning tensor, with size of (batch_size, condition_dim, n_channels)

        Returns:
            torch.Tensor: the output tensor, with size of (batch_size, width*height, n_channels)
        """
        q = self.q(x)
        if cond is None:
            # if there's no condition, it becomes the self-attention
            k = self.k_x(x)
            v = self.v_x(x)
        else:
            k = self.k_cond(cond)
            v = self.v_cond(cond)
        return self.normal_attention(q, k, v)

class BasicTransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int):
        """ Basic transformer block

        Args:
            d_model (int): the input embedding size, i.e. the channel of the input tensor
            d_cond (int): the size of the conditioning vector, i.e. the channel of the conditioning tensor
            n_heads (int): number of heads
            d_heads (int): the size of each head
        """
        super().__init__()
        self.attn1 = CrossAttnBlock(d_model, d_model, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)

        self.attn2 = CrossAttnBlock(d_model, d_cond, n_heads, d_head)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, cond: Union[torch.Tensor, None]) -> torch.Tensor:
        """ do the cross attention between x and cond, when cond is None, do the self attention

        Args:
            x (torch.Tensor): input tensor, with size of (batch_size, width*height, n_channels)
            cond (Union[torch.Tensor, None]): the conditioning tensor, with size of (batch_size, condition_dim, n_channels)

        Returns:
            torch.Tensor: the output tensor, with size of (batch_size, width*height, n_channels)
        """        
        # self-attention
        x = x + self.attn1(self.norm1(x), None)
        # cross-attention
        x = x + self.attn2(self.norm2(x), cond)
        x = x + self.ff(self.norm3(x))
        return x

class SpatialTransformer(nn.Module):
    def __init__(self, d_model: int, d_cond: int, n_heads: int, n_layers: int):
        """ Spatial Transformer

        Args:
            d_model (int): the number of channels of the input
            d_cond (int): the size of conditioning vector
            n_heads (int): the number of heads
            n_layers (int): the number of transformer layers
        """ 
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=d_model, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(d_model, d_cond, n_heads, d_model // n_heads) for _ in range(n_layers)
        ])
        self.proj_out = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, cond: Union[torch.Tensor, None]) -> torch.Tensor:
        """ do the spatial transformation

        Args:
            x (torch.Tensor): the input tensor, with size of (batch_size, n_channels, height, width)
            cond (Union[torch.Tensor, None]): the conditioning tensor, with size of (batch_size, n_channels, d_cond), n_channels should be same as x

        Returns:
            torch.Tensor: the output tensor, with size of (batch_size, channels, height, width)
        """        
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        for block in self.transformer_blocks:
            if cond is not None:
                cond = cond.permute(0, 2, 1).contiguous()
            x = block(x, cond)
        
        x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        x = self.proj_out(x)
        return x + x_in

def _test_cross_attention():
    attn = CrossAttnBlock(d_model=64, d_cond=16, n_heads=4, d_head=32)
    x = torch.randn(2, 64, 64)
    cond = torch.randn(2, 10, 16)
    out = attn(x, cond)
    print(out.shape)

def _test_basic_transformer_block():
    block = BasicTransformerBlock(d_model=64, d_cond=32, n_heads=4, d_head=64)
    x = torch.randn(2, 16, 64)
    cond = torch.randn(2, 1, 32)
    out = block(x, cond)
    print(out.shape)

def _test_spatial_transformer():
    st = SpatialTransformer(d_model=64, d_cond=32, n_heads=4, n_layers=1)
    x = torch.randn(2, 64, 16, 16)
    cond = torch.randn(2, 32, 3)
    out = st(x, cond)
    print(out.shape)

if __name__ == '__main__':
    _test_cross_attention()
    _test_basic_transformer_block()
    _test_spatial_transformer()

