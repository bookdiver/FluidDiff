from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

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

class LayerNorm(nn.Module):
    """ An unlearnable layer norm
    """
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    """ Pre-normalization
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, feat_channels, d_mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(feat_channels, feat_channels * d_mult),
            nn.Dropout(dropout),
            nn.Linear(feat_channels * d_mult, feat_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AttnBlock(nn.Module):
    def __init__(self, feat_channels: int, n_heads: int, d_head: int):
        """ Simple self-attention

        Args:
            feat_channels (int): the channel of the input tensor
            n_heads (int): the number of heads
            d_head (int): the size of each head

        """      
        super().__init__()
        self.n_heads = n_heads
        self.scale = d_head ** -0.5
        d_attn = n_heads * d_head
        self.to_qkv = nn.Conv2d(feat_channels, d_attn * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(d_attn, feat_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ do the self-attention without condition

        Args:
            x (torch.Tensor): input feature map, with size of (batch_size, n_channels, height, width)

        Returns:
            torch.Tensor: the output feature map, with size of (batch_size, n_channels, width, height)
        """
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.n_heads), qkv)
        q = q * self.scale

        similarity = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = similarity.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        out = self.to_out(out)
        return out

class LinearAttnBlock(nn.Module):
    def __init__(self, feat_channels: int, n_heads: int, d_head: int):
        """ A linear attention block
        """
        super().__init__()
        self.feat_channels = feat_channels
        self.n_heads = n_heads
        self.scale = d_head ** -0.5
        d_attn = n_heads * d_head
        self.to_qkv = nn.Conv2d(feat_channels, d_attn * 3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(d_attn, feat_channels, kernel_size=1),
            LayerNorm(feat_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ do the linear self-attention without condition

        Args:
            x (torch.Tensor): input feature map, with size of (batch_size, n_channels, height, width)

        Returns:
            torch.Tensor: the output feature map, with size of (batch_size, n_channels, width, height)
        """
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.n_heads), qkv)
        
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y)  -> b (h c) x y', h=self.n_heads, x=h, y=w)
        out = self.to_out(out)
        return out


class CrossAttnBlock(nn.Module):
    def __init__(self, feat_channels: int, cond_channels: int, n_heads: int, d_head: int, is_inplace: bool=True):
        """ Cross Attention Block

        Args:
            feat_channels (int): the channel of the input tensor
            cond_channels (int): the channel of the condition tensor
            n_heads (int): the number of heads
            d_head (int): the size of each head
            is_inplace (bool, optional): Whether to perform the attention softmax inplace to save memory. Defaults to True.
        """        
        super().__init__()
        self.feat_channels = feat_channels
        self.cond_channels = cond_channels
        self.n_heads = n_heads
        self.is_inplace = is_inplace

        self.scale = d_head ** -0.5

        d_attn = n_heads * d_head
        self.q = nn.Linear(feat_channels, d_attn, bias=False)
        self.k_cond = nn.Linear(cond_channels, d_attn, bias=False)
        self.k_x = nn.Linear(feat_channels, d_attn, bias=False)
        self.v_cond = nn.Linear(cond_channels, d_attn, bias=False)
        self.v_x = nn.Linear(feat_channels, d_attn, bias=False)

        self.to_out = nn.Linear(d_attn, feat_channels)

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
    def __init__(self, feat_channels: int, cond_channels: int, n_heads: int, d_head: int):
        """ Basic transformer block

        Args:
            feat_channels (int): the input embedding size, i.e. the channel of the input tensor
            cond_channels (int): the size of the conditioning vector, i.e. the channel of the conditioning tensor
            n_heads (int): number of heads
            d_heads (int): the size of each head
        """
        super().__init__()
        self.attn1 = CrossAttnBlock(feat_channels, feat_channels, n_heads, d_head)
        self.norm1 = nn.LayerNorm(feat_channels)

        self.attn2 = CrossAttnBlock(feat_channels, cond_channels, n_heads, d_head)
        self.norm2 = nn.LayerNorm(feat_channels)

        self.ff = FeedForward(feat_channels)
        self.norm3 = nn.LayerNorm(feat_channels)
    
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
    def __init__(self, feat_channels: int, cond_channels: int, n_heads: int, n_layers: int):
        """ Spatial Transformer

        Args:
            feat_channels (int): the number of channels of the input
            cond_channels (int): the number of channels of conditioning vector
            n_heads (int): the number of heads
            n_layers (int): the number of transformer layers
        """ 
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=feat_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(feat_channels, feat_channels, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(feat_channels, cond_channels, n_heads, feat_channels // n_heads) for _ in range(n_layers)
        ])
        self.proj_out = nn.Conv2d(feat_channels, feat_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, cond: Union[torch.Tensor, None]) -> torch.Tensor:
        """ do the spatial transformation

        Args:
            x (torch.Tensor): the input tensor, with size of (batch_size, n_channels, height, width)
            cond (Union[torch.Tensor, None]): the conditioning tensor, with size of (batch_size, n_channels, d_cond)

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

def _test_attention():
    attn = AttnBlock(feat_channels=64, n_heads=8, d_head=32)
    x = torch.randn(2, 64, 16, 16)
    print(f"Input shape: {x.shape}")
    out = attn(x)
    print(f"Output shape: {out.shape}")

def _test_linear_attention():
    attn = LinearAttnBlock(feat_channels=64, n_heads=8, d_head=32)
    x = torch.randn(2, 64, 16, 16)
    print(f"Input shape: {x.shape}")
    out = attn(x)
    print(f"Output shape: {out.shape}")

def _test_cross_attention():
    attn = CrossAttnBlock(feat_channels=64, cond_channels=16, n_heads=4, d_head=32)
    x = torch.randn(2, 64, 64)
    cond = torch.randn(2, 10, 16)
    print(f"Input shape: {x.shape}")
    print(f"Conditioning shape: {cond.shape}")
    out = attn(x, cond)
    print(f"Output shape: {out.shape}")

def _test_basic_transformer_block():
    block = BasicTransformerBlock(feat_channels=64, cond_channels=32, n_heads=4, d_head=64)
    x = torch.randn(2, 16, 64)
    cond = torch.randn(2, 1, 32)
    print(f"Input shape: {x.shape}")
    print(f"Conditioning shape: {cond.shape}")
    out = block(x, cond)
    print(f"Output shape: {out.shape}")

def _test_spatial_transformer():
    st = SpatialTransformer(feat_channels=64, cond_channels=3, n_heads=4, n_layers=1)
    x = torch.randn(2, 64, 16, 16)
    print(f"Input feature map shape: {x.shape}")
    cond = torch.randn(2, 3, 64)
    print(f"Conditioning vector shape: {cond.shape}")
    out = st(x, cond)
    print(f"Output feature map shape: {out.shape}")

if __name__ == '__main__':
    # _test_attention()
    # _test_linear_attention()
    # _test_cross_attention()
    # _test_basic_transformer_block()
    _test_spatial_transformer()

