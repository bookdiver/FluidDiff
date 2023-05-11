import torch
from torch import nn
import torch.nn.functional as F
import math
from functools import partial
from einops import rearrange


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def is_odd(n):
    return (n % 2) == 1

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
class SinusoidalPosEmb(nn.Module):
    """ Sinusoidal positional embedding for diffusion steps.
        Notice that the diffusion time here does NOT need to be normalized.
    """
    def __init__(
        self, 
        dim: int
    ):
        super().__init__()
        self.dim = dim

    def forward(
        self, 
        x: torch.Tensor
    ):
        # x input size: (b)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              in_channels,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1))
    def forward(self, x):
        x = F.interpolate(x, scale_factor=(2, 2), mode='nearest')
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              in_channels,
                              kernel_size=(3, 3),
                              stride=(2, 2),
                              padding=(1, 1))
    def forward(self, x):
        x = self.conv(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)



class SpatialLinearAttention(nn.Module):
    """ Do the attention over the spatial dimensions only. At this time, 
        the temporal dimension is ascribed into the batch dimension.
    """
    def __init__(
        self, 
        channel_dim: int, 
        heads: int=4, 
        dim_head: int=32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(channel_dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, channel_dim, 1)

    def forward(
        self, 
        x: torch.Tensor
    ):
        # x input size: (b, c, h, w)
        b, c, h, w = x.shape

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda z: rearrange(z, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x=h, y=w)
        out = self.to_out(out)
        return out
    
class Unet2D_Spatial(nn.Module):
    def __init__(self,
                 channels: int,
                 cond_channels: int=None,
                 out_channels: int=None,
                 channel_mults: tuple=(1, 2, 4, 8),
                 attn_heads: int=8,
                 attn_channels_per_head: int=32,
                 init_conv_channels: int=None,
                 init_conv_kernel_size: int=5,
                 resnet_groups: int=8):
        super().__init__()
        self.channels = channels
        
        init_conv_channels = default(init_conv_channels, channels)
        assert is_odd(init_conv_kernel_size)

        init_padding = init_conv_kernel_size // 2
        self.init_conv = nn.Conv2d(channels, init_conv_channels, kernel_size=(1, init_conv_kernel_size), padding=(0, init_padding))

        self.init_attn = Residual(PreNorm(init_conv_channels, SpatialLinearAttention(init_conv_channels, heads=attn_heads, dim_head=attn_channels_per_head)))

        # dimensions

        channels_list = [2*init_conv_channels, *map(lambda m: 2*init_conv_channels * m, channel_mults)]
        in_out_channels = list(zip(channels_list[:-1], channels_list[1:]))

        # time conditioning

        time_dim = init_conv_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(init_conv_channels*2),
            nn.Linear(init_conv_channels*2, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning

        self.cond_conv = nn.Sequential(
            nn.Conv2d(cond_channels, init_conv_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(init_conv_channels, init_conv_channels, kernel_size=init_conv_kernel_size, stride=1, padding=init_padding)
        )


        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out_channels)

        # block type

        block_klass = partial(ResnetBlock, groups = resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim = time_dim)

        # modules for all layers

        for ind, (channels_in, channels_out) in enumerate(in_out_channels):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(channels_in, channels_out),
                block_klass_cond(channels_out, channels_out),
                Residual(PreNorm(channels_out, SpatialLinearAttention(channels_out, heads = attn_heads))),
                Downsample(channels_out) if not is_last else nn.Identity()
            ]))

        mid_channels = channels_list[-1]
        self.mid_block1 = block_klass_cond(mid_channels, mid_channels)

        self.mid_spatial_attn = Residual(PreNorm(mid_channels, SpatialLinearAttention(mid_channels, heads = attn_heads)))

        self.mid_block2 = block_klass_cond(mid_channels, mid_channels)

        for ind, (channels_in, channels_out) in enumerate(reversed(in_out_channels)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(channels_out * 2, channels_in),
                block_klass_cond(channels_in, channels_in),
                Residual(PreNorm(channels_in, SpatialLinearAttention(channels_in, heads = attn_heads))),
                Upsample(channels_in) if not is_last else nn.Identity()
            ]))

        out_channels = default(out_channels, channels)
        self.final_conv = nn.Sequential(
            block_klass(init_conv_channels * 3, init_conv_channels),
            nn.Conv2d(init_conv_channels, out_channels, 1)
        )

    def forward(
        self,
        x,
        time,
        cond = None
    ):
        _, _, f, _ = x.shape
        device = x.device

        x = self.init_conv(x)

        x = self.init_attn(x)

        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance

        if cond is not None:
            cond_emb = self.cond_conv(cond)
            # cond_emb = repeat(cond_emb, 'b c s -> b c f s', f = f)
        else:
            cond_emb = torch.zeros_like(x)

        x = torch.cat((x, cond_emb), dim = 1)

        h = []

        for block1, block2, spatial_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        return self.final_conv(x)
