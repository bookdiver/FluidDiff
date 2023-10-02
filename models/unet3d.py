import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from einops import rearrange, repeat

from rotary_embedding_torch import RotaryEmbedding

# helpers functions

def exists(x):
    return x is not None

def noop(*args, **kwargs):
    pass

def is_odd(n):
    return (n % 2) == 1

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# relative positional bias

class RelativePositionBias(nn.Module):
    """ Generate relative positional bias for frame position.
        Each head has a separate bias.
    """
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor, 
        num_buckets: int=32, 
        max_distance: int=128
    ):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(
        self, 
        n: int, 
        device: torch.device
    ):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules

class EMA():
    """ Exponential Moving Average tricks for training, in order to improve the model generalization.
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

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
        self.conv = nn.Conv3d(in_channels,
                              in_channels,
                              kernel_size=(1, 3, 3),
                              stride=(1, 1, 1),
                              padding=(0, 1, 1),
                              padding_mode='circular')
    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels,
                              in_channels,
                              kernel_size=(1, 3, 3),
                              stride=(1, 2, 2),
                              padding=(0, 0, 0))
    def forward(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1, 0, 0), mode='circular')
        x = self.conv(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

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

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
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
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
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
        # x input size: (b, c, f, h, w)
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda z: rearrange(z, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)

# attention along space and time

class EinopsToAndFrom(nn.Module):
    """ Rearrange the tensor to fit the format for specific function.
        The order is given as simple string, e.g. 'b c f h w' or 'b c (h w)'.
    """
    def __init__(
        self, 
        from_einops: str, 
        to_einops: str, 
        fn: callable
    ):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(
        self, 
        x: torch.Tensor, 
        **kwargs
    ):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class Attention(nn.Module):
    """
    Do the attention over the temporal or spatial dimension, depends on the input size.
    """
    def __init__(
        self,
        channels: int,
        heads: int=4,
        channels_per_head: int=32,
        rotary_emb: RotaryEmbedding=None
    ):
        super().__init__()
        self.scale = channels_per_head ** -0.5
        self.heads = heads
        hidden_channels = channels_per_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(channels, hidden_channels * 3, bias = False)
        self.to_out = nn.Linear(hidden_channels, channels, bias = False)

    def forward(
        self,
        x: torch.Tensor,
        pos_bias = None,
    ):
        # x input size: (b, (h*w), f, c) for temporal attention
        #               (b, f, (h*w), c) for spatial attention
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        # split out heads

        q, k, v = map(lambda z: rearrange(z, '... n (h d) -> ... h n d', h = self.heads), qkv)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

# model

class Unet3D(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_channels: int=None,
        out_channels: int=None,
        channel_mults: tuple=(1, 2, 4, 8),
        attn_heads: int=8,
        attn_channels_per_head: int=32,
        init_conv_channels: int=None,
        init_conv_kernel_size: int=5,
        use_spatial_attention: bool=True,
        use_temporal_attention: bool=True,
        resnet_groups: int=8
    ):
        super().__init__()
        self.channels = channels

        # temporal attention and its relative positional encoding

        rotary_emb = RotaryEmbedding(min(32, attn_channels_per_head))

        temporal_attn = lambda dimension: EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(dimension, heads = attn_heads, channels_per_head = attn_channels_per_head, rotary_emb = rotary_emb))

        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet

        # initial conv

        init_conv_channels = default(init_conv_channels, channels)
        assert is_odd(init_conv_kernel_size)

        init_padding = init_conv_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_conv_channels, kernel_size=(1, init_conv_kernel_size, init_conv_kernel_size), padding = (0, init_padding, init_padding))

        self.init_temporal_attn = Residual(PreNorm(init_conv_channels, temporal_attn(init_conv_channels)))

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
                Residual(PreNorm(channels_out, SpatialLinearAttention(channels_out, heads = attn_heads))) if use_spatial_attention else nn.Identity(),
                Residual(PreNorm(channels_out, temporal_attn(channels_out))) if use_temporal_attention else nn.Identity(),
                Downsample(channels_out) if not is_last else nn.Identity()
            ]))

        mid_channels = channels_list[-1]
        self.mid_block1 = block_klass_cond(mid_channels, mid_channels)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_channels, heads = attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_channels, spatial_attn)) if use_spatial_attention else nn.Identity()
        self.mid_temporal_attn = Residual(PreNorm(mid_channels, temporal_attn(mid_channels))) if use_temporal_attention else nn.Identity()

        self.mid_block2 = block_klass_cond(mid_channels, mid_channels)

        for ind, (channels_in, channels_out) in enumerate(reversed(in_out_channels)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(channels_out * 2, channels_in),
                block_klass_cond(channels_in, channels_in),
                Residual(PreNorm(channels_in, SpatialLinearAttention(channels_in, heads = attn_heads))) if use_spatial_attention else nn.Identity(),
                Residual(PreNorm(channels_in, temporal_attn(channels_in))) if use_temporal_attention else nn.Identity(),
                Upsample(channels_in) if not is_last else nn.Identity()
            ]))

        out_channels = default(out_channels, channels)
        self.final_conv = nn.Sequential(
            block_klass(init_conv_channels * 3, init_conv_channels),
            nn.Conv3d(init_conv_channels, out_channels, 1)
        )

    def forward(
        self,
        x,
        time,
        cond = None
    ):
        _, _, f, _, _ = x.shape
        device = x.device

        time_rel_pos_bias = self.time_rel_pos_bias(f, device=device)

        x = self.init_conv(x)

        x = self.init_temporal_attn(x, pos_bias = time_rel_pos_bias)

        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance

        if cond is not None:
            cond_emb = self.cond_conv(cond)
            cond_emb = repeat(cond_emb, 'b c h w -> b c f h w', f = f)
        else:
            cond_emb = torch.zeros_like(x)

        x = torch.cat((x, cond_emb), dim = 1)

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias = time_rel_pos_bias)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        return self.final_conv(x)

def test():
    model = Unet3D(channels=1,
                   cond_channels=1,
                   channel_mults=(1, 2, 4, 8, 16),
                   init_conv_channels=32,
                   init_conv_kernel_size=5,
                   use_spatial_attention=False,
                   use_temporal_attention=False
    )
    print(f"the number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # print(model)
    inputs = torch.randn(2, 1, 20, 64, 64)
    time = torch.randn(2)
    cond = torch.randn(2, 1, 64, 64)
    outputs = model(inputs, time, cond)
    print(outputs.shape)

if __name__ == '__main__':
    test()