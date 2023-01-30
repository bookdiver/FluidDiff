import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce

def exists(val):
    return val is not None

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class AttnBlock(nn.Module):
    def __init__(self, 
                 feat_channels: int, 
                 *,
                 num_heads: int=4,
                 head_channels: int=32,
                 linear: bool=False):    
        super().__init__()
        self.feat_channels = feat_channels

        self.num_heads = num_heads
        self.head_channels = head_channels
        attn_channels = num_heads * head_channels

        self.linear = linear

        self.scale = self.head_channels ** -0.5

        self.to_qkv = nn.Conv2d(feat_channels, attn_channels * 3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(attn_channels, feat_channels, kernel_size=1, bias=False),
            LayerNorm(feat_channels) if linear else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.num_heads), qkv)

        if self.linear:
            q = q.softmax(dim=-2)
            k = k.softmax(dim=-1)

            q = q * self.scale
            v = v / (h*w)

            context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
            out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
            out = rearrange(out, 'b h c (x y) -> b (h c) x y', x=h, y=w)
            out = self.to_out(out)
        
        else:
            q = q * self.scale

            sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
            attn = sim.softmax(dim=-1)
            out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

            out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
            out = self.to_out(out)

        return out

class TransformerBlock(nn.Module):
    def __init__(self,
                 feat_channels: int,
                 *,
                 num_heads: int=4,
                 head_channels: int=32,
                 linear: bool=False):
        super().__init__()
        self.attn = AttnBlock(feat_channels, 
                            num_heads=num_heads, 
                            head_channels=head_channels)
        self.norm = LayerNorm(feat_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = self.attn(out)
        out = x + out
        return out

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
                 *,
                 n_groups: int=8):
        super().__init__()
        self.conv = WeightStandardizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(n_groups, out_channels)
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
                 out_channels: int,
                 *,
                 emb_size: int,
                 n_groups: int=8):
        super().__init__()

        self.emb_size = emb_size

        if not exists(out_channels):
            out_channels = in_channels

        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_size, 2 * out_channels)
        )

        self.block1 = Block(in_channels, out_channels, n_groups=n_groups)
        self.block2 = Block(out_channels, out_channels, n_groups=n_groups)
        
        # inner residual connection
        if in_channels == out_channels:
            self.res_connnection = nn.Identity()
        else:
            self.res_connnection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """ foward pass

        Args:
            x (torch.Tensor): input feature map with shape (B, C, H, W)
            emb (torch.Tensor): positional embedding with shape (B, emb_size)

        Returns:
            torch.Tensor: output feature map with shape (B, C, H, W)
        """

        emb = self.emb_mlp(emb)
        emb = rearrange(emb, 'b c -> b c 1 1')
        scale_shift = emb.chunk(2, dim=1)
        
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h, scale_shift=None)
        
        return h + self.res_connnection(x)
        
class DownSample(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int
                 ):
        """ Downsample the feature map
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int
                 ):
        """ Upsample the feature map
        """
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class SinusoidalEmbeddingBlock(nn.Module):
    def __init__(self, embedding_size: int, max_period: int=10000):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_period = max_period
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_size = self.embedding_size // 2
        freqs = torch.exp(
            math.log(self.max_period) * torch.arange(start=0, end=half_size, dtype=torch.float32) / half_size
        ).to(device=x.device)
        args = x[:, None].float() * freqs[None]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding

class UNet4Diffusion(nn.Module):
    def __init__(self,
                *,  
                in_channels: int,
                out_channels: int,
                emb_size: int,
                layer_channels: list=[32, 64, 128, 256, 512],
                n_groups: int=8,
                ):
        super().__init__()

        time_emb_size = 4 * layer_channels[0]
        self.time_mlp = nn.Sequential(
            SinusoidalEmbeddingBlock(emb_size),
            nn.Linear(emb_size, time_emb_size),
            nn.GELU(),
            nn.Linear(time_emb_size, time_emb_size)
        )

        self.in_conv = nn.Conv2d(in_channels, layer_channels[0], kernel_size=3, padding=1)
        in_out = list(zip(layer_channels[:-1], layer_channels[1:]))
        n_layers = len(in_out)

        self.down_blocks = nn.ModuleList([])

        for i, (c_in, c_out) in enumerate(in_out):
            is_last = i == (n_layers - 1)

            self.down_blocks.append(
                nn.ModuleList([
                    ResBlock(c_in, c_in, emb_size=time_emb_size, n_groups=n_groups),
                    ResBlock(c_in, c_in, emb_size=time_emb_size, n_groups=n_groups),
                    TransformerBlock(c_in, linear=True) if not is_last else nn.Identity(),
                    DownSample(c_in, c_out) if not is_last else nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
                ])
            )
        
        mid_channels = layer_channels[-1]

        self.mid_block1 = ResBlock(mid_channels, mid_channels, emb_size=time_emb_size, n_groups=n_groups)
        self.mid_attn = TransformerBlock(mid_channels, num_heads=8, linear=True)
        self.mid_block2 = ResBlock(mid_channels, mid_channels, emb_size=time_emb_size, n_groups=n_groups)

        self.up_blocks = nn.ModuleList([])

        for i, (c_in, c_out) in enumerate(reversed(in_out)):
            is_last = i == (n_layers - 1)

            self.up_blocks.append(
                nn.ModuleList([
                    ResBlock(c_in+c_out, c_out, emb_size=time_emb_size, n_groups=n_groups),
                    ResBlock(c_in+c_out, c_out, emb_size=time_emb_size, n_groups=n_groups),
                    TransformerBlock(c_out, linear=True) if not is_last else nn.Identity(),
                    UpSample(c_out, c_in) if not is_last else nn.Conv2d(c_out, c_in, kernel_size=3, padding=1)
                ])
            )

        self.res_out = ResBlock(layer_channels[0]*2, layer_channels[0], emb_size=time_emb_size, n_groups=n_groups)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, layer_channels[0]),
            nn.SiLU(),
            nn.Conv2d(layer_channels[0], out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_ = torch.cat((x, y), dim=1)
        t = self.time_mlp(time)

        x_ = self.in_conv(x_)
        r = x_.clone()

        h = []

        for block1, block2, attn, down in self.down_blocks:
            x_ = block1(x_, t)
            h.append(x_)

            x_ = block2(x_, t)
            x_ = attn(x_)
            h.append(x_)

            x_ = down(x_)
        
        x_ = self.mid_block1(x_, t)
        x_ = self.mid_attn(x_)
        x_ = self.mid_block2(x_, t)

        for block1, block2, attn, up in self.up_blocks:
            x_ = torch.cat((x_, h.pop()), dim=1)
            x_ = block1(x_, t)

            x_ = torch.cat((x_, h.pop()), dim=1)
            x_ = block2(x_, t)
            x_ = attn(x_)

            x_ = up(x_)

        x_ = torch.cat((x_, r), dim=1)
        x_ = self.res_out(x_, t)
        x_ = self.out_conv(x_)

        return x_

def _test_unet():
    x = torch.randn(16, 1, 64, 64).cuda(1)
    y = torch.randn(16, 2, 64, 64).cuda(1)
    t = torch.randn(16).cuda(1)
    model = UNet(in_channels=3, out_channels=1, emb_size=128).cuda(1)
    # print(f"The number of parameters: {sum(p.numel() for p in model.parameters())}")
    out = model(x, t, y)
    print(out.shape)

if __name__ == '__main__':
    _test_unet()


