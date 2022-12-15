import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from transformer4unet import SpatialTransformer

def exists(val):
    return val is not None

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
                 n_groups: int):
        """ Block for Resnet block, contains 1 weighted convolution layer and 1 normalization layer, followed by a SiLU activation
        """
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
                 emb_channels: int,
                 emb_size: int,
                 n_groups: int=8,
                 *,
                 out_channels: int=None):
        """ Residual block for the UNet

        Args:
            in_channels (int): the input channel
            emb_channels (int): the number of channels for the embedding, the embedding is expected to have the shape (B, emb_channels, emb_size)
            emb_size (int): the size of the embedding
            n_groups (int, optional): the number of groups for the normalization layer. Defaults to 8.
            out_channels (Optional[int], optional): the output channel size. Defaults to be same as in_channels.
        """
        super().__init__()

        self.emb_channels = emb_channels
        self.emb_size = emb_size

        if out_channels is None:
            out_channels = in_channels

        self.emb_mlp = nn.Sequential(
            nn.Flatten(),
            nn.SiLU(),
            nn.Linear(emb_channels * emb_size, 2 * out_channels)
        )

        self.block1 = Block(in_channels, out_channels, n_groups)
        self.block2 = Block(out_channels, out_channels, n_groups)
        
        # inner residual connection
        if in_channels == out_channels:
            self.res_connnection = nn.Identity()
        else:
            self.res_connnection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """ foward pass

        Args:
            x (torch.Tensor): input feature map with shape (B, C, H, W)
            emb (torch.Tensor): embedding with shape (B, emb_channels, emb_size)
            NOTE: emb can be either diffusion step embedding (B, 1, emb_size) or conditional embedding (B, n_conditions, emb_size)
            or both (B, n_conditions + 1, emb_size)

        Returns:
            torch.Tensor: output feature map with shape (B, C, H, W)
        """
        scale_shift = None
        if exists(self.emb_mlp) and exists(emb):
            emb = self.emb_mlp(emb)
            emb = rearrange(emb, 'b c -> b c 1 1')
            scale_shift = emb.chunk(2, dim=1)
        
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h, scale_shift=None)
        
        return h + self.res_connnection(x)
        
class DownSample(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int=None
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
            x (torch.Tensor): input feature map with shape (B, C, H, W)

        Returns:
            torch.Tensor: output feature map with shape (B, C, H/2, W/2)
        """
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int=None
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
            x (torch.Tensor): input feature map with shape (B, C, H, W)

        Returns:
            torch.Tensor: output feature map with shape (B, C, H*2, W*2)
        """
        return self.conv(x)

class UniversialEmbedSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        """ forward pass, using either Residual connection or Spatial Transformer
        to combine the feature map with the embeddings.

        Args:
            x (torch.Tensor): input feature map with shape (B, C, H, W)
            emb (torch.Tensor): can be either single diffusion time step embedding (B, 1, emb_size),
            or conditional embedding (B, n_conditions, emb_size),
            or universal embedding (B, n_conditions+1, emb_size) consisting of both diffusion time step and condition of generation.
    
        Returns:
            torch.Tensor: output feature map
        """
        for module in self:

            if isinstance(module, ResBlock):
                t_emb_channel = 1
                c_emb_channel = c_emb.shape[1]

                if module.emb_channels == t_emb_channel:
                    emb = t_emb
                elif module.emb_channels == c_emb_channel:
                    emb = c_emb
                elif module.emb_channels == t_emb_channel + c_emb_channel:
                    emb = torch.cat([t_emb, c_emb], dim=1)
                else:
                    raise ValueError(f"embedding channels does not match requirements of ResBlock")

                x = module(x, emb)

            elif isinstance(module, SpatialTransformer):

                if module.use_cross_attn:
                    emb = c_emb
                else:
                    emb = None

                x = module(x, emb)
        
            else:
                x = module(x)
        return x

class SinusoidalEmbeddingBlock(nn.Module):
    def __init__(self, embedding_size: int, max_period: int=10000):
        """ Block for embedding condition using Sinusoidal Fourier Features. The condition can
        be either diffusion time steps or condition of the genertion, which depends on the forward
        input.

        Args:
            embedding_dim (int): embedding dimension, decide the length of the vector after embedding
            max_period (int, optional): maximal changing period, determine the minimal frequency of the
            encode, the more the max_period, the more the embedding can capture the details .Defaults to 10000.
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.max_period = max_period
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Embed the input

        Args:
            x (torch.Tensor): input with shape: 
            (B, n_conditions): condition of generation
            or 
            (B, ): diffusion time steps

        Returns:
            torch.Tensor: embedding with shape (batch_size, embedding_dim)
        """

        half_size = self.embedding_size // 2
        freqs = torch.exp(
            math.log(self.max_period) * torch.arange(start=0, end=half_size, dtype=torch.float32) / half_size
        ).to(device=x.device)

        # embed the diffusion time steps
        if x.ndim == 1:
            args = x[:, None].float() * freqs[None]
            embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).unsqueeze(1)

        # embed the condition of generation
        elif x.ndim == 2:

            assert x.shape[1] == 3, "Now only support 3D condition of generation, which is (t, x, y)"
            real_t, pos_x, pos_y = x.chunk(3, dim=1)
            
            args = real_t[:, None].float() * freqs[None]
            pe_t = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

            args = pos_x[:, None].float() * freqs[None]
            pe_x = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

            args = pos_y[:, None].float() * freqs[None]
            pe_y = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

            embedding = torch.cat([pe_t, pe_x, pe_y], dim=1)
        else:
            raise ValueError('Input dimension should be 1 or 2, but got {}'.format(x.ndim))
        
        return embedding
        
def get_embedding_block(*,
                        embedding_object: str, 
                        embedding_size: int, 
                        embedding_channels: int=None,
                        out_size: int,
                        ) -> nn.Module:
    if embedding_object == 'DiffusionStep':
        
        embedding_channels = 1

    elif embedding_object == 'Condition':

        assert embedding_channels is not None and embedding_channels > 1, \
        "Channel multipler should be same as the number of channels of condition"
    
    else:
        raise ValueError('Embedding object should be either DiffusionStep or Condition, but got {}'.format(embedding_object))

    embedding_block = nn.Sequential(
        SinusoidalEmbeddingBlock(embedding_size=embedding_size),
        nn.Flatten(),
        nn.Linear(embedding_channels * embedding_size, embedding_channels * out_size),
        nn.SiLU(),
        nn.Linear(embedding_channels * out_size, embedding_channels * out_size),
        Rearrange('b (c d) -> b c d', c=embedding_channels, d=out_size)
    )

    return embedding_block

def get_down_block(block_type: str,
                   in_channels: int,
                   out_channels: int,
                   res_embedding_size: int,
                   res_embedding_channels: int,
                   *,
                   xattn_channels: int=None,
                   is_last: bool=False,
                   n_res_groups: int=8,
                   n_attn_groups: int=8,
                   n_attn_heads: int=8,
                   n_res_layers: int=2,
                   n_transformer_layers: int=1) -> nn.Module:
    blocks = []
    if block_type == 'ResBlock':
        for _ in range(n_res_layers):
            blocks.append(ResBlock(in_channels=in_channels, 
                                   emb_size=res_embedding_size, 
                                   emb_channels=res_embedding_channels,
                                   n_groups=n_res_groups, 
                                   out_channels=out_channels))
            in_channels = out_channels
        if not is_last:
            blocks.append(DownSample(in_channels=out_channels,
                                     out_channels=out_channels))

    elif block_type == 'AttnBlock':
        for _ in range(n_res_layers):
            blocks.append(ResBlock(in_channels=in_channels, 
                                   emb_size=res_embedding_size,
                                   emb_channels=res_embedding_channels,
                                   n_groups=n_res_groups, 
                                   out_channels=out_channels))
            in_channels = out_channels
            blocks.append(
                SpatialTransformer(vis_feat_channels=out_channels,
                                   text_feat_channels=xattn_channels,
                                   num_heads=n_attn_heads,
                                   num_groups=n_attn_groups,
                                   num_transformers=n_transformer_layers, 
                                   use_cross_attn=False)
            )
        if not is_last:
            blocks.append(DownSample(in_channels=out_channels,
                                     out_channels=out_channels))

    elif block_type == 'CrossAttnBlock':

        for _ in range(n_res_layers):
            blocks.append(ResBlock(in_channels=in_channels, 
                                   emb_size=res_embedding_size, 
                                   emb_channels=res_embedding_channels,
                                   n_groups=n_res_groups, 
                                   out_channels=out_channels))
            in_channels = out_channels
            blocks.append(SpatialTransformer(vis_feat_channels=out_channels,
                                            text_feat_channels=xattn_channels,
                                            num_heads=n_attn_heads,
                                            num_groups=n_attn_groups,
                                            num_transformers=n_transformer_layers, 
                                            use_cross_attn=True)
            )
        if not is_last:
            blocks.append(DownSample(in_channels=out_channels,
                                     out_channels=out_channels))

    return UniversialEmbedSequential(*blocks)

def get_mid_block(block_type: str,
                  channels: int,
                  res_embedding_size: int,
                  res_embedding_channels: int,
                  *,
                  xattn_channels: int=None,
                  n_res_groups: int=8,
                  n_attn_groups: int=8,
                  n_attn_heads: int=8,
                  n_res_layers: int=2,
                  n_transformer_layers: int=1) -> nn.Module:
    assert n_res_layers % 2 == 0, 'n_res_layers should be even'
    blocks = []
    if block_type == 'ResBlock':
        for _ in range(n_res_layers):
            blocks.append(ResBlock(in_channels=channels, 
                                   emb_size=res_embedding_size, 
                                   emb_channels=res_embedding_channels,
                                   n_groups=n_res_groups, 
                                   out_channels=channels))
    elif block_type == 'AttnBlock':
        blocks.append(SpatialTransformer(vis_feat_channels=channels,
                                        text_feat_channels=xattn_channels,
                                        num_heads=n_attn_heads,
                                        num_groups=n_attn_groups,
                                        num_transformers=n_transformer_layers, 
                                        use_cross_attn=False)
        )

        for _ in range(n_res_layers // 2):
            blocks.insert(0, ResBlock(in_channels=channels, 
                                      emb_size=res_embedding_size, 
                                      emb_channels=res_embedding_channels,
                                      n_groups=n_res_groups, 
                                      out_channels=channels))
            blocks.append(ResBlock(in_channels=channels, 
                                   emb_size=res_embedding_size, 
                                   emb_channels=res_embedding_channels,
                                   n_groups=n_res_groups, 
                                   out_channels=channels))
    else:
        raise NotImplementedError('block_type should be either ResBlock or AttnBlock, but got {}'.format(block_type))
    return UniversialEmbedSequential(*blocks)

def get_up_block(block_type: str,
                in_channels: int,
                out_channels: int,
                res_embedding_size: int,
                res_embedding_channels: int,
                *,
                xattn_channels: int=None,
                is_last: bool=False,
                n_res_groups: int=8,
                n_attn_groups: int=8,
                n_attn_heads: int=8,
                n_res_layers: int=2,
                n_transformer_layers: int=1) -> nn.Module:
    blocks = []
    if block_type == 'ResBlock':
        for _ in range(n_res_layers):
            blocks.append(ResBlock(in_channels=in_channels, 
                                   emb_size=res_embedding_size, 
                                   emb_channels=res_embedding_channels,
                                   n_groups=n_res_groups, 
                                   out_channels=out_channels))
            in_channels = out_channels
        if not is_last:
            blocks.append(UpSample(in_channels=out_channels,
                                     out_channels=out_channels))

    elif block_type == 'AttnBlock':
        for _ in range(n_res_layers):
            blocks.append(ResBlock(in_channels=in_channels, 
                                   emb_size=res_embedding_size,
                                   emb_channels=res_embedding_channels,
                                   n_groups=n_res_groups, 
                                   out_channels=out_channels))
            in_channels = out_channels
            blocks.append(
                SpatialTransformer(vis_feat_channels=out_channels,
                                   text_feat_channels=xattn_channels,
                                   num_heads=n_attn_heads,
                                   num_groups=n_attn_groups,
                                   num_transformers=n_transformer_layers, 
                                   use_cross_attn=False)
            )
        if not is_last:
            blocks.append(UpSample(in_channels=out_channels,
                                     out_channels=out_channels))

    elif block_type == 'CrossAttnBlock':

        for _ in range(n_res_layers):
            blocks.append(ResBlock(in_channels=in_channels, 
                                   emb_size=res_embedding_size, 
                                   emb_channels=res_embedding_channels,
                                   n_groups=n_res_groups, 
                                   out_channels=out_channels))
            in_channels = out_channels
            blocks.append(SpatialTransformer(vis_feat_channels=out_channels,
                                            text_feat_channels=xattn_channels,
                                            num_heads=n_attn_heads,
                                            num_groups=n_attn_groups,
                                            num_transformers=n_transformer_layers, 
                                            use_cross_attn=True)
            )
        if not is_last:
            blocks.append(UpSample(in_channels=out_channels,
                                     out_channels=out_channels))

    return UniversialEmbedSequential(*blocks)
                  
def _test_embedding():
    block_t = get_embedding_block(embedding_object='DiffusionStep',
                                  embedding_size=32,
                                  out_size=128)
    block_c = get_embedding_block(embedding_object='Condition',
                                  embedding_size=32,
                                  embedding_channels=3,
                                  out_size=128)
    time_steps = torch.randn((4, ))
    conditions = torch.randn((4, 3))
    print(f"Time steps: {time_steps.shape}")
    print(f"Conditions: {conditions.shape}")
    print(f"Time embedding: {block_t(time_steps).shape}")
    print(f"Condition embedding: {block_c(conditions).shape}")

def _test_down_block():
    print("Testing DownBlock with downsample")
    block = get_down_block(block_type='CrossAttnBlock',
                           in_channels=32,
                           out_channels=64,
                           res_embedding_size=128,
                           res_embedding_channels=1,
                           xattn_channels=3,
                           is_last=True,
                           n_res_groups=8,
                           n_attn_groups=16,
                           n_res_layers=2)
    x = torch.randn((4, 32, 16, 16))
    t_emb = torch.randn((4, 1, 128))
    c_emb = torch.randn((4, 3, 128))
    print(f"Input feature map: {x.shape}")
    print(f"Diffusion step embedding: {t_emb.shape}")
    print(f"Condition embedding: {c_emb.shape}")
    print(f"Output feature map: {block(x, t_emb, c_emb).shape}")

def _test_mid_block():
    print("Testing MidBlock")
    block = get_mid_block(block_type='AttnBlock',
                          channels=32,
                          res_embedding_size=128,
                          res_embedding_channels=1,
                          xattn_channels=None,
                          n_res_groups=8,
                          n_attn_groups=16,
                          n_res_layers=2)
    x = torch.randn((4, 32, 16, 16))
    t_emb = torch.randn((4, 1, 128))
    c_emb = torch.randn((4, 3, 128))
    print(f"Input feature map: {x.shape}")
    print(f"Diffusion step embedding: {t_emb.shape}")
    print(f"Condition embedding: {c_emb.shape}")
    print(f"Output feature map: {block(x, t_emb, c_emb).shape}")

def _test_up_block():
    print("Testing UpBlock")
    block = get_up_block(block_type='ResBlock',
                         in_channels=32,
                         out_channels=16,
                         res_embedding_size=128,
                         res_embedding_channels=1,
                         xattn_channels=3,
                         is_last=False,
                         n_res_groups=8,
                         n_attn_groups=16,
                         n_res_layers=2)
    x = torch.randn((4, 32, 16, 16))
    t_emb = torch.randn((4, 1, 128))
    c_emb = torch.randn((4, 3, 128))
    print(f"Input feature map: {x.shape}")
    print(f"Diffusion step embedding: {t_emb.shape}")
    print(f"Condition embedding: {c_emb.shape}")
    print(f"Output feature map: {block(x, t_emb, c_emb).shape}")


if __name__ == '__main__':
    # _test_embedding()
    # _test_down_block() 
    # _test_mid_block()
    _test_up_block()