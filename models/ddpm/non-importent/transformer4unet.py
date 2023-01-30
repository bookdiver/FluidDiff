import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
    def __init__(self, feat_channels, d_mult=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(feat_channels, feat_channels * d_mult),
            nn.Dropout(dropout),
            nn.Linear(feat_channels * d_mult, feat_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AttnBlock(nn.Module):
    def __init__(self, 
                 vis_feat_channels: int, 
                 num_heads: int=8, 
                 *,
                 text_feat_channels: int, 
                 is_inplace: bool = True,
                 use_cross_attn: bool = False):
        """ Spatial Transformer Block, can do either self-attention or cross-attention

        Args:
            vis_feat_channels (int): the channel of the input visual feature maps, with shape (B, vis_feat_channels, H, W)
            text_feat_channels (int): the channel of the textual conditional embeddings, with shape (B, text_feat_channels, emb_size)
            num_heads (int): the number of heads in the multi-head attention. Defaults to 8.
            is_inplace (bool, optional): whether to do the inplace operation. Defaults to True.
            use_cross_attn (bool, optional): whether to use cross-attention. Defaults to False.
        """        
        super().__init__()
        self.vis_feat_channels = vis_feat_channels

        self.num_heads = num_heads
        self.dim_head = vis_feat_channels // num_heads
        self.is_inplace = is_inplace
        self.use_cross_attn = use_cross_attn

        self.scale = self.dim_head ** -0.5

        self.dim_attn = num_heads * self.dim_head

        self.vis_to_qkv = nn.Linear(vis_feat_channels, self.dim_attn * 3)

        if use_cross_attn:
            assert text_feat_channels is not None, \
                "text_feat_channels must be provided when use_cross_attn is True"
            self.text_to_qkv = nn.Linear(text_feat_channels, self.dim_attn * 3)
        else:
            self.text_to_qkv = None

        self.to_out = nn.Linear(self.dim_attn, vis_feat_channels)
    
    def forward(self, vis_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """ get q, k, v from vis_feat when text_feat is None (i.e. self-attention), or q from vis_feat and k, v from text_feat (i.e. cross-attention)

        Args:
            vis_feat (torch.Tensor): the visual feature maps, with shape (B, (H*W), vis_feat_channels)
            text_feat (torch.Tensor): the textual conditional embeddings, with shape (B, emb_size, text_feat_channels)
        
        Returns:
            torch.Tensor: the output feature maps, with shape (B, (H*W), vis_feat_channels) to fit the feed-forward network
        """
        qkv_vis = self.vis_to_qkv(vis_feat)
        q, k, v = qkv_vis.chunk(3, dim=-1)
        # q_vis, k_vis, v_vis shape [B, (H*W), (num_heads * dim_head)]
        q = rearrange(q, 'b e (h d) -> b e h d', h=self.num_heads, d=self.dim_head)
        # q_vis shape [B, (H*W), num_heads, dim_head]
        # do cross-attention
        if self.use_cross_attn:
            qkv_text = self.text_to_qkv(text_feat)
            _, k, v = qkv_text.chunk(3, dim=-1)
            # k_text, v_text shape [B, emb_size, (num_heads * dim_head)]
        k, v = map(lambda t: rearrange(t, 'b e (h d) -> b e h d', h=self.num_heads, d=self.dim_head), (k, v))
        # k_text, v_text shape [B, emb_size, num_heads, dim_head]
        # k_vis, v_vis shape [B, (H*W), num_heads, dim_head]

        # q, k, v with similar shape [B, (H*W) or emb_size, num_heads, dim_head]

        q = q * self.scale
        attn = torch.einsum('b i h d, b j h d -> b h i j', q, k)
        # output shape [B, num_heads, (H*W), (H*W) or emb_size]

        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        out = torch.einsum('b h i j, b j h d -> b i h d', attn, v)
        # output shape [B,(H*W), num_heads, dim_head]

        out = rearrange(out, 'b e h d -> b e (h d)', h=self.num_heads, d=self.dim_head)
        # output shape [B, (H*W), dim_attn]
        out = self.to_out(out)
        # output shape [B, (H*W), vis_feat_channels]
        return out

class BasicTransformerBlock(nn.Module):
    def __init__(self, 
                 vis_feat_channels: int, 
                 text_feat_channels: int, 
                 num_heads: int, 
                 *,
                 is_inplace: bool=True,
                 use_cross_attn: bool=False):
        """ Basic transformer block with 2 attention blocks and layer normalization, then using a feed-forward network
        to concatenate all the heads.

        Args:
            vis_feat_channels (int): the channel of the input visual feature maps
            text_feat_channels (int): the channel of the textual conditional embeddings
            num_heads (int): number of heads
            is_inplace (bool, optional): whether to use inplace operation to save memory. Defaults to True.
        """
        super().__init__()
        self.use_cross_attn = use_cross_attn

        self.attn1 = AttnBlock(vis_feat_channels=vis_feat_channels,
                               text_feat_channels=text_feat_channels, 
                               num_heads=num_heads, 
                               is_inplace=is_inplace,
                               use_cross_attn=False)
        self.norm1 = nn.LayerNorm(vis_feat_channels)

        self.attn2 = AttnBlock(vis_feat_channels=vis_feat_channels,
                               text_feat_channels=text_feat_channels, 
                               num_heads=num_heads, 
                               is_inplace=is_inplace,
                               use_cross_attn=use_cross_attn)
        self.norm2 = nn.LayerNorm(vis_feat_channels)

        self.ff = FeedForward(vis_feat_channels)
        self.norm3 = nn.LayerNorm(vis_feat_channels)
    
    def forward(self, vis_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """ do the cross attention between vis_feat and text_feat, when cond is None, do the self-attention twice.

        Args:
            vis_feat (torch.Tensor): the visual feature maps, with shape (B, (H*W), vis_feat_channels)
            text_feat (torch.Tensor, optional): the textual conditional embeddings, with shape (B, emb_size, text_feat_channels)

        Returns:
            torch.Tensor: the output tensor, with size of (B, (H*W), vis_feat_channels)
        """
        # self-attention with residual connection
        vis_feat = vis_feat + self.attn1(self.norm1(vis_feat), None)
        # cross-attention if text_feat is not None, else still do self-attention
        vis_feat = vis_feat + self.attn2(self.norm2(vis_feat), text_feat)
        # feed-forward network with residual connection
        vis_feat = vis_feat + self.ff(self.norm3(vis_feat))
        return vis_feat

class SpatialTransformer(nn.Module):
    def __init__(self, 
                 vis_feat_channels: int, 
                 text_feat_channels: int, 
                 num_heads: int,
                 num_groups: int, 
                 num_transformers: int,
                 *,
                 is_inplace: bool=True,
                 use_cross_attn: bool=False):
        """ Spatial Transformer

        Args:
            vis_feat_channels (int): the number of channels of visual feature maps
            text_feat_channels (int): the number of channels of textual conditional embeddings
            num_heads (int): the number of heads
            num_transformers (int): the number of transformer layers
        """ 
        super().__init__()
        assert vis_feat_channels % num_heads == 0, "vis_feat_channels must be divisible by num_heads"
        
        self.use_cross_attn = use_cross_attn
        self.text_feat_channels = text_feat_channels

        self.norm = nn.GroupNorm(num_groups=num_groups, 
                                 num_channels=vis_feat_channels, 
                                 eps=1e-6, 
                                 affine=True)
        self.proj_in = nn.Conv2d(vis_feat_channels, vis_feat_channels, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(vis_feat_channels=vis_feat_channels, 
                                  text_feat_channels=text_feat_channels, 
                                  num_heads=num_heads, 
                                  is_inplace=is_inplace,
                                  use_cross_attn=use_cross_attn) 
                                  for _ in range(num_transformers)
        ])
        self.proj_out = nn.Conv2d(vis_feat_channels, vis_feat_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, vis_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """ do the cross attention between vis_feat and text_feat, when cond is None, do the self-attention

        Args:
            vis_feat (torch.Tensor): the visual feature maps, with shape (B, vis_feat_channels, H, W)
            text_feat (torch.Tensor, optional): the textual conditional embeddings, with shape (B, text_feat_channels, emb_size)

        Returns:
            torch.Tensor: the output tensor, with size of (batch_size, vis_feat_channels, H, W)
        """        
        if self.use_cross_attn:
            assert text_feat is not None, "text_feat must be provided when using cross attention"
        _, _, h, w = vis_feat.shape
        vis_feat_in = vis_feat
        vis_feat = self.norm(vis_feat)
        vis_feat = self.proj_in(vis_feat)

        vis_feat = rearrange(vis_feat, 'b c x y -> b (x y) c', x=h, y=w)
        text_feat = rearrange(text_feat, 'b c e -> b e c') if text_feat is not None else None

        for block in self.transformer_blocks:
            vis_feat = block(vis_feat, text_feat)
        
        vis_feat = rearrange(vis_feat, 'b (x y) c -> b c x y', x=h, y=w)
        vis_feat = self.proj_out(vis_feat)
        return vis_feat + vis_feat_in


def _test_attention():
    attn = AttnBlock(vis_feat_channels=64, 
                     text_feat_channels=3,
                     num_heads=4)
    x = torch.randn(2, 256, 64)
    cond = torch.randn(2, 128, 3)
    print(f"Input feature map shape: {x.shape}")
    print(f"Conditioning shape: {cond.shape}")
    out = attn(x, None)
    print(f"Output shape: {out.shape}")

def _test_basic_transformer_block():
    block = BasicTransformerBlock(vis_feat_channels=64, 
                                  text_feat_channels=3, 
                                  num_heads=4)
    x = torch.randn(2, 256, 64)
    print(f"Input feature map shape: {x.shape}")
    cond = torch.randn(2, 128, 3)
    print(f"Conditioning embedding shape: {cond.shape}")
    out = block(x, cond)
    print(f"Output feature map shape: {out.shape}")

def _test_spatial_transformer():
    st = SpatialTransformer(vis_feat_channels=64, 
                            text_feat_channels=3, 
                            num_heads=4, 
                            num_groups=32, 
                            num_transformers=1,
                            is_inplace=True,
                            use_cross_attn=False)
    x = torch.randn(2, 64, 16, 16)
    print(f"Input feature map shape: {x.shape}")
    cond = torch.randn(2, 3, 128)
    print(f"Conditioning embedding shape: {cond.shape}")
    out = st(x, cond)
    print(f"Output feature map shape: {out.shape}")

if __name__ == '__main__':\
    # _test_attention()
    # _test_basic_transformer_block()
    _test_spatial_transformer()

