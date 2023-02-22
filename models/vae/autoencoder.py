import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def normalization(channels: int, normalization_type: str='group') -> nn.Module:
    if normalization_type == 'batch':
        return nn.BatchNorm3d(num_features=channels)
    elif normalization_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=channels)
    else:
        raise ValueError('normalization_type must be one of batch or group')

def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    """
    ## ResNet Block
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: is the number of channels in the input
        :param out_channels: is the number of channels in the output
        """
        super().__init__()
        # First normalization and convolution layer
        self.norm1 = normalization(in_channels)
        self.act1 = nn.ELU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # Second normalization and convolution layer
        self.norm2 = normalization(out_channels)
        self.act2 = nn.ELU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # `in_channels` to `out_channels` mapping layer for residual connection
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """

        h = x

        # First normalization and convolution layer
        h = self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)

        # Second normalization and convolution layer
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        # Map and add residual
        return self.nin_shortcut(x) + h

class AttnBlock(nn.Module):
    """ Self attention over spatial and temporal dimensions, using 3D convolution
    """
    def __init__(
        self, 
        channel_dim: int,
        heads: int=4,
        dim_head: int=32
    ):
        super().__init__()
        self.norm = normalization(channel_dim)
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(channel_dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, channel_dim, kernel_size=1)
    
    def forward(
        self,
        x: torch.Tensor
    ):
        x = self.norm(x)
        b, c, f, h, w = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (n d) f h w -> b n (f h w) d', n=self.heads), qkv)

        context = torch.einsum('b n i d, b n j d -> b n i j', q, k) * self.scale
        attn = context.softmax(dim=-1)
        out = torch.einsum('b n i j, b n j d -> b n i d', attn, v)
        out = rearrange(out, 'b n (f h w) d -> b (n d) f h w', n=self.heads, f=f, h=h, w=w)
        return self.to_out(out)


class UpSample(nn.Module):
    """ Up-sampling layer
    """
    def __init__(
        self, 
        channels: int
    ):
        super().__init__()
        self.conv = nn.ConvTranspose3d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(
        self, 
        x: torch.Tensor
    ):
        return self.conv(x)


class DownSample(nn.Module):
    """ Down-sampling layer
    """
    def __init__(
        self, 
        channels: int
    ):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(
        self, 
        x: torch.Tensor
    ):
        return self.conv(x)
    

class GaussianLatent:
    """ Gaussian Distribution, used for the variational autoencoder
    """

    def __init__(
        self, 
        parameters: torch.Tensor
    ):
        """
        :param parameters: are the means and log of variances of the embedding of shape
            `[batch_size, z_channels * 2, z_height, z_height]`
        """
        # Split mean and log of variance
        self.mean, log_var = torch.chunk(parameters, 2, dim=1)
        # Clamp the log of variances
        self.log_var = torch.clamp(log_var, -30.0, 20.0)
        # Calculate standard deviation
        self.std = torch.exp(0.5 * self.log_var)

    def sample(self):
        # Sample from the distribution
        return self.mean + self.std * torch.randn_like(self.std)

class Encoder(nn.Module):
    """
    ## Encoder module
    """

    def __init__(
        self, 
        *, 
        channels: int, 
        channel_multipliers: list, 
        n_resnet_blocks: int,
        in_channels: int, 
        z_channels: int,
        use_attn: bool,
        use_variational: bool
    ):
        """
        :param channels: is the number of channels in the first convolution layer
        :param channel_multipliers: are the multiplicative factors for the number of channels in the
            subsequent blocks
        :param n_resnet_blocks: is the number of resnet layers at each resolution
        :param in_channels: is the number of channels in the image
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()

        # Number of blocks of different resolutions.
        # The resolution is halved at the end each top level block
        n_resolutions = len(channel_multipliers)

        # Initial $3 \times 3$ convolution layer that maps the image to `channels`
        self.conv_in = nn.Conv3d(in_channels, channels, 3, stride=1, padding=1)

        # Number of channels in each top level block
        channels_list = [m * channels for m in [1] + channel_multipliers]

        # List of top-level blocks
        self.down = nn.ModuleList()
        # Create top-level blocks
        for i in range(n_resolutions):
            # Each top level block consists of multiple ResNet Blocks and down-sampling
            resnet_blocks = nn.ModuleList()
            # Add ResNet Blocks
            for _ in range(n_resnet_blocks):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i + 1]))
                channels = channels_list[i + 1]
            # Top-level block
            down = nn.Module()
            down.block = resnet_blocks
            # Down-sampling at the end of each top level block except the last
            if i != n_resolutions - 1:
                down.downsample = DownSample(channels)
            else:
                down.downsample = nn.Identity()
            #
            self.down.append(down)

        # Final ResNet blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels) if use_attn else nn.Identity()
        self.mid.block_2 = ResnetBlock(channels, channels)

        # Map to embedding space with a $3 \times 3$ convolution
        self.norm_out = normalization(channels)
        self.act_out = nn.ELU()
        if use_variational:
            self.conv_out = nn.Conv3d(channels, 2 * z_channels, 3, stride=1, padding=1)
        else:
            self.conv_out = nn.Conv3d(channels, z_channels, 3, stride=1, padding=1)

    def forward(self, img: torch.Tensor):
        """
        :param img: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """

        # Map to `channels` with the initial convolution
        x = self.conv_in(img)

        # Top-level blocks
        for down in self.down:
            # ResNet Blocks
            for block in down.block:
                x = block(x)
            # Down-sampling
            x = down.downsample(x)

        # Final ResNet blocks with attention
        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)

        # Normalize and map to embedding space
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)

        #
        return x

class Decoder(nn.Module):
    """
    ## Decoder module
    """

    def __init__(
        self, 
        *, 
        channels: int, 
        channel_multipliers: list, 
        n_resnet_blocks: int,
        out_channels: int, 
        z_channels: int,
        use_attn: bool,
    ):
        """
        :param channels: is the number of channels in the final convolution layer
        :param channel_multipliers: are the multiplicative factors for the number of channels in the
            previous blocks, in reverse order
        :param n_resnet_blocks: is the number of resnet layers at each resolution
        :param out_channels: is the number of channels in the image
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()

        # Number of blocks of different resolutions.
        # The resolution is halved at the end each top level block
        num_resolutions = len(channel_multipliers)

        # Number of channels in each top level block, in the reverse order
        channels_list = [m * channels for m in channel_multipliers]

        # Number of channels in the  top-level block
        channels = channels_list[-1]

        # Initial $3 \times 3$ convolution layer that maps the embedding space to `channels`
        self.conv_in = nn.Conv3d(z_channels, channels, 3, stride=1, padding=1)

        # ResNet blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels) if use_attn else nn.Identity()
        self.mid.block_2 = ResnetBlock(channels, channels)

        # List of top-level blocks
        self.up = nn.ModuleList()
        # Create top-level blocks
        for i in reversed(range(num_resolutions)):
            # Each top level block consists of multiple ResNet Blocks and up-sampling
            resnet_blocks = nn.ModuleList()
            # Add ResNet Blocks
            for _ in range(n_resnet_blocks + 1):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i]))
                channels = channels_list[i]
            # Top-level block
            up = nn.Module()
            up.block = resnet_blocks
            # Up-sampling at the end of each top level block except the first
            if i != 0:
                up.upsample = UpSample(channels)
            else:
                up.upsample = nn.Identity()
            # Prepend to be consistent with the checkpoint
            self.up.insert(0, up)

        # Map to image space with a $3 \times 3$ convolution
        self.norm_out = normalization(channels)
        self.act_out = nn.ELU()
        self.conv_out = nn.Conv3d(channels, out_channels, 3, stride=1, padding=1)

    def forward(self, z: torch.Tensor):
        """
        :param z: is the embedding tensor with shape `[batch_size, z_channels, z_height, z_height]`
        """

        # Map to `channels` with the initial convolution
        h = self.conv_in(z)

        # ResNet blocks with attention
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Top-level blocks
        for up in reversed(self.up):
            # ResNet Blocks
            for block in up.block:
                h = block(h)
            # Up-sampling
            h = up.upsample(h)

        # Normalize and map to image space
        h = self.norm_out(h)
        h = self.act_out(h)
        x = self.conv_out(h)

        #
        return x

class Autoencoder(nn.Module):
    """
    ## Autoencoder
    This consists of the encoder and decoder modules.
    """

    def __init__(
        self, 
        in_channels: int,
        out_channels: int=None,
        n_res_blocks: int=2,
        channel_multipliers: list=[1, 2, 4],
        emb_channels: int=32, 
        z_channels: int=4,
        use_attn_in_bottleneck: bool=True,
        use_variational: bool=True):
        super().__init__()
        self.use_variational = use_variational

        self.encoder = Encoder(channels=32, 
                               channel_multipliers=channel_multipliers, 
                               n_resnet_blocks=n_res_blocks, 
                               in_channels=in_channels, 
                               z_channels=z_channels,
                               use_attn=use_attn_in_bottleneck,
                               use_variational=use_variational)
        self.decoder = Decoder(channels=32, 
                               channel_multipliers=channel_multipliers, 
                               n_resnet_blocks=n_res_blocks, 
                               out_channels=out_channels if out_channels else in_channels, 
                               z_channels=z_channels,
                               use_attn=use_attn_in_bottleneck)
        # Convolution to map from embedding space to
        # quantized embedding space moments (mean and log variance)
        if self.use_variational:
            self.quant_conv = nn.Conv3d(2 * z_channels, 2 * emb_channels, 1)
        else:
            self.quant_conv = nn.Conv3d(z_channels, emb_channels, 1)
        # Convolution to map from quantized embedding space back to
        # embedding space
        self.post_quant_conv = nn.Conv3d(emb_channels, z_channels, 1)

    def encode(self, x: torch.Tensor):
        """
        ### Encode images to latent representation
        :param img: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """
        # Get embeddings with shape `[batch_size, z_channels * 2, z_height, z_height]`
        z = self.encoder(x)
        # Get the moments in the quantized embedding space
        moments = self.quant_conv(z)
        # Return the distribution
        if self.use_variational:
            return GaussianLatent(moments)
        else:
            return moments

    def decode(self, z: torch.Tensor):
        """
        ### Decode images from latent representation
        :param z: is the latent representation with shape `[batch_size, emb_channels, z_height, z_height]`
        """
        # Map to embedding space from the quantized representation
        z = self.post_quant_conv(z)
        # Decode the image of shape `[batch_size, channels, height, width]`
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        ### Forward pass
        :param img: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """
        # Encode the image
        z = self.encode(x)
        # Decode the image
        if self.use_variational:
            x_hat = self.decode(z.sample())
        else:
            x_hat = self.decode(z)
        # Return the results
        return {'x': x, 'x_hat': x_hat, 'z': z}

    def loss(self, x: torch.Tensor, *, beta: float=None, recon_loss_type: str='sum'):
        """ Calculate the MSE loss for vanilla AE or the ELBO loss for VAE
        """
        output = self.forward(x)
        if self.use_variational:
            mean = output['z'].mean.flatten(start_dim=1, end_dim=-1)
            logvar = output['z'].log_var.flatten(start_dim=1, end_dim=-1)
            recon_loss = F.mse_loss(output['x_hat'], output['x'], reduction=recon_loss_type)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1), dim=0)
            loss = recon_loss + beta * kld_loss
        else:
            loss = F.mse_loss(output['x_hat'], output['x'], reduction=recon_loss_type)
            recon_loss = loss
            kld_loss = torch.tensor(0.0)
        
        return loss, recon_loss, beta*kld_loss

def _test_vae():
    vae = Autoencoder(in_channels=3, n_res_blocks=3, emb_channels=4, z_channels=32, use_attn_in_bottleneck=True, use_variational=True)
    x = torch.randn(2, 3, 16, 64, 64)
    # out = vae(x)
    # print("original input shape:", out['x'].shape)
    # print("reconstruction output shape:", out['x_hat'].shape)
    # # print("latent space shape:", out['z'].sample().shape)
    # print("latent space shape:", out['z'].shape)
    print(vae.loss(x, beta=0.5))

if __name__ == '__main__':
    _test_vae()