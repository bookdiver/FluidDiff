import numpy as np
import torch
import torch.nn as nn

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool=False) -> None:
        super(ResidualConvBlock, self).__init__()
        """
        Standard ResNet style convolutional block, when is_res is True, the block is a residual block,
        otherwise it is a standard 2-layer convolutional block. In this block, the size of input won't change.
        """
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased,
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            # do the normalization on the sum of the two convolutions,
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetDown, self).__init__()
        """
        Downscaling block in Unet, which consist of 1 ResidualConvBlock and 1 MaxPool2d layer.
        After this block, the size of the feature maps will be halved.
        """
        layers = [# a normal 2-layer convolutional block
                    ResidualConvBlock(in_channels=in_channels, out_channels=out_channels, is_res=False), 
                    nn.MaxPool2d(kernel_size=2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: (batch_size, in_channels, height, width)
        return self.model(x)
        # output shape: (batch_size, out_channels, height/2, width/2)


class UnetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetUp, self).__init__()
        """
        Upscaling block in Unet, which consist of 1 ConvTranspose2d layer and 2 ResidualConvBlock.
        After this block, the size of the feature maps will be doubled.
        Also it receives the output of the corresponding downscaling block as skip connection.
        """
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor) -> torch.Tensor:
        # input shape: x:(batch_size, in_channels/2, height, width) | skip:(batch_size, in_channels/2, height, width)
        # skip_x is the output from the downscaling path
        x = torch.cat((x, skip_x), dim=1)
        x = self.model(x)
        return x
        # output shape: (batch_size, out_channels, 2*height, 2*width)


class ConditionEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super(ConditionEmbedding, self).__init__()
        """
        Generic 1-layer fc for embedding condition information (onehot encoded), which takes 1 vector as input and outputs a vector with emb_dim. 
        """
        layers = [
            nn.Linear(in_features=input_dim, out_features=embed_dim),
            nn.GELU(),
            nn.Linear(in_features=embed_dim, out_features=embed_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: (batch_size, input_dim)
        return self.model(x)[..., None, None]
        # output shape: (batch_size, embed_dim, H, W)


class GaussianFourierFeatures(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int, learnable: bool=True) -> None:
        super(GaussianFourierFeatures, self).__init__()
        '''
        Gaussian Fourier Features, which takes 1 scalar as input and outputs a vector with projection_dim.
        '''
        self.projection_dim = projection_dim
        self.learnable = learnable
        self.weights = nn.Parameter(torch.randn(input_dim, projection_dim) / np.sqrt(input_dim))
        self.bias = nn.Parameter(torch.randn(projection_dim) / np.sqrt(input_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: (batch_size, input_dim)
        if self.learnable:
            # learnable weights
            x = torch.cos(x @ self.weights + self.bias)
        else:
            # fixed weights
            x = torch.cos(x @ self.weights.detach() + self.bias.detach())
        return x
        # output shape: (batch_size, projection_dim)


class DiffusionEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, projection_dim: int, learnable: bool=True) -> None:
        super(DiffusionEmbedding, self).__init__()
        """
        Diffusion embedding, which takes 1 scalar as input and outputs a vector with embed_dim.
        """
        self.gff = GaussianFourierFeatures(input_dim, projection_dim, learnable)
        self.fc = nn.Linear(projection_dim, embed_dim)
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: (batch_size, input_dim)
        x = self.gff(x)
        x = self.fc(x)
        x = self.gelu(x)
        return x[..., None, None]
        # output shape: (batch_size, embed_dim, H, W)
        
###################################################################################################################################################
class ConditionalUnet(nn.Module):
    def __init__(self, in_channels: int, n_feats: int=256, n_classes: int=10):
        """
        Conditional Unet for generating images conditioned on class labels from MNIST dataset.
        ---------------------------------------------------------------
        |       init_conv ------------------> out                     |
        |            |                       ^                        |
        |            v                       |                        |
        |            down1 -------------> up2                         |
        |                |                  ^ <---- conditionEmbed1   |
        |                v                  | <---- diffusionEmbed1   |
        |                down2 ---------> up1                         |
        |                |               ^  <---- conditionEmbed0     |
        |                v               |  <---- diffusionEmbed0     |
        |                to_vec ------> up0                           |
        |                                                             |
        ---------------------------------------------------------------

        """
        super(ConditionalUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feats = n_feats
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels=in_channels, out_channels=n_feats, is_res=True)

        self.down1 = UnetDown(in_channels=n_feats, out_channels=n_feats)
        self.down2 = UnetDown(in_channels=n_feats, out_channels=2 * n_feats)

        # compress into latent space
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), 
                                    nn.GELU())

        # embed t in diffusion step
        # embed at self.down0
        self.diffusionEmbed0 = DiffusionEmbedding(input_dim=1, embed_dim=2*n_feats, projection_dim=128, learnable=True)
        # embed at self.down1
        self.diffusionEmbed1 = DiffusionEmbedding(input_dim=1, embed_dim=1*n_feats, projection_dim=128, learnable=True)
        # embed t in real time step,
        # embed at self.down0,
        self.conditionEmbed0 = ConditionEmbedding(input_dim=n_classes, embed_dim=2*n_feats)
        # embed at self.down1,
        self.conditionEmbed1 = ConditionEmbedding(input_dim=n_classes, embed_dim=1*n_feats)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 5, 5), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(in_channels=2 * n_feats, out_channels=2 * n_feats, kernel_size=7, stride=7), # otherwise just have 2*n_feat
            nn.GroupNorm(num_groups=8, num_channels=2 * n_feats),
            nn.ReLU(),
        )

        self.up1 = UnetUp(in_channels=4 * n_feats, out_channels=n_feats)
        self.up2 = UnetUp(in_channels=2 * n_feats, out_channels=n_feats)

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=2 * n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=n_feats),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_feats, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """ Condtional Unet for approximating the noise for a given diffusion time t and real time tau.

        Args:
            x (torch.Tensor): noisy samples (B, 1, H, W)
            t (torch.Tensor): diffusion time (B, 1)
            c (torch.Tensor): label of the image (B, 1)

        Returns:
            epsilon (torch.Tensor): the noise added on the input (B, 1, H, W) compared with the clean one
        """        

        c = nn.functional.one_hot(c, num_classes=self.n_classes).float()

        # downscaling path
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # upscaling path
        up1 = self.up0(hiddenvec)
        # first embedding
        t_emb0 = self.diffusionEmbed0(t)
        c_emb0 = self.conditionEmbed0(c)
        # jump connection with down2
        up2 = self.up1(t_emb0*up1+ c_emb0, down2)  # add and multiply embeddings
        # (TODO: use different embedding method such as add extra channels)

        # second embedding
        t_emb1 = self.diffusionEmbed1(t)
        c_emb1 = self.conditionEmbed1(c)
        # jump connection with down1
        up3 = self.up2(t_emb1*up2+ c_emb1, down1)
        # jump connection with x
        out = self.out(torch.cat((up3, x), 1))
        return out

