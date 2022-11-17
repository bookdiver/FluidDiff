from typing import Callable, Optional
import torch
import torch.nn as nn
import numpy as np

########################################################################################################################################################################
###################################################### Time independent score function #################################################################################
########################################################################################################################################################################
class ScoreUnet(nn.Module):
    """Time independent score function"""
    def __init__(self, n_channels: int=2, load: bool=False):
        super().__init__()
        self.n_channels = n_channels
        channels = [8, 16, 32, 64, 64]
        self.Convs = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(n_channels, channels[0], kernel_size=3, padding=1),  # (batch, 8, 28, 28)
            nn.ReLU()
        ),
        nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, 8, 14, 14)
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),  # (batch, 16, 14, 14)
            nn.ReLU()
        ),
        nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, 16, 7, 7) 
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),  # (batch, 32, 7, 7)
            nn.ReLU()
        ),
        nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # (batch, 32, 4, 4)
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1),  # (batch, 64, 4, 4)
            nn.ReLU()
        ),
        nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, 64, 2, 2)
            nn.Conv2d(channels[3], channels[4], kernel_size=3, padding=1),  # (batch, 64, 2, 2)
            nn.ReLU()
        )
        ])

        self.Ups = nn.ModuleList([
        nn.Sequential(
            # input is from convs[4] (batch, 64, 2, 2) 
            nn.ConvTranspose2d(channels[4], channels[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 4, 4)
            nn.ReLU()
        ),
        nn.Sequential(
            # input is from convs[3] (batch, 64, 4, 4) and Ups[0] (batch, 64, 4, 4)
            nn.ConvTranspose2d(channels[3] * 2, channels[2], kernel_size=3, stride=2, padding=1, output_padding=0),  # (batch, 32, 7, 7)
            nn.ReLU()
        ),
        nn.Sequential(
            # input is from convs[2] (batch, 32, 7, 7) and Ups[1] (batch, 32, 7, 7)
            nn.ConvTranspose2d(channels[2] * 2, channels[1], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 16, 14, 14)
            nn.ReLU()
        ),
        nn.Sequential(
            # input is from convs[1] (batch, 16, 14, 14) and Ups[2] (batch, 16, 14, 14)
            nn.ConvTranspose2d(channels[1] * 2, channels[0], kernel_size=3, stride=2, padding=1, output_padding=1),  #  (batch, 8, 28, 28)
            nn.ReLU()
        ),
        nn.Sequential(
            # input is from convs[0] (batch, 8, 28, 28) and Ups[3] (batch, 8, 28, 28)
            nn.Conv2d(channels[0] * 2, channels[0], kernel_size=3, padding=1),  # (batch, 8, 28, 28)
            nn.ReLU(),
            nn.Conv2d(channels[0], n_channels, kernel_size=3, padding=1)   # (batch, 2, 28, 28)
        ),
        ])

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels * 28 * 28, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, n_channels * 28 * 28)
        )
        
        if load:
            model_dict = torch.load('./model/scoreunet.pth')
            self.Convs.load_state_dict(model_dict['Convs'])
            self.Ups.load_state_dict(model_dict['Ups'])
            self.fc.load_state_dict(model_dict['fc'])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2, 28, 28)
        signal = x
        signals = []
        for i, conv in enumerate(self.Convs):
            signal = conv(signal)
            if i < len(self.Convs) - 1:
                signals.append(signal)

        for i, dcov in enumerate(self.Ups):
            if i == 0:
                signal = dcov(signal)
            else:
                signal = torch.cat((signal, signals[-i]), dim=-3)
                signal = dcov(signal)
        score = self.fc(signal)
        score = score.view(-1, self.n_channels, 28, 28) # (batch, n_channels, 28, 28)

        return signal

########################################################################################################################################################################
###################################################### Time dependent score function ###################################################################################
########################################################################################################################################################################

def conv3(in_channels: int, out_channels: int, stride: int=1)-> nn.Module:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def transconv3(in_channels: int, out_channels: int, stride: int=1, output_padding: int=1)-> nn.Module:
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=output_padding, bias=False)

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]


class ConvBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, stride: int=1, direct: str='down', output_padding: Optional[int]=1, embedding_dim: int=256, num_groups: int=4) -> None:
        super().__init__()
        self.direct = direct
        if direct == 'down':
            self.conv = conv3(in_channel, out_channel, stride)
        elif direct == 'up':
            self.transconv = transconv3(in_channel, out_channel, stride, output_padding)
        self.gn = nn.GroupNorm(num_groups, out_channel)
        self.dense = Dense(embedding_dim, out_channel)
        self.swish = lambda x: x * torch.sigmoid(x)
    
    def forward(self, embed: torch.tensor, x: torch.Tensor, x_: torch.Tensor=None) -> torch.Tensor:
        if self.direct == 'down':
            x = self.conv(x)
        elif self.direct == 'up':
            if x_ is not None:
                x = self.transconv(torch.cat([x, x_], dim=1))
            else:
                x = self.transconv(x)
        x += self.dense(embed)
        x = self.gn(x)
        x = self.swish(x)
        return x

class TimeDependentScoreNet(nn.Module):
    def __init__(self, marginal_prob_std: Callable, input_channel: int=1, hidden_channels: list=[32, 64, 128, 256], embed_dim: int=256):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        self.swish = lambda x: x * torch.sigmoid(x)
        self.embed_layer = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        # Encoding layers
        self.convblock1 = ConvBlock(input_channel, hidden_channels[0], stride=1, direct='down', num_groups=4)
        self.convblock2 = ConvBlock(hidden_channels[0], hidden_channels[1], stride=2, direct='down', num_groups=32)
        self.convblock3 = ConvBlock(hidden_channels[1], hidden_channels[2], stride=2, direct='down', num_groups=32)
        self.convblock4 = ConvBlock(hidden_channels[2], hidden_channels[3], stride=2, direct='down', num_groups=32)

        # Decoding layers
        self.transconvblock4 = ConvBlock(hidden_channels[3], hidden_channels[2], stride=2, direct='up', output_padding=0, num_groups=32)
        # Skip connection with convblock3
        self.transconvblock3 = ConvBlock(hidden_channels[2]*2, hidden_channels[1], stride=2, direct='up', output_padding=1, num_groups=32)
        # Skip connection with convblock2
        self.transconvblock2 = ConvBlock(hidden_channels[1]*2, hidden_channels[0], stride=2, direct='up', output_padding=1, num_groups=32)
        # Skip connection with convblock1
        self.transconvblock1 = nn.ConvTranspose2d(hidden_channels[0]*2, input_channel, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        embed = self.swish(self.embed_layer(t))
        # Encoding
        h1 = self.convblock1(embed, x, None)
        h2 = self.convblock2(embed, h1, None)
        h3 = self.convblock3(embed, h2, None)
        h4 = self.convblock4(embed, h3, None)

        # Decoding
        h = self.transconvblock4(embed, h4, None)
        h = self.transconvblock3(embed, h, h3)
        h = self.transconvblock2(embed, h, h2)
        h = self.transconvblock1(torch.cat([h, h1], dim=1))

        # Normalization
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h