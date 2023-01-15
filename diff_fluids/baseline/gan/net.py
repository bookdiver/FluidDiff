from collections import OrderedDict

import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)),
                ('norm', nn.BatchNorm2d(in_channels)),
                ('act', nn.ReLU(inplace=True)),
                ('dropout', nn.Dropout2d(0.5))
            ])
        )
        self.conv2 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
                ('norm', nn.BatchNorm2d(out_channels))
            ])
        )
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False) if in_channels != out_channels else nn.Identity()
        self.out = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.skip_conv(x)
        return self.out(x)

class Generator(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 init_features: int=64,
                 num_resblocks: int=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features

        self.init_conv = self._init_conv()
        self.down_blocks = nn.Sequential(
            self._downsample(init_features, init_features * 2),
            self._downsample(init_features * 2, init_features * 4),
        )
        self.res_blocks = nn.ModuleList([ResBlock(init_features * 4, init_features * 4) for _ in range(num_resblocks)])
        self.up_blocks = nn.Sequential(
            self._upsample(init_features * 4, init_features * 2),
            self._upsample(init_features * 2, init_features),
        )
        self.out_conv = self._out_conv()

    
    def _init_conv(self):
        return nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(self.in_channels, self.init_features, kernel_size=7, stride=1, padding=3, bias=False)),
                ('norm', nn.BatchNorm2d(self.init_features)),
                ('act', nn.ReLU(inplace=True))
            ])
        )
    
    def _out_conv(self):
        return nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(self.init_features, self.out_channels, kernel_size=7, stride=1, padding=3, bias=False)),
                ('act', nn.Tanh())
            ])
        )
    
    def _downsample(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)),
                ('norm', nn.BatchNorm2d(out_channels)),
                ('act', nn.ReLU(inplace=True))
            ])
        )
    
    def _upsample(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            OrderedDict([
                ('conv', nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
                ('norm', nn.BatchNorm2d(out_channels)),
                ('act', nn.ReLU(inplace=True))
            ])
        )
    
    def forward(self, z: torch.Tensor, y: torch.Tensor):
        x = torch.cat([z, y], dim=1)
        x = self.init_conv(x)
        x = self.down_blocks(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.up_blocks(x)
        return self.out_conv(x)

class Discriminator(nn.Module):
    def __init__(self,
                 in_channels: int,
                 init_features: int=64,
                 feature_mults: list=[1, 2, 4, 8]):
        super().__init__()
        self.in_channels = in_channels
        self.init_features = init_features
        self.last_features = init_features * feature_mults[-1]

        self.init_conv = self._init_conv()
        self.convs = nn.ModuleList([])
        channels = [init_features * mult for mult in feature_mults]
        in_out = list(zip(channels[:-1], channels[1:]))
        for (c_in, c_out) in in_out:
            self.convs.append(self._conv_block(c_in, c_out))
        self.out_conv = self._out_conv()
            
    
    def _init_conv(self):
        return nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(self.in_channels, self.init_features, kernel_size=3, stride=2, padding=1, bias=False)),
                ('act', nn.LeakyReLU(0.2, inplace=True))
            ])
        )

    def _conv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)),
                ('norm', nn.BatchNorm2d(out_channels)),
                ('act', nn.LeakyReLU(0.2, inplace=True))
            ])
        )
    
    def _out_conv(self):
        return nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(self.last_features, self.init_features, kernel_size=4, stride=1, padding=0, bias=False)),
                ('norm', nn.BatchNorm2d(self.init_features)),
                ('act1', nn.LeakyReLU(0.2, inplace=True)),
                ('conv2', nn.Conv2d(self.init_features, 1, kernel_size=1, stride=1, padding=0, bias=False)),
                ('act2', nn.Sigmoid())
            ])
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = torch.cat([x, y], dim=1)
        x = self.init_conv(x)
        for conv in self.convs:
            x = conv(x)
        return self.out_conv(x)

def _test():
    x = torch.randn(4, 3, 64, 64)
    y = torch.randn(4, 2, 64, 64)
    z = torch.randn(4, 1, 64, 64)
    generator = Generator(3, 2, 64, 4)
    discriminator = Discriminator(5, 64, [1, 2, 4, 8])
    G_out = generator(z, y)
    print(G_out.shape)
    D_out = discriminator(x, y)
    print(D_out.shape)

if __name__ == '__main__':
    _test()

