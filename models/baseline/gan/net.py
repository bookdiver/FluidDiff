from collections import OrderedDict

import torch
import torch.nn as nn

def get_activation(activation: str):
    if activation == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f'Invalid activation: {activation}')


class UNetGenerator(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 init_features: int=64,
                 activation: str='leaky_relu',):
        super().__init__()
        self.init_conv = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels, init_features, kernel_size=3, stride=2, padding=1, bias=False)),
                ('norm', nn.BatchNorm2d(init_features)),
                ('act', get_activation(activation))
            ])
        )

        self.down_path = nn.Sequential(
            self._down_block(init_features, init_features * 2),
            self._down_block(init_features * 2, init_features * 4),
            self._down_block(init_features * 4, init_features * 8),
            self._down_block(init_features * 8, init_features * 8)
        )

        self.up_path = nn.Sequential(
            self._up_block(init_features * 8 * 2, init_features * 8),
            self._up_block(init_features * 8 * 2, init_features * 4),
            self._up_block(init_features * 4 * 2, init_features * 2),
            self._up_block(init_features * 2 * 2, init_features)
        )
        
        self.out_conv = nn.Sequential(
            OrderedDict([
                ('conv', nn.ConvTranspose2d(init_features * 2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
                ('norm', nn.BatchNorm2d(out_channels)),
                ('act', get_activation(activation))
            ])
        )


    @staticmethod
    def _down_block(in_channels: int, out_channels: int, activation: str='leaky_relu'):
        return nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)),
                ('norm', nn.BatchNorm2d(out_channels)),
                ('act', get_activation(activation))
            ])
        )
    
    @staticmethod
    def _up_block(in_channels: int, out_channels: int, activation: str='leaky_relu'):
        return nn.Sequential(
            OrderedDict([
                ('conv', nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
                ('norm', nn.BatchNorm2d(out_channels)),
                ('act', get_activation(activation))
            ])
        )
    
    def forward(self, x: torch.Tensor):
        hs = []
        h = self.init_conv(x)
        hs.append(h)
        for block in self.down_path:
            h = block(h)
            hs.append(h)
        
        for block in self.up_path:
            h = block(torch.cat([h, hs.pop()], dim=1))
        
        return self.out_conv(torch.cat([h, hs.pop()], dim=1))


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



class Discriminator(nn.Module):
    def __init__(self, in_channels: int, init_features: int=64, activation: str='leaky_relu'):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            self._block(in_channels, init_features, kernel_size=4, stride=2, padding=1, activation=activation),
            self._block(init_features, init_features * 2, kernel_size=4, stride=2, padding=1, activation=activation),
            self._block(init_features * 2, init_features * 4, kernel_size=4, stride=2, padding=1, activation=activation),
            self._block(init_features * 4, init_features * 8, kernel_size=4, stride=1, padding=1, activation=activation),
            nn.Conv2d(init_features * 8, 1, kernel_size=4, stride=1, padding=1, bias=False)
        )
    
    @staticmethod
    def _block(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, activation: str):
        return nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
                ('norm', nn.BatchNorm2d(out_channels)),
                ('act', get_activation(activation))
            ])
        )

    def forward(self, x:torch.Tensor, y: torch.Tensor):
        input = torch.cat((x, y), 1)
        return self.model(input)

def _test():
    x = torch.randn(4, 2, 64, 64)
    y = torch.randn(4, 2, 64, 64)
    generator = UNetGenerator(in_channels=2,
                              out_channels=2)
    discriminator = Discriminator(in_channels=4)
    print(f"The number of parameters in generator: {sum(p.numel() for p in generator.parameters())}")
    print(f"The number of parameters in discriminator: {sum(p.numel() for p in discriminator.parameters())}")
    G_out = generator(x)
    print(G_out.shape)
    D_out = discriminator(x, y)
    print(D_out.shape)
    # print(discriminator)

if __name__ == '__main__':
    _test()

