from collections import OrderedDict

import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 init_features: int=64):
        super().__init__()
        self.in_z_dims = in_z_dims
        self.in_y_channels = in_y_channels
        self.out_channels = out_channels
        self.init_features = init_features

        self.init_z_conv = self._init_z_conv()
        self.init_y_conv = self._init_y_conv()

        self.res_blocks = nn.Sequential(
            ResBlock(init_features * 4 * 2, init_features * 4 * 4),
            ResBlock(init_features * 4 * 4, init_features * 4 * 4),
            ResBlock(init_features * 4 * 4, init_features * 4 * 2),
            ResBlock(init_features * 4 * 2, init_features * 4),
        )
        self.up_blocks = nn.Sequential(
            self._upsample(init_features * 4, init_features * 2),
            self._upsample(init_features * 2, init_features),
            self._upsample(init_features, init_features // 2),
        )
        self.out_conv = self._out_conv()

    
    
    def _out_conv(self):
        return nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(self.init_features // 2, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
                ('act', nn.Tanh())
            ])
        )
    

    @staticmethod
    def _down_block(in_channels: int, out_channels: int):
        return nn.Sequential(
            OrderedDict([
                ('conv', nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
                ('norm', nn.BatchNorm2d(out_channels)),
                ('act', nn.LeakyReLU(0.2, inplace=True))
            ])
        )
    
    def forward(self, z: torch.Tensor, y: torch.Tensor):
        x1 = self.init_z_conv(z)
        x2 = self.init_y_conv(y)
        x = torch.cat([x1, x2], dim=1)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.up_blocks(x)
        return self.out_conv(x)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



class Discriminator(nn.Module):
    def __init__(self, in_x_channels, in_y_channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_x_channels+in_y_channels, 32, normalization=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *discriminator_block(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

def _test():
    x = torch.randn(4, 2, 64, 64)
    y = torch.randn(4, 2, 64, 64)
    z = torch.randn(4, 128, 1, 1)
    generator = Generator(in_z_dims=128,
                            in_y_channels=2,
                            out_channels=2)
    discriminator = Discriminator(in_x_channels=2,
                                  in_y_channels=2)
    print(f"The number of parameters in generator: {sum(p.numel() for p in generator.parameters())}")
    print(f"The number of parameters in discriminator: {sum(p.numel() for p in discriminator.parameters())}")
    G_out = generator(z, y)
    print(G_out.shape)
    D_out = discriminator(x, y)
    print(D_out.shape)
    # print(discriminator)

if __name__ == '__main__':
    _test()

