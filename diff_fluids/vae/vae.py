from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """ A Vanilla VAE implementation for compressing data into a latent space.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: Optional[int]=None,
                 in_dim: Optional[int]=64,
                 latent_dim: int=128, 
                 layer_out_channels: list=None):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        self.layer_out_channels = layer_out_channels if layer_out_channels else [16, 32, 64, 128]
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        # encoder
        layers = []
        for out_channels in self.layer_out_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU()
                )
            )
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
        
        # latent space
        self.fc_mu = nn.Linear(self.layer_out_channels[-1] * ((self.in_dim // 16)**2), self.latent_dim)
        self.fc_var = nn.Linear(self.layer_out_channels[-1] * ((self.in_dim // 16)**2), self.latent_dim)

        # decoder
        self.decode_fc = nn.Linear(self.latent_dim, self.layer_out_channels[-1] * ((self.in_dim // 16)**2))
        self.layer_out_channels
        layers = []
        for i in range(len(self.layer_out_channels)-1, 0, -1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.layer_out_channels[i], 
                                       self.layer_out_channels[i-1], 
                                       kernel_size=3, 
                                       stride=2, 
                                       padding=1, 
                                       output_padding=1),
                    nn.BatchNorm2d(self.layer_out_channels[i-1]),
                    nn.LeakyReLU()
                )
            )
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(self.layer_out_channels[0], 
                                   self.layer_out_channels[0], 
                                   kernel_size=3, 
                                   stride=2, 
                                   padding=1, 
                                   output_padding=1),
                nn.BatchNorm2d(self.layer_out_channels[0]),
                nn.LeakyReLU(),
                nn.Conv2d(self.layer_out_channels[0], self.out_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )
        )
        self.decoder = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return (mu, log_var)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.decode_fc(z)
        z = z.reshape(-1, self.layer_out_channels[-1], self.in_dim // 16, self.in_dim // 16)
        z = self.decoder(z)
        return z
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, input: torch.Tensor) -> dict:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return {'recon': self.decode(z), 
                'input': input,
                'mu': mu, 
                'log_var': log_var}
    
    def loss_function(self, output: dict) -> dict:
        recon = output['recon']
        input = output['input']
        mu = output['mu']
        log_var = output['log_var']

        recon_loss = F.mse_loss(recon, input, reduction='sum')
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)

        loss = recon_loss + kld_loss
        return {'loss': loss, 'reconstruction_loss':recon_loss.detach(), 'kld_loss':-kld_loss.detach()}
    
    def sample(self, num_samples: int, current_device: int) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        return self.decode(z)
    
    def generate(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward(input)['recon']