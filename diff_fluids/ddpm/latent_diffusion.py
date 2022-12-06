from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from unet import UNetModel
from diff_fluids.vae.vae import Autoencoder

class LatentDiffusion(nn.Module):
    eps_model: UNetModel
    vae: Autoencoder
    def __init__(self, 
                 eps_model: UNetModel,
                 vae: Autoencoder,
                 latent_scaling: float,
                 n_steps: int,
                 betas: List[float]):
        super().__init__()
        self.eps_model = eps_model
        self.vae = vae
        self.latent_scaling = latent_scaling
        
        self.n_steps = n_steps
        beta = torch.linspace(betas[0]**0.5, betas[1]**0.5, n_steps, dtype=torch.float32) ** 2
        self.beta = nn.Parameter(beta, requires_grad=False)

        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar, requires_grad=False)
    
    @property
    def device(self):
        return next(iter(self.eps_model.parameters())).device
    
    def get_condition_embedding(self, cond: torch.Tensor) -> torch.Tensor:
        return cond

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.latent_scaling * self.vae.encode(x).sample()
    
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z / self.latent_scaling)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]=None) -> torch.Tensor:
        return self.eps_model(x, t, cond)