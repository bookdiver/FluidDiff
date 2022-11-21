import logging
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from diffusers import UNet2DModel

logging.basicConfig(level=logging.DEBUG)

class DDPM(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 betas: list, 
                 n_T: int, 
                 device: str,
                 pretrained: Optional[str]=None) -> None:
        super(DDPM, self).__init__()
        """ DDPM model

        Args:
            in_channels (int): input channels, 1 for only input density, 3 for input density and velocity
            n_feats (int): hidden channels, for embedding and conditioning
            betas (List[float]): bounds for betas in noise schedule
            n_T (int): diffusion steps
        """
        self.net = UNet2DModel(in_channels=in_channels,
                                   out_channels=in_channels,
                                   down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
                                   up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
                                   block_out_channels=(64, 128, 256)
            )
        if pretrained:
            self.net.load_state_dict(torch.load(pretrained))
            logging.info(f"Load pretrained model from {pretrained}")
        
        self.betas = betas
        self.n_T = n_T
        self.mse_loss = nn.MSELoss()

        for k, v in self.ddpm_schedule().items():
            self.register_buffer(k, v)
        self.device = device
    
    def ddpm_schedule(self) -> dict:
        """ Pre-computed schedules for DDPM training.

        Returns:
            (dict): DDPM schedules
        """    
        beta1, beta2 = self.betas
        assert beta1 < beta2 < 1.0 # beta1 and beta2 must be in (0, 1)

        beta_t = (beta2 - beta1) * torch.arange(0, self.n_T + 1, dtype=torch.float32) / self.n_T + beta1
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            'alpha_t': alpha_t,  # \alpha_t,
            'oneover_sqrta': oneover_sqrta,  # 1/\sqrt{\alpha_t},
            'sqrt_beta_t': sqrt_beta_t,  # \sqrt{\beta_t},
            'alphabar_t': alphabar_t,  # \bar{\alpha_t},
            'sqrtab': sqrtab,  # \sqrt{\bar{\alpha_t}},
            'sqrtmab': sqrtmab,  # \sqrt{1-\bar{\alpha_t}},
            'mab_over_sqrtmab': mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}},
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ This function is used to compute the loss of DDPM.

        Args:
            x (torch.Tensor): (B, 1, H, W), original frame (TODO: now it's only the density, need add velocity as additional channel)

        Returns:
            loss (torch.Tensor): DDPM loss for reconstruct the noise
        """        
        _nts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device)
        _ts = (_nts / self.n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1), 
        sqrtab = self.sqrtab[_nts, None, None, None]    # \sqrt{\bar{\alpha_t}}, extended to (B, 1, 1, 1)
        sqrtmab = self.sqrtmab[_nts, None, None, None]  # \sqrt{1-\bar{\alpha_t}}, extended to (B, 1, 1, 1)

        x_t = (sqrtab * x + sqrtmab * noise)
        # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # the noisy term in loss
        
        # return MSE between added noise, and our predicted noise
        loss = self.mse_loss(noise, self.net(x_t, _ts)[0])
        return loss
    
    def sample(self, n_samples: int, size: Optional[tuple]=None) -> torch.Tensor:
        if size is not None:
            x_i = torch.randn(n_samples, 1, *size).to(self.device)
        else:
            x_i = torch.randn(n_samples, 1, 96, 64).to(self.device)
        for i in tqdm(range(self.n_T, 0, -1)):
            t_is = torch.tensor([i / self.n_T]).to(self.device)

            z = torch.randn_like(x_i) if i > 1 else 0

            with torch.no_grad():
                noise = self.net(x_i, t_is)[0]
                x_i = self.oneover_sqrta[i] * (x_i - noise * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            
        return x_i
