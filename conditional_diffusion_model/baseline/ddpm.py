import logging
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from net import ConditionalUnet

logging.basicConfig(level=logging.DEBUG)

class DDPM(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 n_feats: int, 
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

        self.net = ConditionalUnet(in_channels, n_feats)
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
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """ This function is used to compute the loss of DDPM.

        Args:
            x (torch.Tensor): (B, 1, H, W), input images
            c (torch.Tensor): (B, 1), labels of images

        Returns:
            loss (torch.Tensor): DDPM loss for reconstruct the noise
        """        
        _nts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device)
        _ts = (_nts / self.n_T)[:, None]  # t ~ Uniform(0, n_T), extended to (B, 1)
        noise = torch.randn_like(x)  # eps ~ N(0, 1), 
        sqrtab = self.sqrtab[_nts, None, None, None]    # \sqrt{\bar{\alpha_t}}, extended to (B, 1, 1, 1)
        sqrtmab = self.sqrtmab[_nts, None, None, None]  # \sqrt{1-\bar{\alpha_t}}, extended to (B, 1, 1, 1)

        x_t = (sqrtab * x + sqrtmab * noise)
        # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # the noisy term in loss
        
        # return MSE between added noise, and our predicted noise
        loss = self.mse_loss(noise, self.net(x_t, _ts, c))
        return loss
    
    def sample(self, n_samples: int, c_s: torch.tensor, size: Optional[tuple]=(1, 28, 28), noise_scale: float=0.5) -> torch.Tensor:
        """ Sampling from DDPM.

        Args:
            n_samples (int): number of total samples
            c_s (torch.tensor): conditions, with size of (B, 1) to match the channel dimension
            size (Optional[tuple]): size of the output image, (H, W)

        Returns:
            x_i (torch.Tensor): the generated samples
        """
           
        x_i = torch.randn(n_samples, *size).to(self.device) * noise_scale
        c_s = c_s.to(self.device)
        for i in tqdm(range(self.n_T, 0, -1)):
            t_is = torch.tensor([i / self.n_T])[:, None].to(self.device)

            z = torch.randn_like(x_i) if i > 1 else 0

            with torch.no_grad():
                noise = self.net(x_i, t_is, c_s)
                x_i = self.oneover_sqrta[i] * (x_i - noise * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            
        return x_i
