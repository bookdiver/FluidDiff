from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

class DenoisingDiffusion(nn.Module):
    def __init__(self,
                 eps_model: nn.Module,
                 n_steps: int,
                 betas: tuple = (1e-4, 0.02),
                 device: torch.device = 'cuda:0',
                 ) -> None:
        super().__init__()
        """ DDPM model

        Args:
            in_channels (int): input channels, 1 for only input density, 3 for input density and velocity
            n_feats (int): hidden channels, for embedding and conditioning
            betas (List[float]): bounds for betas in noise schedule
            n_steps (int): diffusion steps
        """
        self.eps_model = eps_model.to(device)
        self.betas = betas
        self.n_steps = n_steps
        self.mse_loss = nn.MSELoss()

        for k, v in self.ddpm_schedule().items():
            self.register_buffer(k, v)

        self.device = device
    
    @property
    def ddpm_schedule(self) -> dict:
        """ Pre-computed schedules for DDPM training.

        Returns:
            (dict): DDPM schedules
        """    
        beta1, beta2 = self.betas
        assert beta1 < beta2 < 1.0 # beta1 and beta2 must be in (0, 1)

        beta_t = (beta2 - beta1) * torch.arange(0, self.n_steps + 1, dtype=torch.float32) / self.n_steps + beta1
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
    
    def ddpm_loss(self, x: torch.Tensor, cond: Optional[torch.Tensor]=None) -> torch.Tensor:
        """ This function is used to compute the loss of DDPM.

        Args:
            x (torch.Tensor): (B, 1, H, W), original frame (TODO: now it's only the density, need add velocity as additional channel)

        Returns:
            loss (torch.Tensor): DDPM loss for reconstruct the noise
        """        
        _nts = torch.randint(1, self.n_steps, (x.shape[0],)).to(x.device)
        _ts = (_nts / self.n_steps)
        noise = torch.randn_like(x)  # eps ~ N(0, 1), 
        sqrtab = self.sqrtab[_nts, None, None, None]    # \sqrt{\bar{\alpha_t}}, extended to (B, 1, 1, 1)
        sqrtmab = self.sqrtmab[_nts, None, None, None]  # \sqrt{1-\bar{\alpha_t}}, extended to (B, 1, 1, 1)

        x_t = (sqrtab * x + sqrtmab * noise)
        # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # the noisy term in loss
        
        # return MSE between added noise, and our predicted noise
        loss = self.mse_loss(noise, self.eps_model(x_t, _ts, cond))
        return loss
    
    @torch.no_grad()
    def sample(self, x0: torch.Tensor, cond: Optional[torch.Tensor]=None) -> torch.Tensor:
        """ This function is used to sample from DDPM.

        Args:
            x0 (torch.Tensor): (B, 1, H, W), initial seed, usually set as N(0, 1)
            cond (torch.Tensor): (B, C), additional conditioning

        Returns:
            x (torch.Tensor): (B, 1, H, W), sampled frame
        """        
        x_t = x0.to(self.device)
        cond = cond.to(self.device) if cond is not None else None

        for i in tqdm(range(self.n_steps, 0, -1)):
            t_is = torch.tensor([i / self.n_steps]).to(self.device)

            z = torch.randn_like(x_t) if i > 1 else 0

            noise_pred = self.eps_model(x_t, t_is, cond)
            x_t = self.oneover_sqrta[i] * (x_t - noise_pred * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
        
        return x_t



    