from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def gather(consts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

# class DenoisingDiffusion:
#     def __init__(self,
#                  eps_model: nn.Module,
#                  n_steps: int,
#                  device: torch.device):
#         super().__init__()
#         self.eps_model = eps_model
#         self.n_steps = n_steps
#         self.device = device
#         self.beta = torch.linspace(1e-4, 0.02, n_steps, dtype=torch.float32, device=device)
#         self.alpha = 1 - self.beta
#         self.alpha_bar = torch.cumprod(self.alpha, dim=0)
#         self.sigma2 = self.beta
    
#     def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple:
#         mean = (gather(self.alpha_bar, t) ** 0.5) * x0
#         var = 1 - gather(self.alpha_bar, t)
#         return (mean, var)
    
#     def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         mean, var = self.q_xt_x0(x0, t)
#         return mean + (var ** 0.5) * torch.randn_like(x0)

#     def p_xtminus1_xt(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]=None) -> Tuple:
#         var = gather(self.sigma2, t)

#         # eps_pred = self.eps_model(x_t, t, cond)
#         eps_pred = self.eps_model(x_t, t)[0]
#         alpha_bar = gather(self.alpha_bar, t)
#         alpha = gather(self.alpha, t)
#         eps_coeff = (1 - alpha) / ((1 - alpha_bar) ** 0.5)
#         mean = 1 / (alpha ** 0.5) * (x_t - eps_coeff * eps_pred)

#         return (mean, var)
    
#     def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]=None) -> torch.Tensor:
#         mean, var = self.p_xtminus1_xt(x_t, t, cond)
#         return mean + (var ** 0.5) * torch.randn_like(x_t)

#     def ddpm_loss(self, x0: torch.Tensor, cond: torch.Tensor=None) -> torch.Tensor:
        
#         batch_size = x0.shape[0]
#         t = torch.randint(0, self.n_steps, (batch_size,), device=self.device, dtype=torch.long)

#         noise = torch.randn_like(x0)
        
#         x_t = self.q_sample(x0, t)
#         # eps_pred = self.eps_model(x_t, t, cond)
#         eps_pred = self.eps_model(x_t, t)[0]
#         return F.mse_loss(noise, eps_pred)
    
#     @torch.no_grad()
#     def sample(self, x_seed: torch.Tensor, cond: Optional[torch.Tensor]=None) -> torch.Tensor:
#         x_t = x_seed
#         cond_emb = self.get_cond_embedding(cond) if cond is not None else None
#         for t_ in range(self.n_steps):
#             t = self.n_steps - t_ - 1
#             x_t = self.p_sample(x_t, x_t.new_full((x_t.shape[0], ), t, dtype=torch.long), cond_emb)
#         return x_t
        


class DenoisingDiffusion(nn.Module):
    def __init__(self,
                 eps_model: nn.Module,
                 n_steps: int,
                 device: torch.device,
                 ) -> None:
        super().__init__()
        """ DDPM model

        Args:
            in_channels (int): input channels, 1 for only input density, 3 for input density and velocity
            n_feats (int): hidden channels, for embedding and conditioning
            betas (List[float]): bounds for betas in noise schedule
            n_T (int): diffusion steps
        """
        self.eps_model = eps_model.to(device)
        self.betas = [1e-4, 0.02]
        self.n_T = n_steps
        self.mse_loss = nn.MSELoss()

        for k, v in self.ddpm_schedule().items():
            self.register_buffer(k, v)
    
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
    
    def ddpm_loss(self, x: torch.Tensor, cond: Optional[torch.Tensor]=None) -> torch.Tensor:
        """ This function is used to compute the loss of DDPM.

        Args:
            x (torch.Tensor): (B, 1, H, W), original frame (TODO: now it's only the density, need add velocity as additional channel)

        Returns:
            loss (torch.Tensor): DDPM loss for reconstruct the noise
        """        
        _nts = torch.randint(1, self.n_T, (x.shape[0],)).to(x.device)
        _ts = (_nts / self.n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1), 
        sqrtab = self.sqrtab[_nts, None, None, None]    # \sqrt{\bar{\alpha_t}}, extended to (B, 1, 1, 1)
        sqrtmab = self.sqrtmab[_nts, None, None, None]  # \sqrt{1-\bar{\alpha_t}}, extended to (B, 1, 1, 1)

        x_t = (sqrtab * x + sqrtmab * noise)
        # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # the noisy term in loss
        
        # return MSE between added noise, and our predicted noise
        loss = self.mse_loss(noise, self.eps_model(x_t, _ts, cond))
        return loss
    