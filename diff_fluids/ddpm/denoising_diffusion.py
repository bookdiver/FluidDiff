from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def gather(consts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

class DenoisingDiffusion:
    def __init__(self,
                 eps_model: nn.Module,
                 n_steps: int,
                 device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, n_steps, dtype=torch.float32, device=device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta
    
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple:
        mean = (gather(self.alpha_bar, t) ** 0.5) * x0
        var = 1 - gather(self.alpha_bar, t)
        return (mean, var)
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * torch.randn_like(x0)

    def p_xtminus1_xt(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]=None) -> Tuple:
        var = gather(self.sigma2, t)

        eps_pred = self.eps_model(x_t, t, cond)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coeff = (1 - alpha) / ((1 - alpha_bar) ** 0.5)
        mean = 1 / (alpha ** 0.5) * (x_t - eps_coeff * eps_pred)

        return (mean, var)
    
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]=None) -> torch.Tensor:
        mean, var = self.p_xtminus1_xt(x_t, t, cond)
        return mean + (var ** 0.5) * torch.randn_like(x_t)

    def ddpm_loss(self, x0: torch.Tensor, cond: torch.Tensor=None) -> torch.Tensor:
        
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device, dtype=torch.long)

        noise = torch.randn_like(x0)
        
        x_t = self.q_sample(x0, t)
        eps_pred = self.eps_model(x_t, t, cond)
        return F.mse_loss(noise, eps_pred)
    
    @torch.no_grad()
    def sample(self, x_seed: torch.Tensor, cond: Optional[torch.Tensor]=None) -> torch.Tensor:
        x_t = x_seed
        cond_emb = self.get_cond_embedding(cond) if cond is not None else None
        for t_ in range(self.n_steps):
            t = self.n_steps - t_ - 1
            x_t = self.p_sample(x_t, x_t.new_full((x_t.shape[0], ), t, dtype=torch.long), cond_emb)
        return x_t
        
        