from collections import namedtuple
import math
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

ModelPrediction = namedtuple('ModelPrediction', ['eps_theta', 'x0_recon'])

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(constant: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    bs, *_ = t.shape
    out = constant.gather(-1, t)

    return out.reshape(bs, *((1, ) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    diffusion_step_scale = 1000 / timesteps
    beta_start = diffusion_step_scale * 0.0001
    beta_end = diffusion_step_scale * 0.02

    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps: int, s: float=0.008) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alpha_bar = torch.cos((t+s) / (1+s) * math.pi * 0.5) **2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(self,
                 denoising_fn: torch.nn.Module,
                 *,
                 sample_size: tuple,
                 timesteps: int=1000,
                 sampling_timesteps: int=None,
                 beta_schedule_type: str='linear',
                 ddim_sampling_eta: float=0.,
                 loss_type: str='L2'
                 ) -> None:
        super().__init__()

        self.denoising_fn = denoising_fn
        self.sample_size = sample_size
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps, 'sampling_timesteps must be less than or equal to timesteps'
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        if beta_schedule_type == 'linear':
            beta_t = linear_beta_schedule(timesteps)
        elif beta_schedule_type == 'cosine':
            beta_t = cosine_beta_schedule(timesteps)
        else:
            raise NotImplementedError(f'beta_schedule_type {beta_schedule_type} not implemented')

        alpha_t = 1 - beta_t
        alpha_bar_t = torch.cumprod(alpha_t, dim=0)
        alpha_bar_t_prev = F.pad(alpha_bar_t[:-1], (1, 0), value=1.0)

        time_steps, = beta_t.shape
        self.num_timesteps = int(time_steps)
        
        register_buffer('beta_t', beta_t)
        register_buffer('alpha_bar_t', alpha_bar_t)
        register_buffer('alpha_bar_t_prev', alpha_bar_t_prev)

        register_buffer('sqrt_alpha_bar_t', torch.sqrt(alpha_bar_t))
        register_buffer('sqrt_one_minus_alpha_bar_t', torch.sqrt(1 - alpha_bar_t))
        register_buffer('log_one_minus_alpha_bar_t', torch.log(1 - alpha_bar_t))
        register_buffer('sqrt_recip_alpha_bar_t', torch.sqrt(1 / alpha_bar_t))
        register_buffer('sqrt_recipm1_alpha_bar_t', torch.sqrt(1 / alpha_bar_t - 1))

        posterior_variance = beta_t * (1. - alpha_bar_t_prev) / (1. - alpha_bar_t)
        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', beta_t * torch.sqrt(alpha_bar_t_prev) / (1. - alpha_bar_t))
        register_buffer('posterior_mean_coef2', (1. - alpha_bar_t_prev) * torch.sqrt(alpha_t) / (1. - alpha_bar_t))

        self.loss_type = loss_type
    
    def get_eps(self, xt: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert xt.device == t.device == y.device == self.betas.device, f'xt, t, y, and eps_model must be on the same device'

        return self.denoising_fn(xt, t, y)
    
    def get_q_xt_x0_mean_variance(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:      
        mean = extract(self.sqrt_alpha_bar_t, t, x0.shape)
        variance = extract(self.one_minus_alpha_bar_t, t, x0.shape)
        log_variance = extract(self.log_one_minus_alpha_bar_t, t, x0.shape)

        return mean, variance, log_variance
    
    def predict_x0_from_xt(self, xt: torch.Tensor, t: torch.LongTensor, eps: torch.Tensor) -> torch.Tensor:
        xt_coeff = extract(self.sqrt_recip_alpha_bar_t, t, xt.shape)
        eps_coeff = extract(self.sqrt_recipm1_alpha_bar_t, t, eps.shape)

        return xt_coeff * xt - eps_coeff * eps
    
    def predict_xt_from_x0(self, xt: torch.Tensor, t: torch.LongTensor, x0_recon: torch.Tensor) -> torch.Tensor:
        xt_coeff = extract(self.sqrt_recip_alpha_bar_t, t, xt.shape)
        scale = 1 / extract(self.sqrt_recipm1_alpha_bar_t, t, xt.shape)

        return (xt_coeff * xt - x0_recon) * scale
    
    
    def get_q_posterior(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.LongTensor) -> tuple:
        mean = extract(self.posterior_mean_coef1, t, x0.shape) * x0 + extract(self.posterior_mean_coef2, t, xt.shape) * xt
        variance = extract(self.posterior_variance, t, x0.shape)
        log_variance = extract(self.posterior_log_variance_clipped, t, x0.shape)

        return mean, variance, log_variance

    def get_model_predictions(self, xt: torch.Tensor, t: torch.LongTensor, cond: torch.Tensor, clip_x0_recon: bool=False, rederive_eps_theta: bool=False) -> tuple:
        eps_theta = self.denoising_fn(xt, t, cond=cond)
        x0_recon = self.predict_x0_from_xt(xt, t, eps=eps_theta)

        if clip_x0_recon:
            x0_recon = x0_recon.clamp(-1.0, 1.0)

        if clip_x0_recon and rederive_eps_theta:
            eps_theta = self.predict_xt_from_x0(xt, t, x0_recon)

        return ModelPrediction(eps_theta=eps_theta, x0_recon=x0_recon)

    def get_p_mean_variance(self, xt: torch.Tensor, t: torch.LongTensor, cond: torch.Tensor):
        preds = self.get_model_predictions(xt=xt, t=t, cond=cond)
        x0 = preds.x0_recon

        mean_theta, posterior_variance, posterior_log_variance = self.get_q_posterior(x0=x0, xt=xt, t=t)
        return mean_theta, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, xt: torch.Tensor, t: int, cond: torch.Tensor) -> torch.Tensor:
        bs, device = xt.shape[0], xt.device
        batched_ts = torch.full((bs, ), t, device=device, dtype=torch.long)
        mean_theta, _, log_variance_theta = self.get_p_mean_variance(xt=xt, t=batched_ts, cond=cond)
        eps = torch.randn_like(xt) if t > 0 else 0

        return mean_theta + (0.5 * log_variance_theta).exp() * eps
    
    @torch.no_grad()
    def p_sample_loop(self, shape: tuple, cond: torch.Tensor, device: torch.device) -> torch.Tensor:
        xt = torch.randn(shape, device=device)

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='DDPM sampling', total=self.num_timesteps):
            xt = self.p_sample(xt=xt, t=t, cond=cond)

        return xt
    
    @torch.no_grad()
    def ddim_sample(self, xt: torch.Tensor, t: int, t_next: int, cond: torch.Tensor) -> torch.Tensor:
        bs, device = xt.shape[0], xt.device
        batched_ts = torch.full((bs, ), t, device=device, dtype=torch.long)
        eps_theta, x0_recon = self.get_model_predictions(xt=xt, t=batched_ts, cond=cond, clip_x0_recon=True, rederive_eps_theta=True)

        if t_next < 0:
            return x0_recon

        alpha_bar = self.alpha_bar_t[t]
        alpha_bar_next = self.alpha_bar_t[t_next]

        sigma = self.ddim_sampling_eta * ((1 - alpha_bar / alpha_bar_next) * (1 - alpha_bar_next) / (1 - alpha_bar)).sqrt()
        c = (1 - alpha_bar_next - sigma ** 2).sqrt()

        eps = torch.randn_like(xt)

        return x0_recon*alpha_bar_next.sqrt() +  c*eps_theta + sigma*eps

    @torch.no_grad()
    def ddim_sample_loop(self, shape: tuple, cond: torch.Tensor, device: torch.device) -> torch.Tensor:
        times = torch.linspace(-1, self.num_timesteps-1, steps=self.sampling_timesteps+1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        xt = torch.randn(shape, device=device)

        for t, t_next in tqdm(time_pairs, desc='DDIM sampling', total=len(time_pairs)):
            xt = self.ddim_sample(xt=xt, t=t, t_next=t_next, cond=cond)

        return xt

    @torch.no_grad()
    def sample(self, cond: torch.Tensor=None) -> torch.Tensor:
        device = next(self.denoising_fn.parameters()).device

        cond = cond.to(device)
        b = cond.shape[0]
        shape = (b, *self.sample_size)

        sampling_fn = self.ddim_sample_loop if self.is_ddim_sampling else self.p_sample_loop

        return sampling_fn(shape=shape, cond=cond, device=device)
    
    def q_sample(self, x0: torch.Tensor, t: torch.LongTensor, eps: torch.Tensor=None) -> torch.Tensor:
        eps = default(eps, lambda: torch.randn_like(x0))
        mean = extract(self.sqrt_alpha_bar_t, t, x0.shape)
        std = extract(self.sqrt_one_minus_alpha_bar_t, t, x0.shape)

        return mean * x0 + std * eps
    
    @property
    def loss_fn(self):
        if self.loss_type == 'L1':
            return F.l1_loss
        elif self.loss_type == 'L2':
            return F.mse_loss
        else:
            raise NotImplementedError(f'Loss type {self.loss_type} not implemented')

    def p_losses(self, x0: torch.Tensor, t: torch.LongTensor, cond: torch.Tensor, eps: torch.Tensor=None, **kwargs) -> torch.Tensor:
        device = x0.device
        eps = default(eps, lambda: torch.randn_like(x0))

        xt = self.q_sample(x0=x0, t=t, eps=eps)

        cond = cond.to(device) if cond is not None else None

        eps_theta = self.denoising_fn(xt, t, cond, **kwargs)

        loss = self.loss_fn(eps_theta, eps)
        
        return loss
    
    def forward(self, x0: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        b, device = x0.shape[0], x0.device
        t = torch.randint(0, self.num_timesteps, (b, ), device=device).long()
        return self.p_losses(x0, t, *args, **kwargs)

def test():
    from unet import Unet3D
    eps_model = Unet3D(
        channels=1,
        cond_channels=1,
        channel_mults=(1, 2, 4, 8),
        init_conv_channels=32
    )
    diffuser = GaussianDiffusion(
        denoising_fn=eps_model,
        sample_size=(1, 20, 64, 64),
        timesteps=100,
        sampling_timesteps=10,
    ).cuda()
    x = torch.randn(2, 1, 20, 64, 64).cuda()
    cond = torch.randn(2, 1, 64, 64).cuda()
    x = diffuser.sample(cond=cond)
    print(x.shape)
    # sample = diffuser.ddpm_sample(cond=cond)
    # print(sample.shape)

if __name__ == '__main__':
    test()


    