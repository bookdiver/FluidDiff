import math
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from einops_exts import check_shape

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

def linear_beta_schedule(n_diffusion_steps: int) -> torch.Tensor:
    actual_steps = n_diffusion_steps + 1

    diffusion_step_scale = 1000 / n_diffusion_steps
    beta_start = diffusion_step_scale * 0.0001
    beta_end = diffusion_step_scale * 0.02

    return torch.linspace(beta_start, beta_end, actual_steps, dtype=torch.float64)

def cosine_beta_schedule(n_diffusion_steps: int) -> torch.Tensor:
    actual_steps = n_diffusion_steps + 1

    def beta_for_alpha_bar(n_diffusion_steps: int, alpha_bar: callable, max_beta: float=0.999):
        betas = []
        for i in range(actual_steps):
            t1 = i / n_diffusion_steps
            t2 = (i + 1) / n_diffusion_steps
            betas.append(min(1-alpha_bar(t2)/alpha_bar(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float64)
    return beta_for_alpha_bar(
        n_diffusion_steps, 
        alpha_bar = lambda t: 1 - math.cos((t+0.008) / (1+0.008) * math.pi / 2) **2
        )

class GaussianDiffusion(nn.Module):
    def __init__(self,
                 eps_model: torch.nn.Module,
                 *,
                 domain_size: tuple,
                 n_frames: int,
                 n_channels: int,
                 n_diffusion_steps: int=1000,
                #  n_ddim_steps: int=50,
                 beta_schedule_type: str='linear',
                 loss_type: str='L2'
                #  ddim_schedule_type: str='uniform',
                #  ddim_eta: float=0.0,
                 ) -> None:
        super().__init__()

        self.eps_model = eps_model
        self.domain_size = domain_size
        self.n_frames = n_frames
        self.n_channels = n_channels

        self.n_steps = n_diffusion_steps + 1
        
        # if ddim_schedule_type == 'uniform':
            # c = n_diffusion_steps // n_ddim_steps
            # self.ddim_time_steps = np.asarray(list(range(0, self.T, c))) + 1
        # elif ddim_schedule_type == 'quad':
            # self.ddim_time_steps = ((np.linspace(0, np.sqrt(self.T * 0.8), n_ddim_steps)) ** 2).astype(int) + 1
        # else:
            # raise NotImplementedError(f'ddim_schedule_type {ddim_schedule_type} not implemented')

        with torch.no_grad():
        
            if beta_schedule_type == 'linear':
                self.beta_t = linear_beta_schedule(n_diffusion_steps).to(torch.float32)
            elif beta_schedule_type == 'cosine':
                self.beta_t = cosine_beta_schedule(n_diffusion_steps).to(torch.float32)
            else:
                raise NotImplementedError(f'beta_schedule_type {beta_schedule_type} not implemented')
            
            self.alpha_t = (1.0 - self.beta_t).to(torch.float32)

            self.alpha_bar_t = torch.cumprod(self.alpha_t, dim=0).to(torch.float32)
            # self.ddim_alpha_bar_t = self.alpha_bar_t[self.ddim_time_steps].clone()

            self.alpha_bar_t_prev = torch.cat([self.alpha_bar_t.new_tensor([1.]), self.alpha_bar_t[:-1]]).to(torch.float32)
            # self.ddim_alpha_bar_t_prev = torch.cat([self.alpha_bar_t[0:1], self.alpha_bar_t[self.ddim_time_steps[:-1]]]).to(torch.float32)

            self.sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar_t).to(torch.float32)
            # self.ddim_sqrt_alpha_bar_t = torch.sqrt(self.ddim_alpha_bar_t).to(torch.float32)

            self.one_minus_alpha_bar_t = (1.0 - self.alpha_bar_t).to(torch.float32)
            self.log_one_minus_alpha_bar_t = torch.log(self.one_minus_alpha_bar_t).to(torch.float32)
            self.sqrt_one_minus_alpha_bar_t = torch.sqrt(self.one_minus_alpha_bar_t).to(torch.float32)
            # self.ddim_sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - self.ddim_alpha_bar_t).to(torch.float32)

            self.one_over_sqrt_alpha_bar_t = (1.0 / self.sqrt_one_minus_alpha_bar_t).to(torch.float32)

            self.sqrt_one_over_alpha_bar_t_minus_one = torch.sqrt(1/self.alpha_bar_t - 1).to(torch.float32)
            self.posterior_variance = (self.beta_t * (1.0 - self.alpha_bar_t_prev) / (1.0 - self.alpha_bar_t)).to(torch.float32)
            self.posterior_log_variance = torch.log(self.posterior_variance.clamp(min=1e-20)).to(torch.float32)

            # self.ddim_sigma = (ddim_eta * (torch.sqrt((1.0 - self.ddim_alpha_bar_t_prev) / (1.0 - self.ddim_alpha_bar_t))) * \
                                # (torch.sqrt(1.0 - self.ddim_alpha_bar_t / self.ddim_alpha_bar_t_prev))).to(torch.float32)

            self.posterior_mean_x0_coeff = (self.beta_t * torch.sqrt(self.alpha_bar_t_prev) / (1.0 - self.alpha_bar_t)).to(torch.float32)
            self.posterior_mean_xt_coeff = ((1 - self.alpha_bar_t_prev) * torch.sqrt(self.alpha_t) / (1.0 - self.alpha_bar_t)).to(torch.float32)

            self.loss_type = loss_type
    
    def get_eps(self, xt: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert xt.device == t.device == y.device == self.device, f'xt, t, y, and eps_model must be on the same device'

        return self.eps_model(xt, t, y)
    
    def get_q_xt_x0_mean_variance(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:      
        mean = extract(self.sqrt_alpha_bar_t, t, x0.shape)
        variance = extract(self.one_minus_alpha_bar_t, t, x0.shape)
        log_variance = extract(self.log_one_minus_alpha_bar_t, t, x0.shape)

        return mean, variance, log_variance
    
    def predict_x0_from_xt(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        xt_coeff = extract(self.one_over_sqrt_alpha_bar_t, t, xt.shape)
        eps_coeff = extract(self.sqrt_one_over_alpha_bar_t_minus_one, t, eps.shape)

        return xt_coeff * xt - eps_coeff * eps
    
    def get_q_xtm1_xt_mean_variance(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> tuple:
        mean = extract(self.posterior_mean_x0_coeff, t, x0.shape) * x0 + extract(self.posterior_mean_xt_coeff, t, xt.shape) * xt
        variance = extract(self.posterior_variance, t, x0.shape)
        log_variance = extract(self.posterior_log_variance, t, x0.shape)

        return mean, variance, log_variance

    def get_p_xtm1_xt_mean_variance(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, cond_scale: float=1.0, clip_denoising: bool=True) -> tuple:
        eps_theta = self.eps_model.forward_with_cond_scale(xt, t, cond=cond, cond_scale=cond_scale)
        x0_recon = self.predict_x0_from_xt(xt, t, eps=eps_theta)

        if clip_denoising:
            x0_recon = x0_recon.clamp(-1.0, 1.0)

        mean, variance, log_variance = self.get_q_xtm1_xt_mean_variance(x0=x0_recon, xt=xt, t=t)

        return mean, variance, log_variance
    
    @torch.inference_mode()
    def ddpm_p_sample(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, cond_scale: float=1.0, clip_denoising: bool=True) -> torch.Tensor:
        b, *_, device = *xt.shape, xt.device
        mean_theta, _, log_variance_theta = self.get_p_xtm1_xt_mean_variance(xt=xt, t=t, cond=cond, cond_scale=cond_scale, clip_denoising=clip_denoising)
        mean_theta, log_variance_theta = mean_theta.to(device), log_variance_theta.to(device)
        eps = torch.randn_like(xt)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1, ) * (len(xt.shape) - 1))).to(device)

        return mean_theta + nonzero_mask * (0.5 * log_variance_theta).exp() * eps
    
    @torch.inference_mode()
    def ddpm_p_sample_loop(self, shape: tuple, cond: torch.Tensor=None, cond_scale: float=1.0, device: torch.device=torch.device('cuda:0')) -> torch.Tensor:
        b = shape[0]
        fields = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.n_steps)), desc='DDPM sampling', total=self.n_steps):
            t = torch.full((b, ), i, device=device, dtype=torch.long)
            fields = self.ddpm_p_sample(xt=fields, t=t, cond=cond, cond_scale=cond_scale)

        return fields 
        
    @torch.inference_mode()
    def ddpm_sample(self, cond: torch.Tensor=None, cond_scale: float=1.0) -> torch.Tensor:
        device = next(self.eps_model.parameters()).device

        cond = cond.to(device)
        b = cond.shape[0]
        domain_size = self.domain_size
        n_frames = self.n_frames
        n_channels = self.n_channels
        shape = (b, n_channels, n_frames, *domain_size)

        return self.ddpm_p_sample_loop(shape=shape, cond=cond, cond_scale=cond_scale, device=device)
    
    @torch.inference_mode()
    def ddpm_interpolate(self, x1: torch.Tensor, x2: torch.Tensor, t: int=None, weight: float=0.5) -> torch.Tensor:
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.n_steps-1)

        assert x1.shape == x2.shape, f'x1 and x2 must have the same shape, got {x1.shape} and {x2.shape}'

        batched_t = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=batched_t), (x1, x2))

        inter_field = weight * xt1 + (1 - weight) * xt2 #TODO: do a spherical interpolation
        for i in tqdm(reversed(range(0, t)), desc="DDPM interpolation", total=t):
            inter_field = self.p_sample(inter_field, torch.full((b, ), i, device=device, dtype=torch.long))

        return inter_field
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor=None) -> torch.Tensor:
        eps = default(eps, lambda: torch.randn_like(x0))
        mean = extract(self.sqrt_alpha_bar_t, t, x0.shape)
        std = extract(self.sqrt_one_minus_alpha_bar_t, t, x0.shape)

        return mean * x0 + std * eps

    def p_losses(self, x0: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, eps: torch.Tensor=None, **kwargs) -> torch.Tensor:
        device = x0.device
        eps = default(eps, lambda: torch.randn_like(x0))

        xt = self.q_sample(x0, t, eps)

        cond = cond.to(device)

        eps_theta = self.eps_model(xt, t, cond, **kwargs)

        if self.loss_type == 'L1':
            loss = F.l1_loss(eps_theta, eps)
        elif self.loss_type == 'L2':
            loss = F.mse_loss(eps_theta, eps)
        else:
            raise NotImplementedError(f'Loss type {self.loss_type} not implemented')
        
        return loss
    
    def forward(self, x0: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        b, device, domain_size = x0.shape[0], x0.device, self.domain_size
        check_shape(x0, 'b c f h w', c=self.n_channels, f=self.n_frames, h=domain_size[0], w=domain_size[1])
        t = torch.randint(0, self.n_steps, (b, ), device=device).long()
        return self.p_losses(x0, t, *args, **kwargs)


    
    # @torch.no_grad()
    # def ddim_get_x_prev_and_x0_pred(self, eps: torch.Tensor, step: int, xt: torch.Tensor) -> torch.Tensor:
    #     alpha_bar = self.ddim_alpha_bar_t[step]
    #     alpha_bar_prev = self.ddim_alpha_bar_t_prev[step]
    #     sigma = self.ddim_sigma[step]
    #     sqrt_one_minus_alpha_bar = self.ddim_sqrt_one_minus_alpha_bar_t[step]

    #     x0_pred = (xt - sqrt_one_minus_alpha_bar * eps) / torch.sqrt(alpha_bar)

    #     dir_xt = torch.sqrt(1.0 - alpha_bar_prev - sigma ** 2) * eps

    #     if sigma == 0:
    #         eps = 0
    #     else:
    #         eps = torch.rand_like(xt)
        
    #     x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * eps

    #     return x_prev, x0_pred
    
    # @torch.no_grad()
    # def ddim_p_sample_step(self, xt: torch.Tensor, t: torch.Tensor, y: torch.Tensor, step: int) -> torch.Tensor:
    #     eps_t = self.get_eps(xt, t, y)

    #     x_prev, _ = self.ddim_get_x_prev_and_x0_pred(eps_t, step, xt)

    #     return x_prev, eps_t
    
    # @torch.no_grad()
    # def ddim_p_sample(self, sample_shape: tuple, y: torch.Tensor, x_start: torch.Tensor=None) -> torch.Tensor:
    #     bs = sample_shape[0]

    #     x = x_start if x_start is not None else torch.randn(sample_shape, device=self.device)

    #     for step in tqdm(np.flip(self.ddim_time_steps), desc='DDIM sampling'):
    #         t = x.new_full((bs, 1, 1, 1), step, dtype=torch.float64) / self.T
    #         x, _ = self.ddim_p_sample_step(x, t, y, step)
        
    #     return x


def test():
    from unet3d import Unet3D
    eps_model = Unet3D(dim=32, cond_dim=64, dim_mults=(1, 2, 4, 8))
    diffuser = GaussianDiffusion(
        eps_model=eps_model,
        domain_size=(32, 32),
        n_frames=16,
        n_channels=3,
        n_diffusion_steps=100,
    )
    # videos = torch.randn(2, 3, 16, 32, 32)
    cond = torch.randn(2, 64)
    # loss = diffuser(videos, cond=cond)
    sample = diffuser.ddpm_sample(cond=cond)
    print(sample.shape)

if __name__ == '__main__':
    test()


    