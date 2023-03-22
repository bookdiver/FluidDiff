import torch
import math
import h5py
from einops import repeat

def navier_stokes_residual_loss(wt: torch.Tensor, nu: float=1e-5, dt: float=1.) -> torch.Tensor:
    bs, c, nt, nx, ny, device = *wt.shape, wt.device
    wt = wt.clone()
    wt.requires_grad_(True)

    w_h = torch.fft.fft2(wt[:, :, 1:-1], dim=[3, 4])
    k_x_max = nx // 2
    k_x = torch.cat((torch.arange(start=0, end=k_x_max, step=1, device=device),
                     torch.arange(start=-k_x_max, end=0, step=1, device=device)), dim=0)
    k_x = repeat(k_x, 'h -> 1 1 1 h w', h=nx, w=nx)
    k_y_max = ny // 2
    k_y = torch.cat((torch.arange(start=0, end=k_y_max, step=1, device=device),
                     torch.arange(start=-k_y_max, end=0, step=1, device=device)), dim=0)
    k_y = repeat(k_y, 'w -> 1 1 1 h w', h=ny, w=ny)
    laplacian = (k_x ** 2 + k_y ** 2)
    laplacian[..., 0, 0] = 1.0
    psi_h = w_h / laplacian

    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlaplacian_h = -laplacian * w_h

    u = torch.fft.irfft2(u_h[..., :, :k_x_max+1], dim=[3, 4])
    v = torch.fft.irfft2(v_h[..., :, :k_y_max+1], dim=[3, 4])
    wx = torch.fft.irfft2(wx_h[..., :, :k_x_max+1], dim=[3, 4])
    wy = torch.fft.irfft2(wy_h[..., :, :k_y_max+1], dim=[3, 4])
    wlaplacian = torch.fft.irfft2(wlaplacian_h[..., :, :k_x_max+1], dim=[3, 4])
    advection = (u * wx + v * wy) * dt

    w_t = (wt[:, :, 2:, :, :] - wt[:, :, :-2, :, :]) / (2 * dt)

    x = torch.linspace(0, 1, nx+1, device=device)[0:-1]
    y = torch.linspace(0, 1, ny+1, device=device)[0:-1]
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    f = 0.1 * (torch.sin(2*math.pi*(xx+yy)) + torch.cos(2*math.pi*(xx+yy)))

    residual = w_t + advection - nu * wlaplacian - f
    loss = torch.mean(residual ** 2)
    return loss

if __name__ == '__main__':
    with h5py.File('../../data/ns_V1e-5_T20_test.h5', 'r') as f:
        u = torch.from_numpy(f['u'][:]).permute(0, 3, 1, 2).unsqueeze(1).to(torch.float32)
    x = u[120:122, ...]
    x += 0.002 * torch.randn(2, 1, 20, 64, 64)
    loss = navier_stokes_residual_loss(x)
    print(loss)