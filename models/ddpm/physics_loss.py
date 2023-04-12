import math
import torch
import torch.fft as fft

def vorticity_residual(w, w_prev, w_next, visc, dt):
    bsize, _, _, nx, ny = w.shape
    device = w.device

    # Wavenumbers in Fourier space
    k_max = nx // 2
    N = nx 
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0). \
                        reshape(N, 1).repeat(1, N).reshape(1, 1, N, N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0). \
                        reshape(1, N).repeat(N, 1).reshape(1, 1, N, N)
    
    # Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0

    # Calculate stream function in Fourier space
    w = w.clone()
    w.requires_grad_(True)
    w_h = fft.fft2(w, dim=[-2, -1])
    psi_h = w_h / lap

    # Velocity field in Fourier space
    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    lap_w_h = - lap * w_h

    # Calculate velocity field in physical space
    u = fft.irfft2(u_h[..., :, :k_max + 1], dim=[-2, -1])
    v = fft.irfft2(v_h[..., :, :k_max + 1], dim=[-2, -1])
    wx = fft.irfft2(wx_h[..., :, :k_max + 1], dim=[-2, -1])
    wy = fft.irfft2(wy_h[..., :, :k_max + 1], dim=[-2, -1])
    lap_w = fft.irfft2(lap_w_h[..., :, :k_max + 1], dim=[-2, -1])
    
    advection = u * wx + v * wy

    # Centeral difference in time
    wt = (w_next - w_prev) / (2 * dt)

    # Force term
    x = torch.linspace(0., 1., nx+1, device=device)
    x = x[0:-1]
    X, Y = torch.meshgrid(x, x, indexing='ij')
    f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    # Calculate the residual
    res = wt + (advection - visc * lap_w) - f
    loss = (res ** 2).mean()

    return loss