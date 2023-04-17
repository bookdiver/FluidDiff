import math
import torch
import torch.fft as fft

def burgers_residual(u, visc=1e-2, dt=1e-2, u0=None):
    _, _, _, nx = u.shape
    device = u.device

    # Wavenumbers in Fourier space
    k_max = nx // 2
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                      torch.arange(start=-k_max, end=0, step=1, device=device)), 0). \
                        reshape(1, 1, 1, nx)
    
    def cal_residual(u):
        u_h = torch.fft.fft(u, dim=-1)
        
        ux_h = 1j * k_x * u_h
        uxx_h = 1j * k_x * ux_h
        ux = torch.fft.ifft(ux_h[:, :, :, :], dim=-1, n=nx).real
        uxx = torch.fft.ifft(uxx_h[:, :, :, :], dim=-1, n=nx).real

        ut = (u[:, :, 2:, :] - u[:, :, :-2, :]) / (2 * dt)
        res = ut + (u * ux - visc * uxx)[:, :, 1:-1, :]
        
        return res
    
    res_u = cal_residual(u)
    loss = torch.mean(res_u ** 2)
    if u0 is not None:
        res_u0 = cal_residual(u0)
        loss = torch.mean((res_u - res_u0) ** 2)
    
    return loss
    


def navier_stokes_residual(w, w_prev, w_next, visc, dt, w0=None):
    _, _, _, nx, ny = w.shape
    device = w.device

    # Wavenumbers in Fourier space
    k_max = nx // 2
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                      torch.arange(start=-k_max, end=0, step=1, device=device)), 0). \
                        reshape(nx, 1).repeat(1, ny).reshape(1, 1, 1, nx, ny)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                      torch.arange(start=-k_max, end=0, step=1, device=device)), 0). \
                        reshape(1, ny).repeat(nx, 1).reshape(1, 1, 1, nx, ny)

    # Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0

    w = w.clone()
    w.requires_grad_(True)

    def cal_residual(w, f):
        w_h = fft.fft2(w, dim=[-2, -1])
        psi_h = w_h / lap

        # Velocity field in Fourier space
        u_h = 1j * k_y * psi_h
        v_h = -1j * k_x * psi_h
        wx_h = 1j * k_x * w_h
        wy_h = 1j * k_y * w_h
        lap_w_h = - lap * w_h

        # Calculate velocity field in physical space
        u = fft.ifft2(u_h[..., :k_max + 1, :, :], dim=[-2, -1]).real
        v = fft.ifft2(v_h[..., :k_max + 1, :, :], dim=[-2, -1]).real
        wx = fft.ifft2(wx_h[..., :k_max + 1, :, :], dim=[-2, -1]).real
        wy = fft.ifft2(wy_h[..., :k_max + 1, :, :], dim=[-2, -1]).real
        lap_w = fft.ifft2(lap_w_h[..., :k_max + 1, :, :], dim=[-2, -1]).real
        
        advection = u * wx + v * wy

        # Centeral difference in time
        wt = (w_next - w_prev) / (2 * dt)

        res = wt + (advection - visc * lap_w) - f

        return res

    # Force term
    x = torch.linspace(0., 1., nx+1, device=device)
    x = x[0:-1]
    X, Y = torch.meshgrid(x, x, indexing='ij')
    f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    res_w = cal_residual(w, f)
    loss = torch.mean(res_w ** 2)
    if w0 is not None:
        res_w0 = cal_residual(w0, f)
        loss = torch.mean((res_w - res_w0) ** 2)

    return loss

if __name__ == '__main__':
    u = torch.rand(200, 1, 100, 128)
    u_prev = torch.rand(200, 1, 100, 128)
    u_next = torch.rand(200, 1, 100, 128)
    loss = burgers_residual(u, u_prev, u_next)
    print(loss)