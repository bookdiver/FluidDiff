import math
import torch
import torch.fft as fft
import torch.nn.functional as F

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

def Burgers_FDM(u, visc=1e-2, L=1.0, T=1.0):
    bsize, _, nt, nx = u.shape
    device = u.device
    u = u.reshape(bsize, nt, nx)

    dx = L / nx
    dt = T / (nt-1)

    k_max = nx // 2
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, 1, nx)
    
    u_h = fft.fft(u, dim=2)
    
    ux_h = 2j * math.pi * k_x * u_h
    uxx_h = 2j * math.pi * k_x * ux_h
    ux = fft.irfft(ux_h[:, :, :k_max+1], dim=2, n=nx)
    uxx = fft.irfft(uxx_h[:, :, :k_max+1], dim=2, n=nx)

    ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    Du = ut + (u * ux - visc * uxx)[:, 1:-1, :]
    
    return Du
    

def NS_vorticity_FDM(w, w_prev, w_next, visc=1e-3, fine_dt=1e-3):
    bsize, _, nt, nx, ny = w.shape
    device = w.device
    w = w.reshape(bsize, nt, nx, ny)
    w_prev = w_prev.reshape(bsize, nt, nx, ny)
    w_next = w_next.reshape(bsize, nt, nx, ny)

    # Wavenumbers in Fourier space
    k_max = nx // 2
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(nx, 1).repeat(1, ny).reshape(1, 1, nx, ny)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, ny).repeat(nx, 1).reshape(1, 1, nx, ny)

    # Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0

    w_h = fft.fft2(w, dim=(2, 3))
    psi_h = w_h / lap

    # Velocity field in Fourier space
    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    lap_w_h = - lap * w_h

    # Calculate velocity field in physical space
    u = fft.irfft2(u_h[:, :k_max + 1, :, :], dim=(2, 3), s=(nx, ny))
    v = fft.irfft2(v_h[:, :k_max + 1, :, :], dim=(2, 3), s=(nx, ny))
    wx = fft.irfft2(wx_h[:, :k_max + 1, :, :], dim=(2, 3), s=(nx, ny))
    wy = fft.irfft2(wy_h[:, :k_max + 1, :, :], dim=(2, 3), s=(nx, ny))
    lap_w = fft.irfft2(lap_w_h[:, :k_max + 1, :, :], dim=(2, 3), s=(nx, ny))
    
    advection = u * wx + v * wy

    # Centeral difference in time
    wt = (w_next - w_prev) / (2 * fine_dt)

    Du = wt + (advection - visc * lap_w)

    return Du

def Darcy_FDM(u, a, L=1):
    bsize, _, nx, ny = u.shape
    u = u.reshape(bsize, nx, ny)
    a = a.reshape(bsize, nx, ny)

    dx = L / (nx-1)
    dy = L / (ny-1)

    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

    a = a[:, 1:-1, 1:-1]

    aux = a * ux
    auy = a * uy
    auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
    auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
    
    Du = - (auxx + auyy)

    return Du


def Burgers_loss(u, u0=None):
    Du = Burgers_FDM(u)
    Du0 = torch.zeros_like(Du) if u0 is None else Burgers_FDM(u0)

    forcing = torch.zeros_like(Du)

    if u0 is None:
        return F.mse_loss(Du, forcing)
    else:
        return F.mse_loss(Du, Du0)

def NS_vorticity_loss(w, w_prev, w_next, w0=None):
    bsize, _, nt, nx, ny = w.shape
    device = w.device
    Du = NS_vorticity_FDM(w, w_prev, w_next)
    Du0 = torch.zeros_like(Du) if w0 is None else NS_vorticity_FDM(w0, w_prev, w_next)

    lploss = LpLoss(size_average=True)

    def get_forcing(nx, ny, L=1.0):
        x = torch.linspace(0, L, nx+1, device=device, dtype=torch.float)[:-1].reshape(nx, 1).repeat(1, ny)
        y = torch.linspace(0, L, ny+1, device=device, dtype=torch.float)[:-1].reshape(1, ny).repeat(nx, 1)
        return 0.1 * (torch.sin(2*math.pi*(x+y)) + torch.cos(2*math.pi*(x+y)))
    
    forcing = get_forcing(nx, ny).repeat(bsize, nt, 1, 1)

    if w0 is None:
        return lploss(Du, forcing)
    else:
        return lploss(Du, Du0)

def Darcy_loss(u, a, a0=None):
    Du = Darcy_FDM(u, a)
    Du0 = torch.zeros_like(Du) if a0 is None else Darcy_FDM(u, a0)

    forcing = torch.ones_like(Du)

    lploss = LpLoss(size_average=True)

    if a0 is None:
        return lploss(Du, forcing)
    else:
        return lploss(Du, Du0)

    