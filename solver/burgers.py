import torch
import math

from random_field import GaussianRF

import scipy.io
from tqdm import tqdm

from timeit import default_timer

class BurgersEq1D():
    def __init__(self,
                 xmin=0,
                 xmax=1,
                 Nx=128,
                 nu=0.01,
                 dt=1e-4,
                 T=1.0,
                 device=None,
                 dtype=torch.float64,
                 ):
        self.xmin = xmin
        self.xmax = xmax
        self.Nx = Nx
        x = torch.linspace(xmin, xmax, Nx + 1, device=device, dtype=dtype)[:-1]
        self.x = x
        self.dx = x[1] - x[0]
        self.nu = nu
        self.u = torch.zeros_like(x, device=device)
        self.u0 = torch.zeros_like(self.u, device=device)
        self.dt = dt
        self.T = T
        self.t = 0
        self.i_t = 0
        self.U = []
        self.device = device
        
    # All Central Differencing Functions are 4th order.  These are used to compute ann inputs.
    def CD_i(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_i = (data_m2 - 8.0*data_m1 + 8.0*data_p1 - data_p2)/(12.0*dx)
        return data_diff_i

    def CD_ii(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_ii = (-data_m2 + 16.0*data_m1 - 30.0*data + 16.0*data_p1 -data_p2)/(12.0*dx**2)
        return data_diff_ii

    def Dx(self, data):
        data_dx = self.CD_i(data=data, axis=0, dx=self.dx)
        return data_dx

    def Dxx(self, data):
        data_dxx = self.CD_ii(data, axis=0, dx=self.dx)
        return data_dxx

    def burgers_calc_RHS(self, u):
        u_xx = self.Dxx(u)
        u2 = u**2.0
        u2_x = self.Dx(u2)
        u_RHS = -0.5*u2_x + self.nu*u_xx
        return u_RHS
        
    def update_field(self, field, RHS, step_frac):
        field_new = field + self.dt*step_frac*RHS
        return field_new

    def rk4_merge_RHS(self, field, RHS1, RHS2, RHS3, RHS4):
        field_new = field + self.dt/6.0*(RHS1 + 2*RHS2 + 2.0*RHS3 + RHS4)
        return field_new

    def burgers_rk4(self, u):
        u_RHS1 = self.burgers_calc_RHS(u)
        u1 = self.update_field(u, u_RHS1, step_frac=0.5)
        
        u_RHS2 = self.burgers_calc_RHS(u1)
        u2 = self.update_field(u, u_RHS2, step_frac=0.5)
        
        u_RHS3 = self.burgers_calc_RHS(u2)
        u3 = self.update_field(u, u_RHS3, step_frac=1.0)
        
        u_RHS4 = self.burgers_calc_RHS(u3)
        
        u_new = self.rk4_merge_RHS(u, u_RHS1, u_RHS2, u_RHS3, u_RHS4)
        
        return u_new

    def burgers_driver(self, u0, save_interval=10):
        self.u0 = u0[:self.Nx]
        self.u = self.u0
        self.i_t = 0
        self.U = []
        
        if save_interval != 0 and self.i_t % save_interval == 0:
            self.U.append(self.u)
            
        # Compute equations
        while self.t < self.T:
#             print(f"t:\t{self.t}")
            self.u = self.burgers_rk4(self.u)
            self.t += self.dt
            
            self.i_t += 1
            if save_interval != 0 and self.i_t % save_interval == 0:
                self.U.append(self.u)

        return torch.stack(self.U)

if __name__ == '__main__':
    device = torch.device('cuda:0')

    #Number of solutions to generate
    N = 200

    #Batch size
    bsize = 100

    Nx = 128
    T = 1.0
    dt = 1e-4
    dt_save = 1e-2
    save_interval = int(dt_save/dt)
    visc = 1e-2

    GRF = GaussianRF(dim=1, size=128, alpha=2, tau=5, sigma=1, device=device)

    burgers = BurgersEq1D(Nx=Nx, T=T, dt=dt, nu=visc, device=device)

    a = torch.zeros(N, Nx, device=device)
    u = torch.zeros(N, int(T/dt_save)+1, Nx, device=device)

    c = 0

    t0 = default_timer()
    for j in range(N//bsize):
        u0 = GRF.sample(N=bsize)
        a[c:c+bsize, :] = u0

        U = torch.vmap(burgers.burgers_driver, in_dims=(0, None))(u0, save_interval)
        u[c:c+bsize, :, :] = U

        c += bsize

        print(f'Progress: {j+1} / {N//bsize}')
        print(f'Elapsed time: {default_timer()-t0:.2f} s')
        print('--'*20)

    scipy.io.savemat(f'../data/burgers_data_v{visc:.0e}_N{N}.mat', \
                    mdict={'a': a.cpu().numpy(),
                           'u': u.cpu().numpy()})

