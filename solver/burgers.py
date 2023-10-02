import torch
from scipy.io import savemat
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

from random_field import GaussianRF, GRF_Mattern


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
        self.dt = dt
        self.T = T
        self.t = 0
        self.i_t = 0
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
        data_dx = self.CD_i(data, axis=-1, dx=self.dx)
        return data_dx

    def Dxx(self, data):
        data_dxx = self.CD_ii(data, axis=-1, dx=self.dx)
        return data_dxx

    def dudt(self, u):
        u_xx = self.Dxx(u)
        u2 = u**2.0
        u2_x = self.Dx(u2)
        u_RHS = -0.5*u2_x + self.nu*u_xx
        return u_RHS
        
    # def update_u(self, u, prev_k, step_frac):
    #     u_new = u + step_frac*prev_k
    #     return u_new

    # def burgers_rk4(self, u):
    #     k1 = self.dt * self.dudt(u)
        
    #     u1 = self.update_u(u, k1, step_frac=0.5)
    #     k2 = self.dt * self.dudt(u1)

    #     u2 = self.update_u(u, k2, step_frac=0.5)
    #     k3 = self.dt * self.dudt(u2)

    #     u3 = self.update_u(u, k3, step_frac=1.0)        
    #     k4 = self.dt * self.dudt(u3)
        
    #     du = (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
    #     u_next = u + du
        
    #     return u_next, du
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
        u = u0.clone()
        self.i_t = 0
        U = []
        
        if save_interval != 0 and self.i_t % save_interval == 0:
            U.append(u)
            
        # Compute equations
        while self.t < self.T:
#             print(f"t:\t{self.t}")
            # u, du = self.burgers_rk4(u)
            u = self.burgers_rk4(u)
            self.t += self.dt
            
            self.i_t += 1
            if save_interval != 0 and self.i_t % save_interval == 0:
                U.append(u)

        self.t = 0

        return torch.stack(U).permute(1, 0, 2)

if __name__ == '__main__':
    device = torch.device('cuda:0')

    #Number of solutions to generate
    N = 1

    #Batch size
    bsize = N

    Nx = 4096
    T = 1.0
    dt = 1/10000
    dt_save = 1/100
    save_interval = int(dt_save/dt)
    visc = 0.01

    GRF = GaussianRF(dim=1, size=Nx, alpha=2, tau=1, sigma=1, device=device)
    # GRF = GRF_Mattern(1, Nx, length=1.0, nu=None, l=0.1, sigma=1, boundary="periodic", device=device)

    burgers = BurgersEq1D(Nx=Nx, T=T, dt=dt, nu=visc, device=device)

    a = torch.zeros(N, Nx, device=device)
    u = torch.zeros(N, int(T/dt_save), Nx, device=device)

    c = 0

    for j in range(N//bsize):
        u0 = GRF.sample(bsize)
        a[c:c+bsize, :] = u0

        # U = torch.vmap(burgers.burgers_driver, in_dims=(0, None))(u0, save_interval)
        U = burgers.burgers_driver(u0, save_interval)
        u[c:c+bsize, :, :] = U[:, 1:, :]

        c += bsize

        print(f'Progress: {j+1} / {N//bsize}')
        print('--'*20)

    # savemat(f'../data/burgers_data_v{visc:.0e}_N{N}.mat', \
    #                 mdict={'a': a.cpu().numpy(),
    #                        'u': u.cpu().numpy()})
    print(a.shape)
    print(u.shape)
    plt.figure()
    plt.subplot(121)
    plt.plot(a.squeeze().cpu())

    plt.subplot(122)
    plt.imshow(u.squeeze()[:, ::32].cpu(), cmap='jet')
    plt.colorbar()

    plt.savefig('./burgers_plot.png')

