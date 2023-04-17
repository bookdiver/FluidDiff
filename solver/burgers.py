import torch
import math

from random_field import GaussianRF

import scipy.io
from tqdm import tqdm

from timeit import default_timer

def central_diff_x(u, delta_x):
    u_m2 = torch.roll(u, shifts=2, dims=-1)
    u_m1 = torch.roll(u, shifts=1, dims=-1)
    u_p1 = torch.roll(u, shifts=-1, dims=-1)
    u_p2 = torch.roll(u, shifts=-2, dims=-1)
    u_x = (u_m2 - 8. * u_m1 + 8. * u_p1 - u_p2) / (12. * delta_x)
    return u_x

def central_diff_xx(u, delta_x):
    u_m2 = torch.roll(u, shifts=2, dims=-1)
    u_m1 = torch.roll(u, shifts=1, dims=-1)
    u_p1 = torch.roll(u, shifts=-1, dims=-1)
    u_p2 = torch.roll(u, shifts=-2, dims=-1)
    u_xx = (-u_m2 + 16. * u_m1 - 30. * u + 16. * u_p1 - u_p2) / (12. * delta_x**2)
    return u_xx

def burgers_rhs(u, delta_x, visc):
    u_xx = central_diff_xx(u, delta_x)
    u2_x = central_diff_x(u**2, delta_x)
    rhs = - 0.5 * u2_x + visc * u_xx
    return rhs

def step(u, rhs, delta_t, step_frac):
    u_next = u + delta_t * step_frac * rhs
    return u_next

def rk4_merge_rhs(u, rhs1, rhs2, rhs3, rhs4, delta_t):
    u_next = u + delta_t * (rhs1 + 2. * rhs2 + 2. * rhs3 + rhs4) / 6.
    return u_next

def burges_1d(u0, visc, T, delta_t=1e-4, sampling_every=10, sub=1):
    
    N = u0.size(-1)
    u = u0.clone()

    steps = math.ceil(T/delta_t)

    sol_u = torch.zeros(u0.size(0), N//sub, steps//sampling_every, device=u0.device)


    c = 0
    pbar = tqdm(range(steps), dynamic_ncols=True)
    for i in pbar:
        u_rhs1 = burgers_rhs(u, 1./N, visc)
        u1 = step(u, u_rhs1, delta_t, 0.5)

        u_rhs2 = burgers_rhs(u1, 1./N, visc)
        u2 = step(u, u_rhs2, delta_t, 0.5)

        u_rhs3 = burgers_rhs(u2, 1./N, visc)
        u3 = step(u, u_rhs3, delta_t, 1.)

        u_rhs4 = burgers_rhs(u3, 1./N, visc)
        u = rk4_merge_rhs(u, u_rhs1, u_rhs2, u_rhs3, u_rhs4, delta_t)

        if (i+1) % sampling_every == 0:
            sol_u[:, :, c] = u[:, ::sub]
            c += 1

    return sol_u

if __name__ == '__main__':
    device = torch.device('cuda:0')

    #Resolution
    s = 1024
    sub = 8

    #Number of solutions to generate
    N = 200

    #Batch size
    bsize = 100

    #Set up random field
    GRF = GaussianRF(1, s, alpha=2.5, tau=7, device=device)

    # Number of snapshots from each solution
    record_steps = 100

    sampling_every = 10

    #Total Simluation time
    T = 1.0
    delta_t = 1e-4

    visc = 1e-4

    total_steps = math.ceil(T/(delta_t*sampling_every))
    record_interval = math.ceil(total_steps/record_steps)

    #Inputs
    a = torch.zeros(N, s//sub, device=device)
    #Solutions
    u_now = torch.zeros(N, s//sub, record_steps, device=device)
    u_prev = torch.zeros(N, s//sub, record_steps, device=device)
    u_next = torch.zeros(N, s//sub, record_steps, device=device)

    c = 0
    t0 = default_timer()
    for j in range(N//bsize):
        u0 = GRF.sample(N=bsize)

        sol_w = burges_1d(u0, visc, T, delta_t=delta_t, sampling_every=sampling_every, sub=sub)

        a[c:c+bsize, :] = u0[:, ::sub]
        u_now[c:c+bsize, :, :] = sol_w[:, :, 1::record_interval]
        u_prev[c:c+bsize, :, :] = sol_w[:, :, :-1:record_interval]
        u_next[c:c+bsize, :, :] = sol_w[:, :, 2::record_interval]

        c += bsize

        print(f'Progress: {j+1} / {N//bsize}')
        print(f'Elapsed time: {default_timer()-t0:.2f} s')
        print('--'*20)
    
    scipy.io.savemat(f'../data/burgers_data_v{visc:.0e}_N{N}.mat', \
                     mdict={'a': a.cpu().numpy(),
                            'u': u_now.cpu().numpy(),
                            'u_prev': u_prev.cpu().numpy(),
                            'u_next': u_next.cpu().numpy()})

