import torch

import math

from random_field import GaussianRF

from timeit import default_timer

import scipy.io
from tqdm import tqdm


#w0: initial vorticity
#f: forcing term
#visc: viscosity (1/Re)
#T: final time
#delta_t: internal time-step for solve (descrease if blow-up)
#record_steps: number of in-time snapshots to record
def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, sampling_every=10, sub=1):

    #Grid size - must be power of 2
    N = w0.size(-1)

    #Maximum frequency
    k_max = math.floor(N/2.0)

    #Number of steps to final time
    steps = math.ceil(T/delta_t)

    #Initial vorticity to Fourier space
    w_h = torch.fft.rfft2(w0)

    #Forcing to Fourier space
    f_h = torch.fft.rfft2(f)

    #If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    #Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)

    #Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    #Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
    lap[0,0] = 1.0
    #Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max, torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    #Saving solution and time
    sol_w = torch.zeros(w0.size(0), N//sub, N//sub, steps//sampling_every, device=w0.device)

    #Record counter
    c = 0
    pbar = tqdm(range(steps), dynamic_ncols=True)
    for j in pbar:
        #Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        #Velocity field in x-direction = psi_y
        q = 2. * math.pi * k_y * 1j * psi_h
        q = torch.fft.irfft2(q, s=(N, N))

        #Velocity field in y-direction = -psi_x
        v = -2. * math.pi * k_x * 1j * psi_h
        v = torch.fft.irfft2(v, s=(N, N))

        #Partial x of vorticity
        w_x = 2. * math.pi * k_x * 1j * w_h
        w_x = torch.fft.irfft2(w_x, s=(N, N))

        #Partial y of vorticity
        w_y = 2. * math.pi * k_y * 1j * w_h
        w_y = torch.fft.irfft2(w_y, s=(N, N))

        #Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.rfft2(q*w_x + v*w_y)

        #Dealias
        F_h = dealias* F_h

        #Crank-Nicolson update
        w_h = (-delta_t*F_h + delta_t*(f_h+0.1*w_h) + (1.0 - 0.5*delta_t*visc*lap)*w_h)/(1.0 + 0.5*delta_t*visc*lap)

        if (j+1) % sampling_every == 0:
        
            #Solution in physical space
            w = torch.fft.irfft2(w_h, s=(N, N))

            #Record solution and time
            sol_w[...,c] = w[:, ::sub, ::sub]
            c += 1

    return sol_w



device = torch.device('cuda')

#Resolution
s = 256
sub = 4

#Number of solutions to generate
N = 200

#Batch size
bsize = 20

#Set up 2d GRF with covariance parameters
GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

#Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, 1, s+1, device=device)
t = t[0:-1]

X,Y = torch.meshgrid(t, t, indexing='ij')
f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))
# f = -4.0 * torch.cos(4.0 * Y)

#Number of snapshots from solution
record_steps = 20

sampling_every = 10

# Total simulation time
T = 20.0

# Time step
delta_t= 1e-4

# viscosity
visc = 1e-3

total_steps = math.ceil(T/(delta_t*sampling_every))
record_interval = math.ceil(total_steps/record_steps)

#Inputs
a = torch.zeros(N, s//sub, s//sub)
#Solutions
w_now = torch.zeros(N, s//sub, s//sub, record_steps)
w_prev = torch.zeros(N, s//sub, s//sub, record_steps)
w_next = torch.zeros(N, s//sub, s//sub, record_steps)

c = 0
t0 =default_timer()
for j in range(N//bsize):

    #Sample random feilds
    w0 = GRF.sample(bsize)

    #Solve NS
    sol_w = navier_stokes_2d(w0, f, visc, T, delta_t, sampling_every, sub)

    a[c:(c+bsize),...] = w0[:, ::sub, ::sub]
    w_now[c:(c+bsize),...] = sol_w[..., 1::record_interval]
    w_prev[c:(c+bsize),...] = sol_w[..., :-1:record_interval]
    w_next[c:(c+bsize),...] = sol_w[..., 2::record_interval]
    # q[c:(c+bsize),...] = sol_q
    # v[c:(c+bsize),...] = sol_v

    c += bsize
    # Print progress
    print(f'Progress: {j+1} / {N//bsize}')
    print(f'Time elapsed: {default_timer() - t0:.2f} s')
    print('--'*20)

scipy.io.savemat(f'../data/ns_data_T{T:.0f}_v{visc:.0e}_N{N}.mat', \
                 mdict={'a': a.cpu().numpy(), 
                        'w': w_now.cpu().numpy(),
                        'w_prev': w_prev.cpu().numpy(),
                        'w_next': w_next.cpu().numpy()})