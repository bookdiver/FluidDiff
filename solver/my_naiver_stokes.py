import torch
import math
import scipy

from FluidDiff.solver.random_field import GaussianRF
from timeit import default_timer
import argparse
from tqdm import tqdm


class NaiverStokesFlow2d(object):

    def __init__(self, s, L=1.0, device=None):
        self.s = s
        self.L = L

        k_max = math.floor(s/2.0)
        # Wavenumbers in x and y directions
        freq_list = torch.cat((torch.arange(start=0, end=k_max, step=1), \
                               torch.arange(start=-k_max, end=0, step=1)), 0)
        self.ky = freq_list.repeat(s, 1).to(device)
        self.kx = self.ky.transpose(0, 1)

        self.kx = self.kx[..., :k_max+1]
        self.ky = self.ky[..., :k_max+1]

        # Negative Laplacian in Fourier space
        self.lap = 4*(math.pi**2)/(L**2) * (self.kx**2 + self.ky**2)
        self.lap[0, 0] = 1.0

        # Dealiasing mask using 2/3 rule
        self.dealias = torch.unsqueeze(torch.logical_and(torch.abs(self.kx) <= (2.0/3.0)*k_max, \
                                                         torch.abs(self.ky) <= (2.0/3.0)*k_max).float(), 0).to(device)
        
        self.device = device

    # Compute stream function from vorticity (Fourier space)
    def stream_function(self, w_h, real_space=False):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / self.lap

        if real_space:
            return torch.fft.irfft2(psi_h, s=(self.s, self.s))
        else:
            return psi_h

    # Compute velocity field from stream function (Fourier space)
    def velocity_field(self, psi_h, real_space=True):
        q_h = 2.0 * math.pi * self.ky * 1j * psi_h / self.L
        v_h = -2.0 * math.pi * self.kx * 1j * psi_h / self.L

        if real_space:
            q = torch.fft.irfft2(q_h, s=(self.s, self.s))
            v = torch.fft.irfft2(v_h, s=(self.s, self.s))
            return q, v
        else:
            return q_h, v_h

    # Compute non-linear term from given vorticity (Fourier space)
    def nonlinear_term(self, w_h):
        w_x_h = 2 * math.pi * self.kx * w_h / self.L
        w_y_h = 2 * math.pi * self.ky * w_h / self.L

        w_x = torch.fft.irfft2(w_x_h, s=(self.s, self.s))
        w_y = torch.fft.irfft2(w_y_h, s=(self.s, self.s))

        # Velocity field in physical space
        q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)

        nonlin = torch.fft.rfft2(q*w_x + v*w_y)

        nonlin = self.dealias * nonlin

        return nonlin

    def advance(self, w, f=None, visc=1e-4, T=1.0, delta_t=1e-4, sub=1, record_steps=100):

        w_h = torch.fft.rfft2(w)

        if f is not None:
            f_h = torch.fft.rfft2(f)
            if len(f_h.size()) < len(w_h.size()):
                f_h = torch.unsqueeze(f_h, 0)
        else:  
            f_h = 0.0

        # Total Steps
        steps = math.ceil(T / delta_t)
        record_time = math.floor(steps / record_steps)

        # Record arrays
        sol_w = torch.zeros((w.size(0), self.s//sub, self.s//sub, record_steps), device=self.device)
        sol_q = torch.zeros((w.size(0), self.s//sub, self.s//sub, record_steps), device=self.device)
        sol_v = torch.zeros((w.size(0), self.s//sub, self.s//sub, record_steps), device=self.device)
        record_counter = 0

        # Advance solution in Fourier space
        for i in tqdm(range(steps)):

            # Non-linear term
            nonlin = self.nonlinear_term(w_h)

            w_h = (-delta_t*nonlin + delta_t*f_h + (1.0-0.5*delta_t*visc*self.lap)*w_h) / (1.0 + 0.5*visc*self.lap*delta_t)

            # Record solution
            if (i + 1) % record_time == 0:
                w = torch.fft.irfft2(w_h, s=(self.s, self.s))
                q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
                sol_w[..., record_counter] = w[:, ::sub, ::sub].clone()
                sol_q[..., record_counter] = q[:, ::sub, ::sub].clone()
                sol_v[..., record_counter] = v[:, ::sub, ::sub].clone()
                record_counter += 1
        
        return sol_w, sol_q, sol_v


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--visc", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=0.1)
    opt = parser.parse_args()

    device = torch.device('cuda:1')
    s = 256
    L = 1.0
    sub = 1
    T = 10.0
    dt = 1e-4
    eta = opt.eta
    visc = opt.visc

    # Number of instances to generate
    N = 20

    # Set up the Gaussian random field as the initial condition
    # GRF = GaussianRF(s=s, L=L, alpha=2.5, tau=7, device=device)
    GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)
    NS = NaiverStokesFlow2d(s=s, L=L, device=device)

    t = torch.linspace(0, L, s+1, device=device)[0:-1]
    X, Y = torch.meshgrid(t, t, indexing='ij')
    f = eta * (torch.sin(2*math.pi*(X+Y)) + torch.cos(2*math.pi*(X+Y)))
    # f = -eta * torch.cos(eta * Y)

    # Number of snapshots to be recorded
    record_time_steps = 50
    
    # Inputs
    a = torch.zeros(N, s//sub, s//sub, device=device)
    # Solutions
    w = torch.zeros(N, s//sub, s//sub, record_time_steps, device=device)
    q = torch.zeros(N, s//sub, s//sub, record_time_steps, device=device)
    v = torch.zeros(N, s//sub, s//sub, record_time_steps, device=device)

    # Batch size
    bsize = 20

    # Instances counter
    c = 0
    # Start time
    t0 = default_timer()

    for j in range(N // bsize):
        # Generate a batch of initial conditions
        w0 = GRF.sample(bsize)
        a[c:c+bsize, ...] = w0[:, ::sub, ::sub].cpu()

        # Advance the solution
        sol_w, sol_q, sol_v = NS.advance(w0, f, visc, T, delta_t=1e-4, sub=sub, record_steps=record_time_steps)
        w[c:c+bsize, ...] = sol_w
        q[c:c+bsize, ...] = sol_q
        v[c:c+bsize, ...] = sol_v
        c += bsize

        # Print progress
        print(f'Progress: {j+1} / {N//bsize}')
        print(f'Time elapsed: {default_timer() - t0:.2f} s')
        print('--'*20)

    # Save the data
    scipy.io.savemat(f'../data/ns_data_v{visc:.0e}.mat',\
                    mdict={'T': T,
                           'visc': visc,
                           'force': f'{eta}*sin(2*pi*(x+y)) + cos(2*pi*(x+y))',
                           'delta_t': dt,
                           'number_of_instances': N,
                           'resolution': f'{record_time_steps} x {s//sub} x {s//sub}',
                           'original_sptial_resolution': f'{s} x {s}',
                           'a': a.cpu().numpy(), 
                           'w': w.cpu().numpy(),
                           'q': q.cpu().numpy(),
                           'v': v.cpu().numpy()})