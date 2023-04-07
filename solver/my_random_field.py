import torch
import math

class GaussianRF(object):
    def __init__(self, s, L=1.0, alpha=2.0, tau=3.0, sigma=None, mean=None, boundary="periodic", device=None):

        self.s = s

        self.mean = mean

        self.device = device

        if sigma is None:
            self.sigma = tau**(0.5*(2*alpha - 2.0))
        else:
            self.sigma = sigma

        k_max = s//2
        freq_list = torch.cat((torch.arange(start=0, end=k_max, step=1),\
                               torch.arange(start=-k_max, end=0, step=1)), 0)
        k2 = freq_list.repeat(s, 1).to(device)
        k1 = k2.transpose(0,1).to(device)

        self.sqrt_eig = s**2 * math.sqrt(2.0) * self.sigma * ((4*(math.pi**2)/(L**2) * (k1**2+k2**2) + tau**2)**(-alpha/2.0))
        self.sqrt_eig[0, 0] = 0.0

    def sample(self, N):

        coeff = torch.randn(N, self.s, self.s, dtype=torch.cfloat, device=self.device)
        coeff = coeff * self.sqrt_eig

        u = torch.fft.ifftn(coeff, dim=list(range(-1, -3, -1))).real

        return u