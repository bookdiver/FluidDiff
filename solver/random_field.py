import torch
from math import pi, gamma, sqrt

from timeit import default_timer


class GaussianRF:
    """ Generate a Gaussain random field with N(0, sigma * (-delta + tau**2) ** (-alpha) )
    """

    def __init__(self, dim, size, length=1.0, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*sqrt(2.0)*sigma*((4*(pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*sqrt(2.0)*sigma*((4*(pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*sqrt(2.0)*sigma*((4*(pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff

        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real
    


class GRF_Mattern(object):
    def __init__(self, dim, size, length=1.0, nu=None, l=0.1, sigma=1.0, boundary="periodic", constant_eig=None, device=None):

        self.dim = dim
        self.device = device
        self.bc = boundary

        a = sqrt(2/length)
        if self.bc == "dirichlet":
            constant_eig = None
        
        
        if nu is not None:
            kappa = sqrt(2*nu)/l
            alpha = nu + 0.5*dim
            self.eta2 = size**dim * sigma*(4.0*pi)**(0.5*dim)*gamma(alpha)/(kappa**dim * gamma(nu))
        else:
            self.eta2 = size**dim * sigma*(sqrt(2.0*pi)*l)**dim
        # if sigma is None:
        #     sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2
        if self.bc == "periodic":
            const = (4.0*(pi**2))/(length**2)
        else:
            const = (pi**2)/(length**2)

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            k2 = k**2
            if nu is not None:
                # self.sqrt_eig = (size**dim)*sqrt(2.0)*sigma*((const*(k**2) + tau**2)**(-alpha/2.0))
                eigs = 1.0 + (const/(kappa*length)**2*k2)
                self.sqrt_eig = self.eta2/(length**dim) * eigs**(-alpha/2.0)
            else:
                self.sqrt_eig = self.eta2/(length**dim)*torch.exp(-(l)**2*const*k2/4.0)

            if constant_eig is not None:
                self.sqrt_eig[0] = constant_eig # (size**dim)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            k2 = k_x**2 + k_y**2 
            if nu is not None:
                # self.sqrt_eig = (size**dim)*sqrt(2.0)*sigma*((const*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
                eigs = 1.0 + (const/(kappa*length)**2*k2)
                self.sqrt_eig = self.eta2/(length**dim) * eigs**(-alpha/2.0)
            else:
                self.sqrt_eig = self.eta2/(length**dim)*torch.exp(-(l)**2*const*k2/4.0)

            if constant_eig is not None:
                self.sqrt_eig[0,0] = constant_eig # (size**dim)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            k2 = k_x**2 + k_y**2 + k_z**2
            if nu is not None:
                # self.sqrt_eig = (size**dim)*sqrt(2.0)*sigma*((const*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
                eigs = 1.0 + (const/(kappa*length)**2*k2)
                self.sqrt_eig = self.eta2/(length**dim) * eigs**(-alpha/2.0)
            else:
                self.sqrt_eig = self.eta2/(length**dim)*torch.exp(-(l)**2*const*k2/4.0)

            if constant_eig is not None:
                self.sqrt_eig[0,0,0] = constant_eig # (size**dim)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        if self.bc == 'dirichlet':
            coeff.real[:] = 0
        if self.bc == 'neumann':
            coeff.imag[:] = 0
        coeff = self.sqrt_eig*coeff

        u = torch.fft.irfftn(coeff, self.size, norm="backward")
        return u