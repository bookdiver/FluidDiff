from dataclasses import dataclass

@dataclass
class PDEConfig:
    """Base inheritance for configuration of PDEs."""
    tmax: float # max simulation time
    skip_t: float # skip the first skip_t seconds of the simulation when recording
    Lx: float # physical domain size in x
    Ly: float # physical domain size in y
    nt: int # number of time steps
    nx: int # number of grid points in x
    ny: int # number of grid points in y
    t_sample_rate: int # sample every t_sample_rate time steps

    def __repr__(self):
        return "PDEConfig"
    
    @property
    def dt(self):
        return self.tmax / self.nt
    
    @property
    def skip_nt(self):
        return int(self.skip_t / self.dt)

    @property
    def trajlen(self):
        return int((self.nt - self.skip_nt) / self.t_sample_rate)
    
    @property
    def grid_size(self):
        return {'t': self.trajlen, 'x': self.nx, 'y': self.ny}
    


@dataclass
class NSPointSmoke2D(PDEConfig):
    nu: float 
    buoyancy: float 
    source_coord: list
    source_strength: float 
    source_radius: float 

    def __repr__(self):
        return "2DNSPointSmoke"
    
    @property
    def n_samples(self):
        return len(self.source_coord)

@dataclass
class NSFullSmoke2D(PDEConfig):
    nu: float
    buoyancy: float
    n_samples: int

    def __repr__(self):
        return "2DNSFullSmoke"

