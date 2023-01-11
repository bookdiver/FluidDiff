from dataclasses import dataclass


@dataclass
class PDEConfig:
    """Base inheritance for configuration of PDEs."""

    pass

@dataclass
class NavierStokes2D(PDEConfig):
    tmin: float
    tmax: float
    Lx: float 
    Ly: float 
    nt: int 
    nx: int
    ny: int 
    skip_nt: int 
    sample_rate: int 
    nu: float 
    buoyancy_x: float 
    buoyancy_y: float 
    source_coord: list
    source_strength: float 
    source_radius: float 
    samples: int

    def __repr__(self):
        return "NavierStokes2D"

    @property
    def trajlen(self):
        return int(self.nt / self.sample_rate)

    @property
    def grid_size(self):
        return (self.trajlen, self.nx, self.ny)

    @property
    def dt(self):
        return (self.tmax - self.tmin) / (self.nt)

    @property
    def dx(self):
        return self.Lx / (self.nx - 1)

    @property
    def dy(self):
        return self.Ly / (self.ny - 1)