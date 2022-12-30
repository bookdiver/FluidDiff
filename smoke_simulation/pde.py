from dataclasses import dataclass


@dataclass
class PDEConfig:
    """Base inheritance for configuration of PDEs."""

    pass

@dataclass
class NavierStokes2D(PDEConfig):
    tmin: float = 0
    tmax: float = 20.0
    Lx: float = 32.0
    Ly: float = 32.0
    nt: int = 500
    nx: int = 128
    ny: int = 128
    skip_nt: int = 0
    sample_rate: int = 5
    nu: float = 0.03
    buoyancy_x: float = 0.0
    buoyancy_y: float = 0.05
    source_ycoord: float = 5.0
    source_strength: float = 1.0
    source_radius: float = 2.0

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