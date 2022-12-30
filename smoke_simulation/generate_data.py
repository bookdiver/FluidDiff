import sys

from omegaconf import OmegaConf
from pytorch_lightning.cli import instantiate_class
from phi.torch.flow import TORCH

from navierstokes2dsmoke import generate_trajectories_smoke

def main(cfg):
    if cfg.experiment == "smoke2d":
        pde = instantiate_class(tuple(), cfg.pdeconfig)
        generate_trajectories_smoke(
            pde=pde,
            num_samples=cfg.samples,
            dirname=cfg.dirname,
        )
    else:
        raise NotImplementedError()

def cli():
    # This is worth it to avoid hydra complexity
    if "--help" in sys.argv:
        print("Usage: python generate_data.py base=<config.yaml>")
        sys.exit(0)
    cfg = OmegaConf.from_cli()

    if "base" in cfg:
        basecfg = OmegaConf.load(cfg.base)
        del cfg.base
        cfg = OmegaConf.merge(basecfg, cfg)
        OmegaConf.resolve(cfg)
        main(cfg)
    else:
        raise SystemExit("Base configuration file not specified! Exiting.")

if __name__ == "__main__":
    TORCH.set_default_device('GPU')
    cli()