import glob
import os
import sys

from omegaconf import OmegaConf
from pytorch_lightning.cli import instantiate_class

from nspointsmoke2d import generate_trajectories_pointsmoke
from nsfullsmoke2d import generate_trajectories_fullsmoke

def main(cfg):
    os.makedirs(cfg.dirname, exist_ok=True)
    exiting_files = glob.glob(os.path.join(cfg.dirname, f"*{cfg.mode}.h5"))
    if cfg.overwrite:
        for f in exiting_files:
            os.remove(f)
    else:
        if len(exiting_files) > 0:
            raise SystemExit("Data files already exist. Exiting.")

    if cfg.experiment == "pointsmoke":
        if cfg.mode == 'train':
            pde = instantiate_class(tuple(), cfg.pdeconfig.train)
            generate_trajectories_pointsmoke(
                pde=pde,
                mode=cfg.mode,
                dirname=cfg.dirname,
            )
        elif cfg.mode == 'test':
            pde = instantiate_class(tuple(), cfg.pdeconfig.test)
            generate_trajectories_pointsmoke(
                pde=pde,
                mode=cfg.mode,
                dirname=cfg.dirname,
            )
        else:
            raise NotImplementedError()
    elif cfg.experiment == "fullsmoke":
        if cfg.mode == 'train':
            pde = instantiate_class(tuple(), cfg.pdeconfig.train)
            generate_trajectories_fullsmoke(
                pde=pde,
                mode=cfg.mode,
                dirname=cfg.dirname,
            )
        elif cfg.mode == 'test':
            pde = instantiate_class(tuple(), cfg.pdeconfig.test)
            generate_trajectories_fullsmoke(
                pde=pde,
                mode=cfg.mode,
                dirname=cfg.dirname,
            )
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

def cli():
    # This is worth it to avoid hydra complexity
    if "--help" in sys.argv:
        print("Usage: python generate_data.py base=<config.yaml> experiment=<smoke2d> mode=<train or test> dirname=<directory to save data>")
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
    cli()