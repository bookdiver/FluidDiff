import os
import argparse
import math
from tqdm import tqdm 

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, lr_scheduler

from utils import SmokePlumeDataset
from diffuser import GaussianDiffusion
from unet3d import Unet3D
from FluidDiff.models.vae.autoencoder import Autoencoder

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

parse = argparse.ArgumentParser(description='Latent Denoising Diffusion Training')

parse.add_argument('--device', type=int, default=0, help='device to use (default 0)')
parse.add_argument('--debug', action='store_true', help='debug mode, default False')
parse.add_argument('--experiment-name', type=str, default='default', help='name of the experiment, default "default"')

parse.add_argument('--from-checkpoint', type=str, default=None, help='path to checkpoint to load from, default None')
parse.add_argument('--epochs', type=int, default=100, help='number of epochs to train, default 100')
parse.add_argument('--batch-size', type=int, default=1, help='batch size, default 1')
parse.add_argument('--lr', type=float, default=1e-4, help='learning rate, default 1e-4')
parse.add_argument('--lrf', type=float, default=0.1, help='the final learning rate factor compared with the original one, default 0.1')

parse.add_argument('--train_physics_variables', type=str, nargs='+', default=['density'], help='physics variables to train on, default density')

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        path: str):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, path)

def save_config(
        args: argparse.Namespace, 
        path: str):
    argsDict = args.__dict__
    with open(path+'/config.txt', 'w') as f:
        f.writelines('---------------------- Config ----------------------' + '\n')
        for key, value in argsDict.items():
            f.writelines(key + ' : ' + str(value) + '\n')
        f.writelines('----------------------------------------------------' + '\n')

def set_random_seed(
        seed: int,
        deterministic: bool = False,
        benchmark: bool = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True

class Trainer:
    def __init__(
            self,
            *,
            args: argparse.Namespace):
        
        self.device = torch.device('cuda', args.device)
        self.experiment_name = args.experiment_name

        os.makedirs(f'./checkpoints/smoke_plume64x64/{self.experiment_name}', exist_ok=True)
        save_config(args, f'./checkpoints/smoke_plume64x64/{self.experiment_name}')
        self.tb_writer = SummaryWriter(log_dir=f'./logs/smoke_plume64x64/{self.experiment_name}') if not args.debug else None

        dataset = SmokePlumeDataset(fileroot='../../data/smoke_plume_64x64',
                                physics_variables=args.train_physics_variables,
                                read_every_frames=5,
                                normalize_type='pm1')
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        self.encoder = Autoencoder(
            in_channels = len(args.train_physics_variables),
            z_channels=1,
            use_variational=True,
            activation_type='relu'
        )
        self.encoder.load_state_dict(
            torch.load(f'../vae/checkpoint/smoke_plume64x64/VAE_2D{args.train_physics_variables[0]}/model_checkpoint.pt')['model_state_dict']
        )
        print('Load pretrained encoder successfully!')
