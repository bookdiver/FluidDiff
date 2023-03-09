import os
import argparse
import math
from tqdm import tqdm 

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, lr_scheduler
from einops import rearrange

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

        self.autoencoder = Autoencoder(
            in_channels = len(args.train_physics_variables),
            z_channels=1,
            use_variational=True,
            activation_type='relu'
        )
        self.autoencoder.load_state_dict(
            torch.load(f'../vae/checkpoint/smoke_plume64x64/VAE_2D{args.train_physics_variables[0]}/model_checkpoint.pt')['model_state_dict']
        )
        self.autoencoder.to(self.device)
        print('Load pretrained encoder successfully!')

        self.eps_model = Unet3D(
            dim=32,
            cond_dim=256,
            dim_mults=(1, 2, 4, 4)
        )

        self.diffuser = GaussianDiffusion(
            eps_model=self.eps_model,
            domain_size=(16, 16),
            n_frames=40,
            n_channels=1,
            n_diffusion_steps=1000
        )
        self.diffuser.to(self.device)

        self.optimizer = Adam(
            self.diffuser.eps_model.parameters(),
            lr=args.lr)
        lf = lambda x: ((1 + math.cos(x*math.pi/args.epochs)) / 2) * (1 - args.lrf) + args.lrf
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        
        print("Trainer initialized")

    def train(self, args: argparse.Namespace):
        for epoch in range(args.epochs):
            self.diffuser.eps_model.train()
            pbar_train = tqdm(self.train_loader)
            for i, data in enumerate(pbar_train):
                x = data['x'].to(self.device)
                with torch.no_grad():
                    b, c, t, h, w = x.shape
                    x = rearrange(x, 'b c t h w -> (b t) c h w')
                    x = self.autoencoder.encode(x).sample()
                    x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
                y = data['y'].to(self.device)
                loss = self.diffuser(x, cond=y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar_train.set_description(f'Epoch {epoch}/{args.epochs}, Loss: {loss.item():.4f}')
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar('train/loss', loss.item(), epoch*len(self.train_loader)+i)
            self.scheduler.step()

            self.diffuser.eps_model.eval()
            pbar_test = tqdm(self.test_loader)
            with torch.no_grad():
                for i, data in enumerate(pbar_test):
                    x = data['x'].to(self.device)
                    b, c, t, h, w = x.shape
                    x = rearrange(x, 'b c t h w -> (b t) c h w')
                    x = self.autoencoder.encode(x).sample()
                    x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
                    y = data['y'].to(self.device)
                    loss = self.diffuser(x, cond=y)
                    pbar_test.set_description(f'Epoch {epoch}/{args.epochs}, Loss: {loss.item():.4f}')
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar('test/loss', loss.item(), epoch*len(self.test_loader)+i)
            
            if not args.debug:
                save_checkpoint(
                    model = self.diffuser.eps_model.state_dict(),
                    optimizer = self.optimizer.state_dict(),
                    scheduler = self.scheduler.state_dict(),
                    epoch = epoch,
                    path=f'./checkpoints/smoke_plume64x64/{self.experiment_name}/model_checkpoint.pt'
                )
            
        if self.tb_writer is not None:
            self.tb_writer.close()
        print('Training finished!')

if __name__ == '__main__':
    args = parse.parse_args()
    trainer = Trainer(args=args)
    trainer.train(args)


