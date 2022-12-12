import argparse
import logging
import math
from typing import Union

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from utils import FluidDataSet
from unet import UNet, UNetXAttn
from denoising_diffusion import DenoisingDiffusion

matplotlib.use('Agg')

parse = argparse.ArgumentParser(description='Denoising Diffusion Training')

parse.add_argument('--exp-name', type=str, help="the name of the experiment, with the format of '(dataset)_(model)'")

parse.add_argument('--epochs', type=int, default=20, help='number of epochs, default: 20')
parse.add_argument('--batch-size', type=int, default=16, help='batch size, default: 16')

parse.add_argument('--device', type=int, default=0, help='device to use (default 0)')
parse.add_argument('--debug', action='store_true', help='debug mode, default False')

class Configs:
    eps_model: Union[UNet, UNetXAttn]
    diffuser: DenoisingDiffusion
    in_channels: int=1
    out_channels: int=1
    channels: int=32
    channel_multpliers: list=[1, 2, 4, 8]
    n_res_blocks: int=2
    attention_levels: list=[0, 1, 2]
    n_heads: int=4
    transformer_layers: int=1
    n_steps: int=400
    lr: float=2e-4
    lrf: float=0.1
    dataset: FluidDataSet
    data_loader: DataLoader
    optimizer: torch.optim.Adam
    eval_interval: int=10
    tb_writer: SummaryWriter
    data_root: str='/media/bamf-big/gefan/DiffFluids/data/smoke/'
    tb_writer_root: str='/media/bamf-big/gefan/DiffFluids/diff_fluids/ddpm/logs/'
    model_save_root: str='/media/bamf-big/gefan/DiffFluids/diff_fluids/ddpm/checkpoint/'
    def __init__(self, args):
        if args.debug:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(asctime)s:%(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')
            self.tb_writer = SummaryWriter(log_dir=self.tb_writer_root + args.exp_name +'/')
        
        self.args = args

        if 'xunet' in args.exp_name:
            self.eps_model = UNetXAttn(
                in_channels = self.in_channels,
                out_channels = self.out_channels,
                channels = self.channels,
                channel_multpliers = self.channel_multpliers,
                n_res_blocks = self.n_res_blocks,
                attention_levels = self.attention_levels,
                n_heads = self.n_heads,
                transformer_layers = self.transformer_layers,
                cond_channels = 3
            ).cuda(args.device)
        elif 'unet' in args.exp_name:
            self.eps_model = UNet(
                in_channels = self.in_channels,
                out_channels = self.out_channels,
                channels = self.channels,
                channel_multpliers = self.channel_multpliers,
                n_res_blocks = self.n_res_blocks,
                attention_levels = self.attention_levels,
                n_heads = self.n_heads,
                cond_channels = 3
            ).cuda(args.device)
        else:
            raise NotImplementedError('Only UNet and UNetXAttn are supported now.')

        self.diffuser = DenoisingDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device = args.device
        ).cuda(args.device)

        if 'smoke_small' in args.exp_name:
            dataset_name = 'smoke_small'
            self.init_seed = torch.randn((4, 1, 64, 64))
            # inital condition for testing [t,    x,    y]
            self.init_cond = torch.tensor([[10.0, 23.0, 6.0],
                                           [20.0, 23.0, 6.0],
                                           [30.0, 23.0, 6.0],
                                           [40.0, 23.0, 6.0]])
        elif 'smoke_medium' in args.exp_name:
            dataset_name = 'smoke_medium'
            self.init_seed = torch.randn((4, 1, 64, 64))
            # inital condition for testing [t,    x,    y]
            self.init_cond = torch.tensor([[10.0, 37.0, 6.0],
                                           [20.0, 37.0, 6.0],
                                           [30.0, 37.0, 6.0],
                                           [40.0, 37.0, 6.0]])
        elif 'smoke_large' in args.exp_name:
            dataset_name = 'smoke_large'
            self.init_seed = torch.randn((4, 1, 96, 96))
            # inital condition for testing [t,    x,    y]
            self.init_cond = torch.tensor([[10.0, 28.0, 13.0],
                                           [20.0, 28.0, 13.0],
                                           [30.0, 28.0, 13.0],
                                           [40.0, 28.0, 13.0]])
        else:
            raise NotImplementedError('Only smoke_small, smoke_medium, smoke_large are supported now.')

        self.dataset = FluidDataSet(self.data_root, dataset_name)
        self.ground_truths = self.dataset.get_ground_truths(self.init_cond)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

        self.optimizer = torch.optim.Adam(self.diffuser.eps_model.parameters(), lr=self.lr)
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - self.lrf) + self.lrf
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

        logging.info('Configs initialized')
        logging.info(f'Dataset: {dataset_name}')
        logging.info(f'Batch size: {args.batch_size}')
        logging.info(f'Epochs: {args.epochs}')
        logging.info(f'Epsilon model: {self.eps_model.__class__.__name__}')
    
    def train(self):
        for epoch in range(1 + self.args.epochs):
            self.diffuser.eps_model.train()
            cum_loss = 0
            pbar = tqdm(self.dataloader)
            for i, batch in enumerate(pbar):
                density = batch[0].cuda(self.args.device)
                cond = batch[-1].cuda(self.args.device)
                loss = self.diffuser.ddpm_loss(density, cond)
                cum_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description(f'Epoch [{epoch}/{self.args.epochs}] | Loss: {(cum_loss/(i+1)):.3f}')
                if not self.args.debug:
                    self.tb_writer.add_scalar('batch loss', loss.item(), epoch * len(self.dataloader) + i)
            if (epoch % self.eval_interval == 0 or epoch == self.args.epochs) and not self.args.debug:
                self.init_seed = self.init_seed.cuda(self.args.device)
                self.init_cond = self.init_cond.cuda(self.args.device)
                logging.info(f"Evaluating at epoch {epoch}, starting sampling...")
                x_pred = self.diffuser.sample(self.init_seed, self.init_cond)
                fig, ax = plt.subplots(2, 4, figsize=(20, 10))
                for i in range(4):
                    cond_ = self.init_cond[i].detach().cpu().numpy()
                    ax[0, i].imshow(self.ground_truths[i][0].squeeze(0).detach().numpy(), cmap='gray', origin='lower')
                    ax[0, i].set_title(f't: {cond_[0]} s, x: {cond_[1]}, y: {cond_[2]}')
                    ax[0, i].axis('off')
                    ax[1, i].imshow(x_pred[i, 0].detach().cpu().numpy(), cmap='gray', origin='lower')
                    ax[1, i].axis('off')
                    if i == 0:
                        ax[0, i].set_ylabel('Ground truth')
                        ax[1, i].set_ylabel('Prediction')
                self.tb_writer.add_figure('sample', fig, epoch)
                plt.close(fig)
            self.scheduler.step()
        torch.save(self.diffuser.eps_model.state_dict(), self.model_save_root + f'{self.args.exp_name}.pth')
        logging.info('Training finished')
        self.tb_writer.close()
                        
if __name__ == '__main__':
    args = parse.parse_args()
    configs = Configs(args)
    configs.train()




