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

dataset_name = ['smoke_small', 'smoke_medium', 'smoke_large']

parse = argparse.ArgumentParser(description='Denoising Diffusion Training')

parse.add_argument('--device', type=int, default=0, help='device to use (default 0)')
parse.add_argument('--data-root', type=str, default='/media/bamf-big/gefan/DiffFluids/data/smoke/', help='path to data root')
parse.add_argument('--dataset', type=str, default='smoke_small', choices=dataset_name, help='dataset name: (default: smoke_small)')

parse.add_argument('--epochs', type=int, default=20, help='number of epochs, default: 20')
parse.add_argument('--batch-size', type=int, default=16, help='batch size, default: 16')

parse.add_argument('--debug', action='store_true', help='debug mode, default False')
parse.add_argument('--xattention', action='store_true', help='use cross attention, default False')

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
    n_steps: int=1000
    lr: float=2e-4
    lrf: float=0.1
    dataset: FluidDataSet
    data_loader: DataLoader
    optimizer: torch.optim.Adam
    tb_writer = SummaryWriter
    def __init__(self, args):
        if args.debug:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(asctime)s:%(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')
        
        self.args = args

        if args.xattention:
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
        else:
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

        self.diffuser = DenoisingDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device = args.device
        )

        self.dataset = FluidDataSet(args.data_root, args.dataset)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        self.optimizer = torch.optim.Adam(self.diffuser.eps_model.parameters(), lr=self.lr)
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - self.lrf) + self.lrf
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

        if not args.debug:
            self.tb_writer = SummaryWriter(log_dir='/media/bamf-big/gefan/DiffFluids/diff_fluids/ddpm/logs/' + args.dataset +'/')

        if args.dataset == 'smoke_small':
            self.init_seed = torch.randn((4, 1, 64, 64)).cuda(args.device)
            # inital condition for testing [t,    x,    y]
            self.init_cond = torch.tensor([[10.0, 6.0, 6.0],
                                           [20.0, 23.0, 6.0],
                                           [30.0, 40.0, 23.0],
                                           [40.0, 58.0, 40.0]]).cuda(args.device)
        else:
            raise NotImplementedError('Only smoke_small dataset is supported now.')
        logging.info('Configs initialized')
    
    def train(self):
        for epoch in range(1 + self.args.epochs):
            self.diffuser.eps_model.train()
            cum_loss = 0
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch}')
            for i, batch in enumerate(pbar):
                density = batch[0].cuda(self.args.device)
                cond = batch[-1].cuda(self.args.device)
                loss = self.diffuser.ddpm_loss(x0=density, cond=cond)
                cum_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description(f'loss: {(cum_loss/(i+1)):.3f}')
                if not self.args.debug:
                    self.tb_writer.add_scalar('loss', loss.item(), epoch * len(self.dataloader) + i)
            if (epoch % 5 == 0 or epoch == self.args.epochs) and not self.args.debug:
                logging.info(f"Evaluating at epoch {epoch}")
                x_pred = self.diffuser.sample(self.init_seed, self.init_cond)
                fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                for i in range(4):
                    ax[i].imshow(x_pred[i, 0].detach().cpu().numpy(), cmap='gray', origin='lower')
                    cond_ = self.init_cond[i, 0].detach().cpu().numpy()
                    ax[i].set_title(f't: {cond_[0]} s, x: {cond_[1]}, y: {cond_[2]}')
                    ax[i].axis('off')
                self.tb_writer.add_figure('sample', fig, epoch)
                plt.close(fig)
            self.scheduler.step()
        torch.save(self.diffuser.eps_model.state_dict(), f'/media/bamf-big/gefan/DiffFluids/diff_fluids/ddpm/checkpoint/{self.args.dataset}_condUnet.pt')
        logging.info('Training finished')
        self.tb_writer.close()
                        
if __name__ == '__main__':
    args = parse.parse_args()
    configs = Configs(args)
    configs.train()




