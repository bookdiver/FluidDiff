import argparse
import logging
import math

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from utils import MyDataSet
from unet import UNetModel
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

class Configs:
    eps_model: UNetModel
    diffuser: DenoisingDiffusion
    in_channels: int=1
    out_channels: int=1
    channels: int=32
    channel_multpliers: list=[1, 2, 4, 8]
    n_res_blocks: int=2
    attention_levels: list=[1, 2]
    n_heads: int=4
    transformer_layers: int=1
    n_steps: int=1000
    lr: float=2e-5
    lrf: float=0.1
    dataset: MyDataSet
    data_loader: DataLoader
    optimizer: torch.optim.Adam
    tb_writer = SummaryWriter
    def __init__(self, args):
        if args.debug:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(asctime)s:%(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')
        
        self.args = args

        self.eps_model = UNetModel(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            channels = self.channels,
            channel_multpliers = self.channel_multpliers,
            n_res_blocks = self.n_res_blocks,
            attention_levels = self.attention_levels,
            n_heads = self.n_heads,
            transformer_layers = self.transformer_layers,
            d_cond = args.batch_size
        ).cuda(args.device)

        self.diffuser = DenoisingDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device = args.device
        )

        self.dataset = MyDataSet(args.data_root, args.dataset)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        self.optimizer = torch.optim.Adam(self.diffuser.eps_model.parameters(), lr=self.lr)
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - self.lrf) + self.lrf
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

        if not args.debug:
            self.tb_writer = SummaryWriter(log_dir='/media/bamf-big/gefan/DiffFluids/logs/' + args.dataset +'/')

        logging.info('Configs initialized')
    
    def train(self):
        for epoch in range(1 + self.args.epochs):
            self.diffuser.eps_model.train()
            self.scheduler.step(epoch)
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch}')
            for i, batch in enumerate(pbar):
                density = batch[0].cuda(self.device)
                cond = batch[-1].cuda(self.device)
                loss = self.diffuser.ddpm_loss(x0=density, noise=None, cond=cond)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix({'loss': loss.item()})
                if not self.args.debug:
                    self.tb_writer.add_scalar('loss', loss.item(), epoch * len(self.dataloader) + i)


if __name__ == '__main__':
    args = parse.parse_args()
    configs = Configs(args)
    configs.train()




