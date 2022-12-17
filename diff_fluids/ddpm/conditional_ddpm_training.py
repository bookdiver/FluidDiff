import argparse
import logging
import math
import yaml

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from utils import FluidDataSet
from unet import UNet
from denoising_diffusion import DenoisingDiffusion

matplotlib.use('Agg')

parse = argparse.ArgumentParser(description='Denoising Diffusion Training')

parse.add_argument('--experiment', type=str, help="the no. of the experiment, e.g. 'exp1' ")

parse.add_argument('--device', type=int, default=0, help='device to use (default 0)')
parse.add_argument('--debug', action='store_true', help='debug mode, default False')

class Trainer:
    def __init__(self, *, args: dict, configs: dict) -> None:

        self.args = args
        self.configs = configs

        self.experiment_name = configs['recording_params']['experiment_name']
        self.eval_interval = configs['recording_params']['eval_interval']
        self.tb_writer_root = configs['recording_params']['tb_writer_root']
        self.model_save_root = configs['recording_params']['model_save_root']

        if args.debug:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(asctime)s:%(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')
            self.tb_writer = SummaryWriter(log_dir=self.tb_writer_root + self.experiment_name +'/')

        self.eps_model = UNet(**configs['unet_architecture'])

        self.diffuser = DenoisingDiffusion(
            eps_model=self.eps_model,
            **configs['ddpm_params']
        ).cuda(args.device)

        if configs['dataset']['dataset_name'] == 'smoke_small':
            self.init_seed = torch.randn((4, 1, 64, 64))
            # inital condition for testing [t,    x,    y]
            self.init_cond = torch.tensor([[10.0, 23.0, 6.0],
                                           [20.0, 23.0, 6.0],
                                           [30.0, 23.0, 6.0],
                                           [40.0, 23.0, 6.0]])
        elif configs['dataset']['dataset_name'] == 'smoke_medium':
            self.init_seed = torch.randn((4, 1, 64, 64))
            # inital condition for testing [t,    x,    y]
            self.init_cond = torch.tensor([[10.0, 37.0, 6.0],
                                           [20.0, 37.0, 6.0],
                                           [30.0, 37.0, 6.0],
                                           [40.0, 37.0, 6.0]])
        elif configs['dataset']['dataset_name'] == 'smoke_large':
            self.init_seed = torch.randn((4, 1, 96, 96))
            # inital condition for testing [t,    x,    y]
            self.init_cond = torch.tensor([[10.0, 28.0, 13.0],
                                           [20.0, 28.0, 13.0],
                                           [30.0, 28.0, 13.0],
                                           [40.0, 28.0, 13.0]])
        else:
            raise NotImplementedError('Only smoke_small, smoke_medium, smoke_large are supported now.')

        self.dataset = FluidDataSet(**configs['dataset'])
        self.ground_truths = self.dataset.get_ground_truths(self.init_cond)

        self.init_seed = self.init_seed.cuda(args.device)
        self.init_cond = self.init_cond.cuda(args.device)

        self.n_epochs = configs['training_params']['n_epochs']
        self.lr = configs['training_params']['learning_rate']
        self.lrf = configs['training_params']['learning_rate_final']
        self.batch_size = configs['training_params']['batch_size']

        self.dataloader = DataLoader(self.dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=True, 
                                    num_workers=4, 
                                    pin_memory=False)

        self.optimizer = torch.optim.Adam(self.diffuser.eps_model.parameters(), lr=self.lr)
        lf = lambda x: ((1 + math.cos(x * math.pi / self.n_epochs)) / 2) * (1 - self.lrf) + self.lrf
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

        logging.info('Configs initialized')
        logging.info(f'Dataset: {configs["dataset"]["dataset_name"]}')
        logging.info(f'Batch size: {self.batch_size}')
        logging.info(f'Epochs: {self.n_epochs}')
        logging.info(f'Learning rate: {self.lr}')
    
    def train(self):
        for epoch in range(1 + self.n_epochs):
            self.diffuser.eps_model.train()
            cum_loss = 0
            pbar = tqdm(self.dataloader)
            for i, batch in enumerate(pbar):
                density = batch[0].cuda(self.args.device)
                cond = batch[-1].cuda(self.args.device)
                cond[:, 0] = cond[:, 0] / 50.0
                loss = self.diffuser.ddpm_loss(density, cond)
                cum_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description(f'Epoch [{epoch}/{self.n_epochs}] | Loss: {(cum_loss/(i+1)):.3f}')
                if not self.args.debug:
                    self.tb_writer.add_scalar('batch loss', loss.item(), epoch * len(self.dataloader) + i)
            if (epoch % self.eval_interval == 0 or epoch == self.n_epochs) and not self.args.debug:
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
                self.tb_writer.add_figure('sample', fig, epoch)
                plt.close(fig)
            self.scheduler.step()

        torch.save(self.diffuser.eps_model.state_dict(), self.model_save_root + f'{self.experiment_name}.pth')
        logging.info('Training finished')
        self.tb_writer.close()
                        
if __name__ == '__main__':
    args = parse.parse_args()
    with open("config.yaml", "r") as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)[str(args.experiment)]
    trainer = Trainer(args=args, configs=configs)
    trainer.train()




