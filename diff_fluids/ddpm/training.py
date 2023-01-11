import argparse
import logging
import math

import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from utils import FluidDataset
from unet2 import UNet
from denoising_diffusion import DenoisingDiffusion

matplotlib.use('Agg')

parse = argparse.ArgumentParser(description='Denoising Diffusion Training')

parse.add_argument('--device', type=int, default=0, help='device to use (default 0)')
parse.add_argument('--debug', action='store_true', help='debug mode, default False')
parse.add_argument('--savename', type=str, default='default', help='name of the experiment, default "default"')

class Trainer:
    def __init__(self, 
                *, 
                args: dict, 
                configs: dict) -> None:

        self.device = args.device
        self.debug = args.debug
        self.savename = args.savename
        self.configs = configs

        self.eval_interval = configs['recording_params']['eval_interval']
        self.tb_writer_root = configs['recording_params']['tb_writer_root']
        self.model_save_root = configs['recording_params']['model_save_root']
        self.dataset_name = configs['dataset']['name']

        if self.debug:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(asctime)s:%(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')
            self.tb_writer = SummaryWriter(log_dir=self.tb_writer_root + self.savename +'/')

        eps_model = UNet(**configs['eps_model'])

        self.diffuser = DenoisingDiffusion(
            eps_model=eps_model,
            **configs['diffuser']
        ).cuda(self.device)

        dataset = FluidDataset(**configs['dataset'])
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.n_epochs = configs['training_params']['n_epochs']
        self.lr = configs['training_params']['learning_rate']
        self.lrf = configs['training_params']['learning_rate_final']
        self.batch_size = configs['training_params']['batch_size']
        self.patience = configs['training_params']['patience']

        self.train_dl = DataLoader(train_dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=True, 
                                    num_workers=4, 
                                    pin_memory=False)
        self.val_dl = DataLoader(val_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    pin_memory=False)

        self.optimizer = torch.optim.Adam(self.diffuser.eps_model.parameters(), lr=self.lr)
        lf = lambda x: ((1 + math.cos(x * math.pi / self.n_epochs)) / 2) * (1 - self.lrf) + self.lrf
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

        logging.info("Trainer initialized")
        logging.info(f"Training set size: {len(self.train_dl.dataset)}")
        logging.info(f"Validation set size: {len(self.val_dl.dataset)}")
    
    def train(self):
        train_losses = []
        val_losses = []
        epoch_len = len(str(self.n_epochs))

        for epoch in range(1, 1 + self.n_epochs):

            # Training
            self.diffuser.eps_model.train()
            pbar_train = tqdm(self.train_dl)
            for _, data in enumerate(pbar_train):
                x = torch.cat((data['u'], data['v']), dim=1).to(self.device)
                y = data['y'].to(self.device)
                loss = self.diffuser.ddpm_loss(x, y)
                train_loss = loss.item()
                train_losses.append(train_loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar_train.set_description(f"[{epoch:>{epoch_len}}/{self.n_epochs:>{epoch_len}}] | " +
                                           f"Train Loss: {train_loss:.5f}")
            self.scheduler.step()

            # Validation
            self.diffuser.eps_model.eval()
            val_loss = 0
            pbar_val = tqdm(self.val_dl)
            for _, data in enumerate(pbar_val):
                x = torch.cat((data['u'], data['v']), dim=1).to(self.device)
                y = data['y'].to(self.device)
                loss = self.diffuser.ddpm_loss(x, y)
                val_loss = loss.item()
                val_losses.append(val_loss)
                pbar_val.set_description(f"[{epoch:>{epoch_len}}/{self.n_epochs:>{epoch_len}}] | " +
                                           f"Val Loss: {val_loss:.5f}")

            # Record losses
            if not self.debug:
                avg_train_loss = sum(train_losses) / len(train_losses)
                avg_val_loss = sum(val_losses) / len(val_losses)
                self.tb_writer.add_scalar("Train Loss vs. Epoch", avg_train_loss, epoch)
                self.tb_writer.add_scalar("Val Loss vs. Epoch", avg_val_loss, epoch)
            train_losses = []
            val_losses = []
            
            # Testing
            if epoch % self.eval_interval == 0 or epoch == self.n_epochs:
                logging.info(f"Evaluating at epoch {epoch}, starting sampling...")
                with torch.no_grad():
                    sample = next(iter(self.val_dl))
                    x0 = torch.cat((sample['u'][:4], sample['v'][:4]), dim=1)
                    y = sample['y'][:4].to(self.device)
                    seed = torch.randn(*x0.shape, device=self.device)
                    x_pred = self.diffuser.sample(seed, y)
                fig, ax = plt.subplots(6, 4, figsize=(12, 18))
                x0 = x0.detach().numpy()
                x_pred = x_pred.detach().cpu().numpy()
                for i in range(4):
                    ax[0, i].imshow(x0[i, 0], cmap='gray', origin='lower')
                    ax[0, i].axis('off')
                    ax[1, i].imshow(x_pred[i, 0], cmap='gray', origin='lower')
                    ax[1, i].axis('off')
                    ax[2, i].imshow(x0[i, 1], origin='lower')
                    ax[2, i].axis('off')
                    ax[3, i].imshow(x_pred[i, 1], origin='lower')
                    ax[3, i].axis('off')
                    ax[4, i].imshow(x0[i, 2], origin='lower')
                    ax[4, i].axis('off')
                    ax[5, i].imshow(x_pred[i, 2], origin='lower')
                    ax[5, i].axis('off')
                self.tb_writer.add_figure("Sampling on Test set", fig, epoch)
                plt.close(fig)
            torch.save(self.diffuser.eps_model.state_dict(), self.model_save_root + f'{self.savename}.pth')

        logging.info('Training finished')
        self.tb_writer.close()
                        
if __name__ == '__main__':
    args = parse.parse_args()
    with open("config.yaml", "r") as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    trainer = Trainer(args=args, configs=configs)
    trainer.train()




