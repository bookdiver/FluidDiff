import argparse
import logging

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from utils import CGANDataset
from net import Discriminator, Generator

matplotlib.use('Agg')

parse = argparse.ArgumentParser(description='Conditional GAN baseline training')

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

        if self.debug:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(asctime)s:%(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')
            self.tb_writer = SummaryWriter(log_dir=self.tb_writer_root + self.savename +'/')

        self.generator = Generator(**configs['model']['generator']).to(self.device)
        self.discriminator = Discriminator(**configs['model']['discriminator']).to(self.device)

        dataset = CGANDataset(**configs['dataset'])
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.n_epochs = configs['training_params']['n_epochs']
        self.lr = configs['training_params']['learning_rate']
        self.batch_size = configs['training_params']['batch_size']

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

        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        self.loss_fn = nn.BCELoss()

        logging.info("Trainer initialized")
        logging.info(f"Training set size: {len(self.train_dl.dataset)}")
        logging.info(f"Validation set size: {len(self.val_dl.dataset)}")

    
    def train(self):
        epoch_len = len(str(self.n_epochs))
        real_label = 1.
        fake_label = 0.

        for epoch in range(1, 1 + self.n_epochs):

            # Training
            self.generator.train()
            self.discriminator.train()

            pbar_train = tqdm(self.train_dl)
            for i, data in enumerate(pbar_train):
                # Train discriminator
                self.discriminator.zero_grad()
                x = data['x'].to(self.device)
                y = data['y'].to(self.device)
                b, _, h, w = x.shape
                z = torch.randn(b, 1, h, w, device=self.device)
                
                label = torch.full((b,), real_label, device=self.device, dtype=torch.float)
                output = self.discriminator(x, y).reshape(-1)
                loss_D_real = self.loss_fn(output, label)
                loss_D_real.backward()

                fake = self.generator(z, y)
                label.fill_(fake_label)
                output = self.discriminator(fake.detach(), y).reshape(-1)
                loss_D_fake = self.loss_fn(output, label)
                loss_D_fake.backward()

                loss_D = loss_D_real + loss_D_fake
                self.optimizerD.step()
                
                # Train generator
                self.generator.zero_grad()
                label.fill_(real_label)
                output = self.discriminator(fake, y).reshape(-1)
                loss_G = self.loss_fn(output, label)
                loss_G.backward()
                self.optimizerG.step()

                pbar_train.set_description(f"Training [{epoch:>{epoch_len}}/{self.n_epochs:>{epoch_len}}] | " +
                                           f"Generator Loss: {loss_D.item():.4f} " +
                                           f"Discriminator Loss: {loss_G.item():.4f}")
                if hasattr(self, 'tb_writer'):
                    self.tb_writer.add_scalar('Generator Train Loss', loss_G.item(), epoch * len(self.train_dl) + i)
                    self.tb_writer.add_scalar('Discriminator Train Loss', loss_D.item(), epoch * len(self.train_dl) + i)

            # Validation
            self.generator.eval()
            self.discriminator.eval()
            with torch.no_grad():
                pbar_val = tqdm(self.val_dl)
                for i, data in enumerate(pbar_val):
                    x = data['x'].to(self.device)
                    y = data['y'].to(self.device)
                    b, _, h, w = x.shape
                    z = torch.randn(b, 1, h, w, device=self.device)

                    label = torch.full((b,), real_label, device=self.device, dtype=torch.float)
                    output = self.discriminator(x, y).reshape(-1)
                    loss_D_real = self.loss_fn(output, label)
                    
                    fake = self.generator(z, y)
                    label.fill_(fake_label)
                    output = self.discriminator(fake.detach(), y).reshape(-1)
                    loss_D_fake = self.loss_fn(output, label)

                    loss_D = loss_D_real + loss_D_fake

                    label.fill_(real_label)
                    output = self.discriminator(fake, y).reshape(-1)
                    loss_G = self.loss_fn(output, label)

                    pbar_val.set_description(f"Validation [{epoch:>{epoch_len}}/{self.n_epochs:>{epoch_len}}] | " +
                                             f"Generator Loss: {loss_G.item():.5f}" +
                                             f"Discriminator Loss: {loss_D.item():.5f}")
            
            # Testing
            if epoch % self.eval_interval == 0 or epoch == self.n_epochs:
                logging.info(f"Evaluating at epoch {epoch}, starting testing...")
                with torch.no_grad():
                    sample = next(iter(self.val_dl))
                    x0 = sample['x'][:4]
                    b, _, h, w = x0.shape
                    z = torch.randn(b, 1, h, w, device=self.device)
                    y = sample['y'][:4].to(self.device)
                    x_pred = self.generator(z, y)
                fig, ax = plt.subplots(6, 4, figsize=(12, 18))
                x0 = x0.detach().numpy()
                x_pred = x_pred.detach().cpu().numpy()
                for i in range(4):
                    ax[0, i].imshow(x0[i, 0], origin='lower')
                    ax[0, i].axis('off')
                    ax[1, i].imshow(x_pred[i, 0], origin='lower')
                    ax[1, i].axis('off')
                    ax[2, i].imshow(x0[i, 1], origin='lower')
                    ax[2, i].axis('off')
                    ax[3, i].imshow(x_pred[i, 1], origin='lower')
                    ax[3, i].axis('off')
                    ax[4, i].imshow(x0[i, 2], origin='lower')
                    ax[4, i].axis('off')
                    ax[5, i].imshow(x_pred[i, 2], origin='lower')
                    ax[5, i].axis('off')
                if hasattr(self, 'tb_writer'):
                    self.tb_writer.add_figure("Sampling on Test set", fig, epoch)
                plt.close(fig)
            torch.save(self.generator.state_dict(), self.model_save_root + f'{self.savename}.pth')

        logging.info('Training finished')
        self.tb_writer.close()
                        
if __name__ == '__main__':
    args = parse.parse_args()
    with open("config.yaml", "r") as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    trainer = Trainer(args=args, configs=configs)
    trainer.train()