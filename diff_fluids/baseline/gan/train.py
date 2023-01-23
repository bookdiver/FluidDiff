import argparse
import logging

import yaml
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from utils import CGANDataset
from net import Discriminator, UNetGenerator, weights_init_normal

matplotlib.use('Agg')

parse = argparse.ArgumentParser(description='Conditional GAN baseline training')

parse.add_argument('--device', type=int, default=0, help='device to use (default 0)')
parse.add_argument('--debug', action='store_true', help='debug mode, default False')
parse.add_argument('--savename', type=str, default='default', help='name of the experiment, default "default"')
parse.add_argument('--start_epoch', type=int, default=1, help='start epoch, default 1')


class Trainer:
    def __init__(self, 
                *, 
                args: dict, 
                configs: dict) -> None:

        self.device = args.device
        self.debug = args.debug
        self.savename = args.savename
        self.start_epoch = args.start_epoch
        self.configs = configs

        self.eval_interval = configs['recording_params']['eval_interval']
        self.tb_writer_root = configs['recording_params']['tb_writer_root']
        self.model_save_root = configs['recording_params']['model_save_root']

        if not self.debug:
            self.tb_writer = SummaryWriter(log_dir=self.tb_writer_root + self.savename +'/')

        self.generator = UNetGenerator(**configs['model']['generator']).to(self.device)
        self.discriminator = Discriminator(**configs['model']['discriminator']).to(self.device)
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        train_dataset = CGANDataset(**configs['dataset'], is_test=False)
        val_dataset = CGANDataset(**configs['dataset'], is_test=True)
        
        self.n_epochs = configs['training_params']['n_epochs']
        self.G_lr = configs['training_params']['G_learning_rate']
        self.D_lr = configs['training_params']['D_learning_rate']
        self.batch_size = configs['training_params']['batch_size']
        self.lambda_GAN = configs['training_params']['lambda_GAN']

        self.train_dl = DataLoader(train_dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=True, 
                                    num_workers=8, 
                                    pin_memory=False)
        self.val_dl = DataLoader(val_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=8,
                                    pin_memory=False)

        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.G_lr, betas=(0.5, 0.999))
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.D_lr, betas=(0.5, 0.999))

        self.GAN_loss = nn.MSELoss()
        self.patch_loss = nn.L1Loss()

        logging.info("Trainer initialized")
        logging.info(f"Training set size: {len(self.train_dl.dataset)}")
        logging.info(f"Validation set size: {len(self.val_dl.dataset)}")

    
    def train(self):
        epoch_len = len(str(self.n_epochs))
        real_label = 1.
        fake_label = 0.

        if self.start_epoch != 1:
            self.generator.load_state_dict(torch.load(self.model_save_root + self.savename + f'_generator.pt'))
            self.discriminator.load_state_dict(torch.load(self.model_save_root + self.savename + '_discriminator.pt'))

        for epoch in range(self.start_epoch, 1 + self.n_epochs):

            # Training
            self.generator.train()
            self.discriminator.train()

            cum_train_loss_G = 0.
            cum_train_loss_D = 0.

            pbar_train = tqdm(self.train_dl)
            for i, data in enumerate(pbar_train):

                real_x = data['x'].to(self.device)
                real_y = data['y'].to(self.device)

                # Train Generator
                self.optimizerG.zero_grad()

                # GAN loss
                fake_x = self.generator(real_y)
                pred_fake = self.discriminator(fake_x, real_y)
                loss_GAN =  self.GAN_loss(pred_fake, Variable(torch.full_like(pred_fake, real_label), requires_grad=False))
                loss_patch = self.patch_loss(fake_x, real_x)

                loss_G = loss_GAN + self.lambda_GAN * loss_patch
                cum_train_loss_G += loss_G.item()
                loss_G.backward()

                self.optimizerG.step()

                # Train Discriminator
                self.optimizerD.zero_grad()

                # Real loss
                pred_real = self.discriminator(real_x, real_y)
                loss_real = self.GAN_loss(pred_real, Variable(torch.full_like(pred_real, real_label), requires_grad=False))

                # Fake loss
                pred_fake = self.discriminator(fake_x.detach(), real_y)
                loss_fake = self.GAN_loss(pred_fake, Variable(torch.full_like(pred_fake, fake_label), requires_grad=False))

                # Total loss
                loss_D = (loss_real + loss_fake) / 2
                cum_train_loss_D += loss_D.item()
                loss_D.backward()

                self.optimizerD.step()

                pbar_train.set_description(f"Training [{epoch:>{epoch_len}}/{self.n_epochs:>{epoch_len}}] | " +
                                           f"Avg G Loss: {cum_train_loss_G / (i+1):.4f} " +
                                           f"Avg D Loss: {cum_train_loss_D / (i+1):.4f} ")
                if hasattr(self, 'tb_writer'):
                    self.tb_writer.add_scalars('Train Loss', 
                                                {"Generator": loss_G.item(), "Discriminator": loss_D.item()}, epoch * len(self.train_dl) + i)

            # Validation
            self.generator.eval()
            self.discriminator.eval()
            cum_val_loss_G = 0.
            cum_val_loss_D = 0.
            with torch.no_grad():
                pbar_val = tqdm(self.val_dl)
                for i, data in enumerate(pbar_val):
                    real_x = data['x'].to(self.device)
                    real_y = data['y'].to(self.device)

                    # GAN loss
                    fake_x = self.generator(real_y)
                    pred_fake = self.discriminator(fake_x, real_y)
                    loss_GAN = self.GAN_loss(pred_fake, Variable(torch.full_like(pred_fake, real_label), requires_grad=False))
                    loss_patch = self.patch_loss(fake_x, real_x)

                    loss_G = loss_GAN + self.lambda_GAN * loss_patch
                    cum_val_loss_G += loss_G.item()

                    # Real loss
                    pred_real = self.discriminator(real_x, real_y)
                    loss_real = self.GAN_loss(pred_real, Variable(torch.full_like(pred_real, real_label), requires_grad=False))

                    # Fake loss
                    pred_fake = self.discriminator(fake_x.detach(), real_y)
                    loss_fake = self.GAN_loss(pred_fake, Variable(torch.full_like(pred_fake, fake_label), requires_grad=False))

                    # Total loss
                    loss_D = (loss_real + loss_fake) / 2
                    cum_val_loss_D += loss_D.item()

                    pbar_val.set_description(f"Validation [{epoch:>{epoch_len}}/{self.n_epochs:>{epoch_len}}] | " +
                                             f"Avg G Loss: {cum_val_loss_G / (i+1):.4f} " +
                                             f"Avg D Loss: {cum_val_loss_D / (i+1):.4f} ")
            
            # Testing
            if epoch % self.eval_interval == 0 or epoch == self.n_epochs:
                logging.info(f"Evaluating at epoch {epoch}, starting testing...")
                with torch.no_grad():
                    sample = next(iter(self.val_dl))
                    x0 = sample['x'][:4]
                    y = sample['y'][:4].to(self.device)
                    x_pred = self.generator(y)
                fig, ax = plt.subplots(4, 4, figsize=(16, 16))
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
                if hasattr(self, 'tb_writer'):
                    self.tb_writer.add_figure("Sampling on Test set", fig, epoch)
                plt.close(fig)
            if not self.debug:
                torch.save(self.generator.state_dict(), self.model_save_root + self.savename + '_generator.pt')
                torch.save(self.discriminator.state_dict(), self.model_save_root + self.savename + '_discriminator.pt')

        logging.info('Training finished')
        self.tb_writer.close()
                        
if __name__ == '__main__':
    args = parse.parse_args()
    with open("config.yaml", "r") as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    trainer = Trainer(args=args, configs=configs)
    trainer.train()