import argparse
import logging
import math

import yaml
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from utils import NavierStokesDataset
from net import UNet

matplotlib.use('Agg')

parse = argparse.ArgumentParser(description='Training UNet for baseline test')

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

        self.model = UNet(**configs['model']).to(self.device)

        train_dataset = NavierStokesDataset(**configs['dataset'], is_test=False)
        test_dataset = NavierStokesDataset(**configs['dataset'], is_test=True)

        self.n_epochs = configs['training_params']['n_epochs']
        self.lr = configs['training_params']['learning_rate']
        self.lrf = configs['training_params']['learning_rate_final']
        self.batch_size = configs['training_params']['batch_size']

        self.train_dl = DataLoader(train_dataset, 
                                   batch_size=self.batch_size, 
                                   shuffle=True, 
                                   num_workers=8, 
                                   pin_memory=False)
        self.test_dl = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=8,
                                 pin_memory=False)
        
        self.l2loss = MSELoss(reduction='mean')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lf = lambda x: ((1 + math.cos(x * math.pi / self.n_epochs)) / 2) * (1 - self.lrf) + self.lrf
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

        logging.info("Trainer initialized")
    
    def train(self):
        epoch_len = len(str(self.n_epochs))

        for epoch in range(1, 1 + self.n_epochs):

            # Training
            self.model.train()
            pbar_train = tqdm(self.train_dl)
            for i, data in enumerate(pbar_train):
                x = data['x'].to(self.device)
                y = data['y'].to(self.device)
                x_pred = self.model(y)
                loss = self.l2loss(x, x_pred)
                train_loss = loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar_train.set_description(f"[{epoch:>{epoch_len}}/{self.n_epochs:>{epoch_len}}] | " +
                                           f"Train Loss: {train_loss:.5f}")
                if hasattr(self, 'tb_writer'):
                    self.tb_writer.add_scalar("Train Loss", train_loss, (epoch-1)*len(self.train_dl) + i)
            self.scheduler.step()

            # Validation
            self.model.eval()
            cum_val_loss = 0
            pbar_val = tqdm(self.test_dl)
            for i, data in enumerate(pbar_val):
                x = data['x'].to(self.device)
                y = data['y'].to(self.device)
                x_pred = self.model(y)
                loss = self.l2loss(x, x_pred)
                cum_val_loss += loss.item()
                pbar_val.set_description(f"[{epoch:>{epoch_len}}/{self.n_epochs:>{epoch_len}}] | " +
                                           f"Avg Val Loss: {cum_val_loss / (i+1):.5f}")

            
            # Testing
            if epoch % self.eval_interval == 0 or epoch == self.n_epochs:
                logging.info(f"Evaluating at epoch {epoch}, starting sampling...")
                with torch.no_grad():
                    sample = next(iter(self.test_dl))
                    x = sample['x'][:4]
                    y = sample['y'][:4].to(self.device)
                    x_pred = self.model(y)
                fig, ax = plt.subplots(4, 4, figsize=(12, 12))
                x_pred = x_pred.detach().cpu().numpy()
                for i in range(4):
                    ax[0, i].imshow(x[i, 0], origin='lower')
                    ax[0, i].axis('off')
                    ax[1, i].imshow(x_pred[i, 0], origin='lower')
                    ax[1, i].axis('off')
                    ax[2, i].imshow(x[i, 1], origin='lower')
                    ax[2, i].axis('off')
                    ax[3, i].imshow(x_pred[i, 1], origin='lower')
                    ax[3, i].axis('off')
                if hasattr(self, 'tb_writer'):
                    self.tb_writer.add_figure("Sampling on Test set", fig, epoch)
                plt.close(fig)
            if not self.debug:
                torch.save(self.model.state_dict(), self.model_save_root + f'{self.savename}.pth')

        logging.info('Training finished')
        self.tb_writer.close()
                        
if __name__ == '__main__':
    args = parse.parse_args()
    with open("./config.yaml", "r") as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    trainer = Trainer(args=args, configs=configs)
    trainer.train()