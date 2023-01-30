import argparse
import logging
import math

import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import NavierStokesPINNDataset, ChainedScheduler
from net import PINN

parse = argparse.ArgumentParser(description='Training PINN for baseline test')

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

        self.tb_writer_root = configs['recording_params']['tb_writer_root']
        self.model_save_root = configs['recording_params']['model_save_root']

        if self.debug:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(asctime)s:%(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')
            self.tb_writer = SummaryWriter(log_dir=self.tb_writer_root + self.savename +'/')

        self.model = PINN(**configs['model']).to(self.device)
        self.l2_loss = torch.nn.MSELoss()

        dataset = NavierStokesPINNDataset(**configs['dataset'])
        train_dataset, test_dataset = random_split(dataset, [int(0.10 * len(dataset)), len(dataset) - int(0.10 * len(dataset))])

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

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lf = lambda x: ((1 + math.cos(x * math.pi / self.n_epochs)) / 2) * (1 - self.lrf) + self.lrf
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        logging.info("Trainer initialized")
    
    def train(self):
        epoch_len = len(str(self.n_epochs))

        for epoch in range(1, 1 + self.n_epochs):

            # Training
            self.model.train()
            cum_train_loss = 0
            pbar_train = tqdm(self.train_dl)
            for i, data in enumerate(pbar_train):
                data = data.to(self.device).requires_grad_(True)
                x, y, t, u, v, p, _ = torch.chunk(data, 7, dim=-1)
                eq_loss = self.model.eq_loss(x, y, t)
                data_loss = self.model.data_loss(x, y, t, u, v, p)
                loss = eq_loss + data_loss
                cum_train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar_train.set_description(f"[{epoch:>{epoch_len}}/{self.n_epochs:>{epoch_len}}] | " +
                                           f"Train Loss: {cum_train_loss / (i+1):.5f}")
                if hasattr(self, 'tb_writer'):
                    self.tb_writer.add_scalars("Train Loss", 
                    {"equation loss": eq_loss.item(), "data loss": data_loss.item(), "total loss": loss.item()}, 
                    (epoch-1)*len(self.train_dl) + i)

            # Validation
            self.model.eval()
            cum_val_loss = 0
            for data in self.test_dl:
                data = data.to(self.device)
                x, y, t, u, v, p, _ = torch.chunk(data, 7, dim=-1)
                loss = self.model.data_loss(x.requires_grad_(True), y.requires_grad_(True), t, u, v, p)
                cum_val_loss += loss.item()
            print(f"Validation Loss: {cum_val_loss/len(self.test_dl):.5f}")
            if not self.debug:
                torch.save(self.model.state_dict(), self.model_save_root + f'{self.savename}.pth')

        self.scheduler.step()

        logging.info('Training finished')
        self.tb_writer.close()
                        
if __name__ == '__main__':
    args = parse.parse_args()
    with open("./config.yaml", "r") as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    trainer = Trainer(args=args, configs=configs)
    trainer.train()