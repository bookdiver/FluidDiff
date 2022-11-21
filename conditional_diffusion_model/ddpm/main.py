import argparse
from typing import Optional
import logging
import sys

sys.path.append('..')

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import MyDataSet
from ddpm import DDPM

logging.basicConfig(level=logging.DEBUG)

parse = argparse.ArgumentParser()
parse.add_argument('--data_path', type=str, default='/media/bamf-big/gefan/DiffFluids/data/smoke/')
parse.add_argument('--n_epochs', type=int, default=20, help='number of epochs, default: 20')
parse.add_argument('--batch_size', type=int, default=16, help='batch size, default: 16 | suggested: 64 for small, 32 for medium, 16 for large')
parse.add_argument('--lr', type=float, default=1e-4, help='learning rate, default: 1e-4')
parse.add_argument('--save', action='store_true', help='save model')
parse.add_argument('--dataset', type=str, default='smoke_small', help='dataset name, default smoke_small')

args = parse.parse_args()


class Trainer:
    def __init__(self, 
                 model: nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 load_path: Optional[str]=None,
                 device: str='cuda:1'
                 ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.load_path = load_path
        self.device = device
        self.writer = SummaryWriter(log_dir='/media/bamf-big/gefan/DiffFluids/logs')

    def train(self) -> None:
        self.model.train()
        if self.load_path is not None:
            self.model.net.load_state_dict(torch.load(self.load_path))
            logging.info(f"Model checkpoint loaded from {self.load_path}")
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

        for ep in range(args.n_epochs):

            pbar = tqdm(self.dataloader)
            ep_cum_loss = 0
            n_batchs = 0
            ep_avg_loss = 0
            for x in pbar:
                n_batchs += 1
                optimizer.zero_grad()
                frame = x[0].to(self.device)
                # condition = x[-1].to(self.device)
                loss = self.model(frame)
                loss.backward()
                self.writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=ep * len(self.dataloader) + n_batchs)
                ep_cum_loss += loss.item()
                ep_avg_loss = ep_cum_loss / n_batchs
                pbar.set_description(f'Epoch [{ep+1:0>2d} / {args.n_epochs}] | Average Loss: {ep_avg_loss:.4f}')
                optimizer.step()
            scheduler.step()
        
        if args.save:
            torch.save(self.model.net.state_dict(), '/media/bamf-big/gefan/DiffFluids/checkpoint' + args.dataset + "_diffuser.pkl")
            logging.info(f"Model saved")
        self.writer.close()

if __name__ == '__main__':
    device = 'cuda:1'
    dataset = MyDataSet(args.data_path + args.dataset + '.npz')
    model = DDPM(in_channels=1, betas=[1e-4, 0.02], n_T=400, device=device)
    logging.info(f"Dataset created, {len(dataset)} sample in total.")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)
    logging.info(f"Model created")
    trainer = Trainer(model, dataloader, device=device)
    logging.info("Start training ...")
    trainer.train()