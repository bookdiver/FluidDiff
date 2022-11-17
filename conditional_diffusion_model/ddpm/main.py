import argparse
from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import MyDataSet
from ddpm import DDPM

logging.basicConfig(level=logging.DEBUG)

parse = argparse.ArgumentParser()
parse.add_argument('--data_path', type=str, default='../data/smoke/')
parse.add_argument('--n_epochs', type=int, default=20)
parse.add_argument('--batch_size', type=int, default=16)
parse.add_argument('--lr', type=float, default=1e-4)
parse.add_argument('--multi_gpu', action='store_true', help='use multi-gpu for training')
parse.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parse.add_argument('--save', action='store_true', help='save model')
parse.add_argument('--save_path', type=str, default='./checkpoint/ddpmunet.pkl')
parse.add_argument('--dataset', type=str, default='smoke1', help='dataset name')

args = parse.parse_args()

if args.multi_gpu:
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)

class Trainer:
    def __init__(self, 
                 model: nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 log_path: str='./logs/',
                 load_path: Optional[str]=None,
                 device: str='cuda:1') -> None:
        self.model = model
        self.dataloader = dataloader
        self.load_path = load_path
        self.device = device
        self.writer = SummaryWriter(log_dir=log_path, comment='DDPM_smoke1')

    def train(self) -> None:
        self.model.train()
        if self.load_path is not None:
            self.model.net.load_state_dict(torch.load(self.load_path))
            logging.info(f"Model checkpoint loaded from {self.load_path}")
        if args.multi_gpu and torch.cuda.device_count() > 1:
            logging.info(f"Now using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model.cuda())
            optimizer = torch.optim.Adam(self.model.module.net.parameters(), lr=args.lr)
        else:
            self.model.to(self.device)
            logging.info(f"Now using 1 GPU!")
            optimizer = torch.optim.Adam(self.model.net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        for ep in range(args.n_epochs):

            pbar = tqdm(self.dataloader)
            ep_cum_loss = 0
            n_batchs = 0
            ep_avg_loss = 0
            for x in pbar:
                n_batchs += 1
                optimizer.zero_grad()
                frame = x[0].to(self.device)
                condition = x[-1].to(self.device)
                if args.multi_gpu:
                    loss = self.model.module(frame, condition)
                else:
                    loss = self.model(frame, condition)
                loss.backward()
                self.writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=ep * len(self.dataloader) + n_batchs)
                ep_cum_loss += loss.item()
                ep_avg_loss = ep_cum_loss / n_batchs
                pbar.set_description(f'Epoch [{ep+1:0>2d} / {args.n_epochs}] | Average Loss: {ep_avg_loss:.4f}')
                optimizer.step()
            scheduler.step()
        
        if args.save:
            torch.save(self.model.net.state_dict(), args.save_path)
            logging.info(f"Model saved to {args.save_path}")
        self.writer.close()

if __name__ == '__main__':
    device = 'cuda:1'
    dataset = MyDataSet(args.data_path + args.dataset + '.npz')
    model = DDPM(in_channels=1, n_feats=256, betas=[1e-3, 0.02], n_T=400, device=device)
    logging.info(f"Dataset created, {len(dataset)} sample in total.")
    if args.multi_gpu:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)    
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, pin_memory=True)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)
    logging.info(f"Model created")
    trainer = Trainer(model, dataloader, device=device)
    logging.info("Start training ...")
    trainer.train()