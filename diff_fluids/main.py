import argparse
import math
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import MyDataSet
from ddpm import DDPM

class Trainer:
    def __init__(self, args: dict) -> None:
        self.device = args.device
        self.writer = SummaryWriter(log_dir='/media/bamf-big/gefan/DiffFluids/logs/' + args.dataset)

        self.dataset = MyDataSet(args.data_root + args.dataset + '.npz')
        nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
        logging.info(f"Using {nw} dataloader workers every process")
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
        self.model = DDPM(in_channels=1, betas=args.betas, n_T=args.T)
        if args.use_pretrained:
            self.model.load_state_dict(torch.load(args.load_root + args.dataset + '.pt'))
            logging.info(f"Load pretrained model from {args.load_root + args.dataset + '.pt'}")
        self.model.to(self.device)


    def train(self) -> None:
        if args.dataset == 'smoke_small':
            init_x = torch.randn((8, 1, 64, 48), device=self.device)
        elif args.dataset == 'smoke_medium':
            init_x = torch.randn((8, 1, 96, 64), device=self.device)
        elif args.dataset == 'smoke_large':
            init_x = torch.randn((8, 1, 128, 96), device=self.device)

        trainable_params = [p for p in self.model.net.parameters() if p.requires_grad]
        logging.info(f"Total number of trainable parameters: {sum([p.numel() for p in trainable_params])}")
        optimizer = optim.Adam(trainable_params, lr=args.lr)
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        for ep in range(args.epochs):
            self.model.train()

            pbar = tqdm(self.dataloader)
            ep_cum_loss = 0
            n_batchs = 0
            ep_avg_loss = 0
            for x in pbar:
                n_batchs += 1
                optimizer.zero_grad()
                density = x[0].to(self.device)
                loss = self.model(density)
                loss.backward()
                self.writer.add_scalar(tag='batch_loss', scalar_value=loss.item(), global_step=ep * len(self.dataloader) + n_batchs)
                ep_cum_loss += loss.item()
                ep_avg_loss = ep_cum_loss / n_batchs
                pbar.set_description(f'Epoch [{ep+1:0>2d} / {args.epochs}] | Average Loss: {ep_avg_loss:.4f}')
                optimizer.step()
            scheduler.step()
            self.writer.add_scalar(tag='learning_rate', scalar_value=optimizer.param_groups[0]["lr"], global_step=ep)
            if ep % 5 == 0 or ep == args.epochs - 1:
                logging.info(f"Test model at epoch {ep+1}")
                with torch.no_grad():
                    self.model.eval()
                    x_gen = self.model.sample(init_x)
                    fig, ax = plt.subplots(2, 4, figsize=(10, 5))
                    for i in range(2):
                        for j in range(4):
                            ax[i, j].imshow(x_gen[i * 4 + j, 0, :, :].detach().cpu().numpy(), cmap='gray', origin='lower')
                            ax[i, j].axis('off')
                    self.writer.add_figure(tag='generated_images', figure=fig, global_step=ep)
        
        torch.save(self.model.state_dict(), '/media/bamf-big/gefan/DiffFluids/checkpoint/ddpm/' + args.dataset + "_diffuser.pkl")
        logging.info(f"Model saved")
        self.writer.close()

if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--device', type=str, default='cuda:1', help='device to use, default cuda:1')

    parse.add_argument('--data-root', type=str, default='/media/bamf-big/gefan/DiffFluids/data/smoke/', help='path to data root')
    parse.add_argument('--dataset', type=str, default='smoke_small', help='dataset name, default smoke_small')

    parse.add_argument('--T', type=int, default=400, help='number of diffusion steps, default 400')
    parse.add_argument('--betas', type=float, nargs='+', default=[1e-4, 0.02], help='bounds for betas in noise schedule, default [1e-4, 0.02]')
    parse.add_argument('--use-pretrained', action='store_true', help='use pretrained model, default False')
    parse.add_argument('--load-root', type=str, default='/media/bamf-big/gefan/DiffFluids/checkpoints/', help='path to load trained model')

    parse.add_argument('--epochs', type=int, default=20, help='number of epochs, default: 20')
    parse.add_argument('--batch-size', type=int, default=16, help='batch size, default: 16 | suggested: 64 for small, 32 for medium, 16 for large')
    parse.add_argument('--lr', type=float, default=1e-4, help='learning rate, default: 1e-4')
    parse.add_argument('--lrf', type=float, default=0.1, help='learning rate factor, default: 0.1')

    args = parse.parse_args()

    trainer = Trainer(args)
    logging.info("Start training ...")
    trainer.train()