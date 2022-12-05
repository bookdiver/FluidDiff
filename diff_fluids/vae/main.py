import argparse
import logging

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchvision.transforms as transforms 
from tqdm import tqdm

from vae import VAE

logging.basicConfig(level=logging.INFO)

dataset_name = ['smoke_small', 'smoke_medium', 'smoke_large']
device_ids = [0, 1]

parse = argparse.ArgumentParser(description='VAE training, used for compression')

parse.add_argument('--debug', action='store_true', help='Debug mode: (default: False)')

parse.add_argument('--dataset', type=str, default='smoke_small', choices=dataset_name, help='dataset name: (default: smoke_small)')
parse.add_argument('--device', type=int, default=1, choices=device_ids, help='device to use (default 1)')
parse.add_argument('--epochs', type=int, default=20, help='number of epochs, default: 20')
parse.add_argument('--batch-size', type=int, default=16, help='batch size, default: 16')

parse.add_argument('--latent-dim', type=int, default=128, help='latent dimension, default: 128')

class FluidDataset(Dataset):
    def __init__(self, data: str='smoke_small'):
        load_path = f'../../data/smoke/{data}.npz'
        self.data = np.load(load_path, allow_pickle=True)
        self.density = torch.from_numpy(self.data['log_density']).float().flatten(start_dim=0, end_dim=1).unsqueeze(1)
        self.transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return self.density.shape[0]

    def __getitem__(self, idx):
        frame = self.density[idx]
        frame = self.transform(frame)
        return frame

def main(args):
    dataset = FluidDataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    logging.info(f'Loaded dataset {args.dataset} with {len(dataset)} frames')
    if not args.debug:
        tb_writer = SummaryWriter(log_dir=f'./logs/{args.dataset}')

    in_dim = dataset[0].shape[-2]
    model = VAE(in_channels=1, in_dim=in_dim, latent_dim=args.latent_dim).cuda(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, args.epochs+1):
        pbar = tqdm(dataloader)
        cum_loss = 0
        for i, input in enumerate(pbar):
            input = input.cuda(args.device)
            optimizer.zero_grad()
            output = model(input)
            losses = model.loss_function(output)
            loss = losses['loss']
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
            pbar.set_description(f'Epoch {epoch}, Average Loss: {(cum_loss/(i+1)):.4f}')
            if not args.debug:
                tb_writer.add_scalar('Loss', losses['loss'].item(), epoch*len(dataloader)+i)
                tb_writer.add_scalar('Reconstruction Loss', losses['reconstruction_loss'].item(), epoch*len(dataloader)+i)
                tb_writer.add_scalar('KL Loss', losses['kld_loss'].item(), epoch*len(dataloader)+i)
        if epoch % 5 == 0 and not args.debug:
            logging.info(f'Evaluating on epoch {epoch}')
            with torch.no_grad():
                samples = next(iter(dataloader))
                output = model(samples.cuda(args.device))
                tb_writer.add_image('Original', torch.flip(make_grid(samples, nrow=4, normalize=True), dims=[1]), epoch)
                tb_writer.add_image('Reconstruction', torch.flip(make_grid(output['recon'], nrow=4, normalize=True), dims=[1]), epoch)
    
    if not args.debug:
        tb_writer.close()
        torch.save(model.state_dict(), f'./checkpoint/{args.dataset}.pt')
        logging.info(f'Saved model to ./checkpoint/{args.dataset}.pt')

if __name__ == '__main__':
    args = parse.parse_args()
    main(args)
                


    
    