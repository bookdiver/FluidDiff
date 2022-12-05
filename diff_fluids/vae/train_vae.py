import argparse
import logging

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchvision.transforms as transforms 
from tqdm import tqdm
import matplotlib.pyplot as plt

from vae import Autoencoder

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

def elbo_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float=0.5, recon_loss_type: str='bce') -> torch.Tensor:
    """
    Calculate the ELBO loss
    """
    if recon_loss_type == 'mse':
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    elif recon_loss_type == 'bce':
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    else:
        raise ValueError('Invalid reconstruction loss type')
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
    return recon_loss + beta * kld_loss

def vis(x: torch.Tensor, x_hat: torch.Tensor) -> plt.figure:
    x_grid = make_grid(x, nrow=2, normalize=True)
    x_hat_grid = make_grid(x_hat, nrow=2, normalize=True)
    x_res_grid = make_grid(torch.abs(x - x_hat), nrow=2)
    fig, ax = plt.subplots(1, 3)
    p1 = ax[0].imshow(x_grid.permute(1, 2, 0), cmap='gray', origin='lower')
    p2 = ax[1].imshow(x_hat_grid.permute(1, 2, 0), cmap='gray', origin='lower')
    p3 = ax[2].imshow(x_res_grid.permute(1, 2, 0), cmap='gray', origin='lower')
    fig.colorbar(p1, ax=ax[0])
    fig.colorbar(p2, ax=ax[1])
    fig.colorbar(p3, ax=ax[2])
    ax[0].set_title('Original')
    ax[1].set_title('Reconstruction')
    ax[2].set_title('Residual')
    return fig

def main(args):
    dataset = FluidDataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    logging.info(f'Loaded dataset {args.dataset} with {len(dataset)} frames')
    if not args.debug:
        tb_writer = SummaryWriter(log_dir=f'./logs/{args.dataset}')

    model = Autoencoder(in_channels=1).cuda(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, args.epochs+1):
        pbar = tqdm(dataloader)
        cum_loss = 0
        for i, input in enumerate(pbar):
            input = input.cuda(args.device)
            optimizer.zero_grad()
            output = model(input)
            loss = elbo_loss(output['x'], output['x_hat'], output['z'].mean, output['z'].log_var)
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
            pbar.set_description(f'Epoch {epoch}, Average Loss: {(cum_loss/(i+1)):.4f}')
            if not args.debug:
                tb_writer.add_scalar('Loss', loss.item(), epoch*len(dataloader)+i)
        if epoch % 5 == 0 and not args.debug:
            logging.info(f'Evaluating on epoch {epoch}')
            with torch.no_grad():
                samples = next(iter(dataloader))[:4].cuda(args.device)
                samples_hat = model(samples)['x_hat']
                fig = vis(samples, samples_hat)
                tb_writer.add_image('Visualition', fig, epoch)
    
    if not args.debug:
        tb_writer.close()
        torch.save(model.state_dict(), f'./checkpoint/{args.dataset}.pt')
        logging.info(f'Saved model to ./checkpoint/{args.dataset}.pt')

if __name__ == '__main__':
    args = parse.parse_args()
    main(args)
                


    
    