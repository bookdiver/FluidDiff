import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vae import Autoencoder
from data import FluidDataset


parse = argparse.ArgumentParser(description='VAE training, used for compression')

parse.add_argument('--debug', action='store_true', help='Debug mode: (default: False)')

parse.add_argument('--device', type=int, default=0, help='device to use (default 0)')
parse.add_argument('--epochs', type=int, default=30, help='number of epochs, default: 20')
parse.add_argument('--batch-size', type=int, default=8, help='batch size, default: 16')

parse.add_argument('--latent-channels', type=int, default=4, help='latent space channel, default: 32')


def elbo_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float=0.7, recon_loss_type: str='bce') -> torch.Tensor:
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
    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss


def main(args):
    dataset = FluidDataset(fileroot='/media/bamf-big/gefan/FluidDiff/data/smoke_plume_64x64', normalize=True)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    if not args.debug:
        tb_writer = SummaryWriter(log_dir=f'./logs/smoke_plume_64x64_latent_train')

    model = Autoencoder(in_channels=2, emb_channels=args.latent_channels).cuda(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(1, args.epochs+1):
        train_pbar = tqdm(train_dl)
        train_cum_loss = 0
        for i, input in enumerate(train_pbar):
            input = input.cuda(args.device)
            optimizer.zero_grad()
            output = model(input)
            mean = output['z'].mean.flatten(start_dim=1, end_dim=-1)
            log_var = output['z'].log_var.flatten(start_dim=1, end_dim=-1)
            loss, recon_loss, kl_loss = elbo_loss(output['x'], output['x_hat'], mean, log_var, recon_loss_type='mse')
            loss.backward()
            optimizer.step()
            train_cum_loss += loss.item()
            train_pbar.set_description(f'Epoch [{epoch}/{args.epochs}] Average Loss: {(train_cum_loss/(i+1)):.4f}')
            if not args.debug:
                tb_writer.add_scalar('Loss', loss.item(), epoch*len(train_dl)+i)
                tb_writer.add_scalar('Reconstruction Loss', 0.7*recon_loss.item(), epoch*len(train_dl)+i)
                tb_writer.add_scalar('KL Loss', kl_loss.item(), epoch*len(train_dl)+i)
        
        with torch.no_grad():
            test_pbar = tqdm(test_dl, desc=f'Epoch {epoch}')
            test_cum_loss = 0
            for i, input in enumerate(test_pbar):
                input = input.cuda(args.device)
                output = model(input)
                mean = output['z'].mean.flatten(start_dim=1, end_dim=-1)
                log_var = output['z'].log_var.flatten(start_dim=1, end_dim=-1)
                loss, _, _ = elbo_loss(output['x'], output['x_hat'], mean, log_var, recon_loss_type='mse')
                test_cum_loss += loss.item()
                test_pbar.set_description(f'Validation Average Loss: {(test_cum_loss/(i+1)):.4f}')
            if not args.debug:
                tb_writer.add_scalar('Validation Loss', test_cum_loss/len(test_dl), epoch)

    if not args.debug:
        tb_writer.close()
        torch.save(model.state_dict(), './checkpoint/smoke_plume_64x64_latent_train.pt')
        print(f'Saved model to ./checkpoint/smoke_plume_64x64_latent_train.pt')

if __name__ == '__main__':
    args = parse.parse_args()
    main(args)
                


    
    