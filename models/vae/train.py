import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from autoencoder import Autoencoder
from data import FluidDataset


parse = argparse.ArgumentParser(description='VAE training, used for compression')

parse.add_argument('--debug', action='store_true', help='Debug mode: (default: False)')
parse.add_argument('--from-checkpoint', type=str, default=None, help='path to checkpoint to load (default: None)')

parse.add_argument('--device', type=int, default=0, help='device to use (default 0)')
parse.add_argument('--epochs', type=int, default=50, help='number of epochs, default: 20')
parse.add_argument('--batch-size', type=int, default=8, help='batch size, default: 16')
parse.add_argument('--lr', type=float, default=1e-4, help='learning rate, default: 1e-4')

parse.add_argument('--use-vae', action='store_true', help='use VAE (default: False)')
parse.add_argument('--vae-beta', type=float, default=0.05, help='beta for VAE loss, default: 0.05')
parse.add_argument('--use-attention', action='store_true', help='use attention (default: False)')
parse.add_argument('--latent-channels', type=int, default=4, help='latent space channel, default: 32')
parse.add_argument('--experiment-name', type=str, default='default name', help='name of the experiment, default: default name')


def main(args):
    dataset = FluidDataset(fileroot='/media/bamf-big/gefan/FluidDiff/data/smoke_plume_64x64', normalize=True)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    if not args.debug:
        tb_writer = SummaryWriter(log_dir=f'./logs/'+args.experiment_name)

    model = Autoencoder(
        in_channels=3, 
        emb_channels=args.latent_channels,
        use_attn_in_bottleneck=args.use_attention,
        use_variational=args.use_vae
        ).cuda(args.device)
    
    if args.from_checkpoint is not None:
        model.load_state_dict(torch.load(args.from_checkpoint))
        print(f'Loaded checkpoint from {args.from_checkpoint}')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs+1):
        train_pbar = tqdm(train_dl)
        train_cum_loss = 0
        for i, input in enumerate(train_pbar):
            input = input.cuda(args.device)
            optimizer.zero_grad()
            loss, recon_loss, kld_loss = model.loss(input, beta=args.vae_beta)
            loss.backward()
            optimizer.step()
            train_cum_loss += loss.item()
            train_pbar.set_description(f'Epoch [{epoch}/{args.epochs}] Average Loss: {(train_cum_loss/(i+1)):.4f}')
            if not args.debug:
                tb_writer.add_scalar('Loss', loss.item(), epoch*len(train_dl)+i)
                tb_writer.add_scalar('Reconstruction Loss', recon_loss.item(), epoch*len(train_dl)+i)
                tb_writer.add_scalar('KLD Loss', kld_loss.item(), epoch*len(train_dl)+i)
        
        with torch.no_grad():
            test_pbar = tqdm(test_dl, desc=f'Epoch {epoch}')
            test_cum_loss = 0
            for i, input in enumerate(test_pbar):
                input = input.cuda(args.device)
                loss, _, _ = model.loss(input, beta=args.vae_beta)
                test_cum_loss += loss.item()
                test_pbar.set_description(f'Validation Average Loss: {(test_cum_loss/(i+1)):.4f}')
            if not args.debug:
                tb_writer.add_scalar('Validation Loss', test_cum_loss/len(test_dl), epoch)
        
        torch.save(model.state_dict(), './checkpoint/' + args.experiment_name + '.pt')

    if not args.debug:
        tb_writer.close()
        print(f'Training finished')

if __name__ == '__main__':
    args = parse.parse_args()
    main(args)
                


    
    