import os
import argparse
import math
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from autoencoder import Autoencoder, Autoencoder3D
from data import FluidDataset


parse = argparse.ArgumentParser(description='VAE training, used for compression')

parse.add_argument('--debug', action='store_true', help='Debug mode: (default: False)')
parse.add_argument('--from-checkpoint', type=str, default=None, help='path to checkpoint to load (default: None)')

parse.add_argument('--device', type=int, default=0, help='device to use (default 0)')
parse.add_argument('--epochs', type=int, default=50, help='number of epochs, default: 50')
parse.add_argument('--batch-size', type=int, default=8, help='batch size, default: 8')
parse.add_argument('--lr', type=float, default=1e-4, help='learning rate, default: 1e-4')
parse.add_argument('--lrf', type=float, default=0.1, help='the final learning rate factor compared to the initial one, default: 0.1')

parse.add_argument('--use-vae', action='store_true', help='use VAE (default: False)')
parse.add_argument('--vae-beta', type=float, default=0.1, help='beta for VAE loss, default: 0.05')
parse.add_argument('--activation-type', type=str, default='tanh', help='activation function, default: tanh')
parse.add_argument('--recon-type', type=str, default='sum', help='reconstruction loss type, default: sum')
parse.add_argument('--latent-channels', type=int, default=1, help='latent space channel, default: 1')

parse.add_argument('--physics-variables', type=str, nargs='+', default=['density', 'vorticity'], help='physics variables to use, default: density vorticity')
parse.add_argument('--train-video', action='store_true', help='train on video data (default: False)')
parse.add_argument('--normalization-type', type=str, default='01', help='normalization type, default: minmax')

parse.add_argument('--experiment-name', type=str, default='default name', help='name of the experiment, default: default name')

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, path)

def save_config(args, path):
    argsDict = args.__dict__
    with open(path+'/config.txt', 'w') as f:
        f.writelines('---------------------- Config ----------------------' + '\n')
        for key, value in argsDict.items():
            f.writelines(key + ' : ' + str(value) + '\n')
        f.writelines('----------------------------------------------------' + '\n')

def set_random_seed(seed=123, deterministic=True, benchmark=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True


def main(args):
    dataset = FluidDataset(
        fileroot='/media/bamf-big/gefan/FluidDiff/data/smoke_plume_64x64', 
        physics_variables=args.physics_variables,
        read_frames=args.train_video,
        read_every_frames=5,
        normalize_type=args.normalization_type)

    save_config(args, f'./checkpoint/smoke_plume64x64/{args.experiment_name}')
    
    print("*"*20)
    print(f"Loaded dataset with {len(dataset)} samples")
    print(f"Keep temporal dimension: {args.train_video}")
    print(f"Single sample shape: {dataset[0].shape}")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    print("*"*20)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    if not args.debug:
        tb_writer = SummaryWriter(log_dir=f'./logs/smoke_plume64x64/'+args.experiment_name)

    if args.train_video:
        model = Autoencoder3D(
            in_channels=len(args.physics_variables),
            z_channels=args.latent_channels,
            use_variational=args.use_vae,
            activation_type=args.activation_type,
        ).cuda(args.device)
    else:
        model = Autoencoder(
            in_channels=len(args.physics_variables), 
            z_channels=args.latent_channels,
            use_variational=args.use_vae,
            activation_type=args.activation_type,
            ).cuda(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    start_epoch = 1

    if args.from_checkpoint is not None:
        checkpoint = torch.load(args.from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f'Loaded checkpoint from {args.from_checkpoint} at epoch {start_epoch}')

    time_start = time.time()
    
    for epoch in range(start_epoch, args.epochs+1):
        train_pbar = tqdm(train_dl)
        train_cum_loss = 0
        for i, input in enumerate(train_pbar):
            input = input.cuda(args.device)
            optimizer.zero_grad()
            loss, recon_loss, kld_loss = model.loss(input, beta=args.vae_beta, recon_loss_type=args.recon_type)
            loss.backward()
            optimizer.step()
            train_cum_loss += loss.item()
            train_pbar.set_description(f'Epoch [{epoch}/{args.epochs}] Average Loss: {(train_cum_loss/(i+1)):.4f}')
            if not args.debug:
                tb_writer.add_scalar('Loss', loss.item(), epoch*len(train_dl)+i)
                tb_writer.add_scalar('Reconstruction Loss', recon_loss.item(), epoch*len(train_dl)+i)
                tb_writer.add_scalar('KLD Loss', kld_loss.item(), epoch*len(train_dl)+i)
        scheduler.step()
        
        with torch.no_grad():
            test_pbar = tqdm(test_dl)
            test_cum_loss = 0
            for i, input in enumerate(test_pbar):
                input = input.cuda(args.device)
                loss, _, _ = model.loss(input, beta=args.vae_beta)
                test_cum_loss += loss.item()
                test_pbar.set_description(f'Validation Average Loss: {(test_cum_loss/(i+1)):.4f}')

        if not args.debug:
            tb_writer.add_scalar('Validation Loss', test_cum_loss/len(test_dl), epoch)
            save_checkpoint(model=model, 
                            optimizer=optimizer, 
                            scheduler=scheduler, 
                            epoch=epoch, 
                            path=f'./checkpoint/smoke_plume64x64/{args.experiment_name}/model_checkpoint.pt')

        print('--'*20)

    if not args.debug:
        tb_writer.close()
        
    print(f'Training finished in {time.time()-time_start:.2f} seconds')

if __name__ == '__main__':
    args = parse.parse_args()
    os.makedirs(f'./checkpoint/smoke_plume64x64/{args.experiment_name}', exist_ok=True)
    set_random_seed(seed=1234, deterministic=False, benchmark=True)
    main(args)
                


    
    