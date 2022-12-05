import argparse
import math
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')

import torch
import torch.nn.parallel as parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DiffFluids.diff_fluids.ddpm.utils import MyDataSet
from DiffFluids.diff_fluids.ddpm.ddpm import DDPM

dataset_name = ['smoke_small', 'smoke_medium', 'smoke_large']
device_ids = [0, 1, -1]

parse = argparse.ArgumentParser(description='DDPM training, use diffuser library')


parse.add_argument('--device', type=int, default=1, choices=device_ids, help='device to use (set -1 means use all the GPUs): (default 1)')
parse.add_argument('--local-rank', type=int, default=-1, help='local rank for distributed training')

parse.add_argument('--data-root', type=str, default='/media/bamf-big/gefan/DiffFluids/data/smoke/', help='path to data root')
parse.add_argument('--dataset', type=str, default='smoke_small', choices=dataset_name, help='dataset name: ' +  '|'.join(dataset_name) + '(default: smoke_small)')

parse.add_argument('--T', type=int, default=400, help='number of diffusion steps, default 400')
parse.add_argument('--betas', type=float, nargs='+', default=[1e-4, 0.02], help='bounds for betas in noise schedule, default [1e-4, 0.02]')
parse.add_argument('--use-pretrained', action='store_true', help='use pretrained model, default False')
parse.add_argument('--load-root', type=str, default='/media/bamf-big/gefan/DiffFluids/checkpoints/', help='path to load trained model')

parse.add_argument('--epochs', type=int, default=20, help='number of epochs, default: 20')
parse.add_argument('--batch-size', type=int, default=16, help='batch size, default: 16 | suggested: 64 for small, 32 for medium, 16 for large')
parse.add_argument('--lr', type=float, default=1e-4, help='learning rate, default: 1e-4')
parse.add_argument('--lrf', type=float, default=0.1, help='learning rate factor, default: 0.1')

parse.add_argument('--not-save', action='store_true', help='save model or not, default False')

def reduce_mean(tensor, nprocs) -> torch.Tensor:
    """ Reduce mean for distributed training 
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def main():
    args = parse.parse_args()
    if args.device == -1:
        logging.info('Using all the GPUs')
        args.nprocs = torch.cuda.device_count()
        mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))
    else:
        logging.info(f'Using GPU: {args.device}')
        args.nprocs = 1
        main_worker(0, args.nprocs, args)

def init_seeds(seed: int = 0, cuda_deterministic: bool = True):
    """ Initialize random seeds for reproducibility 
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True
        

def main_worker(local_rank: int, nprocs: int, args: dict):
    args.local_rank = local_rank if args.device == -1 else args.device
    init_seeds(local_rank+1)
    cudnn.benchmark = True
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=args.nprocs, rank=local_rank)

    model = DDPM(in_channels=1, betas=args.betas, n_T=args.T)
    if args.use_pretrained:
        model.load_state_dict(torch.load(args.load_root + args.dataset + '.pt'))
        logging.info(f"Rank {local_rank}: Load pretrained model from {args.load_root + args.dataset + '.pt'}")
    torch.cuda.set_device(args.local_rank)
    model.cuda(args.local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(args.local_rank)
    model = parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    trainable_params = [p for p in model.module.net.parameters() if p.requires_grad]
    logging.info(f"Device {args.local_rank}: Total number of trainable parameters: {sum([p.numel() for p in trainable_params])}")
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    batch_size = args.batch_size // nprocs
    dataset = MyDataSet(args.data_root + args.dataset + '.npz')
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    nw = 8 if args.device != -1 else 0
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=nw, pin_memory=True, sampler=sampler)

    if local_rank == 0 and args.not_save == False:
        tb_writer = SummaryWriter(log_dir='/media/bamf-big/gefan/DiffFluids/logs/' + args.dataset +'/')
        if args.dataset == 'smoke_small':
            init_x = torch.randn((8, 1, 64, 64)).cuda(args.device)
        elif args.dataset == 'smoke_medium':
            init_x = torch.randn((8, 1, 64, 64)).cuda(args.device)
        elif args.dataset == 'smoke_large':
            init_x = torch.randn((8, 1, 96, 96)).cuda(args.device)

    for epoch in range(args.epochs):
        model.module.net.train()
        sampler.set_epoch(epoch)
        scheduler.step(epoch)
        if local_rank == 0:
            pbar = tqdm(dataloader, total=len(dataloader))
        for i, batch in enumerate(pbar if local_rank == 0 else dataloader):
            density = batch[0].cuda(non_blocking=True)
            loss = model.module(density)
            dist.barrier()
            reduced_loss = reduce_mean(loss, args.nprocs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if local_rank == 0:
                pbar.set_description(f"Epoch {epoch+1} | Loss: {reduced_loss.item():.4f}")
                if args.not_save == False:
                    tb_writer.add_scalar('Train/Loss', reduced_loss.item(), epoch * len(dataloader) + i)
        
        
        if (epoch % 5 == 0 or epoch == args.epochs - 1) and local_rank == 0 and args.not_save == False:
            logging.info(f"Test model at epoch {epoch+1}")
            with torch.no_grad():
                model.module.eval()
                x_gen = model.module.sample(init_x)
                fig, ax = plt.subplots(2, 4, figsize=(10, 5))
                for i in range(2):
                    for j in range(4):
                        ax[i, j].imshow(x_gen[i * 4 + j, 0, :, :].detach().cpu().numpy(), cmap='gray', origin='lower')
                        ax[i, j].axis('off')
                tb_writer.add_figure('generated_images', fig, epoch+1)
    
    if local_rank == 0 and args.not_save == False:
        torch.save(model.module.state_dict(), '/media/bamf-big/gefan/DiffFluids/checkpoint/ddpm/' + args.dataset + "_diffuser.pkl")
        logging.info(f"Model saved")
        tb_writer.close()

if __name__ == '__main__':
    main()