import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from net import ConditionalUnet
from data import MyDataSet

class NSCN:
    def __init__(self, in_channels: int, n_feats: int, sigmas: list, n_sigmas: int):
        """ NSCN model

        Args:
            in_channels (int): input channels, 1 for only input density, 3 for input density and velocity
            n_feats (int): hidden channels, for embedding and conditioning
            sigmas (list): bounds for sigmas in noise schedule
            n_sigmas (int): number of sigmas used in perturbation
        """
        self.net = ConditionalUnet(in_channels, n_feats)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        else:
            raise ValueError("The model must be trained on multiple GPUs!")
        
        self.sigmas = sigmas
        self.n_sigmas = n_sigmas
        self.schedule = self.nscn_schedule()
        self.mse_loss = nn.MSELoss()
    
    def ncsn_schedule(self) -> dict:
        """ Pre-computed schedules for NSCN training.

        Returns:
            (dict): NSCN schedules
        """
        sigma1, sigma2 = self.sigmas
        assert sigma1 < sigma2
        sigma_is = torch.logspace(np.log10(sigma1), np.log10(sigma2), self.n_sigmas, dtype=torch.float32)
        oneover_sigma_sq = 1 / (sigma_is ** 2)
        sigma_sq = sigma_is ** 2
        return {
            'sigma_is': sigma_is,  # \sigma_i
            'oneover_sigma_sq': oneover_sigma_sq,  # 1/\sigma_i^2
            'sigma_sq': sigma_sq,  # \sigma_i^2
            'L': self.n_sigmas,  # number of sigmas
        }

    def loss(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """ This function is to compute the loss of denoising score matching in NCSN.
        NOTE: the empirical mean over all the sigmas is computed by uniformly sampling the sigmas.

        Args:
            x (torch.Tensor): (B, 1, H, W), original frame (TODO: now it's only the density, need add velocity as additional channel)
            c (torch.Tensor): (B, 3), additional conditioning information, contains (real t, src_pos_x, src_pos_y)

        Returns:
            loss (torch.Tensor): NSCN loss for reconstructing the score of perturbed data
        """        
        i = np.random.randint(self.schedule['L'])
        sigma_sq = self.schedule['sigma_sq'][i, None, None, None].cuda()
        oneover_sigma_sq = self.schedule['oneover_sigma_sq'][i, None, None, None].cuda()

        # perturb the input data
        x_perturbed = x + sigma_sq * torch.randn_like(x)
        # compute the score of the perturbed data
        score = (x - x_perturbed) * oneover_sigma_sq
        loss = 0.5 * sigma_sq * self.mse_loss(self.net(x_perturbed, c), score)
        return loss

    def sample(self, x_seed: torch.Tensor, c: torch.Tensor, eps: float=1e-4, n_T: int=200) -> torch.Tensor:
        """ Annealed Langevin dynamics sampling

        Args:
            x_seed (torch.Tensor): (B, 1, H, W), initial frame, pure noise
            c (torch.Tensor): (B, 3), additional conditioning information, contains (real t, src_pos_x, src_pos_y), can be multiple conditions in batch dim
            eps (float, optional): step learning rate. Defaults to 1e-4.
            n_T (int, optional): the number of reverse diffusion steps. Defaults to 200.
            TODO: use Predictor-Corrector scheme to improve the sampling performance (need to know the SDE)

        Returns:
            x_gen (torch.Tensor): generated samples
        """
        assert x_seed.shape[0] == c.shape[0]

        with torch.no_grad():
            x_gen = x_seed
            for sigma in tqdm(self.schedule['sigma_is'], desc='annealed Langevin dynamics sampling'):
                step_size = eps * (sigma/self.schedule['sigma_is'][-1]) ** 2
                for _ in range(n_T):
                    x_gen = x_gen + 0.5 * eps * self.net(x_gen, c) + sigma * torch.randn_like(x_gen)
