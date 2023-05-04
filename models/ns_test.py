import torch
import matplotlib
import matplotlib.pyplot as plt

from unet3d import Unet3D
from diffuser import GaussianDiffusion
from data import NaiverStokes_Dataset

matplotlib.use('Agg')

dataset = NaiverStokes_Dataset('../data/ns_data_T20_v1e-03_N200.mat')

device = torch.device('cuda:0')

model = Unet3D(
    channels=1,
    cond_channels=1,
    channel_mults=(1, 2, 4, 8, 16),
    init_conv_channels=32,
    init_conv_kernel_size=5 
)

diffusion_model = GaussianDiffusion(
    model=model,
    sample_size=(1, 20, 64, 64),
    timesteps=1000,
    objective='pred_x0',
    physics_loss_weight=0.0
).to(device)

def make_multiple_predictions(conds,
                              physics_loss_weight=0.0,
                              error_calibration=False):
    if error_calibration and physics_loss_weight != 0.0:
        ckpt = torch.load(f'./ckpts/ns_{physics_loss_weight:.2f}phyloss_resdiff/ckpt.pt',
                            map_location={'cuda:1':'cuda:0'})
    else:
        ckpt = torch.load(f'./ckpts/ns_{physics_loss_weight:.2f}phyloss/ckpt.pt',
                            map_location={'cuda:1':'cuda:0'})

    diffusion_model.model.load_state_dict(ckpt['model_state_dict'])
    print("Load weight successfully!")

    diffusion_model.model.eval()
    predictions = []
    for i, cond in enumerate(conds):
        print(f"Starting prediction {i+1}/{len(conds)}")
        pred = diffusion_model.sample(cond=cond.unsqueeze(0))
        pred = pred.detach().cpu()
        predictions.append(pred)
    
    return torch.stack(predictions, dim=0)


def cal_nRMSE(xs, x_preds):
    assert len(xs.shape) == len(x_preds.shape) == 4
    rmse = torch.sqrt(torch.sum((xs - x_preds)**2, dim=(1, 2, 3)) / torch.sum(xs**2, dim=(1, 2, 3)))
    mean = torch.mean(rmse)
    std = torch.std(rmse)

    return {
        'mean': mean.item(),
        'std': std.item()
    }


def cal_temporal_mean_abs_error(xs, x_preds):
    assert len(xs.shape) == len(x_preds.shape) == 4
    t_mean_abs_error = torch.mean(torch.abs(xs - x_preds), dim=(2, 3))
    mean = torch.mean(t_mean_abs_error, dim=0)
    std = torch.std(t_mean_abs_error, dim=0)

    return {
        'mean': mean,
        'std': std
    }

if __name__ == "__main__":
    ws = dataset[10:11]['x']
    conds = dataset[10:11]['y']

    # chuck ws into 8 pieces and append them to a list
    ws = torch.split(ws, 1, dim=0)
    conds = torch.split(conds, 1, dim=0)

    # x_preds_000 = make_multiple_predictions(conds, physics_loss_weight=0.0, error_calibration=False)
    # print("x_preds_000.shape:", x_preds_000.shape)
    # x_preds_010 = make_multiple_predictions(conds, physics_loss_weight=0.10, error_calibration=False)
    # print("x_preds_010.shape:", x_preds_010.shape)
    # x_preds_020 = make_multiple_predictions(conds, physics_loss_weight=0.20, error_calibration=False)
    # print("x_preds_020.shape:", x_preds_020.shape)
    # x_preds_050 = make_multiple_predictions(conds, physics_loss_weight=0.50, error_calibration=False)
    # print("x_preds_050.shape:", x_preds_050.shape)
    # x_preds_100 = make_multiple_predictions(conds, physics_loss_weight=1.00, error_calibration=False)
    # print("x_preds_100.shape:", x_preds_100.shape)

    # x_preds_000_EC = x_preds_000.clone()
    # print("x_preds_000_EC.shape:", x_preds_000_EC.shape)
    # x_preds_010_EC = make_multiple_predictions(conds, physics_loss_weight=0.10, error_calibration=True)
    # print("x_preds_010_EC.shape:", x_preds_010_EC.shape)
    x_preds_020_EC = make_multiple_predictions(conds, physics_loss_weight=0.20, error_calibration=True)
    print("x_preds_020_EC.shape:", x_preds_020_EC.shape)
    # x_preds_050_EC = make_multiple_predictions(conds, physics_loss_weight=0.50, error_calibration=True)
    # print("x_preds_050_EC.shape:", x_preds_050_EC.shape)
    # x_preds_100_EC = make_multiple_predictions(conds, physics_loss_weight=1.00, error_calibration=True)
    # print("x_preds_100_EC.shape:", x_preds_100_EC.shape)

    x_span = torch.linspace(0, 1, 64)
    y_span = torch.linspace(0, 1, 64)
    X, Y = torch.meshgrid(x_span, y_span, indexing='ij')

    print("Plotting...")
    fig1, ax1 = plt.subplots(4, 4, figsize=(24, 20))
    im1 = ax1[0, 0].pcolormesh(X, Y, conds[0].squeeze(), cmap='jet')
    ax1[0, 0].set_xlabel('$x$')
    ax1[0, 0].set_ylabel('$y$')
    ax1[0, 0].set_title('Initial condition')
    fig1.colorbar(im1, ax=ax1[0, 0])
    plt.tight_layout()

    ax1[0, 1].set_axis_off()
    ax1[0, 2].set_axis_off()
    ax1[0, 3].set_axis_off()

    for i in range(4):
        im2 = ax1[1, i].pcolormesh(X, Y, ws[0][0, (i+1)*5-1, :, :], cmap='jet')
        ax1[1, i].set_xlabel('$x$')
        ax1[1, i].set_ylabel('$y$')
        ax1[1, i].set_title(f'Ground truth at $t={(i+1)*5}s$')
        fig1.colorbar(im2, ax=ax1[1, i])
        plt.tight_layout()

        im3 = ax1[2, i].pcolormesh(X, Y, x_preds_020_EC[0, 0, (i+1)*5-1, :, :], cmap='jet')
        ax1[2, i].set_xlabel('$x$')
        ax1[2, i].set_ylabel('$y$')
        ax1[2, i].set_title(f'Prediction at $t={(i+1)*5}s$')
        fig1.colorbar(im3, ax=ax1[2, i])
        plt.tight_layout()

        im4 = ax1[3, i].pcolormesh(X, Y, torch.abs(x_preds_020_EC[0, 0, (i+1)*5-1, :, :] - ws[0][0, (i+1)*5-1, :, :]), cmap='jet')
        ax1[3, i].set_xlabel('$x$')
        ax1[3, i].set_ylabel('$y$')
        ax1[3, i].set_title(f'Absolute error at $t={(i+1)*5}s$')
        fig1.colorbar(im4, ax=ax1[3, i])
        plt.tight_layout()

    fig1.savefig('../assets/ns_exp1.png', bbox_inches='tight')



    
