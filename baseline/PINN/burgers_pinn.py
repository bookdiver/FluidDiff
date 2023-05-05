import scipy.io as sio
import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
from pyDOE import lhs

data_path = '../../data/burgers_data_Nt100_v1e-02_N200.mat'
save_name = '../../ckpts/pinn/burgers/Nt100_v1e-02_N200'

device = torch.device('cuda:0')

x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0

N_u = 100
N_f = 5000

data = sio.loadmat(data_path)
u = torch.from_numpy(data['u']).squeeze()
u0 = torch.from_numpy(data['a']).squeeze()
bsize, nt, nx = u.shape

x = torch.linspace(0, 1, nx+1)[:-1]
t = torch.linspace(0, 1, nt+1)[1:]

def create_pinn_data(u, u0, n_sample):
    # initial condition u(x, 0)
    ic_idx = torch.randperm(nx)[:N_u]
    x_ic = x[ic_idx]
    t_ic = torch.zeros(N_u)
    xt_ic = torch.stack([x_ic, t_ic], dim=1)
    u_ic = u0[n_sample, ic_idx].unsqueeze(1)

    # boundary condition u(0, t) = u(1, t)
    x_bc = torch.zeros(N_u)
    x_bc[N_u//2:] = 1
    bc_idx = torch.randperm(nt)[:N_u//2]
    t_bc = t[bc_idx]
    t_bc = torch.cat([t_bc, t_bc])
    xt_bc = torch.stack([x_bc, t_bc], dim=1)

    u_bc1 = u[n_sample, bc_idx, 0]
    u_bc2 = u[n_sample, bc_idx, -1]
    u_bc = torch.cat([u_bc1, u_bc2]).unsqueeze(1)

    xt_u = torch.cat([xt_ic, xt_bc], dim=0)
    u_u = torch.cat([u_ic, u_bc], dim=0)

    xt_u = xt_u.to(torch.float32).to(device)
    u_u = u_u.to(torch.float32).to(device)

    x_f = torch.from_numpy(x_min + (x_max - x_min) * lhs(1, N_f))
    t_f = torch.from_numpy(t_min + (t_max - t_min) * lhs(1, N_f))
    xt_f = torch.cat([x_f, t_f], dim=1).to(torch.float32).to(device)

    return xt_u, u_u, xt_f

class layer(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.layer = nn.Linear(dim_in, dim_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x


class DNN(nn.Module):
    def __init__(self, 
                 dim_in: int, 
                 dim_out: int,
                 n_nodes: tuple, 
                 activation: nn.Module=nn.Tanh()):
        super().__init__()
        
        self.init_fc = layer(dim_in, n_nodes[0], activation)

        self.net = nn.ModuleList()
        for i in range(len(n_nodes) - 1):
            self.net.append(layer(n_nodes[i], n_nodes[i+1], activation))

        self.out_fc = layer(n_nodes[-1], dim_out, None)
    
    def weight_init(self):
        self.init_fc.apply(weights_init)
        self.net.apply(weights_init)
        self.out_fc.apply(weights_init)

    def forward(self, xt):
        xt = self.init_fc(xt)

        for fc in self.net:
            xt = fc(xt)
        
        u = self.out_fc(xt)

        return u


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

class PINN:
    def __init__(self):
        self.net = DNN(dim_in=2, 
                       dim_out=1, 
                       n_nodes=(20, 20, 20, 20, 20, 20, 20)
                    ).to(device)

        self.optimizer = None
        self.iter = 0
        self.xt_u = None
        self.u_u = None
        self.xt_f = None

    def reset(self, xt_u, u_u, xt_f):
        self.xt_u = xt_u
        self.u_u = u_u
        self.xt_f = xt_f
        self.iter = 0
        self.net.weight_init()
        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        print("PINN initialized")

    def f(self, xt, visc=1e-2):
        xt = xt.clone()
        xt.requires_grad = True

        u = self.net(xt)

        u_xt = grad(u.sum(), xt, create_graph=True)[0]
        u_x = u_xt[:, 0:1]
        u_t = u_xt[:, 1:2]

        u_xx = grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]

        f = u_t +  u * u_x - visc * u_xx
        return f

    def closure(self):
        self.optimizer.zero_grad()

        u_pred = self.net(self.xt_u)
        f_pred = self.f(self.xt_f)

        mse_u = torch.mean(torch.square(u_pred - self.u_u))
        mse_f = torch.mean(torch.square(f_pred))

        loss = mse_u + mse_f
        loss.backward()

        self.iter += 1
        print(f"\r{self.iter} loss : {loss.item():.3e}", end="")
        if self.iter % 500 == 0:
            print("")
        return loss

if __name__ == '__main__':
    n_test = 50
    pinn = PINN()

    for n_sample in range(4, n_test):
        print(f"Start training {n_sample+1}th sample")
        xt_u, u_u, xt_f = create_pinn_data(u, u0, n_sample)
        pinn.reset(xt_u, u_u, xt_f)
        pinn.optimizer.step(pinn.closure)
        torch.save(pinn.net.state_dict(), save_name+f'_{n_sample}.pt')
        print("-"*40)