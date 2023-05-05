import scipy.io as sio
import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
from pyDOE import lhs

# data_path = '../../data/ns_data_T20_v1e-03_N200(complete).mat'
data_path = '/home/gefan/FluidDiff/data/ns_data_T20_v1e-03_N200(complete).mat'
save_name = '../../ckpts/pinn/ns/T20_v1e-3_N200'

device = torch.device('cuda:0')
# device = torch.device('cpu')

x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
t_min, t_max = 0.0, 20.0

N_u = 100
N_f = 5000

data = sio.loadmat(data_path)
w = torch.from_numpy(data['w_complete']).squeeze()
w0 = torch.from_numpy(data['a']).squeeze()

bsize, nx, ny, nt = w.shape

x = torch.linspace(x_min, x_max, nx+1)[:-1]
y = torch.linspace(y_min, y_max, ny+1)[:-1]
X, Y = torch.meshgrid(x, y, indexing='ij')
XY = torch.stack([X.flatten(), Y.flatten()], dim=1)
t = torch.linspace(t_min, t_max, nt+1)[1:]

def create_pinn_data(w, w0, n_sample):
    # initial condition w(x, y, 0)
    ic_idx = torch.randperm(nx*ny)[:N_u]
    xy_ic = XY[ic_idx]
    t_ic = torch.zeros(N_u, 1)
    xyt_ic = torch.cat([xy_ic, t_ic], dim=1)
    w_ic = w0.flatten(start_dim=1, end_dim=2)[n_sample, ic_idx].unsqueeze(1)

    # boundary condition w(0, y, t) = w(1, y, t), w(x, 0, t) = w(x, 1, t)
    xy_bc = torch.zeros(N_u, 2)
    xy_bc[N_u//4:2*N_u//4, 0] = 1
    xy_bc[2*N_u//4:3*N_u//4, 1] = 1
    xy_bc[3*N_u//4:, 0] = 1
    xy_bc[3*N_u//4:, 1] = 1
    bc_idx = torch.randperm(nt)[:N_u//4]
    t_bc = t[bc_idx].reshape(-1, 1)
    t_bc = torch.cat([t_bc, t_bc, t_bc, t_bc])
    xyt_bc = torch.cat([xy_bc, t_bc], dim=1)
    w_bc1 = w[n_sample, 0, 0, bc_idx]
    w_bc2 = w[n_sample, 0, -1, bc_idx]
    w_bc3 = w[n_sample, -1, 0, bc_idx]
    w_bc4 = w[n_sample, -1, -1, bc_idx]
    w_bc = torch.cat([w_bc1, w_bc2, w_bc3, w_bc4]).unsqueeze(1)

    xyt_u = torch.cat([xyt_ic, xyt_bc], dim=0)
    w_u = torch.cat([w_ic, w_bc], dim=0)

    xyt_u = xyt_u.to(torch.float32).to(device)
    w_u = w_u.to(torch.float32).to(device)

    x_f = torch.from_numpy(x_min + (x_max - x_min) * lhs(1, N_f))
    y_f = torch.from_numpy(y_min + (y_max - y_min) * lhs(1, N_f))
    t_f = torch.from_numpy(t_min + (t_max - t_min) * lhs(1, N_f))
    xyt_f = torch.cat([x_f, y_f, t_f], dim=1).to(torch.float32).to(device)

    return xyt_u, w_u, xyt_f

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

    def forward(self, xyt):
        xyt = self.init_fc(xyt)

        for fc in self.net:
            xyt = fc(xyt)
        
        psi = self.out_fc(xyt)

        return psi


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

class PINN:
    def __init__(self):
        self.net = DNN(dim_in=3, 
                       dim_out=1, 
                       n_nodes=(20, 20, 20, 20, 20, 20, 20)
                    ).to(device)

        self.optimizer = None
        self.iter = 0
        self.xyt_u = None
        self.w_u = None
        self.xyt_f = None

    def reset(self, xyt_u, w_u, xyt_f):
        self.xyt_u = xyt_u
        self.w_u = w_u
        self.xyt_f = xyt_f
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

    def f(self, xyt, visc=1e-3):
        xyt = xyt.clone()
        xyt.requires_grad = True

        psi = self.net(xyt)
        psi_xyt = grad(psi.sum(), xyt, create_graph=True)[0]
        psi_x = psi_xyt[:, 0:1]
        psi_y = psi_xyt[:, 1:2]

        psi_xx = grad(psi_x.sum(), xyt, create_graph=True)[0][:, 0:1]
        psi_yy = grad(psi_y.sum(), xyt, create_graph=True)[0][:, 1:2]

        u = psi_y
        v = -psi_x
        w = -(psi_xx + psi_yy)

        w_xyt = grad(w.sum(), xyt, create_graph=True)[0]
        w_x = w_xyt[:, 0:1]
        w_y = w_xyt[:, 1:2]
        w_t = w_xyt[:, 2:3]

        w_xx = grad(w_x.sum(), xyt, create_graph=True)[0][:, 0:1]
        w_yy = grad(w_y.sum(), xyt, create_graph=True)[0][:, 1:2]

        force = 0.1 * (torch.sin(2 * np.pi * (xyt[:, 0:1] + xyt[:, 1:2])) + torch.cos(2 * np.pi * (xyt[:, 0:1] + xyt[:, 1:2])))

        f = w_t + u * w_x + v * w_y - visc * (w_xx + w_yy) - force
        return f
    
    def u(self, xyt_u):
        xyt_u = xyt_u.clone()
        xyt_u.requires_grad = True

        psi = self.net(xyt_u)
        psi_xyt = grad(psi.sum(), xyt_u, create_graph=True)[0]
        psi_x = psi_xyt[:, 0:1]
        psi_y = psi_xyt[:, 1:2]

        psi_xx = grad(psi_x.sum(), xyt_u, create_graph=True)[0][:, 0:1]
        psi_yy = grad(psi_y.sum(), xyt_u, create_graph=True)[0][:, 1:2]

        w = -(psi_xx + psi_yy)

        return w - self.w_u

    def closure(self):
        self.optimizer.zero_grad()

        u_pred = self.u(self.xyt_u)
        f_pred = self.f(self.xyt_f)

        mse_u = torch.mean(torch.square(u_pred))
        mse_f = torch.mean(torch.square(f_pred))

        loss = mse_u + mse_f
        loss.backward()

        self.iter += 1
        print(f"\r{self.iter} loss : {loss.item():.3e}", end="")
        if self.iter % 500 == 0:
            print("")
        return loss

if __name__ == '__main__':
    n_test = 1
    pinn = PINN()

    for n_sample in range(n_test):
        print("-"*40)
        print(f"Start training {n_sample+1}th sample")
        xyt_u, w_u, xyt_f = create_pinn_data(w, w0, n_sample)
        pinn.reset(xyt_u, w_u, xyt_f)
        pinn.optimizer.step(pinn.closure)
        torch.save(pinn.net.state_dict(), save_name+f'_{n_sample}.pt')
