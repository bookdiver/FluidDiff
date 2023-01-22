import torch
import torch.nn as nn
from torch.autograd import grad

class PINN(nn.Module):
    def __init__(self, 
                in_features: int=3,
                out_features: int=4,
                n_units: list=[32, 32, 32, 32, 32, 32, 32, 32, 32]):
        super().__init__()
        self.net = nn.Sequential()
        in_unit = in_features
        for i, unit in enumerate(n_units):
            self.net.add_module(f"fc{i}", nn.Linear(in_unit, unit))
            self.net.add_module(f"act{i}", nn.Tanh())
            # if i % 2 == 0:
            #     self.net.add_module(f"drop{i}", nn.Dropout(0.5))
            in_unit = unit
        self.net.add_module("fc", nn.Linear(in_unit, out_features))
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x, y, t):
        X = torch.cat([x, y, t], dim=1).requires_grad_(True)
        out = self.net(X)
        return out
    
    def eq_loss(self, x, y, t):
        predict_out = self.forward(x, y, t)
        psi = predict_out[:, 0].reshape(-1, 1)
        u = grad(psi.sum(), y, create_graph=True)[0]
        v = -grad(psi.sum(), x, create_graph=True)[0]

        p = predict_out[:, 1].reshape(-1, 1)
        # d = predict_out[:, 2].reshape(-1, 1)
        
        u_t = grad(u.sum(), t, create_graph=True)[0]  # du/dt
        v_t = grad(v.sum(), t, create_graph=True)[0]  # dv/dt
        # d_t = grad(d.sum(), t, create_graph=True)[0]  # drho/dt

        u_x = grad(u.sum(), x, create_graph=True)[0]  # du/dx
        v_x = grad(v.sum(), x, create_graph=True)[0]  # dv/dx
        p_x = grad(p.sum(), x, create_graph=True)[0]  # dp/dx
        # d_x = grad(d.sum(), x, create_graph=True)[0]  # drho/dx

        u_y = grad(u.sum(), y, create_graph=True)[0]  # du/dy
        v_y = grad(v.sum(), y, create_graph=True)[0]  # dv/dy
        p_y = grad(p.sum(), y, create_graph=True)[0]  # dp/dy
        # d_y = grad(d.sum(), y, create_graph=True)[0]  # drho/dy

        u_xx = grad(u_x.sum(), x, create_graph=True)[0]  # d^2u/dx^2
        v_xx = grad(v_x.sum(), x, create_graph=True)[0]  # d^2v/dx^2
        u_yy = grad(u_y.sum(), y, create_graph=True)[0]  # d^2u/dy^2
        v_yy = grad(v_y.sum(), y, create_graph=True)[0]  # d^2v/dy^2

        eq1 = u_t + u*u_x + v*u_y + p_x - 0.03*(u_xx+u_yy)
        eq2 = v_t + u*v_x + v*v_y + p_y + 0.5 - 0.03*(v_xx+v_yy)
        # eq3 = d_t + u * d_x + v * d_y

        batch_zeros = torch.zeros_like(eq1)
        loss = self.loss_fn(eq1, batch_zeros) + \
                self.loss_fn(eq2, batch_zeros)
                # self.loss_fn(eq3, batch_zeros)
        return loss
    
    def data_loss(self, x, y, t, u, v, p):
        predict_out = self.forward(x, y, t)
        psi = predict_out[:, 0].reshape(-1, 1)
        u_pred = grad(psi.sum(), y, create_graph=True)[0]
        v_pred = -grad(psi.sum(), x, create_graph=True)[0]

        p_pred = predict_out[:, 1].reshape(-1, 1)
        # d_pred = predict_out[:, 2].reshape(-1, 1)
        
        loss = self.loss_fn(u_pred, u) + \
                self.loss_fn(v_pred, v) + \
                self.loss_fn(p_pred, p)
                # self.loss_fn(d_pred, d)
        return loss
    

def _test():
    net = PINN(in_features=3, out_features=2)
    # print(net)
    print(f"The number of parameters: {sum(p.numel() for p in net.parameters())}")
    data = torch.randn(64*64, 7).requires_grad_(True)
    x, y, t, u, v, p, _ = torch.chunk(data, 7, dim=1)
    print(f"Loss of eq: {net.eq_loss(x, y, t)}")
    print(f"Loss of data: {net.data_loss(x, y, t, u, v, p)}")

if __name__ == "__main__":
    _test()
