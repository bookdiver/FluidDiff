import torch
import torch.autograd as autograd

def ssm_loss(score_net, x, n_slices=1):
    dup_x = x.unsqueeze(0).expand(n_slices, *x.shape).contiguous().view(-1, *x.shape[1:])
    dup_x.requires_grad_(True)
    v = torch.rand_like(dup_x)
    v = v / torch.norm(v, dim=-1, keepdim=True)

    grad1 = score_net(dup_x)
    gradv = torch.sum(grad1 * v)
    loss1 = torch.sum(grad1 * v, dim=-1) ** 2 * 0.5
    grad2 = autograd.grad(gradv, dup_x, create_graph=True)[0]
    loss2 = torch.sum(v * grad2, dim=-1)

    loss1 = loss1.view(n_slices, -1).mean(dim=0)
    loss2 = loss2.view(n_slices, -1).mean(dim=0)
    loss = loss1 + loss2

    return loss.mean()

def ssm_vr_loss(score_net, x, n_particles=1):
    dup_x = x.unsqueeze(0).expand(n_particles, *x.shape).contiguous().view(-1, *x.shape[1:])
    dup_x.requires_grad_(True)
    vectors = torch.randn_like(dup_x)

    grad1 = score_net(dup_x)
    gradv = torch.sum(grad1 * vectors)
    grad2 = autograd.grad(gradv, dup_x, create_graph=True)[0]

    grad1 = grad1.view(dup_x.shape[0], -1)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.

    loss2 = torch.sum((vectors * grad2).view(dup_x.shape[0], -1), dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    
    return loss.mean()

def dsm_loss(score_net, x, sigma=1):
    perturbed_x = x + torch.randn_like(x) * sigma
    perturbed_x.requires_grad_(True)
    target =  1 / (sigma ** 2) * (x - perturbed_x)
    scores = score_net(perturbed_x)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0) / x.numel()

    return loss