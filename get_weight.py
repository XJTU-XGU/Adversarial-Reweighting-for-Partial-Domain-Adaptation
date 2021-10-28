import torch
import numpy as np
import cvxpy as cvx
from network import WassersteinDiscriminator
from torch.utils.data import DataLoader,TensorDataset
from torch.autograd import grad
import tqdm
import random

def gradient_penalty(critic, h_s, h_t):
    alpha = torch.rand(h_s.size(0), 1).to(h_s.device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()
    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty

def get_weight(feature_source,feature_target,rho=5.0,seed=None,max_step=6000,automatical_adjust=True,up=6.0,low=-6.0,
               step=None,multi_process=False,c=1.2):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    if multi_process:
        loader_s = DataLoader(TensorDataset(feature_source), batch_size=36, shuffle=True, drop_last=True,num_workers=4)
        loader_t = DataLoader(TensorDataset(feature_target), batch_size=36, shuffle=True, drop_last=True,num_workers=4)
    else:
        loader_s = DataLoader(TensorDataset(feature_source), batch_size=36, shuffle=True, drop_last=True)
        loader_t = DataLoader(TensorDataset(feature_target), batch_size=36, shuffle=True, drop_last=True)

    adnet = WassersteinDiscriminator(feature_source.size(1), 1024).cuda()
    optimizer = torch.optim.Adam(adnet.parameters(), lr=0.001)
    num_steps = max_step

    for i in tqdm.trange(num_steps):
        if i % len(loader_s) == 0:
            iter_s = iter(loader_s)
        if i % len(loader_t) == 0:
            iter_t = iter(loader_t)
        feat_s = iter_s.__next__()[0].cuda()
        feat_t = iter_t.__next__()[0].cuda()
        out_s = adnet(feat_s)
        out_t = adnet(feat_t)
        gp = gradient_penalty(adnet, feat_s, feat_t)
        wdist = out_s.mean() - out_t.mean()
        lam = i / num_steps * 100.0
        loss = -wdist + lam * gp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    outs_s = adnet(feature_source.cuda()).cpu()
    outs_t = adnet(feature_target.cuda()).cpu()

    ds = np.reshape(outs_s.data.numpy(), (-1,))
    dt = np.reshape(outs_t.data.numpy(), (-1,))
    n = len(ds)
    w = cvx.Variable(n)
    ones = np.ones(n)
    obj = cvx.Minimize(w @ ds)

    if automatical_adjust:
        t = 0
        while True:
            t += 1
            con = [w >= 0,
                   cvx.sum_squares(w - ones) <= rho * n,
                   cvx.sum(w) == n,
                   ]
            prob = cvx.Problem(obj, con)
            prob.solve(cvx.ECOS, max_iters=500)
            op_wdist = w.value @ ds / n - np.mean(dt)
            print("The {:d}st time adjusting rho, rho = {:.1f}".format(t, rho))
            print("status:", prob.status)
            print("original dist:", np.mean(ds) - np.mean(dt))
            print("optimal dist:", op_wdist)
            if op_wdist > up:
                rho *= 1.2
            elif op_wdist > low:
                break
            else:
                rho /= 1.2
    else:
        rho = np.max([1.0 + rho * (1.0 - (step / max_step)), 3.0])
        con = [w >= 0,
               cvx.sum_squares(w - ones) <= rho * n,
               cvx.sum(w) == n,
               ]
        prob = cvx.Problem(obj, con)
        prob.solve(cvx.ECOS)
        op_wdist = w.value @ ds / n - np.mean(dt)
        print("rho = {:.1f}".format(rho))
        print("status:", prob.status)
        print("original dist:", np.mean(ds) - np.mean(dt))
        print("optimal dist:", op_wdist)
    weight = w.value
    return weight