import torch
from torch.nn import functional as F

def nll_loss(y_hat, y, reduce=True):
    y_hat = y_hat.permute(0,2,1)
    y = y.squeeze(-1)
    loss = F.nll_loss(y_hat, y)
    return loss