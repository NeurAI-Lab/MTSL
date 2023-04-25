from torch import nn
import torch
import torch.nn.functional as F

__all__ = ['RMSE']


class RMSE(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(RMSE, self).__init__()

    def forward(self, pred, target, unsqueeze=True):
        if unsqueeze:
            target = target.unsqueeze(1)

        if not pred.shape == target.shape:
            print(pred.shape)
            print(target.shape)
            _, _, H, W = target.shape
            pred = F.upsample(pred, size=(H, W), mode='bilinear')

        mask = torch.where(target > 0)
        loss = torch.sqrt(torch.mean(torch.abs(target[mask] - pred[mask]) ** 2))
        return loss
