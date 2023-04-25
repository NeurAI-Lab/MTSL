import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

__all__ = ["ClassBalancedSegmentationLosses", 'cross_entropy_loss']


def cross_entropy_loss(cfg, *args, **kwargs):
    return nn.functional.cross_entropy


class ClassBalancedSegmentationLosses(nn.Module):
    def __init__(self, cfg, se_loss=False, se_weight=0.2,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=-1, num_outputs=1, aux_indexes=[], beta=1 - 1e-3,
                 **kwargs):
        super(ClassBalancedSegmentationLosses, self).__init__()

        self.se_loss = se_loss
        self.aux = aux
        self.nclass = cfg.NUM_CLASSES.SEGMENT
        self.se_weight = se_weight
        self.bceloss = nn.BCELoss(weight)
        self.normal_loss_indexes = [i for i in range(num_outputs) if i not in aux_indexes]
        self.aux_indexes = aux_indexes
        self.num_outputs = num_outputs
        self.beta = beta

        if len(aux_indexes) > 0:
            self.aux_weight_per_loss = aux_weight / len(aux_indexes)
        else:
            self.aux_weight_per_loss = 0

        self.ignore_index = ignore_index

    def forward(self, *inputs, ignore_index=-1):
        loss = 0
        weights = self._class_balanced_weights(inputs[-1], self.nclass, self.beta).type_as(inputs[0])
        for ind in self.normal_loss_indexes:
            pred = inputs[ind]
            loss += F.cross_entropy(pred, inputs[-1].long(), weight=weights, ignore_index=-1)

        return loss

    @staticmethod
    def _class_balanced_weights(target, nclass, beta=1 - 1e-3):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = torch.zeros(batch, nclass)
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            # vect = hist>0
            tvect[i] = hist
        tvect_sum = torch.sum(tvect, 0)
        tvect_sum = (1 - beta) / (1 - beta ** (tvect_sum))
        tvect_sum[tvect_sum == np.inf] = 0
        return tvect_sum
