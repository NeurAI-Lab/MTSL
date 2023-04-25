import torch
from torch import nn

from encoding_custom.nn.similarity import CKASimilarity


class SimilarityLoss(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(SimilarityLoss, self).__init__()

        self.similarity_type = cfg.MISC.SIMILARITY_TYPE
        if self.similarity_type == 'CKA':
            self.sim_fn = CKASimilarity(**kwargs)
        else:
            raise ValueError('Unknown similarity measure..')

        self.loss_w = kwargs.get('loss_w', None)
        if self.loss_w is None:
            self.loss_w = cfg.MISC.FEATURE_SIMILARITY_LOSS_W
        self.make_loss = kwargs.get('make_loss', True)

    def forward(self, features):
        similarity = 0
        total_sim = 0
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                feat_x = torch.mean(features[i], dim=(2, 3)).flatten(1)
                feat_y = torch.mean(features[j], dim=(2, 3)).flatten(1)
                similarity += self.sim_fn.calculate_similarity(feat_x, feat_y)
                total_sim += 1

        loss = similarity / total_sim
        if self.make_loss:
            if self.similarity_type == 'Procrustes':
                loss = torch.abs(loss)
            else:
                loss = 1 - loss
            loss = loss * self.loss_w
        return {'losses/feat_similarity': loss}, {}


class GenericLosses(nn.Module):

    def __init__(self, cfg, ignore_cfg=False, **kwargs):
        super(GenericLosses, self).__init__()
        if cfg.MISC.FEATURE_SIMILARITY_LOSS or ignore_cfg:
            self.sim_loss = SimilarityLoss(cfg, **kwargs)

    def forward(self, predictions, targets):
        generic_add, generic_record = {}, {}
        if hasattr(self, 'sim_loss'):
            add, record = self.sim_loss(
                list(predictions['activations'].values()))
            generic_add.update(add)
            generic_record.update(record)
        return generic_add, generic_record
