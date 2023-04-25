from torch import nn


class AELossModule(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(AELossModule, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.l1_loss = nn.L1Loss()
        if cfg.MODEL.AE.RECONST_LOSS == 'mse':
            self.reconst_loss = nn.MSELoss()
        elif cfg.MODEL.AE.RECONST_LOSS == 'l1_loss':
            self.reconst_loss = nn.L1Loss()
        else:
            raise ValueError('Unknown reconstruction loss..')

    def forward(self, pred, target):
        reconst = pred['reconst']
        reconst_loss = self.reconst_loss(reconst, target)
        losses = {'ae': {'losses/ae_reconst_loss': reconst_loss}}
        return losses
