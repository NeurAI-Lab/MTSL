from .segment_losses.seg_losses_all import cross_entropy_loss, \
    ClassBalancedSegmentationLosses
from .depth_losses import RMSE
from .aux_losses import BalancedBinaryCrossEntropyLoss, NormalsL1Loss
from .ae_losses import AELossModule


task_to_loss_fn = {
    'segment': {
        'default': ClassBalancedSegmentationLosses,
        'balanced': ClassBalancedSegmentationLosses,
        'cross_entropy': cross_entropy_loss},
    'depth': {
        'default': RMSE, 'rmse': RMSE},
    'sem_cont': {
        'default': BalancedBinaryCrossEntropyLoss,
        'binary_ce': BalancedBinaryCrossEntropyLoss},
    'sur_nor': {
        'default': NormalsL1Loss, 'l1_loss': NormalsL1Loss},
    'ae': {
        'default': AELossModule}}
