import torch.nn as nn

from encoding_custom.nn.common import conv3x3
from encoding_custom.heads.depth_head import DepthHead
from encoding_custom.heads.segment.seg_head import SegHead


class SurfaceNormalHead(DepthHead):
    def __init__(self, cfg, num_features):
        super(SurfaceNormalHead, self).__init__(cfg, 1, num_features)

        in_planes = cfg.MODEL.DEPTH.INPLANES
        self.activation_fn = nn.Tanh()
        self.depth = conv3x3(in_planes * self.num_features, 3, bias=True)


class SemanticContourHead(SegHead):
    def __init__(self, cfg, num_classes, num_en_features):
        super(SemanticContourHead, self).__init__(cfg, num_classes,
                                                  num_en_features)
