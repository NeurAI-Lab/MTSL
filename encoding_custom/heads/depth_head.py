import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding_custom.nn.common import conv1x1, conv3x3


class DepthHead(nn.Module):
    def __init__(self, cfg, out_ch, num_features):
        super(DepthHead, self).__init__()

        in_planes = cfg.MODEL.DEPTH.INPLANES
        out_planes = cfg.MODEL.DEPTH.OUTPLANES
        if cfg.MODEL.DEPTH.ACTIVATION_FN == 'relu':
            self.activation_fn = nn.ReLU(inplace=True)
        elif cfg.MODEL.DEPTH.ACTIVATION_FN == 'sigmoid':
            self.activation_fn = torch.sigmoid
        else:
            raise ValueError('Unknown activation function..')
        self.num_features = num_features

        self.convs = nn.ModuleList()
        for i in range(self.num_features):
            self.convs.append(conv1x1(in_planes * 4, out_planes, bias=False))

        self.depth = conv3x3(in_planes * self.num_features, out_ch, bias=True)

    def forward(self, inputs, mask_size, **kwargs):
        outs = []

        for i in range(self.num_features):
            x = self.convs[i](inputs[i])
            outs.append(
                F.interpolate(x, (mask_size[0] // 4, mask_size[1] // 4),
                              mode="bilinear", align_corners=False))
        all_outs = torch.cat(outs, dim=1)
        out = self.depth(all_outs)
        out = self.activation_fn(out)

        out = F.interpolate(out, size=mask_size, mode='bilinear',
                            align_corners=False)
        return out
