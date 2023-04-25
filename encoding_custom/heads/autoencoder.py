import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding_custom.nn.common import conv1x1, conv3x3


class AutoEncoderHead(nn.Module):
    def __init__(self, cfg, num_features, **kwargs):
        super(AutoEncoderHead, self).__init__()

        in_planes = cfg.MODEL.AE.INPLANES
        out_planes = cfg.MODEL.AE.OUTPLANES
        self.num_features = num_features

        self.convs = nn.ModuleList()
        for i in range(self.num_features):
            self.convs.append(conv1x1(in_planes * 4, out_planes, bias=False))
        self.final_conv = conv3x3(in_planes * self.num_features, 3, bias=True)

    def forward(self, decoder_features, mask_size, **kwargs):
        outs = []
        for i in range(self.num_features):
            x = self.convs[i](decoder_features[i])
            outs.append(
                F.interpolate(x, (mask_size[0] // 4, mask_size[1] // 4),
                              mode="bilinear", align_corners=False))
        all_outs = torch.cat(outs, dim=1)
        out = self.final_conv(all_outs)

        out = F.interpolate(out, size=mask_size, mode='bilinear',
                            align_corners=False)
        return {'reconst': out, 'en_feats': kwargs.get('encoder_features', None)}
