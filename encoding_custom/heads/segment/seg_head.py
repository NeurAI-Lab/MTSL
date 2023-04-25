import torch
import torch.nn as nn
import torch.nn.functional as F


class SegHead(nn.Module):
    def __init__(self, cfg, num_classes, num_en_features):
        super(SegHead, self).__init__()
        in_planes = cfg.MODEL.SEG.INPLANES
        out_planes = cfg.MODEL.SEG.OUTPLANES
        self.num_en_features = num_en_features

        self.convs = nn.ModuleList()
        for i in range(self.num_en_features):
            self.convs.append(nn.Conv2d(
                in_planes * 4, out_planes, kernel_size=1))

        self.final_conv = nn.Conv2d(
            out_planes * self.num_en_features, num_classes, kernel_size=1)

    def forward(self, inputs, mask_size, **kwargs):
        outs = []

        for i in range(self.num_en_features):
            x = self.convs[i](inputs[i])
            outs.append(F.interpolate(
                x, (mask_size[0] // 4, mask_size[1] // 4),
                mode="bilinear", align_corners=False))

        all_outs = torch.cat(outs, dim=1)
        out = self.final_conv(all_outs)

        out = F.interpolate(out, size=mask_size, mode='bilinear',
                            align_corners=False)
        return out
