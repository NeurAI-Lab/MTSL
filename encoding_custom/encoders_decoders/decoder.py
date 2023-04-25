import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding_custom.backbones.resnet import initialize_weights


def _sum_each(x, y):
    assert (len(x) == len(y))
    z = []
    for i in range(len(x)):
        z.append(x[i] + y[i])
    return z


class DecoderBlock(nn.Module):
    # ResNet Bottleneck with skips
    expansion = 4

    def __init__(self, inplanes, planes, norm_layer, stride=1, dilation=1,
                 downsample=None):
        super(DecoderBlock, self).__init__()
        if type(inplanes) is list:
            self.input_projection = nn.Conv2d(
                inplanes[1], planes * DecoderBlock.expansion,
                kernel_size=1, bias=False)
            inplanes = sum(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * DecoderBlock.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * DecoderBlock.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        inputs, skip = x
        inputs = F.interpolate(inputs, skip.size()[-2:],
                               mode="bilinear", align_corners=False)
        residual = inputs
        if hasattr(self, 'input_projection'):
            residual = self.input_projection(residual)

        x = torch.cat([inputs, skip], dim=1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(self, cfg, de_in_channels, norm_layer, decoder_type):
        super(Decoder, self).__init__()
        planes = cfg.MODEL.DECODER.OUTPLANES
        self.num_en_features = len(de_in_channels)
        self.blocks = nn.ModuleList()

        for i in range(self.num_en_features):
            if decoder_type == "resnetv1":
                self.blocks.append(DecoderBlock(de_in_channels[i],
                                                planes, norm_layer))
            else:
                assert False, "wrong decoder type"

        if cfg.MODEL.DECODER.INIT_WEIGHTS:
            initialize_weights(self, norm_layer)

    def forward(self, en_feats, de_feats=None):
        if type(de_feats) is list and len(de_feats) != 0:
            x = de_feats[-1]
        else:
            x = en_feats[-1]
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i]([x, en_feats[self.num_en_features - i - 1]])

            outs.append(x)
        return outs
