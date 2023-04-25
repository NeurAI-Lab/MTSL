import torch.nn as nn
from collections import OrderedDict

from encoding_custom.backbones.resnet import Bottleneck, initialize_weights

__all__ = ['resnet_encoder', 'ResNetEncoder']


class LayerSpec:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


class ResNetEncoder(nn.Module):
    """
    Modified ResNet
    """
    def __init__(self, inplanes, cfg, block, num_blocks, dilated=False,
                 norm_layer=nn.BatchNorm2d, dilate_only_last_layer=False):
        super(ResNetEncoder, self).__init__()
        self.inplanes = inplanes
        self.norm_layer = norm_layer
        self.block = block
        self.feat_channels = []

        out_ch_before_exp = cfg.MODEL.ENCODER.OUT_CHANNELS_BEFORE_EXPANSION
        num_en_features = cfg.MODEL.ENCODER.NUM_EN_FEATURES
        self.conv_layer = nn.Conv2d
        layer_spec = OrderedDict()
        layer_spec.update(
            {'layer1': LayerSpec([out_ch_before_exp, num_blocks[0]], {'stride': 2})})
        layer_spec.update(
            {'layer2': LayerSpec([out_ch_before_exp, num_blocks[1]], {'stride': 2})})

        if dilated:
            if dilate_only_last_layer:
                layer_spec.update({'layer3': LayerSpec(
                    [out_ch_before_exp, num_blocks[2]], {'stride': 2})})
                layer_spec.update({'layer4': LayerSpec(
                    [out_ch_before_exp, num_blocks[3]], {'stride': 2, 'dilation': 2})})
            else:
                layer_spec.update({'layer3': LayerSpec(
                    [out_ch_before_exp, num_blocks[2]], {'stride': 2, 'dilation': 2})})
                layer_spec.update({'layer4': LayerSpec(
                    [out_ch_before_exp, num_blocks[3]], {'stride': 2, 'dilation': 4})})
        else:
            layer_spec.update({'layer3': LayerSpec(
                [out_ch_before_exp, num_blocks[2]], {'stride': 2})})
            layer_spec.update({'layer4': LayerSpec(
                [out_ch_before_exp, num_blocks[3]], {'stride': 2})})

        self.layers = list(layer_spec.keys())
        # only use required layers...
        self.layers = self.layers[:num_en_features - 4]
        for layer, spec in layer_spec.items():
            if layer in self.layers:
                self.add_module(layer, self._make_layer(*spec.args, **spec.kwargs))

        initialize_weights(self, norm_layer)

    def _make_layer(self, planes, num_blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.block.expansion:
            downsample = nn.Sequential(
                self.conv_layer(self.inplanes, planes * self.block.expansion,
                                kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * self.block.expansion))

        block_kwargs = {'previous_dilation': dilation, 'norm_layer': self.norm_layer,
                        'conv_layer': self.conv_layer}
        layers = [self.block(
            self.inplanes, planes, stride=stride, dilation=max(1, dilation // 2),
            downsample=downsample, **block_kwargs)]

        self.inplanes = planes * self.block.expansion

        for i in range(1, num_blocks):
            layers.append(self.block(
                self.inplanes, planes, dilation=dilation, **block_kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        outs = [x]
        for layer in self.layers:
            layer_fn = getattr(self, layer)
            outs.append(layer_fn(outs[-1]))

        return outs[1:]


def resnet_encoder(inplanes, cfg, norm_layer=nn.BatchNorm2d, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        inplanes:
        cfg:
        norm_layer:
        kwargs:
    """
    model = ResNetEncoder(inplanes, cfg, Bottleneck, [2, 2, 2, 2],
                          norm_layer=norm_layer, **kwargs)
    model.feat_channels = cfg.MODEL.ENCODER.FEAT_CHANNELS
    return model
