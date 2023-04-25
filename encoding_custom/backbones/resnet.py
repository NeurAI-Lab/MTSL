import math
import torch
import torch.nn as nn

from encoding_custom.utils.model_store import get_model_file

__all__ = [
    "ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "BasicBlock", "Bottleneck", 'initialize_weights']


def initialize_weights(model, norm_layer):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, norm_layer):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    # ResNet BasicBlock
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=dilation,
            dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=previous_dilation,
            dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d,
                 conv_layer=nn.Conv2d):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_layer(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv_layer(
            planes, planes, kernel_size=3, stride=stride, padding=dilation,
            dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv_layer(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    @staticmethod
    def _sum_each(x, y):
        assert len(x) == len(y)
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        residual = x

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


class ResNet(nn.Module):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    num_classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer :
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
        Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    def __init__(self, block, layers, num_classes=None, dilated=True,
                 norm_layer=nn.BatchNorm2d, add_additional_layers=False,
                 dilate_only_last_layer=False, **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2)
        if dilated:
            if dilate_only_last_layer:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
                self.layer4 = self._make_layer(
                    block, 256, layers[2], stride=1, dilation=2)
            else:
                self.layer3 = self._make_layer(
                    block, 256, layers[2], stride=1, dilation=2)
                self.layer4 = self._make_layer(
                    block, 512, layers[3], stride=1, dilation=4)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if add_additional_layers:
            self.layer5 = self._make_layer(block, 64, layers[3], stride=2)
            self.layer6 = self._make_layer(block, 64, layers[3], stride=1)

            self.layer7 = self._make_layer(block, 64, layers[3], stride=2)

        if num_classes is not None:
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        initialize_weights(self, norm_layer)

        self.backbone_feat_channels = []

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion))

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(
                block(self.inplanes, planes, stride, dilation=1,
                      downsample=downsample, previous_dilation=dilation,
                      norm_layer=self.norm_layer))
        elif dilation == 4:
            layers.append(
                block(self.inplanes, planes, stride, dilation=2,
                      downsample=downsample, previous_dilation=dilation,
                      norm_layer=self.norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation,
                      previous_dilation=dilation, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c2, c3, c4

    def forward_stage(self, x, stage):
        if stage == 'layer1':
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            return x

        else:  # Stage 2, 3 or 4
            layer = getattr(self, stage)
            return layer(x)

    def forward_stage_with_last_block(self, x, stage):
        if stage == 'layer1':
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x1 = self.layer1[:-1](x)
            x = self.layer1[-1](x1)
            return x1, x

        else:  # Stage 2, 3 or 4
            layer = getattr(self, stage)
            x1 = layer[:-1](x)
            return x1, layer[-1](x1)


def resnet18(pretrained=False, root="~/.encoding/models", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        root:
        kwargs:
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.feat_channels = [64, 128, 256, 512]
    model.details = {'block_expansion': BasicBlock.expansion}
    if pretrained:
        model.load_state_dict(
            torch.load(get_model_file("resnet18", root=root)), strict=False)
    return model


def resnet34(pretrained=False, root="~/.encoding/models", **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        root:
        kwargs:
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    model.feat_channels = [64, 128, 256, 512]
    model.details = {'block_expansion': BasicBlock.expansion}
    if pretrained:
        model.load_state_dict(
            torch.load(get_model_file("resnet34", root=root)), strict=False)
    return model


def resnet50(pretrained=False, root="~/.encoding/models", **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        root:
        kwargs:
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model.feat_channels = [256, 512, 1024, 2048]
    model.details = {'block_expansion': Bottleneck.expansion}
    if pretrained:
        model.load_state_dict(
            torch.load(get_model_file("resnet50", root=root)), strict=False)
    return model


def resnet101(pretrained=False, root="~/.encoding/models", **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        root:
        kwargs:
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    model.feat_channels = [256, 512, 1024, 2048]
    model.details = {'block_expansion': Bottleneck.expansion}
    if pretrained:
        model.load_state_dict(
            torch.load(get_model_file("resnet101", root=root)), strict=False)
    return model


def resnet152(pretrained=False, root="~/.encoding/models", **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        root:
        kwargs:
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    model.feat_channels = [256, 512, 1024, 2048]
    model.details = {'block_expansion': Bottleneck.expansion}
    if pretrained:
        model.load_state_dict(
            torch.load(get_model_file("resnet151", root=root)), strict=False)
    return model
