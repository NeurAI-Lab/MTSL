from torch import nn as nn


def conv1x1(in_planes, out_planes, stride=1, bias=False, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=bias, groups=groups)


def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    # 3x3 convolution with padding
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=padding, bias=bias)
