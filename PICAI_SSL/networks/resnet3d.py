import math
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.InstanceNorm3d(out_planes),
        nn.ReLU()
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv3d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(width)
        self.conv2 = nn.Conv3d(width, width, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               groups=groups, bias=False)
        self.bn2 = nn.InstanceNorm3d(width)
        self.conv3 = nn.Conv3d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.InstanceNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

    def __init__(self, block, layers, in_channel=3, width=1, groups=1, width_per_group=64,
                 mid_dim=1024, low_dim=128, avg_down=False, deep_stem=False,
                 head_type='mlp_head', layer4_dilation=1):
        super(ResNet, self).__init__()
        self.avg_down = avg_down
        self.inplanes = 16 * width
        self.base = int(16 * width)
        self.groups = groups
        self.base_width = width_per_group

        if deep_stem:
            self.conv1 = nn.Sequential(
                conv3x3_bn_relu(in_channel, 32, stride=2),
                conv3x3_bn_relu(32, 32, stride=1),
                conv3x3(32, 64, stride=1)
            )
        else:
            self.conv1 = nn.Conv3d(in_channel, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)

        self.bn1 = nn.InstanceNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, self.base*2, layers[0], stride=2)
        self.layer2 = self._make_layer(block, self.base*4, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base*8, layers[2], stride=2)
        if layer4_dilation == 1:
            self.layer4 = self._make_layer(block, self.base*16, layers[3], stride=2)
        elif layer4_dilation == 2:
            self.layer4 = self._make_layer(block, self.base*16, layers[3], stride=1, dilation=2)
        else:
            raise NotImplementedError

        self.avgpool = nn.AvgPool3d(7, stride=1)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool3d(kernel_size=stride, stride=stride),
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                    nn.InstanceNorm3d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.InstanceNorm3d(planes * block.expansion),
                )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [x, c2, c3, c4, c5]


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
