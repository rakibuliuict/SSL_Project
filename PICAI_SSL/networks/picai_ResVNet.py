import torch
from torch import nn
import torch.nn.functional as F
from networks.resnet3d import resnet34


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            in_channels = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_channels, n_filters_out, kernel_size=3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()
        ops = [nn.Conv3d(n_filters_in, n_filters_out, kernel_size=stride, stride=stride)]
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()
        ops = [nn.ConvTranspose3d(n_filters_in, n_filters_out, kernel_size=stride, stride=stride)]
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class ResVNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='instancenorm', has_dropout=False):
        super(ResVNet, self).__init__()
        print("Initialized ResVNet")
        self.resencoder = resnet34(in_channel=n_channels)
        self.has_dropout = has_dropout

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)
        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)
        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)
        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        if has_dropout:
            self.dropout = nn.Dropout3d(p=0.5)

        self.branchs = nn.ModuleList()
        for _ in range(1):
            layers = [ConvBlock(1, n_filters, n_filters, normalization=normalization)]
            if has_dropout:
                layers.append(nn.Dropout3d(p=0.5))
            layers.append(nn.Conv3d(n_filters, n_classes, kernel_size=1))
            self.branchs.append(nn.Sequential(*layers))

    def _resize_and_add(self, up_tensor, skip_tensor):
        if up_tensor.shape != skip_tensor.shape:
            skip_tensor = F.interpolate(skip_tensor, size=up_tensor.shape[2:], mode='trilinear', align_corners=True)
        return up_tensor + skip_tensor

    def encoder(self, x):
        return self.resencoder(x)

    def decoder(self, features):
        x1, x2, x3, x4, x5 = features
        x5_up = self.block_five_up(x5)
        x5_up = self._resize_and_add(x5_up, x4)

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = self._resize_and_add(x6_up, x3)

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = self._resize_and_add(x7_up, x2)

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = self._resize_and_add(x8_up, x1)

        outputs = [branch(x8_up) for branch in self.branchs]
        outputs.append(x6)
        return outputs

    def forward(self, x, turnoff_drop=False):
        if turnoff_drop:
            self.has_dropout = False
        features = self.encoder(x)
        output = self.decoder(features)
        return output
