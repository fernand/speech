import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, num_channels, dropout=0.0):
        super(SELayer, self).__init__()
        self.layer1 = nn.Linear(num_channels, num_channels // 8)
        self.layer2 = nn.Linear(num_channels // 8, num_channels)

    def forward(self, x):
        x_orig = x
        x = torch.mean(x, dim=2, keepdim=False) / x.size(2)
        x = swish(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        # TODO: not sure this is right
        x = x.unsqueeze(2) * x_orig
        return x_orig


class SingleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout=0.0):
        super(SingleConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            stride=stride,
            padding=5 // 2
            # groups=in_channels,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.se_layer = SELayer(out_channels, dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = swish(x)
        x = swish(self.se_layer(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout=0.0):
        super(ConvBlock, self).__init__()
        layers = []
        for i in range(5):
            if i < 4:
                conv_stride = 1
                padding = 5 // 2
            else:
                # Last conv has stride 1 or 2.
                conv_stride = stride
                padding = 0
            if i == 0:
                conv_in_channels = in_channels
                conv_out_channels = out_channels
            else:
                conv_in_channels = out_channels
                conv_out_channels = out_channels
            layers.append(
                nn.Conv1d(
                    in_channels=conv_in_channels,
                    out_channels=conv_out_channels,
                    kernel_size=5,
                    stride=conv_stride,
                    padding=padding,
                    groups=in_channels,
                )
            )
            layers.append(nn.BatchNorm1d(conv_out_channels))
            layers.append(Swish())
        self.conv_layers = nn.Sequential(*layers)
        last_stride = layers[-3].stride[0]
        self.residual = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            padding=0,
            stride=last_stride,
            groups=in_channels,
        )
        self.se_layer = SELayer(out_channels, dropout)

    def forward(self, x):
        x_orig = x
        x = self.conv_layers(x)
        x = swish(self.se_layer(x) + self.residual(x_orig))
        return x


class ContextNet(nn.Module):
    def __init__(self, alpha, n_feats, n_class):
        super(ContextNet, self).__init__()
        l1 = int(256 * alpha)
        l2 = int(512 * alpha)
        l3 = int(640 * alpha)
        conv_blocks = []
        conv_blocks.append(SingleConvBlock(n_feats, l1, 1))
        for i in range(2):
            conv_blocks.append(ConvBlock(l1, l1, 1))
        conv_blocks.append(ConvBlock(l1, l1, 2))
        for i in range(3):
            conv_blocks.append(ConvBlock(l1, l1, 1))
        conv_blocks.append(ConvBlock(l1, l1, 2))
        for i in range(3):
            conv_blocks.append(ConvBlock(l1, l1, 1))
        # C11
        conv_blocks.append(ConvBlock(l1, l2, 1))
        for i in range(2):
            conv_blocks.append(ConvBlock(l2, l2, 1))
        conv_blocks.append(ConvBlock(l2, l2, 2))
        for i in range(7):
            conv_blocks.append(ConvBlock(l2, l2, 1))
        conv_blocks.append(SingleConvBlock(l2, l3, 1))
        self.encoder = nn.Sequential(*conv_blocks)
        self.classifier = nn.Linear(l3, n_class)

    def forward(self, x):
        x = self.encoder(x).transpose(1, 2)
        x = torch.tanh(self.classifier(x))
        t_len = x.size(1)
        return x, t_len
