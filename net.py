import apex
import torch
import torch.nn as nn
import torch.nn.functional as F
import sru


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_s):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_s, stride, padding=kernel_s // 2
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_s, stride, padding=kernel_s // 2
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # See the Pytorch ResNet code or Resnet paper for those.
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 0)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        residual = x  # B, C, F, T
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x  # B, C, F, T


class SRModel(nn.Module):
    def __init__(
        self, n_cnn_layers, n_rnn_layers, rnn_dim, n_vocab, n_feats, dropout=0.1,
    ):
        super().__init__()
        self.cnn = nn.Conv2d(1, 32, 3, stride=2, padding=3 // 2)
        self.resnet_layers = nn.Sequential(
            *[ResidualBlock(32, 32, stride=1, kernel_s=3) for _ in range(n_cnn_layers)]
        )
        n_features = 32 * n_feats // 2
        self.feature_ln = apex.normalization.FusedLayerNorm(n_features)
        self.birnn_layers = sru.SRU(
            input_size=n_features,
            hidden_size=rnn_dim,
            num_layers=n_rnn_layers,
            dropout=dropout,
            rescale=True,
            layer_norm=True,
            bidirectional=True,
            nn_rnn_compatible_return=True,
        )
        self.classifier = nn.Sequential(
            apex.normalization.FusedLayerNorm(rnn_dim),
            nn.Linear(rnn_dim, n_vocab + 1, bias=False),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.resnet_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # B, C, T
        x = x.permute(2, 0, 1).contiguous()  # T, B, C
        x = self.feature_ln(x)
        x, _ = self.birnn_layers(x)  # T, B, C*2
        x = (
            x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        )  # T,B,C*2 -> T,B,C by sum
        x = x.transpose(0, 1).contiguous()  # B, T, C
        x = self.classifier(x)
        return x
