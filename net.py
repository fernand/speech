import torch
import torch.nn as nn
import sru


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_kernel_s):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            (3, t_kernel_s),
            1,
            padding=(3 // 2, t_kernel_s // 2),
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            (3, t_kernel_s),
            1,
            padding=(3 // 2, t_kernel_s // 2),
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
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
        self,
        n_rnn_layers,
        rnn_dim,
        n_vocab,
        n_feats,
        dropout,
        highway_bias,
        projection_size,
    ):
        super().__init__()
        self.cnn = nn.Conv2d(1, 32, 3, stride=2, padding=3 // 2)
        self.resnet_layers = nn.Sequential(
            *[ResidualBlock(32, 32, t_kernel_s=3) for _ in range(3)]
        )
        n_features = 32 * n_feats // 2
        self.sru_layers = sru.SRU(
            input_size=n_features,
            hidden_size=rnn_dim,
            num_layers=n_rnn_layers,
            projection_size=projection_size,
            dropout=dropout,
            rescale=False,
            layer_norm=True,
            bidirectional=True,
            amp_recurrence_fp16=True,
            highway_bias=highway_bias,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(2*rnn_dim),
            nn.Linear(2*rnn_dim, n_vocab + 1, bias=False),
        )

    def forward(self, x):  # B, C, T
        x = self.cnn(x)
        x = self.resnet_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]).contiguous()  # B, C, T
        x = x.permute(2, 0, 1).contiguous()  # T, B, C
        x, _ = self.sru_layers(x)
        x = x.transpose(0, 1).contiguous()  # B, T, C
        x = self.classifier(x)
        return x
