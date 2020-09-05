import torch
import torch.nn as nn
import torch.nn.functional as F
import sru


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, n_feats):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel, stride, padding=kernel // 2
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel, stride, padding=kernel // 2
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
        residual = x  # (batch, channel, feature, time)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x  # (batch, channel, feature, time)


class SRModel(nn.Module):
    def __init__(
        self,
        n_cnn_layers,
        n_rnn_layers,
        rnn_dim,
        n_vocab,
        n_feats,
        stride=2,
        dropout=0.1,
    ):
        super().__init__()
        n_feats = n_feats // stride
        self.stride = stride
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)
        self.resnet_layers = nn.Sequential(
            *[
                ResidualBlock(32, 32, kernel=3, stride=1, n_feats=n_feats)
                for _ in range(n_cnn_layers)
            ]
        )
        self.birnn_layers = sru.SRU(
            input_size=n_feats * 32,
            proj_input_to_hidden_first=True,
            hidden_size=rnn_dim,
            num_layers=n_rnn_layers,
            rnn_dropout=dropout,
            layer_norm=True,
            use_tanh=True,
            bidirectional=True,
            nn_rnn_compatible_return=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(rnn_dim * 2),
            nn.Linear(rnn_dim * 2, rnn_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_dim, n_vocab + 1, bias=False),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.resnet_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.permute(2, 0, 1).contiguous()  # (time, feature, batch)
        x, _ = self.birnn_layers(x)  # (time, batch, feature)
        # TODO: don't do nn_rnn_compatible_return then do it here to only have 1 contiguous.
        # SRU return shape is 4D https://github.com/asappresearch/sru/blob/master/sru/sru_functional.py#L621
        x = x.transpose(0, 1).contiguous()  # (batch, time, feature)
        x = self.classifier(x)
        return x
