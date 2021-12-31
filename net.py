import torch.nn as nn


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


class LSTMBlock(nn.Module):
    def __init__(self, input_dim, lstm_dim, proj_dim, dropout):
        super().__init__()
        self.has_dropout = dropout > 0.0
        if self.has_dropout:
            self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(proj_dim, lstm_dim, batch_first=False, bidirectional=True)
        if proj_dim > 0:
            self.proj = nn.Linear(input_dim, proj_dim)
        else:
            self.proj = None

    def forward(self, x):
        x = self.ln(x)
        if self.proj is not None:
            x = self.proj(x)
        if self.has_dropout:
            x = self.dropout(x)
        x, _ = self.lstm(x)
        return x


class SRModel(nn.Module):
    def __init__(
        self,
        n_rnn_layers,
        rnn_dim,
        n_vocab,
        n_feats,
        dropout,
        proj_dim,
    ):
        super().__init__()
        self.cnn = nn.Conv2d(1, 32, 3, stride=2, padding=3 // 2)
        self.resnet_layers = nn.Sequential(
            *[ResidualBlock(32, 32, t_kernel_s=5) for _ in range(3)]
        )
        n_features = 32 * n_feats // 2
        self.lstm_layers = [LSTMBlock(n_features, rnn_dim, proj_dim, dropout)]
        self.lstm_layers.extend(
            [
                LSTMBlock(2 * rnn_dim, rnn_dim, proj_dim, dropout)
                for _ in range(n_rnn_layers - 1)
            ]
        )
        self.lstm_layers = nn.Sequential(*self.lstm_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * rnn_dim),
            nn.Linear(2 * rnn_dim, n_vocab + 1, bias=False),
        )

    def forward(self, x):  # B, C, T
        x = self.cnn(x)
        x = self.resnet_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]).contiguous()  # B, C, T
        x = x.permute(2, 0, 1).contiguous()  # T, B, C
        x = self.lstm_layers(x)
        x = x.transpose(0, 1).contiguous()  # B, T, C
        x = self.classifier(x)
        return x
