import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SELayer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // 8, num_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_orig = x
        x = torch.mean(x, dim=2, keepdim=False)
        x = self.fc(x)
        x = x.unsqueeze(2) * x_orig
        return x


class SingleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=3 // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        self.se_layer = SELayer(out_channels)
        self.proj_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(
            self.proj_conv.weight, mode="fan_out", nonlinearity="relu"
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.relu(self.se_layer(x) + self.proj_conv(residual))
        return x


class LSTMBlock(nn.Module):
    def __init__(self, input_dim, lstm_dim, dropout):
        super().__init__()
        self.has_dropout = dropout > 0.0
        if self.has_dropout:
            self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, lstm_dim, batch_first=False, bidirectional=True)

    def forward(self, x):
        x = self.ln(x)
        if self.has_dropout:
            x = self.dropout(x)
        x, _ = self.lstm(x)
        x = (
            x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        )  # T,B,C*2 -> T,B,C by sum
        return x


class SRModel(nn.Module):
    def __init__(
        self,
        n_cnn_layers,
        lstm_input_dim,
        n_lstm_layers,
        lstm_dim,
        n_vocab,
        n_feats,
        dropout,
    ):
        super().__init__()
        self.cnn = nn.Conv2d(1, 32, 3, stride=2, padding=3 // 2)
        self.resnet_layers = nn.Sequential(
            *[ResidualBlock(32, 32, stride=1, kernel_s=3) for _ in range(n_cnn_layers)]
        )
        n_features = 32 * n_feats // 2
        self.conv_block = SingleConvBlock(n_features, lstm_input_dim)
        self.lstm_layers = [LSTMBlock(lstm_input_dim, lstm_dim, dropout)]
        for _ in range(n_lstm_layers - 1):
            self.lstm_layers.append(LSTMBlock(lstm_dim, lstm_dim, dropout))
        self.lstm_layers = nn.Sequential(*self.lstm_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_dim),
            nn.Linear(lstm_dim, n_vocab + 1, bias=False),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.resnet_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]).contiguous()  # B, C, T
        x = self.conv_block(x)
        x = x.permute(2, 0, 1).contiguous()  # T, B, C
        x = self.lstm_layers(x)
        x = x.transpose(0, 1).contiguous()  # B, T, C
        x = self.classifier(x)
        return x
