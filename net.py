import torch
import torch.nn as nn
import torch.nn.functional as F
import sru


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(
            in_channels, out_channels, kernel, stride, padding=kernel // 2
        )
        self.cnn2 = nn.Conv2d(
            out_channels, out_channels, kernel, stride, padding=kernel // 2
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class SpeechRecognitionModel(nn.Module):
    """Speech Recognition Model Inspired by DeepSpeech 2"""

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
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats // stride
        self.stride = stride
        self.cnn = nn.Conv2d(
            1, 32, 3, stride=stride, padding=3 // stride
        )  # cnn for extracting hierarchical features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(
            *[
                ResidualCNN(
                    32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats
                )
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
            nn.Linear(rnn_dim * 2, rnn_dim),  # birnn returns rnn_dim*2
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_vocab + 1),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.permute(2, 0, 1).contiguous()  # (time, feature, batch)
        x, _ = self.birnn_layers(x)  # (time, batch, feature)
        # TODO: don't do nn_rnn_compatible_return then do it here to only have 1 contiguous.
        # SRU return shape is 4D https://github.com/asappresearch/sru/blob/master/sru/sru_functional.py#L621
        x = x.transpose(0, 1).contiguous()
        x = self.classifier(x)
        return x
