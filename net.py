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
            padding=5 // 2,
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
        self.residual = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            padding=0,
            stride=stride,
            groups=in_channels,
        )
        self.se_layer = SELayer(out_channels, dropout)

    def forward(self, x):
        x_orig = x
        x = self.conv_layers(x)
        x = swish(self.se_layer(x) + self.residual(x_orig))
        return x


# TODO: Need to pack the sequence since there's a big difference in label counts.
class LabelEncoder(nn.Module):
    def __init__(self, alpha, n_vocab):
        super(LabelEncoder, self).__init__()
        n_embeds = int(640 * alpha)
        l1 = int(2048 * alpha)
        # 0 is essentially the start of sequence token.
        self.embed = nn.Embedding(n_vocab + 1, n_embeds, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=n_embeds, hidden_size=l1, num_layers=1, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(l1)
        self.projection = nn.Linear(l1, n_embeds)

    def forward(self, y):
        y = self.embed(y)
        y, hidden = self.lstm(y)
        y = self.layer_norm(y)
        y = self.projection(y)
        return y, hidden


class ContextNet(nn.Module):
    def __init__(self, alpha, n_feats, n_vocab):
        super(ContextNet, self).__init__()
        self.blank = 0
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
        self.label_encoder = LabelEncoder(alpha, n_vocab)
        self.projection = nn.Linear(l3, l3)
        self.classifier = nn.Linear(l3, n_vocab + 1)

    def joint(self, x, y):
        out = torch.tanh(self.projection(x + y))
        out = self.classifier(out)
        return out

    def forward(self, x, y):
        h_enc = self.encoder(x).transpose(1, 2)  # B, T, F
        y, _ = self.label_encoder(y)  # B, U, F
        x = h_enc.unsqueeze(2)
        y = y.unsqueeze(1)
        out = self.joint(x, y)
        return out, h_enc  # out is B, T, U, n_vocab+1

    # Needs to be fed self.encoder(x).transpose(1, 2)[i]
    def infer_greedy(self, h):
        """Greedy search implementation.
        Args:
            h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
        Returns:
            hyp (list of dicts): 1-best decoding results
    """
        hyp = {"score": 0.0, "yseq": [self.blank]}

        # Start with the start of sentence token (which is 0)
        sos_token = torch.tensor([0], dtype=torch.long).unsqueeze(0).cuda()
        y, hidden = self.label_encoder.forward(sos_token)

        for hi in h:
            ytu = F.log_softmax(self.joint(hi, y[0]), dim=1)
            logp, pred = torch.max(ytu, dim=1)
            pred = int(pred)

            if pred != self.blank:
                hyp["yseq"].append(pred)
                hyp["score"] += float(logp)
                y = (
                    torch.tensor([hyp["yseq"][-1]], dtype=torch.long)
                    .unsqueeze(0)
                    .cuda()
                )
                y, hidden = self.label_encoder.forward(y)
        return hyp
