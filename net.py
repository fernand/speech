import torch
import sru

class LSTMBlock(torch.nn.Module):
    def __init__(self, input_dim, lstm_dim, dropout):
        super().__init__()
        self.has_dropout = dropout > 0.0
        if self.has_dropout:
            self.dropout = torch.nn.Dropout(dropout)
        self.ln = torch.nn.LayerNorm(input_dim)
        self.lstm = torch.nn.LSTM(input_dim, lstm_dim, batch_first=False, bidirectional=False)

    def forward(self, x):
        x = self.ln(x)
        if self.has_dropout:
            x = self.dropout(x)
        x, _ = self.lstm(x)
        return x


class SRModel(torch.nn.Module):
    def __init__(
        self,
        n_rnn_layers,
        rnn_dim,
        n_vocab,
        n_feats,
        dropout=0.1,
    ):
        super().__init__()
        self.first_layer = sru.SRU(
            input_size=n_feats,
            hidden_size=rnn_dim,
            num_layers=8,
            dropout=dropout,
            rescale=False,
            layer_norm=True,
            bidirectional=False,
            amp_recurrence_fp16=True,
        )
        self.lstm_layers = [LSTMBlock(rnn_dim, rnn_dim, dropout)]
        for _ in range(n_rnn_layers - 1):
            self.lstm_layers.append(LSTMBlock(rnn_dim, rnn_dim, dropout))
        self.lstm_layers = torch.nn.Sequential(*self.lstm_layers)
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(rnn_dim),
            torch.nn.Linear(rnn_dim, n_vocab + 1, bias=False),
        )

    def forward(self, x): # B, T, C
        x = x.transpose(0, 1).contiguous() # T, B, C
        x, _ = self.first_layer(x) 
        x = self.lstm_layers(x)
        x = x.transpose(0, 1).contiguous()  # B, T, C
        x = self.classifier(x)
        return x
