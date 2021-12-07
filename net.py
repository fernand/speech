import torch
import sru

class SRModel(torch.nn.Module):
    def __init__(
        self,
        n_rnn_layers,
        rnn_dim,
        n_vocab,
        n_feats,
        dropout=0.1,
        lstm_dim=1024,
    ):
        super().__init__()
        self.first_layer = sru.SRU(
            input_size=n_feats,
            hidden_size=rnn_dim,
            num_layers=n_rnn_layers,
            dropout=0.0,
            rescale=False,
            layer_norm=True,
            bidirectional=False,
            amp_recurrence_fp16=True,
        )
        self.avg_pool = torch.nn.AvgPool1d(2)
        self.sru_layers = sru.SRU(
            input_size=rnn_dim,
            hidden_size=rnn_dim,
            num_layers=n_rnn_layers-1,
            dropout=dropout,
            rescale=False,
            layer_norm=True,
            bidirectional=False,
            amp_recurrence_fp16=True,
        )
        self.lstm = torch.nn.LSTM(
            input_size=rnn_dim,
            hidden_size=lstm_dim,
            num_layers=1,
            batch_first=False,
            dropout=0.0,
            bidirectional=False,
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(lstm_dim),
            torch.nn.Linear(lstm_dim, n_vocab + 1, bias=False),
        )

    def forward(self, x): # B, T, C
        x = x.transpose(0, 1).contiguous() # T, B, C
        x, _ = self.first_layer(x) 
        x = x.permute(1, 2, 0) # B, C, T
        x = self.avg_pool(x)
        x = x.permute(2, 0, 1).contiguous() # T, B, C
        x, _ = self.sru_layers(x)
        x, _ = self.lstm(x)
        x = x.transpose(0, 1).contiguous()  # B, T, C
        x = self.classifier(x)
        return x
