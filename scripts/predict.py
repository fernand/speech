import os
import sys

import torch
import torchaudio

torchaudio.set_audio_backend("sox_io")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import data
import net
import decoder


def get_model(model_path):
    hparams = {
        "n_cnn_layers": 3,
        "lstm_input_dim": 512,
        "n_lstm_layers": 3,
        "lstm_dim": 1024,
        "dropout": 0.1,
        # Does not include the blank.
        "n_vocab": 28,
        "n_feats": data.N_MELS,
    }
    model = net.SRModel(
        hparams["n_cnn_layers"],
        hparams["lstm_input_dim"],
        hparams["n_lstm_layers"],
        hparams["lstm_dim"],
        hparams["n_vocab"],
        hparams["n_feats"],
        hparams["dropout"],
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


@torch.no_grad()
def predict(audio_path, model):
    wav, _ = torchaudio.load(audio_path, normalize=True)
    spect = data.spectrogram_transform(wav).unsqueeze(0)
    output = model(spect)
    output = torch.nn.functional.log_softmax(output, dim=2)
    pred = decoder.greedy_decode(output)
    print(pred)
