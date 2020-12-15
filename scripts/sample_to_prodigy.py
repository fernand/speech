import json
import os
import pickle
import sys
import random

import torch
import torchaudio

torchaudio.set_audio_backend("sox_io")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import data
import decoder
import net


def load_model(path):
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 10,
        "rnn_dim": 512,
        "dropout": 0.1,
        # Does not include the blank.
        "n_vocab": 28,
        "n_feats": data.N_MELS,
    }
    model = net.SRModel(
        hparams["n_cnn_layers"],
        hparams["n_rnn_layers"],
        hparams["rnn_dim"],
        hparams["n_vocab"],
        hparams["n_feats"],
        hparams["dropout"],
    )
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def append_predictions(model, utterances):
    new_utterances = []
    for utterance in utterances:
        waveform, _ = torchaudio.load(utterance[0], normalize=True)
        with torch.no_grad():
            spect = data.spectrogram_transform(waveform).unsqueeze(0)  # 1, 1, C, T
            output = model(spect)  # B, T, n_vocab+1
            output = torch.nn.functional.log_softmax(output, dim=2)
            decoded = decoder.greedy_decode(output)[0]
        new_utterances.append((utterance[0], utterance[1], decoded))
    return new_utterances


# Update prodigy.json to disable cors and set host to the right IP.
# prodigy audio.transcribe utterances audio/utterances.jsonl --loader jsonl
# cd /; python -m http.server 8081 --bind 192.168.1.21
# Use Safari and disable CORS.
def to_jsonl(utterances, output_f):
    if os.path.exists(output_f):
        os.remove(output_f)
    random.shuffle(utterances)
    with open(output_f, "w") as f:
        for utterance in utterances:
            url = "http://192.168.1.21:8081" + utterance[0]
            js = {"audio": url, "transcript": "\n".join([utterance[1], utterance[2]])}
            f.write(json.dumps(js) + "\n")


if __name__ == "__main__":
    cer = float(sys.argv[1])
    output_json_f = sys.argv[2]
    model_path = sys.argv[3]
    num_to_sample = 100
    datasets = ["first", "fourth"]
    utterances = []
    for dataset in [f"datasets/{d}/manifest.pkl" for d in datasets]:
        with open(dataset, "rb") as f:
            m = pickle.load(f)
        candidates = [t[0] for t in m if t[1] >= cer and t[1] < cer + 0.005]
        random.shuffle(candidates)
        samples = candidates[:num_to_sample]
        for path in samples:
            path = path.replace("/data", "/hd1")
            text_f = path.replace(".wav", ".txt")
            with open(text_f) as f:
                utterance = f.read().strip()
                utterances.append((path, utterance))
    model = load_model(model_path)
    utterances = append_predictions(model, utterances)
    to_jsonl(utterances, output_json_f)
