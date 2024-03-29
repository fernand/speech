import os
import pickle
import sys

import torch
import torchaudio
import tqdm

torchaudio.set_audio_backend("sox_io")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import data
import decoder
import net


def load_model(model_path):
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
    model.cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


class MetaTV(data.SortedDataset):
    def __init__(self, manifest_path, batch_size):
        super().__init__(batch_size)
        with open(manifest_path, "rb") as f:
            self.paths = sorted(pickle.load(f), key=lambda t: t[-1])
        self.paths = [m for m in self.paths if m[1] <= 0.3]
        # self.paths = self.paths[:300]

    def get_clip(self, i):
        m = self.paths[i]
        audio_path = m[0].replace("/data", "/hd1")
        transcript_path = audio_path.strip(".wav") + ".txt"

        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        duration = waveform.size(1) / 16000
        if duration > 6.0:
            waveform = torch.zeros(1, 16000, dtype=torch.float32)

        transcript = None
        if os.path.exists(transcript_path):
            with open(transcript_path) as f:
                transcript = f.read().strip()
                if len(transcript) == 0:
                    transcript = None
        return (waveform, transcript, m)


def collate_fn(collated):
    spectrograms = []
    transcripts = []
    manifests = []
    for (waveform, transcript, manifest) in collated:
        spec = data.spectrogram_transform(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        transcripts.append(transcript)
        manifests.append(manifest)
    spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
    spectrograms = spectrograms.unsqueeze(1).transpose(2, 3)
    return spectrograms, transcripts, manifests


def get_loader(manifest_path, batch_size=256):
    dataset = MetaTV(manifest_path, batch_size)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=None,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=3,
        pin_memory=True,
    )
    return loader


def filter_files(model, loader):
    filtered_manifest = []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            spectrograms, transcripts, manifests = batch
            current_batch_size = len(transcripts)
            spectrograms = spectrograms.cuda()
            output = model(spectrograms)  # B, T, n_vocab+1
            output = torch.nn.functional.log_softmax(output, dim=2).cpu()

            preds = decoder.greedy_decode(output)
            for j in range(current_batch_size):
                if transcripts[j] is None:
                    continue
                pred_w = preds[j].split(" ")
                target_w = transcripts[j].split(" ")
                m = manifests[j]
                if (
                    len(target_w) >= 2
                    and len(pred_w) >= 2
                    and pred_w[0] == target_w[0]
                    and pred_w[-1] == target_w[-1]
                ):
                    filtered_manifest.append(m)
    return filtered_manifest


if __name__ == "__main__":
    datasets = [
        "datasets/first",
        "datasets/second",
        "datasets/third",
        "datasets/fourth",
        "datasets/fifth",
        "datasets/sixth",
    ]
    model_path = "good_models/sru-lstm-123456-libri-0.1cer/model_cc3a8ef99e314fe88df830e5bf9c8dff.pth"
    for dataset in datasets:
        print("Processing dataset", dataset.split("/")[-1])
        manifest_path = os.path.join(dataset, "manifest.pkl")
        model = load_model(model_path)
        loader = get_loader(manifest_path)
        filtered_manifest = filter_files(model, loader)
        with open(os.path.join(dataset, "filtered_manifest.pkl"), "wb") as f:
            pickle.dump(filtered_manifest, f)
