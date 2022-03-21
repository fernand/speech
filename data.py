import random

import torch
import torchaudio
import torch.nn as nn
import scipy.io.wavfile

torchaudio.set_audio_backend("sox_io")


N_MELS = 80
spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, hop_length=160, n_mels=N_MELS, power=1.0
)

train_audio_transforms = nn.Sequential(
    spectrogram_transform,
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    # torchaudio.transforms.TimeMasking(time_mask_param=35),
)


def get_seg(audio_path):
    _, wav = scipy.io.wavfile.read(audio_path)
    wav = torch.from_numpy(wav / 32767).float()
    disc_path = audio_path.replace(".wav", ".disc")
    with open(disc_path, "rt") as f:
        disc = f.read()
    return (wav, disc)


class SpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, segments):
        super().__init__()
        self.pairs = pairs
        self.segments = segments

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pair = self.pairs[i]
        p1_segs = self.segments[pair[0]]
        p2_segs = self.segments[pair[1]]
        p1_seg = random.sample(p1_segs, 1)[0]
        p2_seg = random.sample(p2_segs, 1)[0]
        wav1, disc1 = get_seg(p1_seg)
        wav2, disc2 = get_seg(p2_seg)
        disc2 = disc2.replace("1", "2")
        wav1.unsqueeze(0)
        wav2.unsqueeze(0)
        return (torch.cat([wav1, wav2], dim=0), disc1 + disc2)


def collate_fn(data, data_type="train"):
    spectrograms = []
    labels = []
    for (waveform, disc) in data:
        if data_type == "train":
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)  # T, C
        else:
            spec = spectrogram_transform(waveform).squeeze(0).transpose(0, 1)
        if len(spec) > len(disc):
            spec = spec[:-1, :]
        spectrograms.append(spec)
        label = torch.LongTensor([int(c) for c in disc])
        labels.append(label)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)  # B, T, C
    spectrograms = spectrograms.unsqueeze(1).transpose(2, 3)  # B, 1, C, T
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=3)
    return spectrograms, labels
