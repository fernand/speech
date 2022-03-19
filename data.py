import gc
import os
import pickle
import re
import sys

import torch
import torchaudio
import torch.nn as nn

torchaudio.set_audio_backend("sox_io")

N_MELS = 80
spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, hop_length=160, n_mels=N_MELS, power=1.0
)

train_audio_transforms = nn.Sequential(
    spectrogram_transform,
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35),
)

class SortedDataset(torch.utils.data.Dataset):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.paths = []

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __getitem__(self, i):
        return [
            self.get_clip(j)
            for j in range(i * self.batch_size, (i + 1) * self.batch_size)
        ]

    def get_clip(self, i):
        pass


def collate_fn(data, data_type="train"):
    spectrograms = []
    labels = []
    label_lengths = []
    for (waveform, utterance) in data:
        if data_type == "train":
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)  # T, C
        else:
            spec = spectrogram_transform(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.LongTensor(text.text_to_int(utterance))
        labels.append(label)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)  # B, T, C
    spectrograms = spectrograms.unsqueeze(1).transpose(2, 3)  # B, 1, C, T
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, torch.IntTensor(label_lengths)
