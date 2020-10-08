import os
import pickle
import random

import torch
import torchaudio
import torchtext
import torch.nn as nn

from text import TextTransform

N_MELS = 80

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, hop_length=160, n_mels=N_MELS, power=1.0
    ),
    # torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    # torchaudio.transforms.TimeMasking(time_mask_param=35),
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, hop_length=160, n_mels=N_MELS, power=1.0
)

text_transform = TextTransform()


class SortedTV(torch.utils.data.Dataset):
    def __init__(self, dataset_paths, batch_size):
        self.batch_size = batch_size
        self.paths = []
        for dataset_path in dataset_paths:
            with open(dataset_path, "rb") as f:
                paths = [t[0] for t in pickle.load(f)]
                paths = [p.replace("/data", "/hd1") for p in paths]
                self.paths.extend(paths)

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __getitem__(self, i):
        return [
            self.get_clip(j)
            for j in range(i * self.batch_size, (i + 1) * self.batch_size)
        ]

    def get_clip(self, i):
        audio_path = self.paths[i]
        text_path = audio_path.strip(".wav") + ".txt"
        waveform, sample_rate = torchaudio.load(audio_path, normalization=True)
        with open(text_path) as f:
            utterance = f.read().strip()
        return (waveform, utterance)


class SortedLibriSpeech(torch.utils.data.Dataset):
    def __init__(self, dataset_path, batch_size):
        assert dataset_path.endswith(".pkl")
        self.batch_size = batch_size
        with open(dataset_path, "rb") as f:
            self.paths = [t[0] for t in pickle.load(f)]
        if "train" in dataset_path:
            # Remove the longest clips.
            self.paths = self.paths[:-1000]

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __getitem__(self, i):
        return [
            self.get_clip(j)
            for j in range(i * self.batch_size, (i + 1) * self.batch_size)
        ]

    def get_clip(self, i):
        audio_path = self.paths[i]
        path, filename = os.path.split(audio_path)
        fileid = filename.split(".")[0]
        speaker_id, chapter_id, utterance_id = fileid.split("-")

        file_text = speaker_id + "-" + chapter_id + ".trans.txt"
        file_text = os.path.join(path, file_text)

        waveform, sample_rate = torchaudio.load(audio_path, normalization=True)

        with open(file_text) as f:
            for line in f:
                fileid_text, utterance = line.strip().split(" ", 1)
                if fileid == fileid_text:
                    break
            else:
                raise FileNotFoundError("Translation not found for " + audio_path)

        return (waveform, utterance)


def collate_fn(data, data_type="train"):
    spectrograms = []
    labels = []
    label_lengths = []
    for (waveform, utterance) in data:
        if data_type == "train":
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.LongTensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
    spectrograms = spectrograms.unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, torch.IntTensor(label_lengths)
