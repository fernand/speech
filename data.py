import os
import pickle

import numpy as np
import torch
import torchaudio
import torch.nn as nn

from text import TextTransform

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, hop_length=160, n_mels=80, power=1.0
    ),
    # torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    # torchaudio.transforms.TimeMasking(time_mask_param=35),
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, hop_length=160, n_mels=80, power=1.0
)

text_transform = TextTransform()


class SortedTrainLibriSpeech(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        assert dataset_path.endswith(".pkl")
        with open(dataset_path, "rb") as f:
            self.paths = [t[0] for t in pickle.load(f)]
            # If ordering descending need to remove the shortest clips if
            # they're batched up because they won't have enough padding.
            self.paths.reverse()
            if "train" in dataset_path:
                # Remove the longest clips
                self.paths = self.paths[1000:]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        audio_path = self.paths[i]
        path, filename = os.path.split(audio_path)
        fileid = filename.split(".")[0]
        speaker_id, chapter_id, utterance_id = fileid.split("-")

        file_text = speaker_id + "-" + chapter_id + ".trans.txt"
        file_text = os.path.join(path, file_text)

        waveform, sample_rate = torchaudio.load(audio_path)

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
    input_lengths = []
    label_lengths = []
    for (waveform, utterance) in data:
        if data_type == "train":
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.LongTensor([0] + text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        label_lengths.append(len(label) - 1)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
    spectrograms = spectrograms.transpose(1, 2)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, torch.IntTensor(label_lengths)
