import gc
import os
import pickle
import re
import sys

import torch
import torchaudio
import torch.nn as nn

import text

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

UM_REGEXP = re.compile(r"(uh)|(uhh)|(um)|(umm)|(eh)|(hm)|(ah)|(huh)|(ha)|(er)")
MULTI_SPACE_REGEXP = re.compile(r"\s+")


def get_librispeech_paths(dataset_path):
    with open(dataset_path, "rb") as f:
        paths = [(t[0], t[1] / 16000) for t in pickle.load(f)]
    return paths


def get_librispeech_clip(audio_path):
    path, filename = os.path.split(audio_path)
    fileid = filename.split(".")[0]
    speaker_id, chapter_id, utterance_id = fileid.split("-")

    file_text = speaker_id + "-" + chapter_id + ".trans.txt"
    file_text = os.path.join(path, file_text)

    waveform, sample_rate = torchaudio.load(audio_path, normalize=True)

    with open(file_text) as f:
        for line in f:
            fileid_text, utterance = line.strip().split(" ", 1)
            if fileid == fileid_text:
                break

    return (waveform, utterance.lower())


def get_tv_clip(audio_path):
    text_path = audio_path.strip(".wav") + ".txt"
    waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
    with open(text_path) as f:
        utterance = f.read().strip()
    return (waveform, utterance)


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


class IBMDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        audio_paths = [
            os.path.join("/hd1/ibm", f)
            for f in os.listdir("/hd1/ibm")
            if f.endswith("wav")
        ]
        self.paths = audio_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        audio_path = self.paths[i]
        waveform, _ = torchaudio.load(audio_path, normalize=True)
        with open(audio_path.replace(".wav", ".txt")) as f:
            utterance = f.read().strip()
            utterance = re.sub(UM_REGEXP, "", utterance)
            utterance = re.sub(MULTI_SPACE_REGEXP, "", utterance)
        return (waveform, utterance)


class SortedLibriSpeech(SortedDataset):
    def __init__(self, dataset_path, batch_size):
        super().__init__(batch_size)
        assert dataset_path.endswith(".pkl")
        self.paths = [t[0] for t in get_librispeech_paths(dataset_path)]
        if "train" in dataset_path:
            # Remove the longest clips.
            self.paths = self.paths[:-1000]

    def get_clip(self, i):
        return get_librispeech_clip(self.paths[i].replace("/data", "/hd1"))


class SortedTV(SortedDataset):
    def __init__(self, dataset_paths, batch_size, device):
        super().__init__(batch_size)
        hd = "/hd" + str(device + 1)
        for dataset_path in dataset_paths:
            with open(dataset_path, "rb") as f:
                paths = [t[0] for t in pickle.load(f)]
                paths = [p.replace("/data", hd).replace("/hd1", hd) for p in paths]
                self.paths.extend(paths)

    def get_clip(self, i):
        audio_path = self.paths[i]
        return get_tv_clip(audio_path)


class CombinedTVLibriSpeech(SortedDataset):
    def __init__(self, librispeech_dataset_path, tv_dataset_paths, batch_size, device):
        super().__init__(batch_size)
        assert librispeech_dataset_path.endswith(".pkl")
        tuples = get_librispeech_paths(librispeech_dataset_path)[:-2000]
        for dataset_path in tv_dataset_paths:
            with open(dataset_path, "rb") as f:
                tuples.extend(pickle.load(f))
        tuples = sorted(tuples, key=lambda t: t[1])
        self.paths = [t[0] for t in tuples]
        self.hd = "/hd" + str(device + 1)
        self.counter = 0
        self.num_paths = len(self.paths)

    def get_clip(self, i):
        if self.counter == num_paths:
            gc.collect()
            self.counter = 0
        else:
            self.counter += 1
        audio_path = self.paths[i]
        if "LibriSpeech" in audio_path:
            return get_librispeech_clip(audio_path.replace("/data", self.hd))
        elif "clean" in audio_path:
            return get_tv_clip(audio_path.replace("/hd1", self.hd))
        elif "gigaspeech" in audio_path:
            return get_tv_clip(audio_path)
        else:
            print(audio_path, "not in datasets")
            sys.exit(1)


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
