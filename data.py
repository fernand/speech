import random

import numpy as np
import torch
import torchaudio
import torch.nn as nn
import scipy.io.wavfile
import ujson
import uuid

torchaudio.set_audio_backend("sox_io")


N_MELS = 80
spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, hop_length=160, n_mels=N_MELS, power=1.0
)

train_audio_transforms = nn.Sequential(
    spectrogram_transform,
    # torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    # torchaudio.transforms.TimeMasking(time_mask_param=35),
)


def get_seg(audio_path, start_idx):
    _, wav = scipy.io.wavfile.read(audio_path)
    # I want a chunk of at least two seconds
    if start_idx is not None:
        min_stop_idx = 2 * 16000
        if len(wav) <= min_stop_idx:
            stop_idx = len(wav)
        else:
            stop_idx = random.randrange(min_stop_idx, len(wav))
        return wav[start_idx:stop_idx]
    else:
        max_start_idx = len(wav) - 2 * 16000
        if max_start_idx <= 0:
            max_start_idx = 1
        start_idx = random.randrange(0, max_start_idx)
        return wav[start_idx:]


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
        p1_seg = random.sample(p1_segs, 1)[0]
        p1_wav = get_seg(p1_seg, None)
        p1_th = torch.from_numpy(p1_wav / 32767).float()
        same_seg = random.choice([True, False])
        if same_seg:
            # if i % 100 == 0:
            #    scipy.io.wavfile.write('/tmp/0_'+str(uuid.uuid4())+'.wav', 16000, p1_wav)
            return (p1_th, 0)
        else:
            p2_segs = self.segments[pair[1]]
            p2_seg = random.sample(p2_segs, 1)[0]
            p2_wav = get_seg(p2_seg, 0)
            p2_th = torch.from_numpy(p2_wav / 32767).float()
            # if i % 100 == 0:
            #    scipy.io.wavfile.write('/tmp/1_'+str(uuid.uuid4())+'.wav', 16000, np.concatenate([p1_wav,p2_wav]))
            return (torch.cat([p1_th, p2_th], dim=0), 1)


def collate_fn(data, data_type="train"):
    spectrograms = []
    labels = []
    for (waveform, binary_class) in data:
        if data_type == "train":
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)  # T, C
        else:
            spec = spectrogram_transform(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        labels.append(binary_class)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)  # B, T, C
    spectrograms = spectrograms.unsqueeze(1).transpose(2, 3)  # B, 1, C, T
    labels = torch.FloatTensor(labels).unsqueeze(1)
    return spectrograms, labels
