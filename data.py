import torch
import torchaudio
import torch.nn as nn

from text import TextTransform

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, hop_length=160, n_mels=80, power=1.0
    ),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35),
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, hop_length=160, n_mels=80, power=1.0
)

text_transform = TextTransform()


def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == "train":
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0])
        label_lengths.append(len(label))

    spectrograms = (
        nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        .unsqueeze(1)
        .transpose(2, 3)
    )
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths
