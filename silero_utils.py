import torch
import tempfile
import warnings
import torchaudio
from typing import List
from itertools import groupby

# Download model from https://github.com/snakers4/silero-models/blob/master/models.yml#L9


torchaudio.set_audio_backend("sox_io")


def read_batch(audio_paths: List[str]):
    return [read_audio(audio_path) for audio_path in audio_paths]


def split_into_batches(lst: List[str], batch_size: int = 10):
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def read_audio(path: str, target_sr: int = 16000):
    wav, sr = torchaudio.load(path, normalize=True)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = transform(wav)
        sr = target_sr

    assert sr == target_sr
    return wav.squeeze(0)


def prepare_model_input(batch: List[torch.Tensor], device=torch.device("cpu")):
    max_seqlength = max(max([len(_) for _ in batch]), 12800)
    inputs = torch.zeros(len(batch), max_seqlength)
    for i, wav in enumerate(batch):
        inputs[i, : len(wav)].copy_(wav)
    inputs = inputs.to(device)
    return inputs


class Decoder:
    def __init__(self, labels: List[str]):
        self.labels = labels
        self.blank_idx = self.labels.index("_")
        self.space_idx = self.labels.index(" ")

    def process(self, probs, wav_len, word_align):
        assert len(self.labels) == probs.shape[1]
        for_string = []
        argm = torch.argmax(probs, axis=1).numpy()
        align_list = [[]]
        for j, i in enumerate(argm):
            if i == self.labels.index("2"):
                try:
                    prev = for_string[-1]
                    for_string.append("$")
                    for_string.append(prev)
                    align_list[-1].append(j)
                    continue
                except:
                    for_string.append(" ")
                    warnings.warn(
                        'Token "2" detected a the beginning of sentence, omitting'
                    )
                    align_list.append([])
                    continue
            if i != self.blank_idx:
                for_string.append(self.labels[i])
                if i == self.space_idx:
                    align_list.append([])
                else:
                    align_list[-1].append(j)

        string = "".join([x[0] for x in groupby(for_string)]).replace("$", "").strip()

        align_list = list(filter(lambda x: x, align_list))

        if align_list and wav_len and word_align:
            align_dicts = []
            linear_align_coeff = wav_len / len(argm)
            to_move = min(align_list[0][0], 1.5)
            for i, align_word in enumerate(align_list):
                if len(align_word) == 1:
                    align_word.append(align_word[0])
                align_word[0] = align_word[0] - to_move
                if i == (len(align_list) - 1):
                    to_move = min(1.5, len(argm) - i)
                    align_word[-1] = align_word[-1] + to_move
                else:
                    to_move = min(1.5, (align_list[i + 1][0] - align_word[-1]) / 2)
                    align_word[-1] = align_word[-1] + to_move

            for word, timing in zip(string.split(), align_list):
                align_dicts.append(
                    {
                        "word": word,
                        "start_ts": round(timing[0] * linear_align_coeff, 2),
                        "end_ts": round(timing[-1] * linear_align_coeff, 2),
                    }
                )

            return string, align_dicts
        return string

    def __call__(
        self, probs: torch.Tensor, wav_len: float = 0, word_align: bool = False
    ):
        return self.process(probs, wav_len, word_align)


class FastDecoder:
    def __init__(self, labels: List[str]):
        self.labels = labels
        self.blank_idx = self.labels.index("_")
        self.space_idx = self.labels.index(" ")
        self.two_index = self.labels.index("2")

    def __call__(self, probs: torch.Tensor):
        for_string = []
        argm = torch.argmax(probs, axis=1).numpy()
        for j, i in enumerate(argm):
            if i == self.two_index:
                try:
                    prev = for_string[-1]
                    for_string.append("$")
                    for_string.append(prev)
                    continue
                except:
                    for_string.append(" ")
                    warnings.warn(
                        'Token "2" detected a the beginning of sentence, omitting'
                    )
                    continue
            if i != self.blank_idx:
                for_string.append(self.labels[i])

        string = "".join([x[0] for x in groupby(for_string)]).replace("$", "").strip()
        return string


def init_jit_model(device: torch.device = torch.device("cpu"), fast_decoder=True):
    torch.set_grad_enabled(False)
    model = torch.jit.load("silero_en_v2_jit.model", map_location=device)
    model.eval()
    if fast_decoder:
        return model, FastDecoder(model.labels)
    else:
        return model, Decoder(model.labels)


def wav_to_text(f, model, decoder, device):
    batch = read_batch([f])
    inp = prepare_model_input(batch, device=device)
    output = model(inp)
    return decoder(output[0].cpu())
