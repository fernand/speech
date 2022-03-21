import os

from typing import List

import psutil
import scipy.io.wavfile
import torch


def read_audio(path: str):
    sr, wav = scipy.io.wavfile.read(path)
    assert sr == 16000
    assert wav.ndim == 1
    wav = wav / 32767
    return torch.from_numpy(wav)


def read_batch(audio_paths: List[str]):
    return [read_audio(audio_path) for audio_path in audio_paths]


def split_into_batches(lst: List[str], batch_size=10):
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def prepare_model_input(batch: List[torch.Tensor], device):
    max_seqlength = max(max([len(_) for _ in batch]), 12800)
    inputs = torch.zeros(len(batch), max_seqlength)
    for i, wav in enumerate(batch):
        inputs[i, : len(wav)].copy_(wav)
    inputs = inputs.to(device)
    return inputs


def get_model_tuple():
    pp = psutil.Process(os.getppid())
    worker_pids = sorted([p.pid for p in pp.children()])
    device_id = worker_pids.index(os.getpid())
    return (
        torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_stt",
            language="en",
            device=torch.device(f"cuda:{device_id}"),
        ),
        device_id,
    )
