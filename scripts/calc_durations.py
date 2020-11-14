import csv
import multiprocessing
import os
import pickle

import torchaudio
from torchaudio.datasets.utils import walk_files

torchaudio.set_audio_backend("sox_io")


def get_librispeech_audio_files(prefix):
    datasets = [
        "dev-clean"
        # "train-clean-100",
        # "train-clean-360",
        # "train-other-500",
    ]
    dataset_paths = [os.path.join(prefix, "LibriSpeech", d) for d in datasets]
    audio_files = []
    for path in dataset_paths:
        walker = walk_files(path, suffix=".flac", prefix=True)
        audio_files.extend(list(walker))
    return audio_files


def get_common_voice_audio_files(prefix):
    datasets = [
        "train.tsv",
        # "dev.tsv",
        # "test.tsv",
    ]
    dataset_prefix = os.path.join(prefix, "cv-corpus-5.1-2020-06-22/en")
    dataset_paths = [os.path.join(dataset_prefix, d) for d in datasets]
    audio_files = []
    for path in dataset_paths:
        with open(path, newline="") as tsv:
            reader = csv.DictReader(tsv, delimiter="\t")
            for row in reader:
                audio_files.append(os.path.join(dataset_prefix, "clips", row["path"]))
    return audio_files


def get_duration(path):
    y, sample_rate = torchaudio.load(path)
    return y.size(1) / sample_rate


if __name__ == "__main__":
    p = multiprocessing.Pool(6)
    # audio_files = get_librispeech_audio_files("/data")
    audio_files = get_common_voice_audio_files("/data")
    durations = p.map(get_duration, audio_files)
    sorted_durations = sorted(zip(audio_files, durations), key=lambda t: t[1])
    # with open("sorted_train_librispeech.pkl", "wb") as f:
    with open("sorted_train_commonvoice.pkl", "wb") as f:
        pickle.dump(sorted_durations, f)
