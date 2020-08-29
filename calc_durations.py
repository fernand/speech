import multiprocessing
import os
import pickle

import torchaudio
from torchaudio.datasets.utils import walk_files


def get_librispeech_audio_files(prefix):
    datasets = [
        "test-clean"
        #"train-clean-100",
        #"train-clean-360",
        #"train-other-500",
    ]
    dataset_paths = [os.path.join(prefix, "LibriSpeech", d) for d in datasets]
    audio_files = []
    for path in dataset_paths:
        walker = walk_files(path, suffix=".flac", prefix=True)
        audio_files.extend(list(walker))
    return audio_files


def get_duration(path):
    y, sample_rate = torchaudio.load(path)
    assert sample_rate == 16000
    return y.size(1)


if __name__ == "__main__":
    p = multiprocessing.Pool(6)
    audio_files = get_librispeech_audio_files("/data")
    durations = p.map(get_duration, audio_files)
    sorted_durations = sorted(zip(audio_files, durations), key=lambda t: t[1])
    #with open("sorted_train_librispeech.pkl", "wb") as f:
    with open("sorted_test_clean_librispeech.pkl", "wb") as f:
        pickle.dump(sorted_durations, f)
