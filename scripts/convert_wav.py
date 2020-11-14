import os
import pickle
import subprocess

from torchaudio.datasets.utils import walk_files
import joblib
import tqdm


def to_wav(audio_path):
    os.remove(audio_path.replace(".flac", ".wav"))


def get_librispeech_audio_files(prefix):
    datasets = [
        # "dev-clean"
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ]
    dataset_paths = [os.path.join(prefix, "LibriSpeech", d) for d in datasets]
    audio_files = []
    for path in dataset_paths:
        walker = walk_files(path, suffix=".flac", prefix=True)
        audio_files.extend(list(walker))
    return audio_files


if __name__ == "__main__":
    audio_files = get_librispeech_audio_files("/hd1")
    joblib.Parallel(n_jobs=6)(
        joblib.delayed(to_wav)(audio_path) for audio_path in tqdm.tqdm(audio_files)
    )
