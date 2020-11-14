import csv
import os
import pickle
import subprocess

from torchaudio.datasets.utils import walk_files
import joblib
import tqdm


def to_wav(audio_path):
    output_path = audio_path.replace(".mp3", ".wav")
    ffmpeg = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "panic",
        "-i",
        audio_path,
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        output_path,
    ]
    p = subprocess.Popen(ffmpeg, stderr=subprocess.PIPE)
    p.communicate()


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


if __name__ == "__main__":
    # audio_files = get_librispeech_audio_files("/hd1")
    audio_files = get_common_voice_audio_files("/data")
    joblib.Parallel(n_jobs=6)(
        joblib.delayed(to_wav)(audio_path) for audio_path in tqdm.tqdm(audio_files)
    )
