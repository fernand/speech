import gc
import os
import math
import multiprocessing
import pickle
import re
import shutil
import subprocess
import sys
import tempfile

import lz4.frame
import scipy.io.wavfile

sys.path.insert(0, "../..")
import process.int_to_words as int_to_words


def get_meta():
    with lz4.frame.open("/home/fernand/gs/gigaspeech.lzpkl", "rb") as f:
        return pickle.load(f)


def to_wav(audio_path, tmp_dir):
    output_path = os.path.join(
        tmp_dir, os.path.basename(audio_path).replace(".opus", ".wav")
    )
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
    return output_path


NUMBER_REGEXP = re.compile(r"\d+")
ACCENT_REGEXP = re.compile(r"[éèêëà]")
DASH_REGEXP = re.compile(r"-")
NON_ALPHA_QUOTE_REGEXP = re.compile(r"[^a-z\'\s]")
MULTI_SPACE_REGEXP = re.compile(r"\s+")


def replace_num(matchobj):
    return " " + int_to_words.name_number(int(matchobj.group(0))) + " "


ACCENT_DICT = {"é": "e", "è": "e", "ê": "e", "ë": "e", "à": "a"}

# Ignore the transcript if those symbols are included.
BLACKLIST = set(["[", "$", "¢", ".com"])


def remove_accent(matchobj):
    return ACCENT_DICT[matchobj.group(0)]


def process_utterances(audio_meta):
    txt_dir = os.path.join("/nvme/gigaspeech", os.path.dirname(audio_meta["path"]))
    for segment in audio_meta["segments"]:
        transcript = " ".join(
            [w for w in segment["text_tn"].lower().split(" ") if not w.startswith("<")]
        )
        if any([c in transcript for c in BLACKLIST]):
            continue
        transcript = re.sub(NUMBER_REGEXP, replace_num, transcript)
        transcript = re.sub(ACCENT_REGEXP, remove_accent, transcript)
        transcript = re.sub(DASH_REGEXP, " ", transcript)
        transcript = re.sub(NON_ALPHA_QUOTE_REGEXP, " ", transcript)
        transcript = re.sub(MULTI_SPACE_REGEXP, " ", transcript).lstrip()

        with open(os.path.join(txt_dir, segment["sid"] + ".txt"), "wt") as f:
            f.write(transcript)


def process_audio(audio_meta):
    tmp_dir = tempfile.mkdtemp()
    wav_path = to_wav(os.path.join("/nvme", audio_meta["path"]), tmp_dir)
    wav_chunk_dir = os.path.join(
        "/nvme/gigaspeech", os.path.dirname(audio_meta["path"])
    )
    os.makedirs(wav_chunk_dir, exist_ok=True)
    sr, y = scipy.io.wavfile.read(wav_path)
    len_y = len(y)
    assert sr == 16000
    for segment in audio_meta["segments"]:
        start = max(0, int(sr * segment["begin_time"]))
        end = min(len_y, math.ceil(sr * segment["end_time"]))
        output_f = os.path.join(wav_chunk_dir, segment["sid"] + ".wav")
        scipy.io.wavfile.write(output_f, sr, y[start:end])
    shutil.rmtree(tmp_dir)


def get_audio_meta(category, meta):
    category_meta = []
    for audio_meta in meta["audios"]:
        if category in audio_meta["path"]:
            category_meta.append(audio_meta)
    return category_meta


def count_files_processed(category):
    root_dir = f"/nvme/gigaspeech/audio/{category}"
    sub_dirs = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    num_files = 0
    for sub_dir in sub_dirs:
        master_files = set([f.split("_")[0] for f in os.listdir(sub_dir)])
        num_files += len(master_files)
    return num_files


def write_train_dataset(meta, category):
    dataset = []
    for audio_meta in meta:
        wav_dir = os.path.join("/nvme/gigaspeech", os.path.dirname(audio_meta["path"]))
        for segment in audio_meta["segments"]:
            wav_path = os.path.join(wav_dir, segment["sid"] + ".wav")
            duration = segment["end_time"] - segment["begin_time"]
            dataset.append((wav_path, duration))
    dataset = sorted(dataset, key=lambda t: t[1])
    with open(f"../../datasets/gigaspeech/sorted_train_{category}.pkl", "wb") as f:
        pickle.dump(dataset, f)


def write_filtered_dataset(
    dataset_pkl_path, category, min_duration=1.0, max_duration=8.0
):
    with open(dataset_pkl_path, "rb") as f:
        d = pickle.load(f)
    print(len(d))
    filtered_dataset = []
    for path, duration in d:
        if duration >= min_duration and duration <= max_duration:
            filtered_dataset.append((path, duration))
    print(len(filtered_dataset))
    with open(
        f"../../datasets/gigaspeech/sorted_train_{category}_filtered.pkl", "wb"
    ) as f:
        pickle.dump(filtered_dataset, f)


if __name__ == "__main__":
    CATEGORY = "audiobook"
    meta = get_meta()
    category_meta = get_audio_meta(CATEGORY, meta)
    print(len(category_meta))
    del meta
    gc.collect()
    p = multiprocessing.Pool(32)
    p.map(process_audio, category_meta)
    p.map(process_utterances, category_meta)
    write_train_dataset(category_meta, CATEGORY)
    write_filtered_dataset(
        f"../../datasets/gigaspeech/sorted_train_{CATEGORY}.pkl", CATEGORY
    )
