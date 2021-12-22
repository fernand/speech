import gc
import os
import math
import multiprocessing
import pickle
import shutil
import subprocess
import tempfile

import lz4.frame
import scipy.io.wavfile


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


def process_utterances(audio_meta):
    txt_dir = os.path.join("/nvme/gigaspeech", os.path.dirname(audio_meta["path"]))
    for segment in audio_meta["segments"]:
        cleaned_utterance = " ".join(
            [w for w in segment["text_tn"].lower().split(" ") if not w.startswith("<")]
        )
        with open(os.path.join(txt_dir, segment["sid"] + ".txt"), "wt") as f:
            f.write(cleaned_utterance)


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


def get_youtube_audio_meta(meta):
    yt_meta = []
    for audio_meta in meta["audios"]:
        if "youtube" in audio_meta["path"]:
            yt_meta.append(audio_meta)
    return yt_meta


def count_files_processed():
    root_dir = "/nvme/gigaspeech/audio/youtube"
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
    with open(f"sorted_train_{category}.pkl", "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    meta = get_meta()
    yt_meta = get_youtube_audio_meta(meta)
    print(len(yt_meta))
    del meta
    gc.collect()
    # p = multiprocessing.Pool(32)
    # p.map(process_audio, yt_meta)
    # p.map(process_utterances, yt_meta)
    write_train_dataset(yt_meta, "youtube")
