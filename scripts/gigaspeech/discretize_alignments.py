import math
import multiprocessing
import pickle

import scipy.io.wavfile
import ujson

STEP_S = 0.01


def discretize(alignment, wav_len_s):
    result = list("0" * math.ceil(wav_len_s / STEP_S))
    for word in alignment:
        num_bins = int((word["end_ts"] - word["start_ts"]) / STEP_S)
        start_bin_idx = int(word["start_ts"] / STEP_S)
        for bin_idx in range(start_bin_idx, min(start_bin_idx + num_bins, len(result))):
            result[bin_idx] = "1"
    return "".join(result)


def process_alignment(audio_tuple):
    audio_file, _ = audio_tuple
    _, y = scipy.io.wavfile.read(audio_file)
    wav_len_s = len(y) / 16000
    with open(audio_file.replace(".wav", ".silero"), "rt") as f:
        payload = ujson.loads(f.read())
        if len(payload) != 2:
            discretization = "".join(list("0" * math.ceil(wav_len_s / STEP_S)))
        else:
            alignment = payload[1]
            discretization = discretize(alignment, wav_len_s)
    with open(audio_file.replace(".wav", ".disc"), "wt") as f:
        f.write(discretization)


if __name__ == "__main__":
    with open("../../datasets/gigaspeech/sorted_train_youtube_filtered.pkl", "rb") as f:
        dataset = pickle.load(f)
    p = multiprocessing.Pool(32)
    p.map(process_alignment, dataset)
