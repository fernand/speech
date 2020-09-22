import multiprocessing
import os
import pathlib
import pickle
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from decoder import cer
from silero import load_silero_model, wav_to_text


def processor(audio_files, input_dir):
    model, decoder = load_silero_model()
    cers = []
    for audio_f in audio_files:
        audio_f = os.path.join(input_dir, audio_f)
        transcript_f = audio_f.strip(".wav") + ".txt"
        with open(transcript_f, "r") as f:
            transcript = f.read().strip()
        prediction = wav_to_text(audio_f, model, decoder)
        cers.append((audio_f, cer(transcript, prediction)))
    return cers


if __name__ == "__main__":
    input_dir = "/data/clean"
    with open("manifest.pkl", "rb") as f:
        audio_files = pickle.load(f)
    num_workers = 6
    chunks = np.array_split(audio_files, len(audio_files) // num_workers)
    p = multiprocessing.Pool(num_workers)
    res = p.starmap(processor, [(chunk, input_dir) for chunk in chunks])
    with open("cers.pkl", "wb") as f:
        pickle.dump(res, f)
