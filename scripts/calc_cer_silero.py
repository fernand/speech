import multiprocessing
import os
import pickle
import shutil
import sys

import numpy as np
import scipy.io.wavfile

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from decoder import cer, wer
from silero import load_silero_model, wav_to_text


def processor(audio_files, input_dir, chunk_i):
    percent = len(audio_files) // 100
    model, decoder = load_silero_model()
    cers = []
    for i, audio_f in enumerate(audio_files):
        if chunk_i == 0 and i % percent == 0:
            print(i // percent)
        audio_f = os.path.join(input_dir, audio_f)
        transcript_f = audio_f.strip(".wav") + ".txt"
        if not os.path.exists(transcript_f):
            continue
        with open(transcript_f, "r") as f:
            transcript = f.read().strip()
        if len(transcript.strip()) == 0:
            continue
        sr, y = scipy.io.wavfile.read(audio_f)
        assert sr == 16000
        duration = len(y) / 16000
        prediction = wav_to_text(audio_f, model, decoder)
        if len(prediction.strip()) == 0:
            char_error = 1.0
            word_error = 1.0
        else:
            char_error = cer(transcript, prediction)
            word_error = wer(transcript, prediction)
        cers.append((audio_f, char_error, word_error, duration))
    return cers


if __name__ == "__main__":
    # /data/clean2
    input_dir = sys.argv[1]
    # datasets/second/
    output_dir = sys.argv[2]
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    num_workers = 6
    chunks = np.array_split(audio_files, num_workers)
    p = multiprocessing.Pool(num_workers)
    res = p.starmap(
        processor, [(chunk, input_dir, i) for i, chunk in enumerate(chunks)]
    )
    p.close()
    p.join()
    res = [e for t in res for e in t]
    with open(os.path.join(output_dir, "manifest.pkl"), "wb") as f:
        pickle.dump(res, f)
