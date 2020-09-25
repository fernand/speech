import multiprocessing
import os
import pickle
import shutil
import sys
import time

import numpy as np
import scipy.io.wavfile

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from decoder import cer, wer
from silero import load_silero_model, wav_to_text


def processor(audio_files, input_dir):
    model, decoder = load_silero_model()
    manifest = []
    for i, audio_f in enumerate(audio_files):
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
        manifest.append((audio_f, char_error, word_error, duration))
    return manifest


if __name__ == "__main__":
    # /data/clean2
    input_dir = sys.argv[1]
    # datasets/second/
    output_dir = sys.argv[2]
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    audio_files = sorted(audio_files)
    num_chunks = 10
    chunks = np.array_split(audio_files, num_chunks)
    num_workers = 6
    for chunk_i, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_i} of {num_chunks - 1}")
        start = time.time()
        sub_chunks = np.array_split(chunk, num_workers)
        p = multiprocessing.Pool(num_workers)
        res = p.starmap(processor, [(sub_chunk, input_dir) for sub_chunk in sub_chunks])
        p.close()
        p.join()
        res = [e for t in res for e in t]
        with open(os.path.join(output_dir, f"manifest_{chunk_i}.pkl"), "wb") as f:
            pickle.dump(res, f)
        del res
        print(f"Time for chunk: {time.time() - start}")
