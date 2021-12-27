import os
import math
import sys
from multiprocessing import Pool

import stt
import scipy.io.wavfile
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import decoder


def split_into_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def process_chunk(chunk_idx, chunk):
    model = stt.Model("/home/fernand/speech/coqui/model.tflite")
    cers, wers = [], []
    if chunk_idx == 0:
        itr = tqdm(chunk)
    else:
        itr = chunk
    for audio_path in itr:
        sr, y = scipy.io.wavfile.read(audio_path)
        with open(audio_path.replace(".wav", ".txt")) as f:
            utterance = f.read().strip()
        text = model.stt(y)
        cers.append(decoder.cer(utterance, text))
        wers.append(decoder.wer(utterance, text))
    return (cers, wers)


if __name__ == "__main__":
    audio_paths = [
        os.path.join("/hd1/ibm", f) for f in os.listdir("/hd1/ibm") if f.endswith("wav")
    ]
    num_workers = 16
    num_per_chunk = math.ceil(len(audio_paths) / num_workers)
    chunks = list(split_into_chunks(audio_paths, num_per_chunk))
    p = Pool(num_workers)
    res = p.starmap(process_chunk, [(i, chunks[i]) for i in range(len(chunks))])

    test_cer, test_wer = [], []
    for cers, wers in res:
        test_cer.extend(cers)
        test_wer.extend(wers)
    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    print("Test set: Average CER: {:4f} Average WER: {:.4f}\n".format(avg_cer, avg_wer))
