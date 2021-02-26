import os
import sys
from multiprocessing import Pool
from statistics import mean

from deepspeech import Model
from tqdm import tqdm
import scipy.io.wavfile as wav

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from decoder import cer, wer

if __name__ == '__main__':
    audio_paths = [
        os.path.join("/hd1/ibm", f)
        for f in os.listdir("/hd1/ibm")
        if f.endswith("wav")
    ]
    ds = Model('mds/deepspeech-0.9.3-models.pbmm')
    cers, wers = [], []
    for audio_path in tqdm(audio_paths):
        audio = wav.read(audio_path)[1]
        prediction = ds.stt(audio)
        with open(audio_path.replace('.wav', '.txt')) as f:
            transcript = f.read().strip()
        if len(prediction.strip()) == 0:
            char_error = 1.0
            word_error = 1.0
        else:
            char_error = cer(transcript, prediction)
            word_error = wer(transcript, prediction)
        cers.append(char_error)
        wers.append(word_error)
    print('mean cer', mean(cers))
    print('mean wer', mean(wers))
