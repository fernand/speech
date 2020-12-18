import os
import pickle

import joblib
import torchaudio
import tqdm

torchaudio.set_audio_backend("sox_io")


def check_wav(audio_path):
    res = None
    try:
        waveform, sample_rate = torchaudio.load(audio_path, normalization=True)
    except:
        res = audio_path
    return res


if __name__ == "__main__":
    audio_files = [
        os.path.join("/data/cv-corpus-5.1-2020-06-22/en/clips", f)
        for f in os.listdir("/data/cv-corpus-5.1-2020-06-22/en/clips")
        if f.endswith(".wav")
    ]
    results = joblib.Parallel(n_jobs=6)(
        joblib.delayed(check_wav)(audio_path) for audio_path in tqdm.tqdm(audio_files)
    )
    results = [res for res in results if res is not None]
    with open("datasets/commonvoice/bad_files.pkl", "wb") as f:
        pickle.dump(set(results), f)
