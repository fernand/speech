import pickle
import os
import sys

import joblib
import tqdm


def remove_file(audio_f):
    os.remove(audio_f)
    os.remove(audio_f.strip(".wav") + ".txt")


if __name__ == "__main__":
    dataset_dir = sys.argv[1]
    with open(os.path.join(dataset_dir, "manifest.pkl"), "rb") as f:
        manifest = pickle.load(f)
    files_to_trim = [t[0] for t in manifest if t[1] > 0.3]
    joblib.Parallel(n_jobs=6)(
        joblib.delayed(remove_file)(audio_f) for audio_f in tqdm.tqdm(files_to_trim)
    )

