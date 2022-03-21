import os
import pickle

import joblib
import requests
import tqdm
import ujson

URL = "http://localhost:4000/align"
BS = 1456


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def align(audio_paths):
    response = requests.post(
        URL, json=ujson.dumps({"audio_paths": audio_paths, "batch_size": BS})
    )
    alignments = ujson.loads(response.text)
    for audio_path, alignment in alignments.items():
        dirname = os.path.dirname(audio_path)
        filename, _ = os.path.splitext(os.path.basename(audio_path))
        with open(f"{os.path.join(dirname, filename)}.silero", "wt") as f:
            f.write(ujson.dumps(alignment))


if __name__ == "__main__":
    with open("../../datasets/gigaspeech/sorted_train_youtube_filtered.pkl", "rb") as f:
        dataset = pickle.load(f)
    path_chunks = list(chunks([t[0] for t in dataset], BS))
    _ = joblib.Parallel(n_jobs=2)(
        joblib.delayed(align)(path_chunks[i])
        for i in tqdm.tqdm(range(len(path_chunks)))
    )
