import csv
import re
import os

import joblib
import tqdm

rgxp = re.compile(r"[^a-z \']")


def get_common_voice_transcripts(prefix):
    datasets = [
        "train.tsv",
        # "dev.tsv",
        # "test.tsv",
    ]
    dataset_prefix = os.path.join(prefix, "cv-corpus-5.1-2020-06-22/en")
    dataset_paths = [os.path.join(dataset_prefix, d) for d in datasets]
    transcripts = []
    for path in dataset_paths:
        with open(path, newline="") as tsv:
            reader = csv.DictReader(tsv, delimiter="\t")
            for row in reader:
                audio_path = os.path.join(dataset_prefix, "clips", row["path"])
                txt_path = audio_path.replace(".mp3", ".txt")
                transcript = row["sentence"].lower().replace("-", " ")
                transcript = re.sub(rgxp, "", transcript)
                transcripts.append((txt_path, transcript))
    return transcripts


def write_transcript(txt_path, transcript):
    with open(txt_path, "w") as f:
        f.write(transcript)


if __name__ == "__main__":
    transcripts = get_common_voice_transcripts("/data")
    joblib.Parallel(n_jobs=6)(
        joblib.delayed(write_transcript)(txt_path, transcript)
        for txt_path, transcript in tqdm.tqdm(transcripts)
    )
