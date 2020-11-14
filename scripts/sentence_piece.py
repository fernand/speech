import pickle

import joblib
import torchtext
import tqdm


def read_transcript(txt_f):
    with open(txt_f, "r") as f:
        return f.read().strip()


# Only do the TV dataset for now
def gather_sentences(dataset_paths):
    all_paths = []
    for dataset_path in dataset_paths:
        with open(dataset_path, "rb") as f:
            paths = [t[0] for t in pickle.load(f)]
            paths = [p.replace("/data", "/hd1") for p in paths]
            all_paths.extend(paths)
    sentences = joblib.Parallel(n_jobs=3)(
        joblib.delayed(read_transcript)(audio_f.strip(".wav") + ".txt")
        for audio_f in tqdm.tqdm(all_paths)
    )
    with open("sp/sentences.txt", "w") as f:
        f.write("\n".join(sentences))


if __name__ == "__main__":
    train_dataset_path = [
        "datasets/first/sorted_train_cer_0.2.pkl",
        "datasets/second/sorted_train_cer_0.2.pkl",
        "datasets/third/sorted_train_cer_0.2.pkl",
    ]
    # gather_sentences(train_dataset_path)
    torchtext.data.functional.generate_sp_model(
        "sp/sentences.txt", vocab_size=999, model_type="bpe", model_prefix="sp/tv"
    )
