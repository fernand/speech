import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import data
import silero
from decoder import cer, wer


def collate_fn(data):
    return (
        silero.prepare_model_input(
            [t[0].squeeze(0) for t in data], device=torch.device("cpu")
        ),
        [t[1] for t in data],
    )


if __name__ == "__main__":
    dataset_type = sys.argv[1]

    device = torch.device(f"cuda:0")
    model, model_decoder = silero.load_silero_model(device)

    batch_size = None
    if dataset_type == "libri":
        dataset = data.SortedLibriSpeech(
            "datasets/librispeech/sorted_test_clean_librispeech.pkl",
            32,
        )
    elif dataset_type == "tv":
        train_dataset_paths = [
            "datasets/first/sorted_train_cer_0.2.pkl",
            "datasets/second/sorted_train_cer_0.2.pkl",
            "datasets/third/sorted_train_cer_0.2.pkl",
            "datasets/fourth/sorted_train_cer_0.2.pkl",
        ]
        eval_datasets = [
            dataset.replace("train", "eval") for dataset in train_dataset_paths
        ]
        dataset = data.SortedTV(eval_datasets, 32)
    elif dataset_type == "ibm":
        dataset = data.IBMDataset()
        batch_size = 32
    elif dataset_type == "eval_high_cer":
        dataset = data.SortedTV(["eval_paths_0.21cer.pkl"], 32)
    else:
        print("Unkown dataset", dataset_type)
        sys.exit(1)

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
    )

    test_cer, test_wer = [], []
    for batch in test_loader:
        wavs, transcripts = batch
        wavs = wavs.to(device)
        output = model(wavs).cpu()
        predictions = [model_decoder(output[i]) for i in range(len(output))]
        for transcript, prediction in zip(transcripts, predictions):
            # print(transcript, prediction)
            if len(prediction.strip()) == 0:
                char_error = 1.0
                word_error = 1.0
            else:
                char_error = cer(transcript, prediction)
                word_error = wer(transcript, prediction)
            test_cer.append(char_error)
            test_wer.append(word_error)

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    print("Average CER: {:4f} Average WER: {:.4f}\n".format(avg_cer, avg_wer))
