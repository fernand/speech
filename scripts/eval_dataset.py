import os
import pickle
import sys
import time

# import ctcdecode
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import data
import net
import decoder
import text


def test(dataset_type, batch_size, model, test_loader, beam_decode):
    # if beam_decode:
    #    if dataset_type == "tv" or dataset_type == "ibm":
    #        model_path = "lm/tv-1234-lm.arpa"
    #    elif dataset_type == "libri":
    #        model_path = "lm/libri-lm.arpa"
    #    else:
    #        print("No LM for dataset.")
    #        sys.exit(1)
    #    beam_decoder = ctcdecode.CTCBeamDecoder(
    #        labels=list(text.CHARS),
    #        model_path=model_path,
    #        beta=0.1,
    #        blank_id=0,
    #        beam_width=400,
    #        num_processes=4,
    #        log_probs_input=True,
    #    )
    print("\nevaluating...")
    model.eval()
    test_cer, test_wer = [], []
    bad_cers = []
    with torch.no_grad():
        for I, batch in enumerate(test_loader):
            spectrograms, labels, label_lengths = batch
            current_batch_size = labels.size(0)
            spectrograms = spectrograms.half().cuda()

            output = model(spectrograms)  # B, T, n_vocab+1
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1).contiguous()  # T, B, n_vocab+1

            output = output.cpu().transpose(0, 1)
            labels = labels.cpu()
            label_lengths = label_lengths.cpu()
            # if beam_decode:
            #    beam_results, _, _, out_len = beam_decoder.decode(output)
            #    decoded_preds, decoded_targets = [], []
            #    for j in range(current_batch_size):
            #        decoded_preds.append(
            #            text.int_to_text(beam_results[j][0][: out_len[j][0]].numpy())
            #        )
            #        decoded_targets.append(
            #            text.int_to_text(labels[j][: label_lengths[j]].tolist())
            #        )
            # else:
            decoded_preds, decoded_targets = decoder.greedy_decoder(
                output, labels, label_lengths
            )
            for j in range(current_batch_size):
                # print(decoded_targets[j], decoded_preds[j])
                cer = decoder.cer(decoded_targets[j], decoded_preds[j])
                test_cer.append(cer)
                if cer >= 0.05:
                    bad_cers.append((decoded_targets[j], decoded_preds[j], cer))
                test_wer.append(decoder.wer(decoded_targets[j], decoded_preds[j]))
    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    print("Test set: Average CER: {:4f} Average WER: {:.4f}\n".format(avg_cer, avg_wer))
    # with open("bad_cers.pkl", "wb") as f:
    #    pickle.dump(sorted(bad_cers, key=lambda t: t[2], reverse=True), f)


if __name__ == "__main__":
    beam_decode = eval(sys.argv[1])
    dataset_type = sys.argv[2]
    model_file = sys.argv[3]
    hparams = {
        "shuffle": True,
        "batch_size": 64,
        "epochs": 15,
        "learning_rate": 3e-4,
        "n_cnn_layers": 3,
        "n_rnn_layers": 10,
        "rnn_dim": 512,
        "dropout": 0.1,
        # Does not include the blank.
        "n_vocab": 28,
        "n_feats": data.N_MELS,
    }
    model = net.SRModel(
        hparams["n_cnn_layers"],
        hparams["n_rnn_layers"],
        hparams["rnn_dim"],
        hparams["n_vocab"],
        hparams["n_feats"],
        hparams["dropout"],
    )
    model.cuda()
    batch_size = None
    if dataset_type == "libri":
        dataset = data.SortedLibriSpeech(
            "datasets/librispeech/sorted_test_clean_librispeech.pkl",
            hparams["batch_size"],
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
        dataset = data.SortedTV(eval_datasets, hparams["batch_size"])
    elif dataset_type == "ibm":
        dataset = data.IBMDataset()
        batch_size = hparams["batch_size"]
    elif dataset_type == "eval_high_cer":
        dataset = data.SortedTV(["eval_paths_0.21cer.pkl"], hparams["batch_size"])
    else:
        print("Unkown dataset", dataset_type)
        sys.exit(1)
    model.load_state_dict(torch.load(model_file))
    model.half()
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: data.collate_fn(x, "valid"),
        num_workers=3,
        pin_memory=True,
    )
    test(dataset_type, hparams["batch_size"], model, test_loader, beam_decode)
