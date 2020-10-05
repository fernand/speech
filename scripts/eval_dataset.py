import os
import sys
import time

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import data
import net
import decoder


def test(batch_size, model, test_loader, criterion):
    print("\nevaluating…")
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for I, batch in enumerate(test_loader):
            spectrograms, labels, label_lengths = batch
            spectrograms = spectrograms.cuda()

            output = model(spectrograms)  # B, T, n_vocab+1
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1).contiguous()  # T, B, n_vocab+1

            input_lengths = torch.full(
                (batch_size,), output.size(0), dtype=torch.int32
            ).cuda()
            label_lengths = label_lengths.cuda()
            labels = labels.cuda()
            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            output = output.cpu()
            labels = labels.cpu()
            label_lengths = label_lengths.cpu()
            decoded_preds, decoded_targets = decoder.greedy_decoder(
                output.transpose(0, 1), labels, label_lengths
            )
            for j in range(len(decoded_preds)):
                test_cer.append(decoder.cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(decoder.wer(decoded_targets[j], decoded_preds[j]))
    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    print(
        "Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n".format(
            test_loss, avg_cer, avg_wer
        )
    )


if __name__ == "__main__":
    model_file = sys.argv[1]
    hparams = {
        "shuffle": True,
        "batch_size": 32,
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
    dataset = data.SortedLibriSpeech(
        "datasets/librispeech/sorted_test_clean_librispeech.pkl", hparams["batch_size"]
    )
    model = net.SRModel(
        hparams["n_cnn_layers"],
        hparams["n_rnn_layers"],
        hparams["rnn_dim"],
        hparams["n_vocab"],
        hparams["n_feats"],
        hparams["dropout"],
    )
    model.cuda()
    model.load_state_dict(torch.load(model_file))
    criterion = torch.nn.CTCLoss(blank=0).cuda()
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=None,
        # Also shuffling at the clip level in data.py
        shuffle=True,
        collate_fn=lambda x: data.collate_fn(x, "valid"),
        num_workers=2,
        pin_memory=True,
    )
    test(hparams["batch_size"], model, test_loader, criterion)
