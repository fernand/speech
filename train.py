import time

from comet_ml import Experiment
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from warprnnt_pytorch import RNNTLoss

import words
import net
import data


class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(
    model, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment,
):
    model.train()
    data_len = len(train_loader.dataset)
    start = time.time()
    batch_start = start
    with experiment.train():
        for batch_idx, batch in enumerate(train_loader):
            spectrograms, labels, label_lengths = batch
            spectrograms = spectrograms.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            output = model(spectrograms, labels)  # B, T, U, n_class
            output = F.log_softmax(output, dim=3)

            act_lens = torch.full(
                (output.size(0),), output.size(1), dtype=torch.int32
            ).cuda()
            labels = labels.int().cuda()
            label_lengths = label_lengths.cuda()
            loss = criterion(output, labels, act_lens, label_lengths)
            loss.backward()

            experiment.log_metric("loss", loss.item(), step=iter_meter.get())
            experiment.log_metric(
                "learning_rate", scheduler.get_lr(), step=iter_meter.get()
            )

            optimizer.step()
            scheduler.step()
            iter_meter.step()
            if batch_idx % 100 == 0 or batch_idx == data_len:
                time_for_100_batches = round(time.time() - batch_start)
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tT100B: {}".format(
                        epoch,
                        batch_idx * len(spectrograms),
                        data_len,
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                        time_for_100_batches,
                    )
                )
                batch_start = time.time()
    epoch_time = round(time.time() - start)
    experiment.log_metric("epoch_time", epoch_time)


def test(model, test_loader, criterion, epoch, iter_meter, experiment):
    print("\nevaluating…")
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with experiment.test():
        with torch.no_grad():
            for I, batch in enumerate(test_loader):
                spectrograms, labels, label_lengths = batch
                spectrograms = spectrograms.cuda()
                labels = labels.cuda()

                output = model(spectrograms, labels)  # B, T, U, n_class
                output = F.log_softmax(output, dim=3)

                act_lens = torch.full(
                    (output.size(0),), output.size(1), dtype=torch.int32
                ).cuda()
                labels = labels.int().cuda()
                label_lengths = label_lengths.cuda()
                loss = criterion(output, labels, act_lens, label_lengths)

                decoded_preds, decoded_targets = GreedyDecoder(
                    output.transpose(0, 1), labels, label_lengths
                )
                for j in range(len(decoded_preds)):
                    test_cer.append(words.cer(decoded_targets[j], decoded_preds[j]))
                    test_wer.append(words.wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    experiment.log_metric("test_loss", test_loss, step=iter_meter.get())
    experiment.log_metric("cer", avg_cer, step=iter_meter.get())
    experiment.log_metric("wer", avg_wer, step=iter_meter.get())

    print(
        "Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n".format(
            test_loss, avg_cer, avg_wer
        )
    )


def GreedyDecoder(output, labels, label_lengths, blank_label=0, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(
            data.text_transform.int_to_text(labels[i][: label_lengths[i]].tolist())
        )
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(data.text_transform.int_to_text(decode))
    return decodes, targets


def main(hparams, experiment):
    experiment.log_parameters(hparams)
    torch.manual_seed(7)

    test_dataset = data.SortedTrainLibriSpeech("sorted_dev_clean_librispeech.pkl")
    train_dataset = data.SortedTrainLibriSpeech("sorted_train_librispeech.pkl")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        collate_fn=lambda x: data.collate_fn(x, "train"),
        num_workers=1,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        collate_fn=lambda x: data.collate_fn(x, "valid"),
        num_workers=5,
        pin_memory=True,
    )

    model = net.ContextNet(hparams["alpha"], hparams["n_feats"], hparams["n_class"])
    model.cuda()

    # print(model)
    print(
        "Num Model Parameters", sum([param.nelement() for param in model.parameters()])
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), hparams["learning_rate"], weight_decay=1e-6
    )
    criterion = RNNTLoss(blank=0).cuda()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams["learning_rate"],
        steps_per_epoch=int(len(train_loader)),
        epochs=hparams["epochs"],
        anneal_strategy="linear",
    )

    iter_meter = IterMeter()
    for epoch in range(1, hparams["epochs"] + 1):
        train(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            epoch,
            iter_meter,
            experiment,
        )
        test(model, test_loader, criterion, epoch, iter_meter, experiment)


if __name__ == "__main__":
    experiment = Experiment(
        api_key="IJIo1bzzY2MAGvPlhq9hA7qsb",
        project_name="general",
        workspace="fernand",
        disabled=True,
    )
    hparams = {
        "alpha": 0.5,
        "batch_size": 4,
        "epochs": 2,
        "learning_rate": 2.5e-3,
        "n_class": 29,
        "n_feats": 80,
        "dropout": 0.1,
    }
    main(hparams, experiment)
