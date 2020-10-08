import math
import time

from comet_ml import Experiment
import apex
import torch
import torchaudio
import torch.nn.functional as F

import data
import net
import decoder


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return torch.optim.lr.LambdaLR(optimizer, lr_lambda, last_epoch)


class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(
    batch_size,
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    epoch,
    iter_meter,
    num_epochs,
    experiment,
):
    model.train()
    data_len = len(train_loader.dataset)
    start = time.time()
    batch_start = start
    with experiment.train():
        for batch_idx, batch in enumerate(train_loader):
            spectrograms, labels, label_lengths = batch
            spectrograms = spectrograms.cuda()

            optimizer.zero_grad()

            output = model(spectrograms)  # B, T, n_vocab+1
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1).contiguous()  # T, B, n_vocab+1

            input_lengths = torch.full(
                (batch_size,), output.size(0), dtype=torch.int32
            ).cuda()
            label_lengths = label_lengths.cuda()
            labels = labels.cuda()
            loss = criterion(output, labels, input_lengths, label_lengths)
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            experiment.log_metric("loss", loss.item(), step=iter_meter.get())
            experiment.log_metric(
                "learning_rate", scheduler.get_lr(), step=iter_meter.get()
            )

            optimizer.step()
            scheduler.step()
            iter_meter.step()
            if batch_idx % 100 == 0:
                time_for_100_batches = round(time.time() - batch_start)
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tT100B: {}".format(
                        epoch,
                        batch_idx,
                        data_len,
                        100.0 * batch_idx / data_len,
                        loss.item(),
                        time_for_100_batches,
                    )
                )
                batch_start = time.time()
    epoch_time = round(time.time() - start)
    experiment.log_metric("epoch_time", epoch_time)
    if epoch == num_epochs:
        exp_id = experiment.url.split("/")[-1]
        torch.save(model.state_dict(), f"model_{exp_id}_{epoch}.pth")


def test(batch_size, model, test_loader, criterion, epoch, iter_meter, experiment):
    print("\nevaluating…")
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with experiment.test():
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
    experiment.log_metric("test_loss", test_loss, step=iter_meter.get())
    experiment.log_metric("cer", avg_cer, step=iter_meter.get())
    experiment.log_metric("wer", avg_wer, step=iter_meter.get())

    print(
        "Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n".format(
            test_loss, avg_cer, avg_wer
        )
    )


def main(hparams, experiment):
    experiment.log_parameters(hparams)
    torch.manual_seed(7)
    torch.backends.cudnn.benchmark = True

    eval_datasets = [
        dataset.replace("train", "eval") for dataset in hparams["train_dataset"]
    ]
    test_dataset = data.SortedTV(eval_datasets, hparams["batch_size"])
    train_dataset = data.SortedTV(hparams["train_dataset"], hparams["batch_size"])
    # train_dataset = data.SortedLibriSpeech(
    #     "datasets/librispeech/sorted_train_librispeech.pkl", hparams["batch_size"]
    # )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=None,
        shuffle=hparams["shuffle"],
        collate_fn=lambda x: data.collate_fn(x, "train"),
        num_workers=3,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=None,
        # Also shuffling at the clip level in data.py
        shuffle=True,
        collate_fn=lambda x: data.collate_fn(x, "valid"),
        num_workers=2,
        pin_memory=True,
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
    optimizer = apex.optimizers.FusedAdam(
        model.parameters(),
        lr=hparams["learning_rate"],
        adam_w_mode=False,
        weight_decay=0.01,
        # Different than default Pytorch.
        amsgrad=False,
    )
    model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O2")

    print(
        "Num Model Parameters", sum([param.nelement() for param in model.parameters()])
    )

    criterion = torch.nn.CTCLoss(blank=0).cuda()
    # scheduler = get_linear_schedule_with_warmup(
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 7000, hparams["epochs"] * len(train_loader)
    )

    iter_meter = IterMeter()
    for epoch in range(1, hparams["epochs"] + 1):
        train(
            hparams["batch_size"],
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            epoch,
            iter_meter,
            hparams["epochs"],
            experiment,
        )
        test(
            hparams["batch_size"],
            model,
            test_loader,
            criterion,
            epoch,
            iter_meter,
            experiment,
        )


if __name__ == "__main__":
    train_dataset_path = [
        "datasets/first/sorted_train_cer_0.2.pkl",
        "datasets/second/sorted_train_cer_0.2.pkl",
        "datasets/third/sorted_train_cer_0.2.pkl",
    ]
    experiment = Experiment(
        api_key="IJIo1bzzY2MAGvPlhq9hA7qsb",
        project_name="general",
        workspace="fernand",
        # disabled=True,
    )
    hparams = {
        "shuffle": True,
        "batch_size": 32,
        "epochs": 20,
        "learning_rate": 3e-4,
        "n_cnn_layers": 3,
        "n_rnn_layers": 10,
        "rnn_dim": 512,
        "dropout": 0.1,
        # Does not include the blank.
        "n_vocab": 28,
        "n_feats": data.N_MELS,
        "train_dataset": train_dataset_path,
    }
    main(hparams, experiment)
