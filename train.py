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


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The totale number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


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
                test_loss += loss.item() / len(test_loader)

    experiment.log_metric("test_loss", test_loss, step=iter_meter.get())
    print("Test set: Average loss: {:.4f}\n".format(test_loss))


def main(hparams, experiment):
    experiment.log_parameters(hparams)
    torch.manual_seed(7)

    test_dataset = data.SortedTrainLibriSpeech("sorted_dev_clean_librispeech.pkl")
    train_dataset = data.SortedTrainLibriSpeech("sorted_train_librispeech.pkl")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=hparams["shuffle"],
        collate_fn=lambda x: data.collate_fn(x, "train"),
        num_workers=1,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        collate_fn=lambda x: data.collate_fn(x, "valid"),
        num_workers=5,
        pin_memory=True,
    )

    model = net.ContextNet(hparams["alpha"], hparams["n_feats"], hparams["n_class"])
    model = nn.DataParallel(model)
    model.cuda()

    # print(model)
    print(
        "Num Model Parameters", sum([param.nelement() for param in model.parameters()])
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), hparams["learning_rate"], weight_decay=1e-6
    )
    criterion = RNNTLoss(blank=0).cuda()
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, 15000, hparams["epochs"] * len(train_loader)
    # )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams["learning_rate"],
        steps_per_epoch=len(train_loader),
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
        # disabled=True,
    )
    hparams = {
        "alpha": 0.5,
        "shuffle": True,
        "batch_size": 22,
        "epochs": 2,
        "learning_rate": 2.5e-3,
        "n_class": 29,
        "n_feats": 80,
        "dropout": 0.1,
    }
    main(hparams, experiment)
