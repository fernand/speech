import code
import math
import signal
import sys
import time

from comet_ml import Experiment
import apex
import torch
import torch.nn.functional as F

import data
import net
import decoder
import text


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
                time_for_100_batches = round(time.time() - batch_start, 1)
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


def eval_dataset(experiment, model, criterion, loader, name, iter_meter):
    eval_loss = 0
    eval_cer, eval_wer = [], []
    for I, batch in enumerate(loader):
        spectrograms, labels, label_lengths = batch
        current_batch_size = labels.size(0)
        spectrograms = spectrograms.cuda()

        output = model(spectrograms)  # B, T, n_vocab+1
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1).contiguous()  # T, B, n_vocab+1

        input_lengths = torch.full(
            (current_batch_size,), output.size(0), dtype=torch.int32
        ).cuda()
        label_lengths = label_lengths.cuda()
        labels = labels.cuda()
        loss = criterion(output, labels, input_lengths, label_lengths)
        eval_loss += loss.item() / len(loader)

        output = output.cpu()
        labels = labels.cpu()
        label_lengths = label_lengths.cpu()
        decoded_preds, decoded_targets = decoder.greedy_decoder(
            output.transpose(0, 1), labels, label_lengths
        )
        for j in range(len(decoded_preds)):
            eval_cer.append(decoder.cer(decoded_targets[j], decoded_preds[j]))
            eval_wer.append(decoder.wer(decoded_targets[j], decoded_preds[j]))
    avg_cer = sum(eval_cer) / len(eval_cer)
    avg_wer = sum(eval_wer) / len(eval_wer)
    experiment.log_metric(f"{name}_loss", eval_loss, step=iter_meter.get())
    experiment.log_metric(f"{name}_cer", avg_cer, step=iter_meter.get())
    experiment.log_metric(f"{name}_wer", avg_wer, step=iter_meter.get())
    print(
        "{}: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n".format(
            name, eval_loss, avg_cer, avg_wer
        )
    )
    return avg_cer


def eval(
    batch_size,
    model,
    eval_loader,
    ibm_loader,
    criterion,
    epoch,
    iter_meter,
    experiment,
    last_cer,
):
    print("\nevaluating…")
    model.eval()
    with experiment.test():
        with torch.no_grad():
            _ = eval_dataset(
                experiment, model, criterion, eval_loader, "tv", iter_meter
            )
            avg_cer = eval_dataset(
                experiment, model, criterion, ibm_loader, "ibm", iter_meter
            )
    if avg_cer < last_cer:
        exp_id = experiment.url.split("/")[-1]
        torch.save(model.state_dict(), f"model_{exp_id}.pth")
    return avg_cer


def main(hparams, experiment):
    experiment.log_parameters(hparams)
    torch.manual_seed(7)
    torch.backends.cudnn.benchmark = True

    datasets = hparams["datasets"].split("-")
    if "tv" in datasets:
        tv_train_dataset_paths = [
            "datasets/first/sorted_train_cer_0.1.pkl",
            "datasets/second/sorted_train_cer_0.1.pkl",
            "datasets/third/sorted_train_cer_0.1.pkl",
            "datasets/fourth/sorted_train_cer_0.1.pkl",
            "datasets/fifth/sorted_train_cer_0.1.pkl",
            "datasets/sixth/sorted_train_cer_0.1.pkl",
        ]
        tv_eval_datasets = [
            dataset.replace("train", "eval") for dataset in tv_train_dataset_paths
        ]
        eval_dataset = data.SortedTV(tv_eval_datasets, hparams["batch_size"])
        if "libri" in datasets:
            if "cv" in datasets:
                train_dataset = data.CombinedTVLibriSpeechCommonVoice(
                    "datasets/librispeech/sorted_train_librispeech.pkl",
                    "datasets/commonvoice/sorted_train_commonvoice.pkl",
                    tv_train_dataset_paths,
                    hparams["batch_size"],
                )
            else:
                train_dataset = data.CombinedTVLibriSpeech(
                    "datasets/librispeech/sorted_train_librispeech.pkl",
                    tv_train_dataset_paths,
                    hparams["batch_size"],
                )
        else:
            train_dataset = data.SortedTV(tv_train_dataset_paths, hparams["batch_size"])
    else:
        print("Unkown dataset", hparams["dataset"])
        sys.exit(1)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=None,
        shuffle=True,
        collate_fn=lambda x: data.collate_fn(x, "train"),
        num_workers=3,
        pin_memory=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=None,
        # Also shuffling at the clip level in data.py
        shuffle=True,
        collate_fn=lambda x: data.collate_fn(x, "valid"),
        num_workers=3,
        pin_memory=True,
    )
    ibm_loader = torch.utils.data.DataLoader(
        dataset=data.IBMDataset(),
        batch_size=32 * hparams["multiplier"] * 2,
        shuffle=False,
        collate_fn=lambda x: data.collate_fn(x, "valid"),
        num_workers=3,
        pin_memory=True,
    )

    model = net.SRModel(
        hparams["n_cnn_layers"],
        hparams["lstm_input_dim"],
        hparams["n_lstm_layers"],
        hparams["lstm_dim"],
        hparams["n_vocab"],
        hparams["n_feats"],
        hparams["dropout"],
    )
    # model.load_state_dict(
    #    torch.load(
    #        "good_models/v2-sru-1234-0.1cer/model_f4dd9b8968b94ee6b278266af30dfcef.pth"
    #    )
    # )
    model.cuda()
    optimizer = apex.optimizers.FusedAdam(
        model.parameters(), lr=hparams["learning_rate"]
    )
    model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O2")

    print(
        "Num Model Parameters", sum([param.nelement() for param in model.parameters()])
    )

    criterion = torch.nn.CTCLoss(blank=0).cuda()
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 7000 // hparams["multiplier"], hparams["epochs"] * len(train_loader)
    )

    iter_meter = IterMeter()
    last_cer = 2.0
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
        last_cer = eval(
            hparams["batch_size"],
            model,
            eval_loader,
            ibm_loader,
            criterion,
            epoch,
            iter_meter,
            experiment,
            last_cer,
        )


if __name__ == "__main__":
    signal.signal(signal.SIGUSR2, lambda sig, frame: code.interact())
    datasets = sys.argv[1]
    multiplier = int(sys.argv[2])
    experiment = Experiment(
        api_key="IJIo1bzzY2MAGvPlhq9hA7qsb",
        project_name="general",
        workspace="fernand",
        # disabled=True,
    )
    hparams = {
        "datasets": datasets,
        "multiplier": multiplier,
        "batch_size": 32 * multiplier,
        "epochs": 45,
        "learning_rate": 3e-4,
        "n_cnn_layers": 3,
        "lstm_input_dim": 512,
        "n_lstm_layers": 3,
        "lstm_dim": 1024,
        "dropout": 0.1,
        # Does not include the blank.
        "n_vocab": 28,
        "n_feats": data.N_MELS,
    }
    main(hparams, experiment)
