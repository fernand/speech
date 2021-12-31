import argparse
import time

from comet_ml import Experiment
import torch
import torch.nn.functional as F
import bitsandbytes as bnb

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
    scaler,
    optimizer,
    scheduler,
    epoch,
    iter_meter,
    num_epochs,
    clip_grad_norm,
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

            with torch.cuda.amp.autocast():
                output = model(spectrograms)  # B, T, n_vocab+1
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1).contiguous()  # T, B, n_vocab+1

                input_lengths = torch.full(
                    (batch_size,), output.size(0), dtype=torch.int32
                ).cuda()
                label_lengths = label_lengths.cuda()
                labels = labels.cuda()
                loss = criterion(output, labels, input_lengths, label_lengths)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_grad_norm, norm_type=2
            )
            scaler.step(optimizer)
            scaler.update()
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
    for _, batch in enumerate(loader):
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


TV_TRAIN_DATASETS_PATHS = [
    "datasets/first/sorted_train_cer_0.1.pkl",
    "datasets/second/sorted_train_cer_0.1.pkl",
    "datasets/third/sorted_train_cer_0.1.pkl",
    "datasets/fourth/sorted_train_cer_0.1.pkl",
    "datasets/fifth/sorted_train_cer_0.1.pkl",
    "datasets/sixth/sorted_train_cer_0.1.pkl",
    "datasets/gigaspeech/sorted_train_youtube_filtered.pkl",
    "datasets/gigaspeech/sorted_train_podcast_filtered.pkl",
    "datasets/gigaspeech/sorted_train_audiobook_filtered.pkl",
]


def main(hparams, experiment, device):
    experiment.log_parameters(hparams)
    torch.manual_seed(7)

    datasets = hparams["datasets"].split("-")
    assert "tv" in datasets
    assert "libri" in datasets
    train_dataset = data.CombinedTVLibriSpeech(
        "datasets/librispeech/sorted_train_librispeech.pkl",
        TV_TRAIN_DATASETS_PATHS,
        hparams["batch_size"],
        device,
    )
    tv_eval_datasets = [
        dataset.replace("train", "eval")
        for dataset in TV_TRAIN_DATASETS_PATHS
        if "gigaspeech" not in dataset
    ]
    eval_dataset = data.SortedTV(tv_eval_datasets, hparams["batch_size"], device)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=None,
        shuffle=True,
        collate_fn=lambda x: data.collate_fn(x, "train"),
        num_workers=6,
        pin_memory=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=None,
        # Also shuffling at the clip level in data.py
        shuffle=True,
        collate_fn=lambda x: data.collate_fn(x, "valid"),
        num_workers=4,
        pin_memory=True,
    )
    ibm_loader = torch.utils.data.DataLoader(
        dataset=data.IBMDataset(),
        batch_size=32 * hparams["multiplier"] * 2,
        shuffle=False,
        collate_fn=lambda x: data.collate_fn(x, "valid"),
        num_workers=4,
        pin_memory=True,
    )

    model = net.SRModel(
        hparams["n_rnn_layers"],
        hparams["rnn_dim"],
        hparams["n_vocab"],
        hparams["n_feats"],
        hparams["dropout"],
        hparams["projection_size"],
    )
    # model.load_state_dict(
    #    torch.load(
    #        "good_models/v2-sru-1234-0.1cer/model_f4dd9b8968b94ee6b278266af30dfcef.pth"
    #    )
    # )
    model.cuda()
    optimizer = bnb.optim.Adam8bit(
        model.parameters(),
        lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
    )

    print(
        "Num Model Parameters", sum([param.nelement() for param in model.parameters()])
    )

    criterion = torch.nn.CTCLoss(blank=0).cuda()
    scaler = torch.cuda.amp.GradScaler()
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 7000 // hparams["multiplier"], hparams["epochs"] * len(train_loader)
    )

    iter_meter = IterMeter()
    last_cer = 2.0
    if hparams["one_iter"]:
        num_epochs = 1
    else:
        num_epochs = hparams["epochs"]
    for epoch in range(1, num_epochs + 1):
        train(
            hparams["batch_size"],
            model,
            train_loader,
            criterion,
            scaler,
            optimizer,
            scheduler,
            epoch,
            iter_meter,
            hparams["epochs"],
            hparams["clip_grad_norm"],
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
    p = argparse.ArgumentParser()
    p.add_argument("--one_iter", action="store_true")
    p.set_defaults(one_iter=False)
    p.add_argument("--datasets", type=str, default="tv-libri")
    p.add_argument("--multiplier", type=int, default=2)
    p.add_argument("--device", type=int)
    p.add_argument("--weight_decay", type=float, default=0.001)
    p.add_argument("--clip_grad_norm", type=float, default=2.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--num_epochs", type=int, default=45)
    p.add_argument("--projection_size", type=int, default=0)
    args = p.parse_args()
    device = args.device
    experiment = Experiment(
        api_key="IJIo1bzzY2MAGvPlhq9hA7qsb",
        project_name="general",
        workspace="fernand",
        auto_metric_logging=False,
        log_env_gpu=False,
        log_env_cpu=False,
        log_env_host=False,
        log_env_details=False,
        # disabled=True,
    )
    hparams = {
        "datasets": args.datasets,
        "multiplier": args.multiplier,
        "batch_size": 32 * args.multiplier,
        "epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "n_rnn_layers": 10,
        "rnn_dim": 512,
        "dropout": args.dropout,
        # Does not include the blank.
        "n_vocab": 28,
        "n_feats": data.N_MELS,
        "weight_decay": args.weight_decay,
        "clip_grad_norm": args.clip_grad_norm,
        "one_iter": args.one_iter,
        "projection_size": args.projection_size,
    }
    main(hparams, experiment, device)
