import argparse
import os
import pickle
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

import data
import net


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


def train(
    batch_size,
    model,
    train_loader,
    criterion,
    scaler,
    optimizer,
    scheduler,
    epoch,
    num_epochs,
    clip_grad_norm,
):
    model.train()
    data_len = len(train_loader)
    start = time.time()
    for batch_idx, batch in enumerate(train_loader):
        spectrograms, labels = batch
        spectrograms = spectrograms.cuda()

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output = model(spectrograms)  # B, 1
            labels = labels.cuda()
            loss = criterion(output, labels)

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_grad_norm, norm_type=2
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if batch_idx % 100 == 0:
            time_for_100_batches = round(time.time() - start, 1)
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
            start = time.time()


def main(hparams, device):
    torch.manual_seed(7)

    with open("datasets/gigaspeech/sorted_train_youtube_filtered.pkl", "rb") as f:
        dataset = pickle.load(f)
    sources = set()
    segments = defaultdict(lambda: [])
    for audio_path, audio_len in dataset:
        source = os.path.dirname(audio_path).split("/")[-1]
        sources.add(source)
        segments[source].append(audio_path)
    pairs = [(s1, s2) for s1 in sources for s2 in sources if s1 != s2]
    train_dataset = data.SpeakerDataset(pairs, segments)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        collate_fn=lambda x: data.collate_fn(x, "train"),
        num_workers=6,
        pin_memory=True,
    )

    model = net.Model()
    # checkpoint = torch.load()
    checkpoint = None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(
        "Num Model Parameters", sum([param.nelement() for param in model.parameters()])
    )

    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    scaler = torch.cuda.amp.GradScaler()
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 5 * len(train_loader), hparams["epochs"] * len(train_loader)
    )

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
            hparams["epochs"],
            hparams["clip_grad_norm"],
        )
    torch.save(
        {
            "hparams": hparams,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        f"model_{device}.pth",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--one_iter", action="store_true")
    p.set_defaults(one_iter=False)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=int)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_grad_norm", type=float, default=100.0)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--num_epochs", type=int, default=45)
    args = p.parse_args()
    device = args.device
    hparams = {
        "batch_size": args.batch_size,
        "epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "n_feats": data.N_MELS,
        "weight_decay": args.weight_decay,
        "clip_grad_norm": args.clip_grad_norm,
        "one_iter": args.one_iter,
    }
    main(hparams, device)
