import argparse
import sys
import shutil
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from baselines.cnncap.common import AverageMeter, clean_state_dict, seed_everything
from baselines.cnncap.data import load_split
from baselines.cnncap.model import resnet34

WARMUP_ITERS = 1000
MAX_ITERS = 300000
MIN_LR = 1e-5


def learning_rate(step, peak_lr):
    if step < WARMUP_ITERS:
        return peak_lr * (step + 1) / (WARMUP_ITERS + 1)
    if step > MAX_ITERS:
        return MIN_LR
    coefficient = (MAX_ITERS - step) / (MAX_ITERS - WARMUP_ITERS)
    return MIN_LR + coefficient * (peak_lr - MIN_LR)


def run_epoch(loader, model, device, optimizer=None, step=0, peak_lr=1.5e-4):
    training = optimizer is not None
    model.train(training)
    losses = AverageMeter()
    errors = []

    for features, targets, masks, _ in loader:
        features, targets, masks = features.to(device), targets.to(device), masks.bool().to(device)
        with torch.set_grad_enabled(training):
            prediction = model(features)
            loss = ((1 - prediction / (targets + 1e-8)).masked_select(masks) ** 2).mean()
        losses.update(loss.item(), targets.size(0))

        if training:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, peak_lr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

        errors.extend(torch.abs((prediction - targets) / targets).masked_select(masks).detach().cpu().tolist())

    return losses.avg, max(errors), np.mean(errors), step


def save_checkpoint(path, model, optimizer, epoch, max_error, is_best):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "isbest": is_best,
        "loss": max_error,
    }
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, path.with_name(f"best.{path.name}"))


def main():
    parser = argparse.ArgumentParser(description="Train the CNNCap ResNet34 baseline.")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--goal", choices=("total", "env"), required=True)
    parser.add_argument("--window_width", type=float, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=11037)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(args)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_data = load_split(args.data_dir, "train", args.goal, args.window_width)
    val_data = load_split(args.data_dir, "val", args.goal, args.window_width)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = resnet34(val_data.in_channel).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    from thop import profile

    macs, params = profile(model, inputs=(torch.randn(1, val_data.in_channel, 1024, device=device),), verbose=False)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(clean_state_dict(checkpoint["state_dict"]))
        optimizer.load_state_dict(checkpoint["optimizer"])

    writer = SummaryWriter(args.out_dir)
    writer.add_text("Args", str(args))
    checkpoint_path = args.out_dir / "model.pth"
    best_max_error = float("inf")
    step = 0

    with (args.out_dir / "train.log").open("w", buffering=1) as log:
        log.write(f"{args}\n")
        log.write(f"Loaded dataset with {len(train_data)} training samples and {len(val_data)} validation samples.\n")
        log.write(f"FLOPs: {macs / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

        for epoch in range(args.epochs):
            train_loss, train_max, train_avg, step = run_epoch(train_loader, model, device, optimizer, step, args.lr)
            val_loss, val_max, val_avg, _ = run_epoch(val_loader, model, device)
            print(epoch, train_loss, train_max, train_avg)
            print(epoch, val_loss, val_max, val_avg)
            log.write(f"Training {epoch} {train_loss} {train_max} {train_avg}\n")
            log.write(f"Testing {epoch} {val_loss} {val_max} {val_avg}\n")
            for split, loss, maximum, average in (
                ("Train", train_loss, train_max, train_avg),
                ("Test", val_loss, val_max, val_avg),
            ):
                writer.add_scalar(f"{split}/Loss", loss, epoch)
                writer.add_scalar(f"{split}/MaxErr", maximum, epoch)
                writer.add_scalar(f"{split}/AvgErr", average, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            is_best = val_max < best_max_error
            best_max_error = min(best_max_error, val_max)
            save_checkpoint(checkpoint_path, model, optimizer, epoch, val_max, is_best)
            if step >= MAX_ITERS:
                break
    writer.close()


if __name__ == "__main__":
    main()
