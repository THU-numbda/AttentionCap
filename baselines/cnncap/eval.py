import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from baselines.cnncap.common import AverageMeter, clean_state_dict, seed_everything
from baselines.cnncap.data import load_split
from baselines.cnncap.model import resnet34


@torch.no_grad()
def evaluate(loader, model, device, high_error_threshold, log):
    errors = AverageMeter()
    maximum = 0
    messages = []
    model.eval()

    for features, targets, masks, totals in loader:
        features, targets, masks = features.to(device), targets.to(device), masks.bool().to(device)
        relative_errors = torch.abs((model(features) - targets) / targets)
        for row, target, mask, total in zip(relative_errors, targets, masks, totals):
            selected = row.masked_select(mask)
            errors.update(selected, 1)
            for error, value in zip(selected, target.masked_select(mask)):
                messages.append((error.item(), value.item(), total))
        maximum = max(maximum, selected.max().item())

    high_errors = sum(error > high_error_threshold for error, _, _ in messages)
    for error, value, total in messages:
        log.write(f"{error},{value},{total}\n")
    log.write(f"# High error rate (> {high_error_threshold:.2f}): {high_errors / len(messages):.4f}\n")
    return errors.avg, maximum


@torch.no_grad()
def benchmark(loader, model, device):
    from thop import profile

    total_flops = 0
    for features, targets, masks, _ in loader:
        features, targets, masks = features.to(device), targets.to(device), masks.bool().to(device)
        flops, _ = profile(model, inputs=(features,), verbose=False)
        total_flops += flops
    print(f"Total FLOPs: {total_flops / 1e9:.6f} G, Total Samples: {len(loader.dataset)}")
    print(f"FLOPs per sample: {total_flops / len(loader.dataset) / 1e6:.6f} M")

    synchronize(device)
    start = time.perf_counter()
    for features, targets, masks, _ in loader:
        features, targets, masks = features.to(device), targets.to(device), masks.bool().to(device)
        model(features)
    synchronize(device)
    elapsed = time.perf_counter() - start
    print(f"Total inference time: {elapsed:.6f} s, Total Samples: {len(loader.dataset)}")
    print(f"Time per sample: {elapsed / len(loader.dataset) * 1000:.6f} ms")
    return total_flops, elapsed


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the CNNCap ResNet34 baseline.")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--goal", choices=("total", "env"), required=True)
    parser.add_argument("--window_width", type=float, required=True)
    parser.add_argument("--logfile", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=11037)
    args = parser.parse_args()

    print(args)
    args.logfile.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    synchronize(device)
    start = time.perf_counter()
    dataset = load_split(args.data_dir, args.split, args.goal, args.window_width)
    synchronize(device)
    grid_time = time.perf_counter() - start
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = resnet34(dataset.in_channel).to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(clean_state_dict(checkpoint["state_dict"]), strict=False)
    optimizer = checkpoint["optimizer"]["param_groups"][0]
    print("Model info:")
    print(f"epoch {checkpoint['epoch']}")
    print("Optimizer: Adam" if "betas" in optimizer else "Optimizer: Unknown")
    print(f"  Learning Rate: {optimizer.get('lr')}")
    print(f"  Weight Decay: {optimizer.get('weight_decay')}")
    if "betas" in optimizer:
        print(f"  Betas: {optimizer['betas']}")
    print(f"loss {checkpoint['loss']}")

    with args.logfile.open("w", buffering=1) as log:
        average, maximum = evaluate(loader, model, device, 0.05 if args.goal == "total" else 0.10, log)
        total_flops, model_time = benchmark(loader, model, device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        example, _, _, _ = next(iter(loader))
        from thop import profile

        flops, _ = profile(model, inputs=(example.to(device),), verbose=False)
        batch_size = len(example)
        log.write(f"# Params (total): {parameters / 1e6:.3f} M\n")
        log.write(f"# Params (trainable): {parameters / 1e6:.3f} M\n")
        log.write(f"# FLOPs per sample: {flops / batch_size / 1e6:.3f} M, total {len(dataset) * flops / batch_size / 1e9:.3f} G\n")
        log.write(f"# FLOPs per batch (bs={batch_size}): {flops / 1e6:.3f} M\n")
        log.write(f"# Test all FLOPs:{total_flops / 1e12:.6f} B\n")
        log.write(f"# Test avg {average} max {maximum}\n")
        log.write(f"# Batch size: {args.batch_size}\n")
        log.write(f"# Samples in test set: {len(dataset)}\n")
        log.write(f"# Batches in test set: {len(loader)}\n")
        log.write(f"# Total time: {grid_time + model_time:.4f} s\n")
        log.write(f"# Grid time: {grid_time:.4f} s\n")
        log.write(f"# Model time: {model_time:.4f} s\n")


if __name__ == "__main__":
    main()
