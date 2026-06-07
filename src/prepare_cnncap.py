import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def recover_intervals(density, scale, eps=1e-6):
    density = density.flatten().float()
    negative = density < -eps
    positive = density > 1 + eps
    density = torch.where(negative, -density, torch.where(positive, density - 1, density.clamp(0, 1)))

    occupied = density > eps
    partial = occupied & (density < 1 - eps)
    previous = torch.zeros_like(occupied)
    following = torch.zeros_like(occupied)
    previous[1:] = occupied[:-1]
    following[:-1] = occupied[1:]
    previous_partial = torch.zeros_like(partial)
    following_partial = torch.zeros_like(partial)
    previous_partial[1:] = partial[:-1]
    following_partial[:-1] = partial[1:]

    starts = occupied & (~previous | (partial & previous_partial))
    ends = occupied & (~following | (partial & following_partial))
    start_index = starts.nonzero().flatten()
    end_index = ends.nonzero().flatten()
    if len(start_index) != len(end_index):
        raise ValueError("Malformed CNNCap density intervals")

    left = torch.where(
        density[start_index] < 1 - eps,
        start_index + 1 - density[start_index],
        start_index.float(),
    )
    right = torch.where(
        density[end_index] < 1 - eps,
        end_index + density[end_index],
        end_index + 1,
    )
    intervals = torch.stack((left, right), dim=1) * scale

    labels = starts.long().cumsum(0) - 1
    labels = torch.cummax(labels, dim=0).values
    labels = torch.where(occupied, labels, -1)
    special_labels = labels[negative & occupied]
    if not len(special_labels):
        special_labels = labels[positive & occupied]
    special_index = int(special_labels[0]) if len(special_labels) else -1
    return intervals, special_index


def decompress(compressed, buffer):
    for layer, left, right, value in compressed:
        buffer[layer, round(left) : round(right) + 1] = value
    return buffer


def convert_sample(sample, n_channel, dim=1024, window_size=10, cap_scale=1e15):
    size = len(sample["env_data"]) + 2
    buffer = decompress(sample["x_compress_total"], torch.zeros(n_channel, dim))
    features = torch.zeros(size, 3)
    outputs = torch.zeros(size)

    master_index = -1
    start = 0
    for layer in range(n_channel):
        intervals, special = recover_intervals(buffer[layer], window_size / dim)
        end = start + len(intervals)
        features[start:end, 0] = intervals.mean(dim=1) - window_size / 2
        features[start:end, 1] = intervals[:, 1] - intervals[:, 0]
        features[start:end, 2] = layer
        if special >= 0:
            master_index = start + special
            outputs[master_index] = sample["y_total"] * cap_scale
        start = end

    for environment in sample["env_data"]:
        decompress(environment["x_compress"], buffer)
        start = 0
        for layer in range(n_channel):
            intervals, special = recover_intervals(buffer[layer], window_size / dim)
            if special >= 0 and start + special != master_index:
                outputs[start + special] = -abs(environment["y"] * cap_scale)
            start += len(intervals)

    if master_index < 0 or start + 1 != size:
        raise ValueError(f"Malformed CNNCap sample: master={master_index}, conductors={start + 1}, expected={size}")

    features[-1] = torch.tensor([0, window_size, n_channel])
    outputs[-1] = -outputs[:-1].sum()
    order = torch.arange(size)
    order = torch.cat((order[master_index : master_index + 1], order[order != master_index]))
    return features[order], outputs[order]


def preprocess_data(input_path, output_dir):
    print(f"Reading CNNCap data from '{input_path}'...")
    with Path(input_path).open() as file:
        samples = json.load(file)
    n_channel = max(item[0] for item in samples[0]["x_compress_total"]) + 1
    bins = defaultdict(lambda: {"features": [], "outputs": []})

    for sample in samples:
        features, outputs = convert_sample(sample, n_channel)
        bins[len(features)]["features"].append(features)
        bins[len(features)]["outputs"].append(outputs)

    splits = {"train": {}, "val": {}, "test": {}}
    for size, data in bins.items():
        features = torch.stack(data["features"])
        outputs = torch.stack(data["outputs"])
        train_end = int(0.9 * len(features))
        val_end = len(features)
        ranges = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, len(features))}
        for split, (start, end) in ranges.items():
            if start < end:
                splits[split][size] = {"features": features[start:end].clone(), "outputs": outputs[start:end].clone()}
        print(f"  size={size}: {len(features)} samples -> {train_end} train, {val_end - train_end} val, {len(features) - val_end} test")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, data in splits.items():
        torch.save(data, output_dir / f"{split}_data.pt")
    print(f"CNNCap data saved in '{output_dir}'.")


def main():
    parser = argparse.ArgumentParser(description="Convert raw CNNCap JSON into AttentionCap tensors.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    preprocess_data(args.input_file, args.output_dir or args.input_file.with_suffix(""))


if __name__ == "__main__":
    main()
