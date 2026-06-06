import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


@torch.no_grad()
def preprocess_data(jsonl_path, output_dir, train_ratio=0.9, val_ratio=0.1, shuffle=False, plot=False):
    bins = defaultdict(lambda: {"features": [], "outputs": []})
    print(f"Reading data from '{jsonl_path}'...")
    with Path(jsonl_path).open() as lines:
        for line in lines:
            sample = json.loads(line)
            bins[sample["size"]]["features"].append(sample["conductors"])
            bins[sample["size"]]["outputs"].append(sample["capacitances"])

    splits = {"train": {}, "val": {}, "test": {}}
    for size, data in bins.items():
        features = torch.tensor(data["features"], dtype=torch.float32)
        outputs = torch.tensor(data["outputs"], dtype=torch.float32).view(len(features), size, size)
        if shuffle:
            order = torch.randperm(len(features))
            features, outputs = features[order], outputs[order]

        train_end = int(train_ratio * len(features))
        val_end = int(min(1, train_ratio + val_ratio) * len(features))
        if train_end == 0:
            train_end = val_end = len(features)
        elif train_end == val_end and val_end < len(features):
            val_end = len(features)

        ranges = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, len(features))}
        counts = []
        for split, (start, end) in ranges.items():
            if start < end:
                splits[split][size] = {
                    "features": features[start:end].clone(),
                    "outputs": outputs[start:end].clone(),
                }
            counts.append(end - start)
        print(f"  size={size}: {len(features)} samples -> {counts[0]} train, {counts[1]} val, {counts[2]} test")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, data in splits.items():
        torch.save(data, output_dir / f"{split}_data.pt")

    if plot:
        plot_distribution(splits, output_dir / "dataset_distribution.png")
    print(f"Preprocessing complete. Data saved in '{output_dir}'.")


def plot_distribution(splits, output_path):
    import matplotlib.pyplot as plt

    sizes = sorted(set().union(*(data.keys() for data in splits.values())))
    counts = {
        split: [len(data[size]["features"]) if size in data else 0 for size in sizes]
        for split, data in splits.items()
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(sizes, counts["train"], label="Train", alpha=0.7)
    ax.bar(sizes, counts["val"], label="Val", alpha=0.7, bottom=counts["train"])
    bottom = [train + val for train, val in zip(counts["train"], counts["val"])]
    ax.bar(sizes, counts["test"], label="Test", alpha=0.7, bottom=bottom)
    ax.set(xlabel="Size", ylabel="Samples", title="Dataset Distribution by Size")
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Convert a JSONL dataset into binned PyTorch tensors.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    preprocess_data(
        args.input_file,
        args.output_dir or args.input_file.with_suffix(""),
        args.train_ratio,
        args.val_ratio,
        args.shuffle,
        args.plot,
    )


if __name__ == "__main__":
    main()
