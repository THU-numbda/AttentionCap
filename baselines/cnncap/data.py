from pathlib import Path

import numpy as np
import torch


def coordinate_to_grid(conductors, meta, target_id):
    feature = torch.zeros((len(meta["y_pos"]), meta["dim"]), dtype=torch.float32)
    y_map = {y: i for i, y in enumerate(meta["y_pos"])}

    for i, (x_pos, y_pos, width, _) in enumerate(conductors):
        channel = y_map[y_pos]
        center = (x_pos + meta["window_width"] / 2) / meta["window_width"] * meta["dim"]
        width = width / meta["window_width"] * meta["dim"]
        start, end = center - width / 2, center + width / 2
        start_index, end_index = int(np.floor(start)), int(np.floor(end))
        left, right = max(0, start_index), min(meta["dim"] - 1, end_index)
        if left > right:
            continue

        def mark(density):
            if i == 0:
                return density + 1
            if i == target_id:
                return -density
            return density

        if left < right:
            feature[channel, left + 1 : right] = mark(1.0)
        if start_index == end_index:
            if 0 <= start_index < meta["dim"]:
                feature[channel, start_index] = mark(end - start)
        else:
            if 0 <= start_index < meta["dim"]:
                feature[channel, start_index] = mark(1 - (start - start_index))
            if 0 <= end_index < meta["dim"]:
                feature[channel, end_index] = mark(end - end_index)
    return feature


class CoordinateCapDataset(torch.utils.data.Dataset):
    def __init__(self, groups, goal, meta):
        if goal not in ("total", "env"):
            raise ValueError(f"Unknown goal: {goal}")
        self.in_channel = len(meta["y_pos"])
        self.data = []

        for group in groups.values():
            for features, outputs in zip(group["features"], group["outputs"]):
                conductors = features[:-1].tolist()
                length = len(conductors)
                if goal == "total":
                    for master in range(length):
                        conductors[0], conductors[master] = conductors[master], conductors[0]
                        y = outputs[master, master].item()
                        self.data.append(self._sample(coordinate_to_grid(conductors, meta, 0), y, y))
                        conductors[0], conductors[master] = conductors[master], conductors[0]
                    continue

                for master in range(length):
                    caps = outputs[master, :length].tolist()
                    caps[0], caps[master] = caps[master], caps[0]
                    conductors[0], conductors[master] = conductors[master], conductors[0]
                    for target in range(1, length):
                        actual_target = 0 if target == master else target
                        target_self = outputs[actual_target, actual_target].item()
                        if abs(caps[target]) >= max(caps[0], target_self) * 0.01:
                            self.data.append(
                                self._sample(coordinate_to_grid(conductors, meta, target), abs(caps[target]), caps[0])
                            )
                    conductors[0], conductors[master] = conductors[master], conductors[0]

    @staticmethod
    def _sample(feature, y, total):
        return (
            feature,
            torch.tensor([y], dtype=torch.float32),
            torch.tensor([1], dtype=torch.float32),
            torch.tensor(total, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_groups(data_dir, split):
    return torch.load(Path(data_dir) / f"{split}_data.pt", map_location="cpu", weights_only=True)


def load_meta(data_dir, window_width):
    y_positions = set()
    for split in ("train", "val"):
        for group in load_groups(data_dir, split).values():
            y_positions.update(float(y) for y in group["features"][:, :-1, 1].flatten())
    return {"y_pos": sorted(y_positions), "dim": 512, "window_width": window_width}


def load_split(data_dir, split, goal, window_width):
    return CoordinateCapDataset(load_groups(data_dir, split), goal, load_meta(data_dir, window_width))
