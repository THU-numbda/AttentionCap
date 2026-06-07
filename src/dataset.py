import collections
import logging
import os

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BucketBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, bucket_width=4, shuffle=True, drop_last=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        buckets = collections.defaultdict(list)
        for index, (features, _) in enumerate(dataset):
            buckets[len(features) // bucket_width].append(index)
        self.buckets = list(buckets.values())

    def __len__(self):
        return sum(
            len(indices) // self.batch_size
            + (not self.drop_last and len(indices) % self.batch_size > 0)
            for indices in self.buckets
        )

    def __iter__(self):
        batches = []
        for indices in self.buckets:
            if self.shuffle:
                order = torch.randperm(len(indices)).tolist()
                indices = [indices[i] for i in order]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            for index in torch.randperm(len(batches)).tolist():
                yield batches[index]
        else:
            yield from batches


def collate_and_pad_with_mask(batch, no_padding=False):
    features, outputs = zip(*batch)
    if no_padding:
        return torch.stack(features), torch.stack(outputs), None

    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    lengths = torch.tensor([len(item) for item in features])
    max_length = padded_features.size(1)
    mask = (torch.arange(max_length)[None, :] < lengths[:, None]).float()

    padded_outputs = torch.stack(
        [
            torch.nn.functional.pad(output, (0, max_length - output.size(1), 0, max_length - output.size(0)))
            for output in outputs
        ]
    )
    return padded_features, padded_outputs, mask


def _dataset_dirs(data_dir):
    subdirs = [
        os.path.join(data_dir, name)
        for name in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, name))
    ]
    return [path for path in subdirs or [data_dir] if os.path.isfile(os.path.join(path, "train_data.pt"))]


def get_dataset_dict(data_dir):
    return {os.path.basename(path): index for index, path in enumerate(_dataset_dirs(data_dir))}


def get_dataloader(
    split: str,
    data_dir: str,
    dataset_dict,
    batch_size: int,
    bucket_width: int = 2,
    num_workers: int = 0,
    pin_memory: bool = False,
    is_train: bool = False,
) -> torch.utils.data.DataLoader:
    dataset_dict = {os.path.basename(key): value for key, value in dataset_dict.items()}
    samples = []

    for dataset_dir in _dataset_dirs(data_dir):
        path = os.path.join(dataset_dir, f"{split}_data.pt")
        if not os.path.isfile(path):
            logging.info("Skipping %s; missing '%s' split", dataset_dir, split)
            continue

        name = os.path.basename(dataset_dir)
        if name not in dataset_dict:
            dataset_dict[name] = len(dataset_dict)
            logging.info("Added dataset label: %s -> %s", name, dataset_dict[name])
        dataset_index = dataset_dict[name]

        groups = torch.load(path)
        logging.info("Loaded %s, label=%s, '%s' groups: %s", dataset_dir, dataset_index, split, list(groups))
        for size, tensors in groups.items():
            features = tensors.get("features")
            outputs = tensors.get("outputs")
            if features is None or outputs is None:
                logging.warning("Skipping size %s; missing features or outputs", size)
                continue
            if outputs.ndim != 3:
                raise ValueError(f"Expected matrix outputs, got shape {tuple(outputs.shape)}")

            if len(dataset_dict) > 1:
                index_channel = torch.full((*features.shape[:-1], 1), float(dataset_index), dtype=features.dtype)
                features = torch.cat([features, index_channel], dim=-1)

            if is_train:
                mirrored = features.clone()
                mirrored[..., 0] = -mirrored[..., 0]
                features = torch.cat([features, mirrored])
                outputs = torch.cat([outputs, outputs])

            samples.extend(zip(features.unbind(), outputs.unbind()))

    sampler = BucketBatchSampler(samples, batch_size, bucket_width, shuffle=is_train)
    loader = torch.utils.data.DataLoader(
        samples,
        batch_sampler=sampler,
        collate_fn=lambda batch: collate_and_pad_with_mask(batch, no_padding=bucket_width == 1),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    logging.info("Created '%s' loader with %s samples", split, len(samples))
    return loader
