#!/usr/bin/env python3
from pathlib import Path

import run_train as runner

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "training_output/cnncap"
GPUS = runner.GPUS
MAX_CONCURRENCY = 4
DATASETS = [
    REPO_ROOT / "data/cnncap/55nm_C_2_3_6",
    REPO_ROOT / "data/cnncap/55nm_C_2_4_6",
    REPO_ROOT / "data/cnncap/15nm_C_2_4_6",
    REPO_ROOT / "data/cnncap/15nm_C_2_4_9",
]
MODEL_CONFIG = {
    "use_transformer": "True",
    "n_embd": "256",
    "n_head": "4",
    "n_layer": "6",
    "input_dim": "3",
    "head_mode": "first_row",
    "batch_size": "128",
}
RUNS = [{**MODEL_CONFIG, "data_dir": data_dir} for data_dir in DATASETS]


if __name__ == "__main__":
    runner.OUTPUT_ROOT = OUTPUT_ROOT
    runner.GPUS = GPUS
    runner.MAX_CONCURRENCY = MAX_CONCURRENCY
    runner.RUNS = RUNS
    runner.main()
