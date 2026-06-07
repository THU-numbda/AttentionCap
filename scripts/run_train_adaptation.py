#!/usr/bin/env python3
import run_train as runner

from config_adaptation import GPUS, MAX_CONCURRENCY, OUTPUT_ROOT, RUNS
from config_pretrain import OUTPUT_ROOT as PRETRAIN_OUTPUT_ROOT
from config_pretrain import RUNS as PRETRAIN_RUNS
from run_eval import latest_checkpoint


if __name__ == "__main__":
    checkpoint = latest_checkpoint(PRETRAIN_RUNS[0], PRETRAIN_OUTPUT_ROOT)
    if checkpoint is None:
        raise SystemExit("No pretrain checkpoint found")
    runs = [{**run, "checkpoint_path": checkpoint} for run in RUNS]
    runner.main(runs, OUTPUT_ROOT, GPUS, MAX_CONCURRENCY)
