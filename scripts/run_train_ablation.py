#!/usr/bin/env python3
import run_train as runner

from config_ablation import GPUS, MAX_CONCURRENCY, OUTPUT_ROOT, RUNS


if __name__ == "__main__":
    runner.main(RUNS, OUTPUT_ROOT, GPUS, MAX_CONCURRENCY)
