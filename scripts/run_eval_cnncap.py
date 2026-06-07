#!/usr/bin/env python3
from run_eval import main
from run_train_cnncap import GPUS, OUTPUT_ROOT, RUNS


if __name__ == "__main__":
    main(RUNS, OUTPUT_ROOT, GPUS)
