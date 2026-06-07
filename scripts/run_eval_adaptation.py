#!/usr/bin/env python3
from config_adaptation import GPUS, OUTPUT_ROOT, RUNS
from run_eval import main


if __name__ == "__main__":
    main(RUNS, OUTPUT_ROOT, GPUS)
