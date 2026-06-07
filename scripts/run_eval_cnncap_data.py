#!/usr/bin/env python3
from run_eval import main
from config_cnncap_data import GPUS, OUTPUT_ROOT, RUNS


if __name__ == "__main__":
    main(RUNS, OUTPUT_ROOT, GPUS)
