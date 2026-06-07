#!/usr/bin/env python3
import multiprocessing as mp
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_CMD = [sys.executable, REPO_ROOT / "src/train.py"]
OUTPUT_ROOT = REPO_ROOT / "training_output/main_results"
GPUS = ["cuda:7"]
MAX_CONCURRENCY = 3
DATASETS = [
    REPO_ROOT / "data/real65_50K",
    REPO_ROOT / "data/asap7_50K",
    REPO_ROOT / "data"
]

MODEL_CONFIGS = [
    {
        "use_transformer": "True",
        "n_embd": "256",
        "n_head": "4",
        "n_layer": "6",
    },
    {
        "use_transformer": "True",
        "n_embd": "384",
        "n_head": "4",
        "n_layer": "8",
    }
]

RUNS = [
    {**cfg, "data_dir": data_dir, "batch_size": "512" if data_dir == REPO_ROOT / "data" else "128"}
    for data_dir in DATASETS
    for cfg in MODEL_CONFIGS
]


def kvflag(key: str, value: str) -> str:
    return f"--{key}={value}"


def run_name_from(params):
    dataset = Path(params["data_dir"]).name
    transformer = str(params["use_transformer"]).lower() == "true"
    model = f"{'transformer' if transformer else 'mlp'}-d{params['n_embd']}-l{params['n_layer']}"
    if transformer:
        model += f"-h{params['n_head']}"
    task = "" if params.get("head_mode", "matrix") == "matrix" else f"__{params['head_mode'].replace('_', '-')}"
    return f"{dataset}__{model}__b{params['batch_size']}{task}"


def run_one(idx, base_params, sem, output_root, gpus):
    with sem:
        params = dict(base_params)
        device = params.get("device", gpus[idx % len(gpus)] if gpus else "cpu")
        run_name = run_name_from(params) or f"run{idx}"
        run_dir = Path(output_root) / run_name / time.strftime("%Y%m%d%H%M%S")

        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "train.log")

        cmd = BASE_CMD + [f"--device={device}", f"--out_dir={run_dir}"] + [kvflag(k, v) for k, v in params.items()]
        print(f"[LAUNCH] {' '.join(shlex.quote(str(c)) for c in cmd)}\n  -> {log_path}", flush=True)
        with open(log_path, "w", buffering=1) as lf:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                lf.write(line)
                sys.stdout.write(f"[{run_name}] {line}")
            ret = proc.wait()
        print(f"[DONE] {run_name} (exit={ret})", flush=True)
        if ret:
            raise SystemExit(ret)


def main():
    mp.set_start_method("spawn", force=True)
    output_root = Path(OUTPUT_ROOT)
    gpus = list(GPUS)
    os.makedirs(output_root, exist_ok=True)
    sem = mp.Semaphore(MAX_CONCURRENCY)
    procs = []
    for i, params in enumerate(RUNS):
        p = mp.Process(target=run_one, args=(i, params, sem, output_root, gpus))
        p.start()
        procs.append(p)

    for process in procs:
        process.join()
    failed = [process.exitcode for process in procs if process.exitcode]
    if failed:
        raise SystemExit(max(failed))
    print("All runs finished.")

if __name__ == "__main__":
    main()
