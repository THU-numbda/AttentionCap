#!/usr/bin/env python3
import multiprocessing as mp
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_CMD = [sys.executable, REPO_ROOT / "src/train.py"]
OUTPUT_ROOT = REPO_ROOT / "training_output/main_results"
GPUS = ["cuda:7"]
MAX_CONCURRENCY = 4
EVAL = False
DATASETS = [REPO_ROOT / "data/asap7_50K", REPO_ROOT / "data/real65_50K"]

MODEL_CONFIGS = [
    {
        "use_transformer": "True",
        "n_embd": "256",
        "n_head": "4",
        "n_layer": "6",
        "batch_size": "128",
        "compile": "False",
    },
    {
        "use_transformer": "True",
        "n_embd": "384",
        "n_head": "4",
        "n_layer": "8",
        "batch_size": "128",
        "compile": "False",
    }
]

RUNS = [{**cfg, "data_dir": data_dir} for data_dir in DATASETS for cfg in MODEL_CONFIGS]


os.makedirs(OUTPUT_ROOT, exist_ok=True)


def kvflag(key: str, value: str) -> str:
    return f"--{key}={value}"


def run_name_from(params):
    dataset = Path(params["data_dir"]).name
    transformer = str(params["use_transformer"]).lower() == "true"
    model = f"{'transformer' if transformer else 'mlp'}-d{params['n_embd']}-l{params['n_layer']}"
    if transformer:
        model += f"-h{params['n_head']}"
    name = f"{dataset}__{model}__b{params['batch_size']}"
    return f"{name}__compiled" if str(params["compile"]).lower() == "true" else name


def run_one(idx, base_params, sem):
    with sem:
        params = dict(base_params)
        device = params.get("device", GPUS[idx % len(GPUS)] if GPUS else "cpu")
        run_name = run_name_from(params) or f"run{idx}"
        run_dir = os.path.join(OUTPUT_ROOT, f"{run_name}")

        if EVAL:
            params["eval_only"] = "True"
            params["checkpoint_path"] = os.path.join(run_dir, "ckpt.pt")

        os.makedirs(run_dir, exist_ok=True)
        log_name = "eval.log" if params.get("eval_only", "False") == "True" else "train.log"
        log_path = os.path.join(run_dir, log_name)

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
    sem = mp.Semaphore(MAX_CONCURRENCY)
    procs = []
    for i, params in enumerate(RUNS):
        p = mp.Process(target=run_one, args=(i, params, sem))
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
