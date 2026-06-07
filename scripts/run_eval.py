#!/usr/bin/env python3
import shlex
import subprocess
import sys
from pathlib import Path

from run_train import BASE_CMD, GPUS, OUTPUT_ROOT, RUNS, kvflag, run_name_from


def latest_checkpoint(params, output_root=OUTPUT_ROOT):
    checkpoints = (Path(output_root) / run_name_from(params)).glob("*/ckpt.pt")
    return max(checkpoints, key=lambda path: path.stat().st_mtime, default=None)


def main(runs=RUNS, output_root=OUTPUT_ROOT, gpus=GPUS):
    device = gpus[0] if gpus else "cpu"
    for params in runs:
        run_name = run_name_from(params)
        checkpoint = latest_checkpoint(params, output_root)
        if checkpoint is None:
            print(f"[SKIP] {run_name}: no checkpoint", flush=True)
            continue

        log_path = checkpoint.parent / "eval.log"
        cmd = BASE_CMD + [
            f"--device={device}",
            f"--out_dir={checkpoint.parent}",
            kvflag("data_dir", params["data_dir"]),
            kvflag("checkpoint_path", checkpoint),
            "--eval_only=True",
        ]
        print(f"[EVAL] {' '.join(shlex.quote(str(item)) for item in cmd)}\n  -> {log_path}", flush=True)
        with open(log_path, "w", buffering=1) as log:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                sys.stdout.write(f"[{run_name}] {line}")
            if process.wait():
                raise SystemExit(f"Evaluation failed: {run_name}")


if __name__ == "__main__":
    main()
