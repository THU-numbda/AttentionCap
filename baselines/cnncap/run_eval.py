import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.cnncap.config import DATASETS, OUTPUT_ROOT, RUNS, run_name

EVAL_SCRIPT = Path(__file__).with_name("eval.py")


def main():
    for run in RUNS:
        dataset = DATASETS[run["dataset"]]
        output = OUTPUT_ROOT / run_name(run)
        model = output / "best.model.pth"
        if not model.exists():
            print(f"[SKIP] {run_name(run)}: no checkpoint", flush=True)
            continue
        for split in ("val", "test"):
            command = [
                sys.executable,
                str(EVAL_SCRIPT),
                str(dataset["data_dir"]),
                str(model),
                f"--split={split}",
                f"--goal={run['goal']}",
                f"--window_width={dataset['window_width']}",
                f"--logfile={output / f'{split}.log'}",
                f"--device={run['device']}",
            ]
            print("[EVAL]", " ".join(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
