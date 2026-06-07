import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.cnncap.config import DATASETS, OUTPUT_ROOT, RUNS, run_name

TRAIN_SCRIPT = Path(__file__).with_name("train.py")


def main():
    for run in RUNS:
        dataset = DATASETS[run["dataset"]]
        output = OUTPUT_ROOT / run_name(run)
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            str(dataset["data_dir"]),
            f"--goal={run['goal']}",
            f"--window_width={dataset['window_width']}",
            f"--batch_size={run['batch_size']}",
            f"--device={run['device']}",
            f"--out_dir={output}",
        ]
        print("[LAUNCH]", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
