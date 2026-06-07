from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "training_output/cnncap_baseline"

DATASETS = {
    "asap7": {"data_dir": REPO_ROOT / "data/asap7_50K", "window_width": 2.736},
    "real65": {"data_dir": REPO_ROOT / "data/real65_50K", "window_width": 10.692},
}

RUNS = [
    {"dataset": "asap7", "goal": "total", "batch_size": 448, "device": "cuda:7"},
    {"dataset": "real65", "goal": "total", "batch_size": 448, "device": "cuda:7"},
    {"dataset": "asap7", "goal": "env", "batch_size": 2048, "device": "cuda:7"},
    {"dataset": "real65", "goal": "env", "batch_size": 2048, "device": "cuda:7"},
]


def run_name(run):
    return f"{run['dataset']}__cnncap-resnet34__{run['goal']}__b{run['batch_size']}"
