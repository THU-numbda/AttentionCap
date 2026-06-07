from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "training_output/main_results"
GPUS = ["cuda:7"]
MAX_CONCURRENCY = 3
DATASETS = [
    REPO_ROOT / "data/real65_50K",
    REPO_ROOT / "data/asap7_50K",
    REPO_ROOT / "data",
]
MODEL_CONFIGS = [
    {"use_transformer": "True", "n_embd": "256", "n_head": "4", "n_layer": "6"},
    {"use_transformer": "True", "n_embd": "384", "n_head": "4", "n_layer": "8"},
]
RUNS = [
    {**model, "data_dir": data_dir, "batch_size": "512" if data_dir == REPO_ROOT / "data" else "128"}
    for data_dir in DATASETS
    for model in MODEL_CONFIGS
]
