from config import GPUS, REPO_ROOT

OUTPUT_ROOT = REPO_ROOT / "training_output/adaptation"
MAX_CONCURRENCY = len(GPUS)
DATASETS = [
    REPO_ROOT / f"data/adaptation/asap7_{percent}"
    for percent in (10, 30, 50, 70)
] + [REPO_ROOT / "data/asap7_50K"]
RUNS = [
    {
        "use_transformer": "True",
        "n_embd": "256",
        "n_head": "4",
        "n_layer": "6",
        "batch_size": "128",
        "data_dir": data_dir,
        "finetune": "True",
    }
    for data_dir in DATASETS
]
