from config import GPUS, REPO_ROOT

OUTPUT_ROOT = REPO_ROOT / "training_output/pretrain"
MAX_CONCURRENCY = len(GPUS)
RUNS = [
    {
        "use_transformer": "True",
        "n_embd": "256",
        "n_head": "4",
        "n_layer": "6",
        "batch_size": "384",
        "data_dir": REPO_ROOT / "data/adaptation/pretrain",
    }
]
