from config import GPUS, OUTPUT_ROOT, REPO_ROOT
MAX_CONCURRENCY = 4
DATASETS = [
    REPO_ROOT / "data/cnncap/55nm_C_2_3_6",
    REPO_ROOT / "data/cnncap/55nm_C_2_4_6",
    REPO_ROOT / "data/cnncap/15nm_C_2_4_6",
    REPO_ROOT / "data/cnncap/15nm_C_2_4_9",
]
MODEL_CONFIG = {
    "use_transformer": "True",
    "n_embd": "256",
    "n_head": "4",
    "n_layer": "6",
    "input_dim": "3",
    "head_mode": "first_row",
    "batch_size": "128",
}
RUNS = [{**MODEL_CONFIG, "data_dir": data_dir} for data_dir in DATASETS]
