from config import GPUS, REPO_ROOT

OUTPUT_ROOT = REPO_ROOT / "training_output/ablation_study"
MAX_CONCURRENCY = len(GPUS)
DATA_DIR = REPO_ROOT / "data"
MODEL_CONFIGS = [
    {"use_transformer": "True", "n_embd": "384", "n_head": "4", "n_layer": "8"}
]
ABLATIONS = [
    {"ffn_type": "mlp", "loss_f": "laplacian"},
    {"ffn_type": "swiglu", "loss_f": "mse"},
]
RUNS = [
    {**model, **ablation, "data_dir": DATA_DIR, "batch_size": "512"}
    for model in MODEL_CONFIGS
    for ablation in ABLATIONS
]
