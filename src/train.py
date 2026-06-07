import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from dataset import get_dataloader, get_dataset_dict
from model import GPT, GPTConfig
from utils import log_embedding, log_loss_by_size, log_prediction_comparison

SRC_PATH = Path(__file__).resolve().parent
# I/O
out_dir = 'training_output/default'
eval_interval = 2000
plot_interval = 10 * eval_interval
log_interval = 500
eval_only = False
checkpoint_path = ''
tensorboard_log = True
# data
data_dir = 'data/asap7_50K'
gradient_accumulation_steps = 1
batch_size = 128
bucket_width = 1
# model
input_dim = 4
n_layer = 3
n_head = 4
n_embd = 256
use_transformer = False
exp_ratio = 3
dropout = 0.0
bias = True
attention_type = 'standard'
ffn_type = 'swiglu'
norm_type = 'rmsnorm'
head_activation = 'none'
head_mode = 'matrix'
loss_f = "laplacian"

# adamw optimizer
learning_rate = 1.5e-4
max_iters = 300000
weight_decay = 1e-4
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 1000
lr_decay_iters = max_iters
min_lr = 1e-5
# system
device = 'cuda:0'
dtype = 'float32'
compile = False
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec((SRC_PATH / 'configurator.py').read_text())
config = {k: globals()[k] for k in config_keys}


os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
device_type = 'cuda' if 'cuda' in device else 'cpu'
if device_type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

iter_num = 0
best_val_loss = 1e9

model_args = dict(
    input_dim=input_dim,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    bias=bias,
    dropout=dropout,
    use_transformer=use_transformer,
    exp_ratio=exp_ratio,
    attention_type=attention_type,
    norm_type=norm_type,
    ffn_type=ffn_type,
    head_activation=head_activation,
    head_mode=head_mode,
)
if not checkpoint_path:
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    dataset_dict = get_dataset_dict(data_dir=data_dir)
else:
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = dict(checkpoint['model_args'])
    model_args.setdefault("head_mode", head_mode)
    gptconf = GPTConfig(**model_args)
    head_mode = gptconf.head_mode
    model_args["head_mode"] = head_mode
    config["head_mode"] = head_mode
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    unwanted_prefix = '_orig_mod.'
    for key in list(state_dict):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("[load_state_dict] missing keys:", missing)
    print("[load_state_dict] unexpected keys:", unexpected)
    dataset_dict = checkpoint['dataset_dict']
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    learning_rate /= 10
    min_lr /= 10
    print(f"[finetuning] Lower learning rate to {learning_rate}, {min_lr}")

model.to(device)

n_params = sum(param.numel() for param in model.parameters())
print("number of parameters: %.2fM" % (n_params / 1e6))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if checkpoint_path:
    del checkpoint

if eval_only:
    tensorboard_log = False
    batch_size = 128
    compile = False
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

dataloaders = {
    split: get_dataloader(
        split=split,
        data_dir=data_dir,
        dataset_dict=dataset_dict,
        batch_size=batch_size,
        bucket_width=bucket_width,
        num_workers=1,
        pin_memory=device_type == "cuda",
        is_train=split == "train",
        head_mode=head_mode,
    )
    for split in ("train", "val", "test")
}


@torch.no_grad()
def benchmark_test_set():
    from thop import profile

    model.eval()
    test_loader = dataloaders["test"]
    sample_count = len(test_loader.dataset)
    if not sample_count:
        print("No test samples; skipping benchmark")
        return
    total_flops = 0
    for X, _, _ in test_loader:
        X = X.to(device)
        with ctx:
            flops, _ = profile(model, inputs=(X,), verbose=False)
            total_flops += flops
    print(f"Total FLOPs: {total_flops / 1e9:.6f} G, Total Samples: {sample_count}")
    print(f"FLOPs per sample: {total_flops / sample_count / 1e6:.6f} M")

    if device_type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for X, _, _ in test_loader:
        with ctx:
            model(X.to(device))
    if device_type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Total inference time: {elapsed:.6f} s, Total Samples: {sample_count}")
    print(f"Time per sample: {elapsed / sample_count * 1000:.6f} ms")


@torch.no_grad()
def estimate_loss(writer=None, max_batches=200):
    out = {}
    model.eval()
    for split, loader in dataloaders.items():
        totals = dict(loss=0.0, self_relerr=0.0, self_higherr=0, coupling_relerr=0.0, coupling_higherr=0, count=0, coupling_count=0)
        size_loss = {}
        size_count = {}

        for batch_idx, (X, Y, mask) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            X, Y = X.to(device), Y.to(device)
            mask = mask.to(device) if mask is not None else None
            with ctx:
                pred, loss = model(X, Y, mask, loss_f=loss_f)

            weights = torch.nn.functional.relu(-pred)
            pred = torch.diag_embed(weights.sum(dim=-1)) - weights
            b, t, _ = X.shape

            if head_mode == "first_row":
                pred = pred[:, 0]
                count = b
                self_relerr = pred[:, 0] / (Y[:, 0] + 1e-9) - 1
                coupling_mask = Y.abs() > 0.01 * Y[:, :1].abs()
                coupling_mask[:, 0] = False
                if mask is not None:
                    coupling_mask &= mask.bool()
                coupling_relerr = (pred / (Y + 1e-9) - 1).abs()
            else:
                count = mask.sum().item() if mask is not None else b * t
                diag = torch.diagonal(Y, dim1=1, dim2=2)
                self_relerr = torch.diagonal(pred, dim1=1, dim2=2) / (diag + 1e-9) - 1
                if mask is not None:
                    self_relerr *= mask
                diag = diag.abs()
                coupling_mask = (Y.abs() > 0.01 * diag.unsqueeze(-1)) & (Y.abs() > 0.01 * diag.unsqueeze(-2))
                coupling_mask &= ~torch.eye(t, dtype=torch.bool, device=Y.device).unsqueeze(0)
                if mask is not None:
                    coupling_mask &= mask.unsqueeze(-1).bool() & mask.unsqueeze(-2).bool()
                coupling_relerr = (pred / (Y + 1e-9) - 1).abs()

            coupling_count = coupling_mask.sum().item()
            weighted_loss = loss.item() * count
            totals["loss"] += weighted_loss
            totals["self_relerr"] += self_relerr.abs().sum().item()
            totals["self_higherr"] += (self_relerr.abs() > 0.05).sum().item()
            totals["coupling_relerr"] += (coupling_relerr * coupling_mask).sum().item()
            totals["coupling_higherr"] += ((coupling_relerr > 0.1) & coupling_mask).sum().item()
            totals["count"] += count
            totals["coupling_count"] += coupling_count
            size_loss[t] = size_loss.get(t, 0.0) + weighted_loss
            size_count[t] = size_count.get(t, 0) + count

        count = totals["count"]
        coupling_count = totals["coupling_count"]
        out[f"{split}/loss"] = totals["loss"] / count if count else float("nan")
        out[f"{split}/self_relerr"] = totals["self_relerr"] / count if count else float("nan")
        out[f"{split}/self_higherr_ratio"] = totals["self_higherr"] / count if count else float("nan")
        out[f"{split}/coupling_relerr"] = totals["coupling_relerr"] / coupling_count if coupling_count else float("nan")
        out[f"{split}/coupling_higherr_ratio"] = totals["coupling_higherr"] / coupling_count if coupling_count else float("nan")

        if writer is not None and iter_num % plot_interval == 0 and count:
            loss_by_size = {size: size_loss[size] / size_count[size] for size in sorted(size_count)}
            log_loss_by_size(writer, loss_by_size, split, iter_num)
            if split == "val":
                log_embedding(writer, model, iter_num)
                if head_mode == "matrix":
                    log_prediction_comparison(writer, pred, Y, iter_num)

    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    coeff = (lr_decay_iters - it) / (lr_decay_iters - warmup_iters)
    return min_lr + coeff * (learning_rate - min_lr)

writer = None
if tensorboard_log:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=out_dir)


def format_metrics_table(metrics):
    columns = [
        ("split", lambda split: split),
        ("loss", lambda split: f"{metrics[f'{split}/loss']:.4e}"),
        ("self relerr", lambda split: f"{metrics[f'{split}/self_relerr']:.2%}"),
        ("self >5%", lambda split: f"{metrics[f'{split}/self_higherr_ratio']:.2%}"),
        ("coupling relerr", lambda split: f"{metrics[f'{split}/coupling_relerr']:.2%}"),
        ("coupling >10%", lambda split: f"{metrics[f'{split}/coupling_higherr_ratio']:.2%}"),
    ]
    rows = [[value(split) for _, value in columns] for split in ("train", "val", "test")]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, (header, _) in enumerate(columns)]
    header = " | ".join(name.ljust(width) for (name, _), width in zip(columns, widths))
    divider = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(value.ljust(width) if index == 0 else value.rjust(width) for index, (value, width) in enumerate(zip(row, widths)))
        for row in rows
    ]
    return "\n".join((header, divider, *body))


def evaluate(step, lr, max_batches=200):
    global best_val_loss
    metrics = estimate_loss(writer, max_batches=max_batches)
    is_best = metrics["val/loss"] < best_val_loss
    print(f"{'[best]' if is_best else ''}[eval] step {step}\n{format_metrics_table(metrics)}")

    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(key, value, step)
        writer.add_scalar("LearningRate", lr, step)

    if not is_best or eval_only:
        return
    best_val_loss = metrics["val/loss"]
    if step == 0:
        return

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": step,
        "best_val_loss": best_val_loss,
        "config": config,
        "dataset_dict": dataset_dict,
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))


if eval_only:
    benchmark_test_set()
    evaluate(iter_num, get_lr(iter_num) if decay_lr else learning_rate, max_batches=None)
else:
    start_time = time.time()
    micro_step = 0
    epoch = 0
    while iter_num <= max_iters:
        for X, Y, mask in dataloaders["train"]:
            X, Y = X.to(device), Y.to(device)
            mask = mask.to(device) if mask is not None else None
            with ctx:
                _, loss = model(X, Y, mask, loss_f=loss_f)
                loss = loss / gradient_accumulation_steps
            loss.backward()

            micro_step += 1
            if micro_step % gradient_accumulation_steps:
                continue

            lr = get_lr(iter_num) if decay_lr else learning_rate
            for group in optimizer.param_groups:
                group["lr"] = lr
            if iter_num % eval_interval == 0:
                evaluate(iter_num, lr)

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if iter_num % log_interval == 0:
                elapsed_ms = (time.time() - start_time) * 1000
                train_loss = loss.item() * gradient_accumulation_steps
                print(f"[log] iter {iter_num}, epoch {epoch}, ncond {X.size(-2)}, lr {lr}: loss {train_loss:.4f}, time {elapsed_ms:.2f}ms")
            start_time = time.time()

            iter_num += 1
            if iter_num > max_iters:
                break
        epoch += 1

if writer is not None:
    writer.close()
