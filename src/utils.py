import matplotlib.pyplot as plt


def log_loss_by_size(writer, loss_by_size, split, step):
    sizes = list(loss_by_size)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(sizes, loss_by_size.values(), color="blue", alpha=0.7)
    ax.set(xlabel="Sequence Length", ylabel="Loss", title=f"Loss by Sequence Length for {split} set")
    ax.set_xticks(sizes)
    fig.tight_layout()
    writer.add_figure(f"loss_by_size_{split}/histogram", fig, global_step=step)
    plt.close(fig)


def log_embedding(writer, model, step):
    embedding = model.extra_embedding.weight.detach().cpu().numpy()[:5]

    fig, ax = plt.subplots(figsize=(16, 4))
    image = ax.imshow(embedding, aspect="auto", origin="lower")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(title="Extra Embedding Weights", xlabel="embedding dimension", ylabel="embedding index")
    fig.tight_layout()
    writer.add_figure("extra_embedding/weights", fig, global_step=step)
    plt.close(fig)


def log_prediction_comparison(writer, pred, target, step):
    pred = pred[0].detach().cpu()
    target = target[0].detach().cpu()
    vmin = min(pred.min(), target.min())
    vmax = max(pred.max(), target.max())

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(pred, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
    image = axes[1].imshow(target, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[0].set_title("Prediction")
    axes[1].set_title("Target")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    writer.add_figure("pred_vs_target/comparison", fig, global_step=step)
    plt.close(fig)
