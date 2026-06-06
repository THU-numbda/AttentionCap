import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def visualize_sample(rects: torch.Tensor, colors_mat: torch.Tensor, fname: str) -> None:
    """Save conductor rectangles colored by the capacitance-matrix diagonal."""
    if rects.size(0) != colors_mat.size(0):
        raise ValueError("rects and colors_mat must have the same first dimension")

    rects = rects.cpu().numpy()
    colors = np.diagonal(colors_mat.cpu().numpy())
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=colors.min(), vmax=colors.max())
    ax.set_facecolor("#F0F0F0")
    ax.grid(True, linestyle="--", linewidth=0.5)

    positions = []
    for rect in rects:
        if len(rect) == 4:
            positions.append(rect)
        elif len(rect) == 3:
            x, width, y = rect
            positions.append((x, y, width, 0.1))
        else:
            raise ValueError("rects must have shape (T,4) or (T,3)")

    for (x, y, width, height), color in zip(positions, colors):
        ax.add_patch(patches.Rectangle(
            (x - width / 2, y - height / 2), width, height,
            facecolor=cmap(norm(color)), alpha=0.8,
        ))

    ax.set_xlim(-3.06, 3.06)
    ax.set_ylim(min((y - h / 2 for _, y, _, h in positions), default=0) - 0.06, 10.06)
    ax.set(title="Visualization of Rectangles", xlabel="X-coordinate", ylabel="Y-coordinate")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label="Color Value")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
