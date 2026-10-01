from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .preprocessing import display_normalize


def save_overlay_png(volumes, aux, out_path: str | Path):
    out_path = Path(out_path)
    pred_counts = aux["pred_pixels_per_slice"]
    idx = int(np.argmax(pred_counts)) if np.max(pred_counts) > 0 else len(pred_counts) // 2
    z = int(aux["z_indices"][idx])
    flair = display_normalize(volumes["flair"][:, :, z])
    pred = aux["pred_volume_original"][:, :, z]
    prob = aux["prob_volume_original"][:, :, z]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(flair, cmap="gray"); axes[0].set_title(f"FLAIR - slice {z}"); axes[0].axis("off")
    axes[1].imshow(flair, cmap="gray"); axes[1].imshow(pred, cmap="Reds", alpha=0.38); axes[1].set_title("Predicted WT overlay"); axes[1].axis("off")
    im = axes[2].imshow(prob, cmap="magma", vmin=0, vmax=1); axes[2].set_title("Probability map"); axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    return out_path


def save_burden_profile_png(aux, out_path: str | Path):
    out_path = Path(out_path)
    fig = plt.figure(figsize=(8, 4.2))
    plt.plot(aux["z_indices"], aux["slice_tumor_ratio"], marker="o", markersize=2, linewidth=1.4)
    plt.xlabel("Axial slice index"); plt.ylabel("Predicted tumor / brain pixels")
    plt.title("Predicted tumor burden across axial slices")
    plt.grid(True, alpha=0.25); plt.tight_layout(); fig.savefig(out_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    return out_path
