from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import torch

from .metrics import compute_metrics_from_counts
from .preprocessing import PreprocessConfig, inverse_map_2d, preprocess_slice_from_arrays


@torch.no_grad()
def run_case_inference(
    model,
    volumes: Dict[str, np.ndarray],
    device: torch.device,
    threshold: float,
    cfg: PreprocessConfig,
    batch_size: int = 16,
    case_id: str = "case",
    include_gt: bool = False,
):
    depth = int(volumes["flair"].shape[-1])
    images, gts, brains, metas, z_indices = [], [], [], [], []

    for z in range(depth):
        image, gt, brain, meta = preprocess_slice_from_arrays(volumes, z, cfg, include_gt=include_gt)
        if meta["brain_pixels"] <= 10:
            continue
        images.append(image)
        gts.append(gt)
        brains.append(brain)
        metas.append(meta)
        z_indices.append(z)

    if not images:
        raise RuntimeError("No valid axial brain slices were found after preprocessing")

    probs_chunks, preds_chunks = [], []
    for start in range(0, len(images), batch_size):
        batch = torch.stack(images[start:start + batch_size]).to(device)
        batch = torch.clamp(torch.nan_to_num(batch, nan=0.0, posinf=cfg.zscore_clip, neginf=-cfg.zscore_clip), -cfg.zscore_clip, cfg.zscore_clip)
        probs = torch.sigmoid(model(batch))
        preds = (probs >= float(threshold)).float()
        probs_chunks.append(probs.cpu())
        preds_chunks.append(preds.cpu())

    probs = torch.cat(probs_chunks, 0)
    preds = torch.cat(preds_chunks, 0)
    gt_tensor = torch.stack(gts).float()

    pred_pixels = preds.sum(dim=(1, 2, 3)).numpy()
    gt_pixels = gt_tensor.sum(dim=(1, 2, 3)).numpy() if include_gt else np.zeros_like(pred_pixels)
    brain_pixels = np.asarray([m["brain_pixels"] for m in metas], dtype=np.float32)

    total_pred = float(pred_pixels.sum())
    total_brain = float(brain_pixels.sum())
    burden = total_pred / max(total_brain, 1.0)
    affected_flags = pred_pixels > 10
    affected_ratio = float(affected_flags.mean())
    slice_ratios = pred_pixels / np.maximum(brain_pixels, 1.0)
    max_slice_ratio = float(np.max(slice_ratios))
    mean_slice_ratio = float(np.mean(slice_ratios))

    mean_conf = float(probs[preds.bool()].mean().item()) if total_pred > 0 else 0.0
    uncertainty_band = ((probs >= max(float(threshold) - 0.10, 0.0)) & (probs <= min(float(threshold) + 0.10, 1.0))).float()
    uncertainty_ratio = float(uncertainty_band.mean().item())

    result = {
        "case_id": case_id,
        "num_valid_slices": int(len(images)),
        "threshold": float(threshold),
        "predicted_tumor_pixels": total_pred,
        "brain_pixels": total_brain,
        "predicted_tumor_burden": float(burden),
        "affected_slice_ratio": affected_ratio,
        "max_slice_tumor_ratio": max_slice_ratio,
        "mean_slice_tumor_ratio": mean_slice_ratio,
        "mean_confidence_in_prediction": mean_conf,
        "uncertainty_ratio": uncertainty_ratio,
    }

    if include_gt:
        tp = float((preds * gt_tensor).sum().item())
        fp = float((preds * (1 - gt_tensor)).sum().item())
        fn = float(((1 - preds) * gt_tensor).sum().item())
        tn = float(((1 - preds) * (1 - gt_tensor)).sum().item())
        metrics = compute_metrics_from_counts(tp, fp, fn, tn)
        result.update({f"case_{k}": v for k, v in metrics.items()})
        total_gt = float(gt_pixels.sum())
        result["gt_tumor_pixels"] = total_gt
        result["gt_tumor_burden"] = total_gt / max(total_brain, 1.0)
        result["gt_affected_slice_ratio"] = float((gt_pixels > 10).mean())

    h, w, d = volumes["flair"].shape
    pred_vol = np.zeros((h, w, d), dtype=np.uint8)
    prob_vol = np.zeros((h, w, d), dtype=np.float32)
    for idx, z in enumerate(z_indices):
        pred_2d = preds[idx, 0].numpy()
        prob_2d = probs[idx, 0].numpy()
        pred_vol[:, :, z] = (inverse_map_2d(pred_2d, metas[idx], "nearest") > 0.5).astype(np.uint8)
        prob_vol[:, :, z] = inverse_map_2d(prob_2d, metas[idx], "bilinear").astype(np.float32)

    aux = {
        "z_indices": np.asarray(z_indices, dtype=np.int32),
        "images": torch.stack(images),
        "gt_masks": gt_tensor if include_gt else None,
        "probs": probs,
        "preds": preds,
        "pred_pixels_per_slice": pred_pixels,
        "gt_pixels_per_slice": gt_pixels,
        "brain_pixels_per_slice": brain_pixels,
        "slice_tumor_ratio": slice_ratios,
        "pred_volume_original": pred_vol,
        "prob_volume_original": prob_vol,
    }
    return result, aux
