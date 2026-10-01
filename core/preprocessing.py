from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover
    nib = None
    _NIB_IMPORT_ERROR = exc
else:
    _NIB_IMPORT_ERROR = None


@dataclass
class PreprocessConfig:
    img_size: int = 192
    percentile_low: float = 1.0
    percentile_high: float = 99.5
    zscore_clip: float = 5.0
    use_brain_crop: bool = True
    crop_padding: int = 8


def require_nibabel() -> None:
    if nib is None:
        raise ImportError("nibabel is required for NIfTI input. Install dependencies from requirements.txt") from _NIB_IMPORT_ERROR


def robust_percentile_zscore(x, brain_mask=None, p_low=1.0, p_high=99.5, clip_value=5.0, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if brain_mask is None:
        brain_mask = x > 0
    brain_mask = np.asarray(brain_mask).astype(bool)
    finite_mask = brain_mask & np.isfinite(x)
    if finite_mask.sum() > 20:
        vals = x[finite_mask]
        lo, hi = np.percentile(vals, [p_low, p_high])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x, dtype=np.float32)
        x = np.clip(x, lo, hi)
        vals = x[finite_mask]
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if not np.isfinite(std) or std < eps:
            std = 1.0
        x = (x - mean) / std
    else:
        mean = float(np.mean(x))
        std = float(np.std(x))
        if not np.isfinite(std) or std < eps:
            return np.zeros_like(x, dtype=np.float32)
        x = (x - mean) / std
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    return np.clip(x, -clip_value, clip_value).astype(np.float32)


def get_2d_brain_bbox(brain_mask: np.ndarray, padding: int = 8):
    ys, xs = np.where(np.asarray(brain_mask).astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max() + 1)
    x1, x2 = int(xs.min()), int(xs.max() + 1)
    y1, x1 = max(0, y1 - padding), max(0, x1 - padding)
    y2 = min(brain_mask.shape[0], y2 + padding)
    x2 = min(brain_mask.shape[1], x2 + padding)
    if (y2 - y1) < 16 or (x2 - x1) < 16:
        return None
    return y1, y2, x1, x2


def resize_tensor_2d(x: torch.Tensor, size: int, mode: str) -> torch.Tensor:
    x = x.unsqueeze(0)
    if mode == "nearest":
        y = F.interpolate(x, size=(size, size), mode=mode)
    else:
        y = F.interpolate(x, size=(size, size), mode=mode, align_corners=False)
    return y.squeeze(0).contiguous()


def load_case(paths: Dict[str, str | Path], require_seg: bool = False):
    require_nibabel()
    required = ["flair", "t1", "t1ce", "t2"] + (["seg"] if require_seg else [])
    for key in required:
        if key not in paths or not Path(paths[key]).exists():
            raise FileNotFoundError(f"Missing required {key.upper()} NIfTI file")

    imgs = {k: nib.load(str(paths[k])) for k in required}
    shapes = {k: tuple(imgs[k].shape) for k in required}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) != 1:
        raise ValueError(f"MRI volumes must have identical dimensions. Found: {shapes}")
    if len(next(iter(unique_shapes))) != 3:
        raise ValueError("Expected 3D NIfTI volumes")

    ref = imgs["flair"]
    affine_warnings = []
    for key, img in imgs.items():
        if not np.allclose(img.affine, ref.affine, atol=1e-3):
            affine_warnings.append(key)
    if affine_warnings:
        raise ValueError("Input volumes are not spatially aligned with FLAIR: " + ", ".join(affine_warnings))

    arrays = {k: np.asarray(img.dataobj, dtype=np.float32) for k, img in imgs.items()}
    for k in arrays:
        arrays[k] = np.nan_to_num(arrays[k], nan=0.0, posinf=0.0, neginf=0.0)
    return arrays, ref


def preprocess_slice_from_arrays(
    volumes: Dict[str, np.ndarray],
    z: int,
    cfg: PreprocessConfig,
    include_gt: bool = False,
):
    flair = volumes["flair"][:, :, z]
    t1 = volumes["t1"][:, :, z]
    t1ce = volumes["t1ce"][:, :, z]
    t2 = volumes["t2"][:, :, z]
    brain_mask = (flair > 0) | (t1 > 0) | (t1ce > 0) | (t2 > 0)

    norm = [
        robust_percentile_zscore(m, brain_mask, cfg.percentile_low, cfg.percentile_high, cfg.zscore_clip)
        for m in (flair, t1, t1ce, t2)
    ]
    image_np = np.stack(norm, axis=0).astype(np.float32)

    if include_gt:
        seg = volumes["seg"][:, :, z]
        gt_np = np.isin(seg, [1, 2, 4]).astype(np.float32)[None]
    else:
        gt_np = np.zeros((1, *brain_mask.shape), dtype=np.float32)

    bbox = get_2d_brain_bbox(brain_mask, cfg.crop_padding) if cfg.use_brain_crop else None
    if bbox is not None:
        y1, y2, x1, x2 = bbox
        image_np = image_np[:, y1:y2, x1:x2]
        gt_np = gt_np[:, y1:y2, x1:x2]
        brain_crop = brain_mask[y1:y2, x1:x2].astype(np.float32)[None]
    else:
        brain_crop = brain_mask.astype(np.float32)[None]

    image = resize_tensor_2d(torch.from_numpy(image_np).float(), cfg.img_size, "bilinear")
    gt = resize_tensor_2d(torch.from_numpy(gt_np).float(), cfg.img_size, "nearest")
    brain = resize_tensor_2d(torch.from_numpy(brain_crop).float(), cfg.img_size, "nearest")
    image = torch.clamp(torch.nan_to_num(image, nan=0.0, posinf=cfg.zscore_clip, neginf=-cfg.zscore_clip), -cfg.zscore_clip, cfg.zscore_clip)
    gt = (gt > 0.5).float()
    brain = (brain > 0.5).float()

    meta = {
        "bbox": bbox,
        "original_shape": tuple(brain_mask.shape),
        "brain_pixels": int(brain.sum().item()),
    }
    return image, gt, brain, meta


def inverse_map_2d(processed: np.ndarray, meta: dict, mode: str = "nearest") -> np.ndarray:
    """Map a 192x192 prediction/probability back into the original slice coordinates."""
    h, w = meta["original_shape"]
    bbox = meta["bbox"]
    if bbox is None:
        target_h, target_w = h, w
    else:
        y1, y2, x1, x2 = bbox
        target_h, target_w = y2 - y1, x2 - x1

    ten = torch.from_numpy(processed.astype(np.float32))[None, None]
    if mode == "nearest":
        resized = F.interpolate(ten, size=(target_h, target_w), mode="nearest")[0, 0].numpy()
    else:
        resized = F.interpolate(ten, size=(target_h, target_w), mode="bilinear", align_corners=False)[0, 0].numpy()

    out = np.zeros((h, w), dtype=np.float32)
    if bbox is None:
        out[:, :] = resized
    else:
        y1, y2, x1, x2 = bbox
        out[y1:y2, x1:x2] = resized
    return out


def display_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x)
    vals = x[finite]
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)
