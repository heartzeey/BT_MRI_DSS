from __future__ import annotations
import math
import numpy as np


def safe_metric(num, den, default=0.0, eps=1e-7):
    if den <= eps:
        return default
    return float(num / (den + eps))


def compute_metrics_from_counts(tp, fp, fn, tn):
    dice = safe_metric(2 * tp, 2 * tp + fp + fn)
    iou = safe_metric(tp, tp + fp + fn)
    precision = safe_metric(tp, tp + fp)
    recall = safe_metric(tp, tp + fn)
    specificity = safe_metric(tn, tn + fp)
    accuracy = safe_metric(tp + tn, tp + fp + fn + tn)
    balanced_accuracy = 0.5 * (recall + specificity)
    npv = safe_metric(tn, tn + fn)
    fpr = safe_metric(fp, fp + tn)
    fnr = safe_metric(fn, fn + tp)
    fdr = safe_metric(fp, fp + tp)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
    mcc = safe_metric(tp * tn - fp * fn, denom)
    volume_similarity = 1.0 - safe_metric(abs(fn - fp), 2 * tp + fp + fn)
    pred_vol, gt_vol = tp + fp, tp + fn
    rve = float(abs(pred_vol - gt_vol) / (gt_vol + 1e-7)) if gt_vol > 0 else float("nan")
    return {
        "dice": dice, "iou": iou, "precision": precision, "recall_sensitivity": recall,
        "specificity": specificity, "accuracy": accuracy, "balanced_accuracy": balanced_accuracy,
        "npv": npv, "fpr": fpr, "fnr": fnr, "fdr": fdr, "mcc": mcc,
        "volume_similarity": volume_similarity, "relative_volume_error": rve,
    }
