from __future__ import annotations
import json
from pathlib import Path


def load_calibration(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    required = {"q20", "q40", "q60", "q80"}
    if not required.issubset(data):
        return None
    return data


def assign_severity_score(row, thresholds):
    burden = float(row["predicted_tumor_burden"])
    affected = float(row["affected_slice_ratio"])
    max_slice = float(row["max_slice_tumor_ratio"])
    very_small_detection = burden < max(float(thresholds["q20"]) * 0.10, 1e-5)
    no_affected_slices = affected < 0.01
    if very_small_detection or no_affected_slices:
        return 0
    if burden <= thresholds["q20"]:
        score = 1
    elif burden <= thresholds["q40"]:
        score = 2
    elif burden <= thresholds["q60"]:
        score = 3
    elif burden <= thresholds["q80"]:
        score = 4
    else:
        score = 5
    if affected >= 0.45 and score < 5:
        score += 1
    if max_slice >= 0.25 and score < 5:
        score += 1
    return int(min(score, 5))


def severity_label(score):
    return {
        0: "No tumor detected by model",
        1: "Very low imaging-derived burden",
        2: "Low imaging-derived burden",
        3: "Moderate imaging-derived burden",
        4: "High imaging-derived burden",
        5: "Very high imaging-derived burden",
    }.get(int(score), "Unknown")


def confidence_category(row):
    conf = float(row["mean_confidence_in_prediction"])
    uncert = float(row["uncertainty_ratio"])
    if int(row["severity_score"]) == 0:
        return "Low confidence" if uncert > 0.08 else "Moderate confidence"
    if conf >= 0.85 and uncert < 0.05:
        return "High confidence"
    if conf >= 0.70 and uncert < 0.10:
        return "Moderate confidence"
    return "Low confidence"
