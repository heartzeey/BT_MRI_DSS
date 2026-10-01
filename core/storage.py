from __future__ import annotations
import csv
import hashlib
import json
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:
    nib = None

from .report import build_pdf_report
from .visualization import save_burden_profile_png, save_overlay_png


def sha256_file(path: str | Path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def init_db(results_root: str | Path):
    root = Path(results_root).expanduser(); (root / "database").mkdir(parents=True, exist_ok=True)
    db = root / "database" / "analysis_history.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS analyses (
            analysis_id TEXT PRIMARY KEY, case_id TEXT, timestamp TEXT, analysis_mode TEXT,
            model_name TEXT, threshold REAL, severity_score INTEGER, severity_label TEXT,
            confidence_category TEXT, urgency_category TEXT, predicted_tumor_burden REAL,
            affected_slice_ratio REAL, max_slice_tumor_ratio REAL, uncertainty_ratio REAL,
            case_dice REAL, case_iou REAL, report_path TEXT, case_dir TEXT
        )""")
    return db


def history_dataframe(results_root: str | Path):
    db = init_db(results_root)
    with sqlite3.connect(db) as con:
        return pd.read_sql_query("SELECT * FROM analyses ORDER BY timestamp DESC", con)


def persist_case(result: dict, aux: dict, volumes: Dict[str, np.ndarray], ref_img, results_root: str | Path, model_info: dict, checkpoint_path: str | Path):
    root = Path(results_root).expanduser(); root.mkdir(parents=True, exist_ok=True)
    (root / "cases").mkdir(exist_ok=True); (root / "exports").mkdir(exist_ok=True)
    case_dir = root / "cases" / result["analysis_id"]; case_dir.mkdir(parents=True, exist_ok=True)

    result = dict(result)
    result["checkpoint_sha256"] = sha256_file(checkpoint_path)
    result["result_folder"] = str(case_dir)

    if nib is not None:
        hdr = ref_img.header.copy(); hdr.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(aux["pred_volume_original"].astype(np.uint8), ref_img.affine, hdr), str(case_dir / "predicted_whole_tumor_mask.nii.gz"))
    np.savez_compressed(case_dir / "probability_volume.npz", probability=aux["prob_volume_original"].astype(np.float16))

    slice_df = pd.DataFrame({
        "z_index": aux["z_indices"],
        "predicted_tumor_pixels": aux["pred_pixels_per_slice"],
        "brain_pixels": aux["brain_pixels_per_slice"],
        "predicted_tumor_ratio": aux["slice_tumor_ratio"],
        "gt_tumor_pixels": aux["gt_pixels_per_slice"],
    })
    slice_df.to_csv(case_dir / "slice_measurements.csv", index=False)
    pd.DataFrame([result]).to_csv(case_dir / "result_summary.csv", index=False)
    (case_dir / "result_summary.json").write_text(json.dumps(result, indent=2, default=str))

    overlay = save_overlay_png(volumes, aux, case_dir / "tumor_overlay_and_probability.png")
    burden = save_burden_profile_png(aux, case_dir / "tumor_burden_profile.png")
    report = build_pdf_report(result, model_info, case_dir / "report.pdf", overlay, burden)

    zip_path = case_dir / "analysis_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in case_dir.iterdir():
            if f.name != zip_path.name and f.is_file():
                zf.write(f, arcname=f.name)

    db = init_db(root)
    with sqlite3.connect(db) as con:
        con.execute("""INSERT OR REPLACE INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            result.get("analysis_id"), result.get("case_id"), result.get("timestamp"), result.get("analysis_mode"),
            model_info.get("model_name", "Model 3 - 2D U-Net++"), result.get("threshold"), result.get("severity_score"),
            result.get("severity_label"), result.get("confidence_category"), result.get("urgency_category"),
            result.get("predicted_tumor_burden"), result.get("affected_slice_ratio"), result.get("max_slice_tumor_ratio"),
            result.get("uncertainty_ratio"), result.get("case_dice"), result.get("case_iou"), str(report), str(case_dir)
        ))
    return {"case_dir": case_dir, "report": report, "zip": zip_path, "overlay": overlay, "burden": burden}
