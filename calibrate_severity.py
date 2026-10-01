from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from core.model import choose_device, load_model
from core.preprocessing import PreprocessConfig, load_case
from core.inference import run_case_inference


def discover_cases(root: Path):
    cases = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("BraTS2021_")):
        cid = d.name
        paths = {k: d / f"{cid}_{suffix}.nii.gz" for k,suffix in {"flair":"flair","t1":"t1","t1ce":"t1ce","t2":"t2","seg":"seg"}.items()}
        if all(p.exists() for p in paths.values()):
            cases.append((cid, paths))
    if not cases:
        raise RuntimeError("No complete BraTS2021 cases found")
    return cases


def main():
    ap = argparse.ArgumentParser(description="Calibrate U-Net++ severity thresholds from validation-predicted tumor burden.")
    ap.add_argument("--data-root", required=True, help="BraTS2021_Training_Data folder")
    ap.add_argument("--checkpoint", default="models/best_model3_unetplusplus_brats2021.pt")
    ap.add_argument("--threshold-file", default="config/selected_threshold.json")
    ap.add_argument("--output", default="config/severity_calibration.json")
    ap.add_argument("--max-cases", type=int, default=40, help="Matches the thesis prototype calibration size")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    cases = discover_cases(root)
    ids = [cid for cid,_ in cases]
    train_ids, temp_ids = train_test_split(ids, test_size=0.30, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)
    val_ids = sorted(val_ids)[:args.max_cases]
    lookup = dict(cases)

    threshold_info = json.loads(Path(args.threshold_file).read_text())
    threshold = float(threshold_info["selected_threshold"])
    device = choose_device(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    c = ckpt.get("config", {})
    pcfg = PreprocessConfig(img_size=int(c.get("img_size",192)), percentile_low=float(c.get("percentile_low",1.0)), percentile_high=float(c.get("percentile_high",99.5)), zscore_clip=float(c.get("zscore_clip",5.0)), use_brain_crop=bool(c.get("use_brain_crop",True)), crop_padding=int(c.get("crop_padding",8)))

    rows=[]
    for i,cid in enumerate(val_ids,1):
        volumes, _ = load_case(lookup[cid], require_seg=True)
        result,_ = run_case_inference(model, volumes, device, threshold, pcfg, args.batch_size, cid, include_gt=True)
        rows.append(result)
        print(f"[{i}/{len(val_ids)}] {cid}: B={result['predicted_tumor_burden']:.6f}")

    df=pd.DataFrame(rows)
    detected=df[df["predicted_tumor_burden"]>0]
    if len(detected)>=5:
        qs=np.percentile(detected["predicted_tumor_burden"].to_numpy(), [20,40,60,80])
        fallback=False
    else:
        qs=np.array([0.005,0.015,0.030,0.060],dtype=np.float32); fallback=True

    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    calibration={
        "model":"Model 3 - 2D U-Net++",
        "segmentation_threshold":threshold,
        "q20":float(qs[0]),"q40":float(qs[1]),"q60":float(qs[2]),"q80":float(qs[3]),
        "source":"validation-predicted tumor burden percentiles",
        "num_reference_cases":int(len(df)),
        "validation_case_selection":"sorted validation IDs; first max_cases",
        "fallback_thresholds_used":bool(fallback),
        "seed":42
    }
    output.write_text(json.dumps(calibration,indent=2))
    df.to_csv(output.with_name("severity_calibration_reference.csv"),index=False)
    print(json.dumps(calibration,indent=2))
    print(f"Saved: {output.resolve()}")

if __name__ == "__main__":
    main()
