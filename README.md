BT-MRI DSS

Brain Tumor MRI Decision-Support System built around the thesis-selected Model 3 - 2D U-Net++ checkpoint.

What the system does

- Accepts BraTS-compatible, co-registered FLAIR, T1, T1CE, and T2 NIfTI volumes.
- Reproduces the U-Net++ preprocessing used in the experiment.
- Performs binary whole-tumor segmentation with the validation-selected threshold ('0.65').
- Reconstructs and saves the predicted whole-tumor mask in original NIfTI coordinates.
- Computes predicted tumor burden (B), affected-slice ratio (A), maximum slice tumor ratio (M), prediction confidence, and uncertainty.
- Applies the thesis severity and rule-guided recommendation logic after U-Net++ severity calibration is generated.
- Provides interactive MRI, probability-map, tumor-profile, evaluation, and 3D-mask visualizations.
- Saves every analysis to a selected results folder and maintains an SQLite history database.
- Generates a PDF report, JSON/CSV summaries, slice-level measurements, NIfTI mask, probability volume, plots, and a ZIP bundle.

## Important methodological safeguard

The original severity prototype was calibrated using Model 2 (Attention U-Net). Those percentile thresholds are not reused here. The final U-Net++ system requires a new 'config/severity_calibration.json' generated from U-Net++ validation predictions.

The supplied 'selected_threshold.json' sets the final U-Net++ segmentation operating threshold to 0.65, selected by validation Dice.

## Installation

Python 3.10-3.12 is recommended. If you already have the thesis PyTorch/CUDA installation, keep it and install the remaining requirements.

bash
cd BT_MRI_DSS
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt


For an NVIDIA GPU, use the PyTorch installation compatible with your CUDA driver if the generic pip package does not provide the desired CUDA build.

## One-time severity calibration

Run this on the same BraTS2021_Training_Data collection used for the experiment:

bash
python calibrate_severity.py \
  --data-root /path/to/BraTS2021_Training_Data


The script reproduces the seed-42 70/15/15 patient split, uses the first 40 sorted validation IDs as in the prototype, processes all valid axial brain slices, and calculates q20/q40/q60/q80 from U-Net++ predicted tumor burden. It saves:

- 'config/severity_calibration.json'
- 'config/severity_calibration_reference.csv'

Restart the application after calibration.

## Run the application

bash
streamlit run app.py


or use 'run_app.sh' / 'run_app.bat'.

## Analysis modes

Research Analysis requires FLAIR, T1, T1CE, and T2 only. It generates segmentation, severity, confidence/uncertainty, and recommendation output once calibration is available.

Evaluation Mode additionally accepts the BraTS segmentation mask and reports case-level segmentation metrics. The Dice-based safety modifier is only used in this mode because a new clinical case has no ground-truth segmentation.

## Results archive

The default folder is '~/BT_MRI_DSS_RESULTS', but it can be changed in the sidebar.

text
BT_MRI_DSS_RESULTS/
├── database/
│   └── analysis_history.sqlite
├── cases/
│   └── <analysis_id>/
│       ├── predicted_whole_tumor_mask.nii.gz
│       ├── probability_volume.npz
│       ├── result_summary.json
│       ├── result_summary.csv
│       ├── slice_measurements.csv
│       ├── tumor_overlay_and_probability.png
│       ├── tumor_burden_profile.png
│       ├── report.pdf
│       └── analysis_bundle.zip
└── exports/

## Scope

This is a research decision-support prototype. The imaging-derived severity score is not a WHO tumor grade or histopathological classification. The recommendation output is not a diagnosis or treatment prescription and does not replace review by a qualified radiologist or clinical specialist.
