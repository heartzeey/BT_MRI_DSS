# Thesis alignment

The application intentionally mirrors the current thesis methodology rather than introducing a new prediction method.

- **Input/target:** FLAIR + T1 + T1CE + T2; binary whole tumor = BraTS labels 1, 2, and 4.
- **Preprocessing:** NaN/Inf replacement, union nonzero brain mask, 1st-99.5th percentile clipping, per-slice z-score normalization within the brain region, clipping to [-5,5], 8-pixel brain-region crop, bilinear MRI resize and nearest mask resize to 192x192.
- **Inference:** all valid axial brain slices are processed for case-level severity analysis; slices with <=10 resized brain pixels are excluded.
- **Final segmentation threshold:** 0.65 from Model 3 validation Dice sweep.
- **Severity variables:** B, A, M, with validation-derived q20/q40/q60/q80. Score 0 test, percentile scoring, A>=0.45 and M>=0.25 sequential adjustments, cap at 5.
- **Confidence/uncertainty:** C and U follow the rules in the thesis prototype, using the final model's operating threshold.
- **Recommendation mapping:** 0 = routine review; 1-2 = non-urgent specialist review; 3 = timely specialist review; 4-5 = prompt specialist review.
- **Safety:** low confidence, U>=0.10, sparse predicted involvement, and evaluation-only Dice<0.70 modifier.
- **Deployment distinction:** the system saves an inverse-mapped NIfTI mask in original coordinates for usability, while the case-level burden and evaluation calculations remain based on the 192x192 processed slices used by the thesis inference procedure.
