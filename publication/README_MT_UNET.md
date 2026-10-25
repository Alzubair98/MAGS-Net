# Lightweight MT-U-Net: joint classification and segmentation

Open [`03_Lightweight_MT_UNet_NestedCV.ipynb`](../03_Lightweight_MT_UNet_NestedCV.ipynb)
in the Conda **dl** kernel. The notebook is self-contained, with English notes and both
training and outer-test inference disabled by default.

Section **22.1** combines patient classification, segmentation, and computational cost
for MAGS-Net and MT-U-Net using their saved OOF CSVs. It validates matched samples,
patients, labels, outer folds, seeds, and slice counts. All differences are **MAGS-Net
minus MT-U-Net**. Both models use the same 233 OOF patients and 3,064 slices.

Patient classification averages slice probabilities before argmax. The exact two-sided
McNemar test compares paired correctness; no Holm correction is used for this single
model comparison. There are 10,000 paired patient-bootstrap draws (seed 42, stratified
by true tumor class), sharing patient indices across both models and all four metrics.
The section recomputes patient Accuracy and macro-F1 in each draw. Mean Dice and Mean
IoU average slice scores within patients, then weight patients equally. IoU is averaged
directly, not derived from mean Dice. These are means of 2D scores, not volumetric scores.

The study's `analysis/` receives:

- `paired_patient_statistics.csv`: model scores, the correctness contingency counts,
  McNemar exact p, all four differences, and percentile 95% CI bounds.
- `paired_patient_statistics_table.csv`: four formatted rows for classification and segmentation.
- `multitask_comparison_summary.csv`: two numeric model rows with patient Accuracy,
  macro-F1, mean Dice/IoU, Parameters, MACs, and FP32 weight size.
- `multitask_comparison_table.csv`: the final seven-row paper table, including computational cost.
- `paired_patient_statistics_metadata.json`: settings, metric definitions, counting scope,
  input hashes, and the limits of the statistical interpretation.

Accuracy differences/CIs use percentage points; macro-F1, Dice, and IoU differences/CIs
use the 0–1 scale. The intervals are pointwise, not simultaneous or multiplicity-adjusted.
Uncertainty is conditional on fixed fitted OOF models/folds and excludes retraining and
partition variability. Full-model MACs count Conv2d, ConvTranspose2d, and Linear operations
for one 1×224×224 FP32 input; they exclude normalization, activations, pooling, interpolation,
and elementwise operations. FP32 size is parameters × 4 / 2**20 MiB, excluding buffers,
activations, optimizer state, and checkpoint overhead. The table does not imply measured latency.

The current architecture uses the narrower channels **32, 64, 128, 256, 512** with
explicit `EncoderBlock`, `DecoderBlock`, and `LightweightMTUNet` classes. The notebook
reports measured complexity; it is not parameter-matched to MAGS-Net. This replaces
the initial full-width version and receives a distinct protocol/data-preparation hash.

## Model and loss

Each encoder level uses two bias-free 3x3 convolutions, each followed by BatchNorm and
ReLU. Four max-pooling layers reduce a 224x224 grayscale input to a 14x14 bottleneck.
The decoder uses four learned 2x2 stride-2 transposed convolutions, concatenated encoder
skips, and double-convolution blocks. Bilinear interpolation aligns a skip if spatial
sizes differ. A final 1x1 convolution produces one segmentation logit per pixel. The
512-channel bottleneck classification head is global average pooling, FC256, ReLU,
dropout 0.5, and FC3. The model returns `logits`, `seg_logits`, and sigmoid `seg_mask`.

All layers start from random PyTorch initialization. There is no pretrained encoder,
attention, squeeze-and-excitation, gating, tumor-aware pooling, extra multiscale branch,
or deep supervision. The model receives MRI images only; masks are training targets.

```text
Total loss = CrossEntropy + 0.5 * (BCEWithLogits + soft Dice loss)
```

`DiceLoss(smooth=1.0)` implements per-slice soft Dice loss averaged over
the batch. Losses use raw logits, with FP32 reductions for mixed-precision stability.
`compute_loss` returns `total`, `classification`, `segmentation`, `bce`, and `dice_loss`.
BCE and Dice loss are also printed each epoch and saved in training/validation histories.
The segmentation weight remains 0.5; there is no auxiliary loss.

Dice smoothing is **1.0** for both this model and the existing MAGS-Net reference.
The segmentation-loss formula and weight match MAGS-Net. The configuration, protocol,
and comparison CSV record the smoothing value; MAGS-Net's additional auxiliary localization
supervision remains explicitly labeled.

## Same dataset and saved folds

Use the existing `brainDataset_cropped/metadata_clean.csv`, `images/`, and `masks/`.
The reference experiment is selected by `MAGS_STUDY_DIR`, currently
`results/magsnet_nested_cv/20260916T210025Z_bc27ff71`. Keep that complete study and its
matching preparation directory `results/magsnet_nested_cv/preparation/04ddbd02826f`.
The latter supplies `outer_patients.csv`, `inner_patients.csv`, and `sample_index.csv`.
The MAGS-Net analysis must also contain `complexity.csv` for the complexity comparison.

The notebook verifies the source protocol, data fingerprint, exact saved patient
assignments, labels, image/mask alignment, and absence of patient leakage. **No new
folds are generated.** The cohort has 3,064 slices from 233 patients. Labels remain
Meningioma=0, Glioma=1, Pituitary=2.

Preprocessing is the same as MAGS-Net: existing cleaned/cropped grayscale PNGs, linear
image resize to 224x224, nearest-neighbor mask resize, image scaling by 255, and binary
masks. Training augmentation applies the same paired horizontal/vertical flips and
rotation from -10 to +10 degrees. Validation/test images have no augmentation.

## Training and independent duration selection

| Setting | Value |
| --- | --- |
| Seed | 42 |
| Batch size | 32 |
| Optimizer | Adam |
| Initial LR / weight decay | 0.001 / 1e-4 |
| Maximum inner epochs / patience | 50 / 10 |
| Scheduler | StepLR, step 10, gamma 0.5 |
| Segmentation weight | 0.5 |
| Dice loss smoothing | 1.0 (same as MAGS-Net) |
| Best inner checkpoint | Minimum validation total multitask loss |
| Outer duration | Median of the five inner best epochs |

Mixed precision and gradient clipping at norm 1 follow MAGS-Net. Its sampler policy
is retained: a final singleton joins the preceding batch, producing 33 images when
needed. No sample is dropped. Each inner fit and outer refit creates a fresh model,
optimizer, and scheduler. Ties retain the earlier best epoch and count toward patience.

For each outer fold, train five inner models, select the median best epoch, and refit
on the full outer-development set for that fixed duration. The outer model has no
validation/test loader during fitting. There are **25 inner fits + five outer refits**.
Neither MAGS-Net's selected durations nor outer-test performance selects MT-U-Net epochs.

## Running the notebook

1. Run all cells with the default switches for dataset/fold checks, a synthetic forward
   check, complexity profiling, and analysis of the existing MAGS-Net reference only.
2. To train, set `RUN_TRAINING=True`, `RUN_OUTER_TEST=False`, and `STUDY_DIR=None`, then
   run from the top. Record the printed MT-U-Net study directory.
3. After training completes, set `STUDY_DIR` to that path, disable training, enable
   `RUN_OUTER_TEST`, and run from the top. All five completed refits are checked first.
4. For subsequent analysis, retain the path and disable both switches. Saved OOF
   predictions are loaded. Even when outer testing is enabled again, matching cached
   predictions and masks are verified and reused without another inference pass.

A new training run with `STUDY_DIR=None` creates a separate study. An explicit existing
study reuses verified completed fits. Interrupted fits are detected and require an
explicit restart; optimizer-state resumption is not implemented. To restart only one
partial fit, retain other completed fits and move that incomplete fit directory aside
before rerunning with the same study path. Fix all settings before outer-test evaluation.

## Results

Outputs live under `results/mt_unet_nested_cv/`, separate from previous experiments.
Every fit saves histories, summaries, and weights. Each outer folder contains its
`inner_summary.csv`, `selection.json`, refit files, `outer_predictions.csv`,
`outer_metrics.csv`, evaluation provenance, and `predicted_masks/` PNG files.

The study's `analysis/` contains:

- `MT-UNet_oof.csv`: identifiers, fold, seed, true/predicted class, all three class
  probabilities, per-slice `dice`, `iou`, and GT `tumor_fraction`.
- `MT-UNet_patient_oof.csv`: averaged slice probabilities, patient classification,
  and mean 2D slice Dice/IoU per patient.
- `classification_comparison.csv`: slice/patient Accuracy, macro-F1, macro OVR AUC,
  balanced accuracy, and training objective for MT-U-Net and saved MAGS-Net.
- `segmentation_comparison.csv`: overall and per-class Dice/IoU means and standard
  deviations at slice and patient levels.
- `per_class_classification.csv`, `outer_fold_metrics.csv`, `epoch_selections.csv`,
  `oof_coverage.csv`, and `model_complexity.csv`.
- `model_parameters.csv`: total/trainable parameter counts, parameter count in millions,
  and FP32 parameter size. These values are also printed in section 8.1.
- Confusion-matrix, ROC, and per-class Dice-distribution figures as PNG/PDF once both
  models have complete results.

Patient classification averages probabilities before argmax. Patient segmentation
averages slice scores and is not 3D volumetric Dice. Overall segmentation averages
per-slice metrics, rather than pooling pixels. Each complete model must have 3,064
OOF slices, 233 unique patients, and no duplicate samples or multiple test folds per patient.

Complexity includes both heads and the full decoder for both models. MACs count Conv2d,
ConvTranspose2d, and Linear operations; bias additions, normalization, activations,
pooling, interpolation, and elementwise operations are excluded. FP32 size is parameter
count times four bytes divided by 2**20 MiB, excluding buffers and checkpoint overhead.

Before training, MT-U-Net performance is marked unavailable. Only measured complexity
and actual MAGS-Net reference results are exported under `preparation/<hash>/reference/`.
No synthetic performance is saved as an experiment result.

## Source and package verification

The percent-format Python source and corresponding build script are in this directory.
Rebuilding a notebook replaces its manual edits and outputs. The local development
verification harnesses and unrelated legacy experiments are not required by this release.
From the repository root, run `python scripts/check_release.py` and
`python scripts/smoke_test_demo.py` for the included package checks. Neither trains a model.
Restore the exact supplied MAGS-Net reference with `python scripts/restore_reference.py`
before running comparison notebooks. Dataset setup is documented in
[REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).
