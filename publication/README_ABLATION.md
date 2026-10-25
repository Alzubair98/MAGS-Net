# MAGS-Net scale-gate ablations

Open [`04_MAGSNet_Ablation_NestedCV.ipynb`](../04_MAGSNet_Ablation_NestedCV.ipynb) in
Conda **dl**. The notebook contains the model, fitting, evaluation, and analysis code,
with English explanations. Training and new outer-test inference are disabled by default.

| Variant | Main scale branches | Gate | Gate input |
| --- | --- | --- | --- |
| A0: Full MAGS-Net | 3x3 and 5x5 | Learned | Global + predicted-lesion-masked descriptors |
| A1: Fixed 50/50 | 3x3 and 5x5 | 0.5 / 0.5 | None |
| A2: 3x3 only | 3x3 | 1 / 0 | None |
| A3: 5x5 only | 5x5 | 0 / 1 | None |
| A4: No lesion guidance in scale gate | 3x3 and 5x5 | Learned | Global descriptor only |

A0 imports the existing completed MAGS-Net predictions from
`results/magsnet_nested_cv/20260916T210025Z_bc27ff71`. It is not retrained or reevaluated.
Only A1–A4 perform new fits, each initialized from scratch.

## A4 isolates the lesion-to-gate connection

A4 keeps the entire segmentation decoder, classification head, localization heads, and
original auxiliary supervision. It removes only the predicted-lesion-masked descriptor
from the **scale gate**. The classification head still uses its original predicted-mask
pooling; changing that would test a different hypothesis.

The first gate layer in A4 takes C global features instead of 2C concatenated features.
Its hidden width and two-scale output remain unchanged. This slightly reduces the
parameter count, which is reported explicitly; the variants are not claimed to have
identical parameter counts.

All variants first construct the same randomly initialized full architecture. A4 retains
the global-input columns and bias of that fresh first gate layer, removing the lesion
columns. Unchanged modules therefore start with identical values for the same seed.
This does not transfer trained A0 weights. A1–A3 remove the unused gate MLP, and A2/A3
also remove the unused main scale branch before optimizer creation.

Localization heads remain in all variants, including their original 3x3 convolutions.
Thus A3's "5x5 only" refers to its main scale branch. As in the original MAGS-Net,
auxiliary loss directly supervises the final localization head of each stage. Earlier
localizers have no loss path when lesion-to-gate guidance is disabled; no extra supervision
is added to change that behavior. `trainable_parameters` counts parameters with
`requires_grad=True`, which is distinct from counting parameters on an active loss path.

## Identical data, folds, losses, and general training protocol

The notebook reads `brainDataset_cropped/metadata_clean.csv`, MRI images, and masks.
It verifies their content against the MAGS-Net dataset fingerprint, then reads the exact
saved outer/inner patient assignments from the matching MAGS-Net preparation directory.
No new folds are generated. The current cohort contains 3,064 slices and 233 patients,
with Meningioma=0, Glioma=1, Pituitary=2.

Grayscale 224x224 preprocessing, normalization by 255, paired flips/rotation, mask resizing,
batching, and all three loss functions match MAGS-Net. GT masks never enter the model or
the gate descriptor.

```text
L_total = L_classification + 0.5 L_segmentation + 0.2 L_auxiliary
L_classification = CrossEntropy
L_segmentation = BCEWithLogits + mean per-slice soft Dice loss (smooth=1)
L_auxiliary = mean localization loss over the three final stage heads
```

Seed 42; Adam; LR 0.001; weight decay 1e-4; batch size 32; maximum 50 inner epochs;
patience 10; StepLR(step=10, gamma=0.5); mixed precision and gradient clipping at norm 1.
The original final-singleton batching policy is retained without dropping samples.

For each variant and outer fold, select five inner best epochs using the **minimum total
validation loss**, then take their median. Train a fresh model on the entire outer-development
set for that duration. The outer refit receives no validation or test loader. A0's selected
epochs are never used to fit another variant.

Four new variants mean **100 inner fits + 20 outer refits = 120 new fits**.

## Running later

1. Run with both switches False for preparation, synthetic forward checks, complexity,
   and the saved A0 reference. This performs no training or new outer-test inference.
2. To train, set `RUN_TRAINING=True`, `RUN_OUTER_TEST=False`, and `STUDY_DIR=None`, then
   run from the top. Save the printed ablation study path.
3. After all 120 fits finish, set `STUDY_DIR` to that path, disable training, enable
   `RUN_OUTER_TEST`, and run from the top. All 20 refits are verified before testing.
4. For later analysis, keep the study path and disable both switches. Matching cached
   predictions and masks are verified and reused without repeated inference.

An explicit existing study path reuses verified completed fits. Interrupted fits require
an explicit restart; optimizer-state resumption is not implemented. Preserve completed
fits and move the incomplete fit directory aside before restarting that fit. A new run
with `STUDY_DIR=None` creates a separate experiment. Fix variants/settings before testing.

## Outputs

New outputs live under `results/magsnet_ablation_nested_cv/`. Each variant/outer-fold folder
contains inner histories, best/last checkpoints, `inner_summary.csv`, `selection.json`,
outer refit files, `outer_predictions.csv`, `outer_metrics.csv`, evaluation provenance,
and predicted masks. Old studies and notebooks are not modified.

The study's `analysis/` contains:

- `A0_oof.csv` through `A4_oof.csv`, and corresponding patient OOF files.
- `classification_comparison.csv`: slice/patient Accuracy, macro-F1, balanced accuracy,
  and macro one-vs-rest AUC.
- `segmentation_comparison.csv`: overall and per-class Dice/IoU means and standard deviations.
- `per_class_classification.csv`, `outer_fold_metrics.csv`, and `epoch_selections.csv`.
- `gate_summary.csv`: stage-level gate means/std overall and by tumor class.
- `paired_slice_changes_vs_a0.csv`: matched-sample Dice/IoU and classification-correctness
  changes, with patient IDs retained for subsequent patient-aware statistical analysis.
- `oof_coverage.csv`, `ablation_plan.csv`, and `model_complexity.csv`.
- `paired_patient_statistics.csv`: A0-minus-ablation patient accuracy and Dice differences,
  exact McNemar raw/Holm-adjusted p-values, discordant-pair counts, and paired bootstrap CIs.
- `paired_patient_statistics_table.csv`: the formatted four-row table for the paper;
  `paired_patient_statistics_metadata.json` records the methods, settings, and source OOF hashes.
- Confusion, ROC, and per-class Dice figures in PNG/PDF once all five OOF sets are complete.

Each complete variant must have 3,064 unique OOF slices and 233 patients, each tested
in exactly one outer fold. Pending variants have no fabricated performance. Patient
classification averages probabilities before argmax; patient segmentation averages 2D
slice scores and is not 3D volumetric Dice. No significance test is implied by the paired
change CSV alone.

Section **24.1** adds inference from the completed, saved A0–A4 OOF CSVs without training
or test inference. It matches patient IDs, labels, folds, seeds, and slice counts before
comparing models. Differences in this table are **A0 minus the ablation**, unlike the older
`paired_slice_changes_vs_a0.csv`, which stores ablation-minus-A0 slice differences.
Patient accuracy uses mean slice probabilities before argmax. Dice first averages each
patient's slice scores, then averages equally across patients; this differs from the
pooled slice Dice statistic.

The exact two-sided McNemar test compares paired patient correctness; Holm adjusts the
four McNemar tests as one family. The paired patient bootstrap uses 10,000 draws with
seed 42, resampling within true tumor class and using the same patients across all
models. Percentile 95% CIs describe patient accuracy and Dice differences. These are
pointwise intervals, not multiplicity-adjusted or simultaneous intervals; Holm applies
only to McNemar p-values. Inference is conditional on the fitted OOF models and saved
folds and does not estimate retraining or CV-partition variability.

MACs count Conv2d/Linear operations for one 1x224x224 input and the complete multitask
forward pass. They exclude bias additions, normalization, activations, pooling,
interpolation, and elementwise operations. FP32 size is parameter count times four bytes,
divided by 2**20 MiB, excluding buffers and checkpoint overhead.

## Source and package verification

The percent-format Python source and corresponding build script are in this directory.
Rebuilding a notebook replaces its manual edits and outputs. The local development
verification harnesses and unrelated legacy experiments are not required by this release.
From the repository root, run `python scripts/check_release.py` and
`python scripts/smoke_test_demo.py` for the included package checks. Neither trains a model.
Restore the exact supplied MAGS-Net reference with `python scripts/restore_reference.py`
before running comparison notebooks. Dataset setup is documented in
[REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).
