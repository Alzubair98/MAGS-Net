# MAGS-Net: nested patient-level cross-validation

Open [`MAGSNet-Nested-5x5-CV.ipynb`](../MAGSNet-Nested-5x5-CV.ipynb).
The standalone notebook contains the complete model and follows sections 0–28, from
configuration and paired image/mask checks to nested CV, OOF analysis, and saved results.
It uses short functions and English notes, without importing the earlier experiment package.

For the five scratch classification baselines using the same saved patient folds, open
[`02_Baseline_Classification_NestedCV.ipynb`](../02_Baseline_Classification_NestedCV.ipynb)
and follow [the baseline instructions](README_BASELINES.md). The baseline notebook reads
the existing MAGS-Net predictions as its reference and does not retrain MAGS-Net.

For the standard U-Net with joint classification and segmentation, see
[`03_Lightweight_MT_UNet_NestedCV.ipynb`](../03_Lightweight_MT_UNet_NestedCV.ipynb)
and [the MT-U-Net instructions](README_MT_UNET.md). It uses the same saved patient folds
and selects its own durations by total multitask validation loss.

For A0–A4 scale-gate ablations, open
[`04_MAGSNet_Ablation_NestedCV.ipynb`](../04_MAGSNet_Ablation_NestedCV.ipynb) and follow
[the ablation instructions](README_ABLATION.md). A0 reuses the completed full model;
A1–A4 use the same saved folds and preserve segmentation/localization supervision.

For ImageNet-pretrained ResNet-50, EfficientNet-B0, and MobileNetV3-Large, open
[`05_Pretrained_Classification_NestedCV.ipynb`](../05_Pretrained_Classification_NestedCV.ipynb)
and follow [the pretrained instructions](README_PRETRAINED.md). It retains the same MRI
preprocessing and saved outer/inner folds, with architecture-specific duration selection.
Section 22.1 of the MT-U-Net notebook compares saved MAGS-Net/MT-U-Net OOF predictions
using exact McNemar and paired patient-bootstrap intervals, without retraining.

## Data

Place the notebook beside this directory:

```text
brainDataset_cropped/
  metadata_clean.csv
  images/<filename>.png
  masks/<filename>.png
```

The existing metadata contains `sample_id`, `patient_id`, `filename`, and zero-based `label`:
**0 = Meningioma, 1 = Glioma, 2 = Pituitary**. The current dataset contains 3,064 paired slices
from 233 patients. Class slice counts are 708, 1,426, and 930, respectively.

All image-mask pairs are required. Images are already cropped; the notebook resizes/normalizes
them and applies joint augmentation to training pairs only. It does not perform another crop
or denoising pass. Tumor area refers to the processed image, not physical tumor volume.

When `brainDataset/*.mat` is available, the provenance audit also checks every source label,
patient ID, and crop/resize-aligned GT mask. This uses the original crop-coordinate columns
in the supplied metadata. Raw files are optional for users who only have the processed data.

## Protocol

- Outer 5-fold and inner 5-fold stratification use a table with one row per patient. All slices
  of a patient remain together. Outer test patients never enter that outer fold's inner CV.
- The patient partitions are fixed with split seed 42. Training uses **seed 42 only**.
- The default has one predefined learned-gate candidate: Adam LR 0.001, weight decay 1e-4,
  segmentation weight 0.5, auxiliary weight 0.2, batch size 32, and StepLR (10, 0.5).
- Each inner fit runs at most 50 epochs with patience 10. Its best checkpoint minimizes
  validation total multitask loss. Ties count as no improvement.
- Select a candidate by the mean of its five best inner-validation losses. With the one
  predefined candidate, inner CV selects training duration: the median of its five best epochs.
- Refit a fresh model on every outer-training patient for this fixed duration. Its final
  checkpoint is used for outer testing; the outer test set never selects a checkpoint.
- This means **25 inner fits + 5 outer refits = 30 fits** for seed 42.

This distinction between selection and held-out evaluation follows the
[scikit-learn nested-CV example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
All architecture/candidate/analysis decisions should be fixed before outer-test evaluation.
Nested partitioning does not erase prior model-development use of this same cohort; report
the previous experiments when describing the study design.

## Execution

1. Select the Conda `dl` kernel and run all cells with the default `False` switches. This
   indexes/audits the data, saves fold definitions, and counts model parameters without training.
2. To start training later, set `RUN_TRAINING = True`, leave `RUN_OUTER_TEST = False`, and
   run from the top. Save the printed study path.
3. After all refits finish, set `STUDY_DIR` to that path, set `RUN_TRAINING = False`, and
   enable `RUN_OUTER_TEST`. Run from the top to evaluate and analyze all outer folds/seeds.
4. To recompute analysis from saved predictions, disable both switches and retain `STUDY_DIR`.

Each new training execution with `STUDY_DIR=None` creates a separate timestamped directory.
Selecting an existing study reuses verified completed fits. Partial fits are detected and
must be restarted explicitly; automatic checkpoint resumption is not implemented.
Final test evaluation can be rerun explicitly; analysis alone reuses saved OOF predictions.

## Outputs and interpretation

Outputs live under `results/magsnet_nested_cv/`. Preparation includes sample metadata, patient
fold definitions, class counts, source verification, protocol, and data fingerprints.
Training saves inner best/last checkpoints, outer final checkpoints, histories, and selections.
Evaluation saves one OOF record per slice per seed and binary predicted masks.

The final analyses include per-class and pooled classification/segmentation metrics, patient
probability aggregation, tumor-size/quality summaries, patient-cluster
bootstrap intervals, exploratory Spearman correlations, gate reports, partial Conv/Linear MACs,
optional inference timing, and six PNG/PDF figure pairs.
Across-seed SD is not reported for this single-seed experiment.

### Fold summary and coverage audit

The subsection **Fold Summary and OOF Coverage Audit**, immediately after section 20,
prints every outer fold's development/test patient and slice counts, test class distribution,
patient overlap, and all five inner train/validation counts and best epochs. It checks the
median inner best epoch against the selected duration and actual outer-refit history.

The final coverage report uses saved OOF records to verify that every patient is tested
exactly once and every slice has one OOF prediction per seed. Patient/fold assignments are
counted after collapsing slices within each patient/fold; normal repeated patient IDs across
that patient's slices are not duplicates. Unavailable OOF results are explicitly marked as
unavailable, rather than reported as zero errors.

The export cell saves the following under `<study>/analysis/fold_summary/`:

- `fold_summary.csv`: one row per outer fold, including the inner best-epoch list and median.
- `inner_fold_summary.csv`: one row per inner fold, including its best validation loss/epoch.
- `outer_test_class_distribution.csv`: patient and slice counts per test class/fold.
- `oof_coverage_summary.csv`: total counts, test coverage, missing slices, and duplicates.
- `fold_summary.txt`: the complete readable report shown in the notebook.

Re-running only these cells reads saved files without training or inference. To run the entire
notebook purely for saved-result analysis, set both `RUN_TRAINING` and `RUN_OUTER_TEST` to `False`
and retain the completed `STUDY_DIR`.

Bootstrap intervals are conditional on the fitted models and fixed folds. Patient segmentation
averages 2D slice scores and is not 3D Dice. Seeds/folds are not treated as extra independent
patients. Pooled OOF results remain separate per seed; no seed is selected by test performance.

## Source and package verification

The percent-format Python source and corresponding build script are in this directory.
Rebuilding a notebook replaces its manual edits and outputs. The local development
verification harnesses and unrelated legacy experiments are not required by this release.
From the repository root, run `python scripts/check_release.py` and
`python scripts/smoke_test_demo.py` for the included package checks. Neither trains a model.
Restore the exact supplied MAGS-Net reference with `python scripts/restore_reference.py`
before running comparison notebooks. Dataset setup is documented in
[REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).
