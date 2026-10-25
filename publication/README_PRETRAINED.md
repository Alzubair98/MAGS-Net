# ImageNet-pretrained classification baselines

Open [`05_Pretrained_Classification_NestedCV.ipynb`](../05_Pretrained_Classification_NestedCV.ipynb)
in Conda **dl**. The notebook is self-contained, with English notes. Training and new
outer-test inference are disabled by default.

| Classifier | Pinned TorchVision weights |
| --- | --- |
| ResNet-50 | `ResNet50_Weights.IMAGENET1K_V2` |
| EfficientNet-B0 | `EfficientNet_B0_Weights.IMAGENET1K_V1` |
| MobileNetV3-Large | `MobileNet_V3_Large_Weights.IMAGENET1K_V2` |

Every independent fit reloads the same ImageNet weights, creates a fresh three-class
output layer, and fine-tunes **all** layers. There is no frozen-backbone phase or scratch
fallback. The first fit downloads official weights if they are missing from Torch's cache.
Evaluation reconstructs the architecture without pretrained weights, then loads the
complete trained checkpoint; it does not need another ImageNet download.

## Identical MRI protocol

The notebook reads the exact saved MAGS-Net outer **and inner** patient folds. It verifies
the same master dataset and image/mask fingerprint; masks are used only in that provenance
audit and never enter classifier datasets, forward passes, losses, or classification metrics.
No new split is generated. Labels remain Meningioma=0, Glioma=1, Pituitary=2.

Preprocessing and augmentation match the scratch experiment: cleaned/cropped grayscale
MRI, 224×224 bilinear resize, division by 255, training flips, and rotation within ±10°.
To retain one input channel, the pretrained RGB stem kernels are **summed** across channels.
This preserves the stem response to the same grayscale tensor repeated across RGB channels.
Other backbone weights are retained; only the last output layer is newly initialized.

The experiment deliberately retains the MRI normalization rather than the weight enum's
RGB ImageNet normalization/resize/crop transforms. It compares initialization under the
fixed MRI protocol, not independently tuned best transfer-learning recipes.

Seed 42; batch size 32; Adam with LR 0.001 and weight decay 1e-4; StepLR(10, 0.5);
maximum 50 epochs; patience 10. Loss and validation selection use **Cross-Entropy only**.
Each architecture selects its own five inner best epochs per outer fold, takes their
median, then fits a fresh pretrained model on the complete outer-development set. The
outer test is evaluated once and its cached predictions are verified on later runs.
Three models require **75 inner fits + 15 outer refits = 90 fits**.

## Run and evaluate

1. Run All with the default switches for preparation, synthetic shape checks, and the
   existing MAGS-Net reference. This downloads no weights and trains no model.
2. Set `RUN_TRAINING=True`, `RUN_OUTER_TEST=False`, `PRETRAINED_STUDY_DIR=None`; Run All.
3. Retain the printed study path. After training, set `PRETRAINED_STUDY_DIR` to it,
   `RUN_TRAINING=False`, `RUN_OUTER_TEST=True`; Run All for outer evaluation.
4. For later analysis, retain the path and set both switches False.

Results are separate under `results/pretrained_classification_nested_cv/<study>/analysis/`:

- `<model>_oof.csv`: sample/patient IDs, outer fold, true/predicted labels, three probabilities, seed.
- `<model>_patient_oof.csv`: mean probabilities per patient, then argmax.
- `<model>_classification_metrics.csv`: slice and patient Accuracy, macro-F1, balanced accuracy,
  and macro one-vs-rest AUC.
- Slice/patient confusion matrices in CSV and combined figures in PNG/PDF.
- `classification_comparison.csv`, `oof_coverage.csv`, `epoch_selections.csv`, and method metadata.
- `paired_patient_statistics.csv`: numeric differences in patient Accuracy and macro-F1,
  their paired bootstrap CIs, exact McNemar raw/Holm-adjusted p-values, and correctness counts.
- `paired_patient_statistics_table.csv`: formatted paper table;
  `paired_patient_statistics_metadata.json`: methods, analysis settings, and input OOF hashes.

Every completed classifier must cover exactly 3,064 slices and 233 patients with no
duplicate samples and one outer-test fold per patient. Pending models have no fabricated
performance. The saved MAGS-Net reference is labeled scratch/multitask. No Dice/IoU is
calculated for these classification-only models.

## Paired patient statistics

Section **18.1** reads saved OOF predictions and matches patients against the master index
and saved outer folds before comparing MAGS-Net with each of the three pretrained models.
It uses patient classifications obtained by averaging slice probabilities before argmax.
All differences are **MAGS-Net minus the pretrained model**.

The two-sided exact McNemar test uses paired patient correctness. Holm adjusts the three
comparisons as one family; this does not adjust every test in the paper. There are 10,000
paired patient-bootstrap draws with seed 42, stratified by true tumor class, using the
same sampled patients across all models and metrics. Accuracy and macro-F1 are recomputed
in each draw before taking differences. Macro-F1 averages class F1 over labels 0, 1, 2
with `zero_division=0`; it is not an average of individual patient F1 scores.

Accuracy differences/CIs use percentage points; macro-F1 differences/CIs use the 0–1 scale.
Percentile 95% intervals are pointwise and not Holm-adjusted. They are conditional on
the fitted OOF models and fixed folds and do not include retraining/partition variability.
No training or test inference is needed to run this section after the OOF export cells.
The comparison cells require the corresponding saved OOF predictions.

## Source and package verification

The percent-format Python source and corresponding build script are in this directory.
Rebuilding a notebook replaces its manual edits and outputs. The local development
verification harnesses and unrelated legacy experiments are not required by this release.
From the repository root, run `python scripts/check_release.py` and
`python scripts/smoke_test_demo.py` for the included package checks. Neither trains a model.
Restore the exact supplied MAGS-Net reference with `python scripts/restore_reference.py`
before running comparison notebooks. Dataset setup is documented in
[REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).
