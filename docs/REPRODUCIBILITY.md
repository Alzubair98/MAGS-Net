# Reproducing the seed-42 experiment

## What the saved reference contains

research_reference/results/magsnet_nested_cv/ contains the completed MAGS-Net study
20260916T210025Z_bc27ff71 and its matching preparation directory 04ddbd02826f.
It preserves original bytes for hashes, outer/inner patient assignments, sample
metadata, all five final outer checkpoints, refit and inner histories/selections,
OOF rows, and compact analysis CSVs. It contains no dataset images or masks.

Do not normalize line endings in these reference files: completed-study checks hash
their bytes. .gitattributes protects this folder from Git text conversion.

```bash
python scripts/restore_reference.py
```

Restoration is local, checks all packaged hashes and refuses to replace different
existing files. It does not assert that the archive contains every training artifact:
inner model checkpoints, predicted mask files and other architectures' runs are
not included. Do not treat it as a resumable complete training directory. The
provided subset supports the baseline reference loaders and independent OOF checks.

## Dataset

Obtain the original [Cheng dataset, version 8](https://doi.org/10.6084/m9.figshare.1512427.v8)
and place its 3064 MAT files in brainDataset/. Review the provider's terms.
The fixed public dataset IDs are used solely to reproduce grouping; identity and
clinical independence are not independently established by this repository.

The following script reconstructs the original processed image/mask pairs from
the saved image-derived crop coordinates. It verifies labels, dataset patient IDs
and decoded pixel hashes against the completed study before accepting each pair:

```bash
python scripts/prepare_cheng.py --raw-dir brainDataset --output-dir brainDataset_cropped
```

For checking without writing image files, add --verify-only. This reconstructs the
historical preprocessing, not a new GT-guided crop or fold split. The ground-truth
mask is cropped/resized with the saved image coordinates and nearest interpolation.

Expected layout:

```text
brainDataset_cropped/
  metadata_clean.csv
  images/1.png ...
  masks/1.png ...
```

## Notebook order

Use a separate research environment with publication/requirements.txt and the
appropriate PyTorch build. Open the notebooks from the repository root.

1. Inspect the MAGS-Net notebook. To run a new experiment, retain STUDY_DIR=None,
   enable RUN_TRAINING and keep RUN_OUTER_TEST=False. All new fits create a new study.
2. The supplied baseline notebooks read the restored original MAGS-Net folds/reference.
   Each architecture selects its own duration from its five inner validation folds.
3. After new fits finish, set the relevant study path, disable training and explicitly
   enable outer-test inference. Save its new predictions before running analyses.
4. BRISC requires separately obtained official test image/mask pairs and explicitly
   enabling RUN_INFERENCE. The no-tumor class is outside this model's class set.
   Source-overlap screening is included, not a guarantee of external independence.

The default false switches do not make an entire research notebook data-free:
indexing and integrity checks still require the datasets. Use the local demo or
scripts/smoke_test_demo.py for a dataset-free check.

The five packaged final refit durations are 35, 37, 34, 35 and 23 epochs, with seed 42.
No baseline inherits these durations. The archive's OOF coverage is 3064 slices,
233 patients, one outer-test assignment per patient and zero duplicated sample IDs.

## Scope of reproducibility

Model/data hashes verify supplied artifacts, not clinical generalization. Exact
floating-point outcomes may depend on framework, hardware and numeric kernels.
Bootstrap intervals are conditional on fitted models and partitions; patient
segmentation averages slice scores rather than computing 3D volumetric Dice.
Full independent re-training of all models was not run when preparing this package.
