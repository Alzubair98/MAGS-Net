# MAGS-Net evaluation on BRISC2025

Open `06_BRISC2025_External_Validation.ipynb` using Python (dl). Run all cells.
The notebook contains inference and analysis only. It never creates an optimizer,
computes training gradients, updates BatchNorm statistics, or fine-tunes weights.
All comments and explanations are in English.

## Fixed protocol

- Official test segmentation subset: 860 MRI/GT pairs, excluding no-tumor cases.
- Labels: Meningioma=0, Glioma=1, Pituitary=2.
- Equal-probability ensemble of all five completed MAGS-Net outer-refit models, seed 42.
- Source study: `results/magsnet_nested_cv/20260916T210025Z_bc27ff71`.
- Original image-derived brain contour crop (padding 5), 224×224 linear MRI resize,
  nearest-neighbor mask resize, grayscale [0,1] input. No inference augmentation.
- BRISC mask encoding is decoded with a fixed midpoint threshold; original binary
  masks from MAGS-Net use the original nonzero rule.
- Binary prediction threshold 0.5; individual fold models reported separately without selection.
- 2,000 class-stratified slice-bootstrap percentile resamples, seed 42, nominal 95% CIs.

The preprocessing audit reproduced all 3,064 original stored MRI/mask pairs exactly.
Both parameter tensors and BatchNorm buffers were unchanged after each inference pass.

## Interpretation is essential

[The BRISC paper](https://arxiv.org/html/2506.14318v5) states that BRISC derives from a
collection including the Cheng dataset and that patient identifiers are unavailable.
The local image screen flagged 742/860 slices at normalized correlation >=0.995 under
rotation/reflection, although none were exactly identical on the cropped pixel grid.
These are near-match candidates, not 742 verified patient matches. Four strongest raw
matches are visualized for review. JPEG conversion and recropping can defeat exact hashes.

These results must **not be presented as independent patient-level external validation**.
Even the slices below the similarity cutoff are not proven independent. No samples were
removed from the original all-test analysis based on correlation, label, prediction, or Dice. Bootstrap CIs are exploratory
slice-level intervals and cannot account for unknown within-patient dependence.

Five brain crops lost some annotated tumor area, including four with no GT tumor left in
the cropped field. They remain in all primary results. The original MAGS-Net metric assigns
Dice/IoU=1 when prediction and processed GT are both empty. Supplementary original-field
scores restore predictions to the original image and score against the complete GT so
tumor outside a crop is counted as missed. See `crop_retention_failures.csv`.

## Saved outputs

Completed run: `results/brisc2025_external_validation/test_seed42_537c9da9a6e4/`.

| File | Contents |
| --- | --- |
| `MAGSNet_BRISC_test_predictions.csv` | Ensemble label probabilities, predictions, Dice/IoU, class, plane |
| `individual_model_predictions.csv` | Predictions and scores from each of the five frozen models |
| `overall_metrics.csv`, `overall_bootstrap_ci.csv` | Overall accuracy, macro-F1, macro OvR AUC, mean Dice/IoU |
| `per_class_metrics_ci.csv` | Class precision/recall/F1, OvR AUC and segmentation with CIs |
| `per_plane_metrics_ci.csv`, `per_class_plane_metrics_ci.csv` | Plane and class-by-plane results |
| `original_fov_predictions.csv`, `original_fov_segmentation_ci.csv` | Supplementary full-field segmentation scores |
| `confusion_matrices.csv`, `roc_coordinates.csv` | Reusable figure data |
| `preprocessing_audit.csv`, `crop_retention_failures.csv` | Mask decoding and crop retention |
| `source_overlap_audit.csv`, `independence_audit.json` | Exact/near-image source-overlap screen and limitations |
| `checkpoint_provenance.csv`, `frozen_state_verification.csv` | Checkpoint identities and unchanged model states |
| `ensemble_masks.npz`, `ensemble_segmentation_probabilities.npy` | Binary masks and float32 pixel probabilities |
| `figures/*.png`, `figures/*.pdf` | Confusion/ROC, segmentation examples, distributions, overlap review |
| `protocol.json`, `environment.json`, `result_inventory.csv` | Reproducibility settings and output checksums |

Completed predictions are reused after checksum validation. `RUN_INFERENCE=False` requires
existing predictions; it never starts inference if results are absent. Original datasets,
training notebooks and checkpoints remain unchanged. The original MAGS-Net reference is bundled separately; new outputs are Git-ignored.

The readable source is `publication/brisc_external_validation.py`. Rebuilding is protected
against overwriting an executed notebook. The public release supplies package-level
checks under `scripts/`; original BRISC data and completed predictions are not bundled.
Enable RUN_INFERENCE explicitly to generate predictions after dataset preparation.

## Cohort curation and exploratory subset evaluation (sections 15–19)

The appended sections preserve every original all-test result. They add the requested
cohort flow, provenance review, saved subset definitions, and classification/segmentation
summaries with 95% confidence intervals by class and plane. They reuse frozen predictions;
no training or new inference is performed.

Run sections 0–2 to establish imports and paths, then jump to sections 15–19. The full
notebook can also be rerun normally. The current flow is:

| Stage | Slices |
| --- | ---: |
| All available BRISC tumor image/mask pairs | 4,793 |
| Official test retained for evaluation | 860 |
| Potential Cheng overlap excluded | 742 |
| Ambiguous processed-image duplicates held out | 12 |
| Source-disjoint candidates, unverified | 106 |
| Final clean external subset with documented provenance | 0 |

The 12 ambiguous records form six duplicate groups. Every member is held out; no copy
is chosen according to segmentation or classification performance. The 106 remaining
candidates are all Meningioma. Consequently, three-class macro-F1 and macro-AUC are NA;
candidate accuracy is conditional Meningioma recall. Plane and segmentation results can
still be described. Missing classes/planes and the empty verified cohort are explicitly
reported, never converted to zero metrics.

This curation was added after the aggregate evaluation, so it is post-hoc and exploratory.
The 0.995/0.980 similarity thresholds are heuristics, not validated identity thresholds.
The sensitivity CSV varies the lower cutoff using counts only. Patient independence
cannot be established from negative image-similarity findings.

`<study>/curation/provenance_review.csv` is the editable review template. Existing entries
are preserved. Document an actual source linkage and source/patient-disjointness evidence
with the reviewer and date before marking `source_disjoint`. The notebook checks complete
fields and conflicts, not the truth of external documents. Positive review entries cannot
override automatic overlap or ambiguity flags. Nothing is automatically promoted to clean.

Each review/configuration change creates a separate directory keyed to its input hashes;
`<study>/curation/latest_curation.json` points to the current one. Files include:

- `sample_curation_ledger.csv`: status and inclusion/exclusion reasons for all 860 test slices.
- `potential_cheng_overlap_samples.csv`, `ambiguous_samples.csv`,
  `source_disjoint_candidate_samples.csv`, `final_clean_external_subset.csv`: membership.
- `cohort_flow_counts.csv`, `cohort_flow.png`, `cohort_flow.pdf`: the requested flow diagram.
- `screening_threshold_sensitivity.csv`: count-only sensitivity analysis.
- `candidate_exploratory/`: predictions and overall/class/plane/class-by-plane metrics and CIs.
- `final_clean_external/`: the same schema; currently `not_evaluated` with NA scores.
- `provenance_review_snapshot.csv`, `review_conflicts.csv`, `curation_protocol.json`,
  `curation_complete.json`, `result_inventory.csv`: reproducible review and result records.

The empty final-clean cohort is a documented limitation, not an inference failure. A new
three-class external cohort with traceable independent provenance is needed to support
a three-class independent external-validation claim.

