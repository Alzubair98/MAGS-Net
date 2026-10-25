# SARTAJ–BRISC–Cheng image-source audit

This is an image-provenance audit, not model training, classification evaluation or
segmentation evaluation. Original MRI files, masks and MAGS-Net weights are never written.
The scripts run in the existing conda `dl` environment. No network is downloaded or used;
Torch accelerates low-resolution matrix multiplication only.

## Completed scope

- Inventory: 3,264 SARTAJ images (2,764 tumor, 500 no-tumor), all 4,793 available BRISC
  tumor image/mask pairs, and 3,064 original Cheng MAT files.
- Primary BRISC–SARTAJ search: all 106 prior candidates and 12 ambiguous duplicate records.
- SARTAJ–Cheng search: all 2,764 SARTAJ tumor images, across class labels.
- BRISC–Cheng: fresh candidate retrieval for the 118 unresolved images; high-precision
  verification of saved nearest-source IDs for the other 742 images.
- Additional direct BRISC–SARTAJ checks: source links supplied by the two Cheng audits.
  Remaining Cheng-only BRISC rows are not asserted to be absent from SARTAJ.

The original BRISC train split is inventoried but not searched as a query cohort. The
500 SARTAJ no-tumor images participate in BRISC candidate retrieval, but the separate
SARTAJ–Cheng total refers to tumor images only. No labels are used to restrict retrieval.

## Reproduce

Run from the repository root. The audit requires separately obtained datasets and completed BRISC curation outputs. Dataset roots are discovered by
directory/schema structure, and ambiguous discovery fails instead of selecting silently.
Use a **new output directory for changed data or algorithms**. Cached pair scores are
reused within a completed run. Keep dataset and algorithm versions fixed when reusing caches.

```powershell
conda activate dl
python publication/source_provenance_audit.py --stage all --out results/source_provenance_audit/NEW_RUN --workers 6
python publication/summarize_source_audit.py --out results/source_provenance_audit/NEW_RUN --workers 6
```

The completed local run is `results/source_provenance_audit/20260926_sartaj_brisc_cheng`.
Outputs include the requested inventories, two pair audits, complete BRISC provenance,
deduplicated external **candidates**, comparison figures, and `source_audit_summary.txt`
in Arabic. Full scored candidate pairs, calibration results, transformations and matrices
remain available for inspection. The existing BRISC notebook/results were not changed.

## Matching method and decision limits

Images are decoded to grayscale and normalized using their own 1st/99th percentiles.
We retain a full-field representation and a comparison-only content representation after
constant-border handling and an MRI-derived contour crop. No GT mask determines matching.
Source arrays remain unchanged. Comparison views are 256×256; retrieval uses normalized
64×64 vectors under eight rotation/reflection transforms and both views. The six best
reference IDs per query undergo foreground NCC, SSIM, normalized MAE, 63-bit AC pHash
distance and edge-correlation checks. Two best orientation/view hypotheses are compared
at high resolution after all eight orientations are screened at retrieval resolution.

Bounded affine ECC handles small geometric shifts. Difficult matches receive SIFT/RANSAC
initialization, requiring at least 12 inliers, 70% inlier fraction, distributed keypoints,
0.8–1.2 affine scales, near-right-angle rotation and at least 85% valid field coverage.
There is no nonrigid registration. Positive linear intensity adjustment is recorded.
References: [original SSIM authors](https://ece.uwaterloo.ca/~z70wang/research/ssim/),
[OpenCV ECC documentation](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html).

Confirmation requires exact file/decoded-pixel equality OR all of: NCC≥0.995,
SSIM≥0.980, edge NCC≥0.980, MAE≤0.020, pHash≤4/63, valid field≥85% and sufficient
nonbackground support. These precision-first boundaries were checked against the observed
score distribution and 80 known modified positives plus 80 nearest different-patient
Cheng negatives. The negative-control maxima were NCC≈0.793 and SSIM≈0.651. All 80
negative controls were unmatched. All 80 modified positives were detected: 5 confirmed,
75 probable. We intentionally retain lower-quality true copies as probable rather than
relaxing confirmation to increase the count. This limited control experiment does not
establish a population error rate or guarantee retrieval sensitivity.

All 118 focus cases are confirmed SARTAJ matches. After preferring exact-file/decoded
matches among tied source copies, the summary records both types of exact evidence.
There are also 17 confirmed SARTAJ–Cheng label conflicts and one BRISC–SARTAJ conflict;
these are retained as identity matches, exported for manual review, and visualized.
No class is corrected automatically.

## Interpret the results correctly

- The BRISC provenance table is mutually exclusive. `SARTAJ_MATCH` means confirmed
  SARTAJ identity without a **confirmed** Cheng identity; nine such rows still have a
  probable Cheng match and are excluded from clean candidates.
- `NO_MATCH` means no convincing image match was detected in this search. It never
  proves absence of another slice of the same patient, nor complete source independence.
- Shared images establish image identity, not which dataset copied from which one.
- A candidate needs a confirmed SARTAJ match, no detected direct Cheng match, no detected
  Cheng match for any confirmed SARTAJ alias, a valid BRISC mask, an allowed class and
  agreeing labels. Probable/ambiguous/unaudited Cheng links prevent inclusion.
- All 118 eligible image records are preserved before deduplication. Six repeated copies
  are omitted from the 112-row canonical candidate file using lexicographic sample ID,
  never model outcomes or annotation quality. The duplicate-group table records this.
- These 112 candidates contain only Meningioma: axial 68, coronal 28, sagittal 16.
  They support an exploratory source-screened segmentation/within-class assessment,
  not independent three-class validation or verified patient-level independence.

The old provenance-review template remains untouched. These automated findings do not
fill a human reviewer signature or assert patient-disjointness evidence that is unavailable.
