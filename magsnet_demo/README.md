# Local MAGS-Net demonstration

**Research use only — under development.** This app must not be used for clinical
diagnosis or treatment decisions. It cannot verify that an upload is a brain MRI
or establish whether a tumor is present. Tumor-free scans and unrelated images
may receive a tumor label and a predicted mask, even with a high model probability.
These limitations appear before upload, remain visible with results, and are
included in JSON summaries. The notice does not automatically reject such images.

From the repository root, install the demo requirements and run:

```bash
python -m streamlit run magsnet_demo/app.py
```

The root .streamlit/config.toml binds the application to 127.0.0.1, limits uploads
to 10 MiB, disables telemetry and hides detailed exceptions. The app is local only.
On Windows, run the root run_demo.ps1 with PowerShell if you prefer a launcher.

Upload an original 8-bit PNG/JPEG MRI export, not an already processed model crop.
Click Analyze MRI to run the five frozen seed-42 outer-refit models. The app shows
class probabilities, a binary mask, a restored overlay, a probability map and
stage gate summaries. The three downloads are created in memory. Clearing the
session removes the uploaded image and its results; model resources remain cached.

## Reproducible inference

The architecture is read from publication/magsnet_nested_cv.py using the original
audited AST adapter. Only the seven named architecture definitions are compiled;
dataset, training and evaluation cells are never executed by this adapter.

Original checkpoint files are included under research_reference/results/.
checkpoint_manifest.json binds their SHA-256 hashes, seed, selected duration,
experiment fingerprint, class mapping and expected parameter count. All five
models are required and loaded strictly with weights_only=True. Failed integrity
checks stop inference; random weights are never substituted.

Preprocessing is image-only: OpenCV grayscale, float32 min-max conversion to uint8,
Gaussian 5x5/Otsu/erosion/dilation to identify the largest contour, five-pixel crop
padding, linear resize to 224x224 and division by 255. The normalized unblurred
image is cropped. No ground-truth mask, class label, augmentation, ImageNet
normalization, or test-time tuning is used.

Class probabilities are the equal mean of five softmax vectors. Segmentation
probabilities are the equal mean of five sigmoid maps, thresholded at 0.5.
Gate values average channels, blocks within each stage, then models. Predicted
lesion area is a fraction of the 224x224 grid, not volume or physical size.

For a different verified storage location, MAGSNET_STUDY_DIR points to the study
root or MAGSNET_WEIGHTS_DIR points to outer_1.pt through outer_5.pt with the same
expected file hashes. MAGSNET_PROJECT_ROOT changes the source repository root.

## Limits

There is no tumor-free class, DICOM support or clinical calibration. Contour cropping
can exclude lesions; black padding outside the crop means not analyzed. A dataset
image evaluated by this five-model ensemble is not necessarily held out from the
ensemble's training. Use the OOF notebooks for the study's held-out evaluation.

To benchmark an image you supply locally:

```bash
python -m magsnet_demo.benchmark --image path/to/image.png --device cpu
```

The benchmark does not reproduce a published latency without matching its hardware,
input and timing protocol. Do not commit personal uploads or generated reports.
