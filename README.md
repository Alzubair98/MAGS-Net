# MAGS-Net

**A Lightweight Adaptive Multi-Scale Gated Network for Joint Brain Tumor Classification and Segmentation in Contrast-Enhanced MRI**

Alzubair Algaraghuli and Serkan Ozturk, Computer Engineering, Erciyes University.

Research implementation, reproducibility notebooks, saved MAGS-Net reference artifacts,
and a Streamlit demonstration. The manuscript is in preparation; this repository
does not claim journal acceptance or clinical validation.

## Live demo

**[Open the MAGS-Net live demo](https://mags-net-demo.streamlit.app/)**

Try the application in your browser on Streamlit Community Cloud. Upload a
de-identified contrast-enhanced T1-weighted brain MRI image and click **Analyze MRI**
to view tumor-class probabilities, a predicted segmentation mask, and an overlay.

**Research use only — under development.** The app does not verify that an upload
is a brain MRI or determine whether a tumor is present. Tumor-free and unrelated
images may produce misleading predictions, even with high model probabilities.
It is not intended for clinical diagnosis or treatment decisions.

## Model and evaluation

MAGS-Net jointly predicts three tumor classes and a 2D lesion mask. Its adaptive gates
combine 3x3 and 5x5 branches using global and predicted-lesion descriptors. The class
order is **0: Meningioma, 1: Glioma, 2: Pituitary**. There is no tumor-free class.

- Input: one grayscale 224x224 image.
- Per model: **688,270 parameters**, **0.788 G convolution/linear MACs**, **2.626 MiB FP32 parameter storage**.
- Main experiment: **seed 42**, 5 outer patient folds with 5 inner folds per outer fold.
- Inner selection: minimum validation total multitask loss; median best inner epoch
  determines the duration of a fresh outer-development refit.
- Each original OOF patient is evaluated only with its held-out outer model.
- The demo instead averages the five frozen models' probabilities. This is
  a demonstration ensemble, not the OOF evaluation procedure.

BRISC is reported as a cross-dataset assessment with source-overlap concerns. Its
results do not establish independent external patient-level validation.

## Run the local demo

Use Python 3.11 in a clean environment. The root requirements.txt targets the
Linux/Python 3.11 hosted deployment; for a local CPU installation, use:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install streamlit==1.64.0 numpy==2.2.6 pillow==11.1.0 opencv-python-headless==4.12.0.88
python -m streamlit run magsnet_demo/app.py --server.address 127.0.0.1
```

Run from the repository root and open the localhost URL printed by Streamlit.
The five required checkpoints are included. No dataset download, credentials,
external prediction service, training, or manual machine-path editing is needed
for the demo. See [local demo details](magsnet_demo/README.md).

## Research notebooks

| Notebook                                                              | Purpose                                                                        |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| [MAGS-Net](MAGSNet-Nested-5x5-CV.ipynb)                               | Complete joint model, nested CV, OOF analysis and figures                      |
| [Scratch classifiers](02_Baseline_Classification_NestedCV.ipynb)      | ResNet-50, DenseNet-121, EfficientNet-B0, MobileNetV3-Large and RegNet-Y-400MF |
| [MT-U-Net](03_Lightweight_MT_UNet_NestedCV.ipynb)                     | Task-matched classification/segmentation baseline and paired statistics        |
| [Ablations](04_MAGSNet_Ablation_NestedCV.ipynb)                       | Full, fixed 50/50, 3x3, 5x5, and no lesion guidance                            |
| [Pretrained classifiers](05_Pretrained_Classification_NestedCV.ipynb) | Three ImageNet-initialized classifiers using the same folds                    |
| [BRISC assessment](06_BRISC2025_External_Validation.ipynb)            | Frozen ensemble inference, provenance and crop-retention analyses              |

All notebook outputs are cleared and training/new test-inference switches are off.
Readable percent-format sources and notebook builders are in [publication](publication).
Run notebooks from the repository root. Research dependencies are separate from the
minimal demo dependencies: [publication/requirements.txt](publication/requirements.txt).
Install OpenCV or OpenCV-headless in a given environment, not both distributions.

For the completed study's baseline reference, first run:

```bash
python scripts/restore_reference.py
```

This restores the supplied reference files into the expected local results layout,
without overwriting different files or training. The published sample IDs and
patient IDs are the original public Cheng dataset identifiers used for grouping.
Saved folds, original checkpoint hashes and held-out prediction rows are preserved.
See [reproducibility and dataset setup](docs/REPRODUCIBILITY.md).

## Check the package

```bash
python scripts/check_release.py
python scripts/smoke_test_demo.py
```

The first command checks notebook defaults, code syntax, hashes, patient separation
and OOF coverage. The second uses a synthetic image to check frozen CPU inference,
exports and the Streamlit interaction flow; synthetic output is not a scientific
performance result. Neither command trains a model.

## Included and separate materials

Included: six notebooks, their source, the local demo, five original MAGS-Net outer
refit checkpoints, fixed fold definitions, reference OOF tables and supporting CSVs.
Original checkpoints include optimizer/scheduler state to preserve existing SHA-256
provenance; the demo loads only the model state for inference.

Dataset images/masks, other architectures' checkpoints, manuscript drafts, temporary
experiments, local environments and uploaded images are not distributed here.
Obtain datasets from their original providers under their own terms. Additional
model comparisons can be reproduced by running their notebooks with the same folds.

## Citation and licensing

Use [CITATION.cff](CITATION.cff) for authors and title. No publication DOI is assigned
in this package. The authors' code, documentation and included MAGS-Net checkpoints
are released under the [MIT License](LICENSE). Dataset-derived materials and
third-party dependencies remain subject to their original providers' terms.
The standard license text follows the [Open Source Initiative MIT license](https://opensource.org/license/mit).

## Research use

**Research use only — under development.** This app is not a medical device and
must not be used for clinical diagnosis or treatment decisions. It does not verify
that uploads are brain MRI scans and cannot determine whether a tumor is present.
Tumor-free scans and unrelated images may receive a tumor label and a predicted
mask, even with a high model probability. Use only de-identified research images
with a known tumor from one of the three supported classes.

The notice appears before upload and remains visible with results; JSON summaries
also record these limitations. The notice does not automatically reject unsupported
images. Softmax probabilities are not clinical confidence. Cropping can remove
lesion tissue, and acquisition quality is not verified.
The application processes uploads and creates downloads in session memory; it
does not implement an uploaded-image history or call an external inference API.
In the hosted demo, processing takes place on the Streamlit server. Running the
local demo keeps inference on your own machine.
