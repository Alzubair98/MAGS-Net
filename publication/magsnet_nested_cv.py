# %% [markdown]
# # MAGS-Net: Nested Patient-Level Cross-Validation
# A readable, self-contained workflow for joint tumor classification and segmentation.
# All inputs come from `brainDataset_cropped`; images and binary masks stay paired.
# Run the notebook from top to bottom with the Conda `dl` kernel.
# Training is disabled by default. No previous notebook or checkpoint is required.

# %% [markdown]
# ## 0. Experiment Configuration
# Training uses seed 42 only, with fixed outer and inner splits. Each candidate is selected
# using inner-validation total multitask loss only. With the single predefined candidate,
# the inner loop selects training duration. Add candidates **before** running any outer tests.
# The experiment comprises 25 inner fits and 5 outer refits (30 fits in total).

# %%
CONFIG = {
    "protocol": "magsnet_nested_patient_cv_v1",
    "seeds": [42],
    "split_seed": 42,
    "outer_folds": 5,
    "inner_folds": 5,
    "image_size": 224,
    "batch_size": 32,
    "max_epochs": 50,
    "patience": 10,
    "scheduler_step": 10,
    "scheduler_gamma": 0.5,
    "mask_threshold": 0.5,
    "bootstrap_samples": 2000,
    "use_amp": True,
}
CANDIDATES = [
    {"learning_rate": 0.001, "weight_decay": 1e-4, "lambda_seg": 0.5, "lambda_aux": 0.2},
]
CLASS_NAMES = {0: "Meningioma", 1: "Glioma", 2: "Pituitary"}
RUN_TRAINING = False
RUN_OUTER_TEST = False
RUN_BENCHMARK = False
STUDY_DIR = None  # Set your own completed study path for analysis.

# %% [markdown]
# ## 1. Imports & Environment Setup
# Select the Conda `dl` kernel. No packages or model weights are downloaded here.

# %%
import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from IPython.display import display
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENVIRONMENT = {"python": platform.python_version(), "torch": str(torch.__version__),
               "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
               "sklearn": sklearn.__version__, "opencv": cv2.__version__, "device": str(DEVICE),
               "gpu": torch.cuda.get_device_name() if DEVICE.type == "cuda" else None}
print("Python environment:", Path(sys.prefix).name)
print("Device:", DEVICE, "| PyTorch:", torch.__version__)

# %% [markdown]
# ## 2. Reproducibility & Random Seeds
# Seeds affect initialization, shuffling, and augmentation. They do not move a patient
# between folds. Exact numerical reproducibility across hardware/software is not guaranteed.

# %%
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cv2.setNumThreads(0)


set_seed(CONFIG["split_seed"])

# %% [markdown]
# ## 3. Paths & Dataset Configuration
# Expected layout: `brainDataset_cropped/{images,masks,metadata_clean.csv}`.
# Metadata paths from another computer are ignored; paired paths are built from filenames.
# `brainDataset/*.mat`, when present, is used only to verify label/patient/mask provenance.

# %%
PROJECT_ROOT = next((p for p in (Path.cwd(), Path.cwd() / "ZubNet-B", Path.cwd().parent)
                     if (p / "brainDataset_cropped").is_dir()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Place the notebook beside brainDataset_cropped, or update PROJECT_ROOT.")
DATA_ROOT = PROJECT_ROOT / "brainDataset_cropped"
RESULT_ROOT = PROJECT_ROOT / "results" / "magsnet_nested_cv"


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# %% [markdown]
# ## 4. Dataset Indexing
# Labels are already zero-based: **0=Meningioma, 1=Glioma, 2=Pituitary**.
# Do not subtract one again. A missing file raises an error instead of silently removing a sample.

# %%
def index_dataset(root):
    metadata = root / "metadata_clean.csv"
    if not metadata.exists():
        metadata = root / "metadata.csv"
    frame = pd.read_csv(metadata, dtype={"sample_id": str, "patient_id": str, "filename": str})
    required = ["sample_id", "patient_id", "filename", "label"]
    if frame[required].isna().any().any() or frame["sample_id"].duplicated().any():
        raise ValueError("Missing metadata or duplicate sample IDs.")
    if not frame.label.isin(CLASS_NAMES).all() or set(frame.label) != set(CLASS_NAMES):
        raise ValueError("Expected all three labels: 0, 1, 2.")
    if frame.filename.duplicated().any() or (frame.groupby("patient_id").label.nunique() != 1).any():
        raise ValueError("Duplicate filenames or inconsistent patient labels.")
    frame["label"] = frame.label.astype(int)
    for name in frame.filename:
        if Path(name).name != name or not all((root / folder / name).is_file() for folder in ("images", "masks")):
            raise FileNotFoundError(f"Missing/invalid image-mask pair: {name}")
    return frame.sort_values("sample_id").reset_index(drop=True)


INDEX = index_dataset(DATA_ROOT)
print(f"Indexed {len(INDEX):,} paired images and masks; {INDEX.patient_id.nunique()} patients.")

# %% [markdown]
# ## 5. Dataset Integrity Checks
# Check every pair: readable grayscale images, matching shapes, binary nonempty masks,
# consistent patient labels, and exact-image duplicates across patients. Duplicate slices
# within one patient cannot cross folds. SHA-256 fingerprints bind results to the input data.
# If raw MAT files are available, verify **every** label (`MAT label - 1`), patient ID, and
# cropped/resized GT mask against the source. This optional audit is not a model input.

# %%
def audit_dataset(frame, root):
    records = []
    for row in frame.itertuples():
        image = cv2.imread(str(root / "images" / row.filename), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(root / "masks" / row.filename), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None or image.ndim != 2 or image.shape != mask.shape:
            raise ValueError(f"Unreadable or unaligned pair: {row.filename}")
        if image.dtype != np.uint8 or not set(np.unique(mask)).issubset({0, 1, 255}) or not mask.any():
            raise ValueError(f"Expected uint8 MRI and a nonempty binary mask: {row.filename}")
        records.append({"image_hash": hashlib.sha256(image.tobytes()).hexdigest(),
                        "mask_hash": hashlib.sha256(mask.tobytes()).hexdigest(),
                        "height": image.shape[0], "width": image.shape[1],
                        "gt_tumor_pixels": int((mask > 0).sum()), "tumor_fraction": float((mask > 0).mean())})
    checked = pd.concat([frame.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    if (checked.groupby("image_hash").patient_id.nunique() > 1).any():
        raise ValueError("Identical images belong to different patients; resolve before partitioning.")
    columns = ["sample_id", "patient_id", "label", "filename", "image_hash", "mask_hash", "height", "width"]
    digest = signature(checked[columns].to_dict("records"))
    return checked, digest


def verify_raw_metadata(frame, raw_root):
    if not raw_root.is_dir():
        return {"status": "not_available", "verified_samples": 0}
    for row in frame.itertuples():
        with h5py.File(raw_root / row.original_mat, "r") as handle:
            raw = handle["cjdata"]
            label = int(np.asarray(raw["label"]).squeeze()) - 1
            patient = "".join(chr(int(c)) for c in np.asarray(raw["PID"]).flatten() if c).strip()
            mask = (np.asarray(raw["tumorMask"]).squeeze() > 0).astype(np.uint8)
        assert label == row.label and patient == row.patient_id, f"Source metadata mismatch: {row.filename}"
        cropped = mask[row.crop_y1:row.crop_y2, row.crop_x1:row.crop_x2]
        expected = cv2.resize(cropped, (row.width, row.height), interpolation=cv2.INTER_NEAREST)
        actual = cv2.imread(str(DATA_ROOT / "masks" / row.filename), cv2.IMREAD_GRAYSCALE) > 0
        assert np.array_equal(expected > 0, actual), f"Source mask mismatch: {row.filename}"
    return {"status": "verified", "verified_samples": len(frame)}


INDEX, DATA_HASH = audit_dataset(INDEX, DATA_ROOT)
RAW_AUDIT = verify_raw_metadata(INDEX, PROJECT_ROOT / "brainDataset")
print("Labels:", CLASS_NAMES, "| Raw-source verification:", RAW_AUDIT)
print("All image-mask pairs passed. Cross-patient exact-image duplicates: 0.")

# %% [markdown]
# ## 6. Image & Mask Preprocessing
# These files are already cropped using an MRI-derived brain contour. We do not crop again
# or use the GT mask to choose image content. Resize both arrays together (linear MRI,
# nearest-neighbor mask), scale MRI to `[0,1]`, and binarize the mask. No cohort-wide fitting
# or normalization uses validation/test data. Tumor size means GT area / processed image area,
# **not** physical volume. No additional denoising is applied in this notebook.

# %%
def preprocess_pair(filename):
    size = (CONFIG["image_size"], CONFIG["image_size"])
    image = cv2.imread(str(DATA_ROOT / "images" / filename), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(DATA_ROOT / "masks" / filename), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mask = (cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
    return image, mask

# %% [markdown]
# ## 7. Patient-Level Dataset Summary
# One row per patient is the unit of stratification. All slices of a patient share one label.

# %%
PATIENTS = INDEX.groupby("patient_id", sort=True).agg(label=("label", "first"), slices=("sample_id", "size")).reset_index()
DATASET_SUMMARY = INDEX.groupby("label").agg(slices=("sample_id", "size"), patients=("patient_id", "nunique"))
DATASET_SUMMARY["class"] = DATASET_SUMMARY.index.map(CLASS_NAMES)
display(DATASET_SUMMARY[["class", "patients", "slices"]])

# %% [markdown]
# ## 8. Nested Stratified Patient-Level Partitioning
# StratifiedKFold operates on the **unique-patient table**, not the slice table. This gives
# patient-level class stratification while keeping every patient's slices together.
# Fold definitions are fixed independently of model training. The outer test patients never
# appear in that outer fold's inner train/validation sets.
# See [scikit-learn: nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
# and [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).

# %% [markdown]
# ### 08.1 Outer 5-Fold Split

# %%
def outer_partition(patients):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG["split_seed"])
    result = patients.copy()
    result["outer_fold"] = 0
    for fold, (_, test) in enumerate(splitter.split(patients, patients.label), 1):
        result.loc[test, "outer_fold"] = fold
    return result


OUTER = outer_partition(PATIENTS)

# %% [markdown]
# ### 08.2 Inner 5-Fold Split

# %%
def inner_partition(outer):
    parts = []
    for fold in range(1, 6):
        training = outer.loc[outer.outer_fold != fold, ["patient_id", "label"]].reset_index(drop=True)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG["split_seed"] + fold)
        training["outer_fold"] = fold
        training["inner_fold"] = 0
        for inner, (_, validation) in enumerate(splitter.split(training, training.label), 1):
            training.loc[validation, "inner_fold"] = inner
        parts.append(training)
    return pd.concat(parts, ignore_index=True)


INNER = inner_partition(OUTER)

# %% [markdown]
# ### 08.3 Leakage Checks

# %%
def check_partitions(index, outer, inner):
    assert outer.patient_id.is_unique and set(outer.patient_id) == set(index.patient_id)
    assert set(outer.outer_fold) == set(range(1, 6))
    for table, column in [(outer, "outer_fold")] + [(inner[inner.outer_fold == k], "inner_fold") for k in range(1, 6)]:
        counts = pd.crosstab(table[column], table.label)
        assert (counts.max() - counts.min()).le(1).all(), "Patient classes are not stratified."
    for fold in range(1, 6):
        test = set(outer.loc[outer.outer_fold == fold, "patient_id"])
        training = set(outer.loc[outer.outer_fold != fold, "patient_id"])
        part = inner[inner.outer_fold == fold]
        assert part.patient_id.is_unique and set(part.patient_id) == training
        for k in range(1, 6):
            validation = set(part.loc[part.inner_fold == k, "patient_id"])
            fitting = set(part.loc[part.inner_fold != k, "patient_id"])
            assert not (fitting & validation or fitting & test or validation & test)
            assert fitting | validation | test == set(index.patient_id)
            for patients in (fitting, validation, test):
                subset = index[index.patient_id.isin(patients)]
                assert set(subset.label) == {0, 1, 2}
    return "All 25 inner partitions are disjoint from their corresponding outer test patients."


print(check_partitions(INDEX, OUTER, INNER))

# %% [markdown]
# ### 08.4 Class Distribution Checks
# Patient counts per class differ by at most one across stratified validation/test folds.
# Slice counts can differ because patients contribute different numbers of slices.

# %%
def fold_distribution(index, outer, inner):
    rows = []
    for fold in range(1, 6):
        partition = outer[outer.outer_fold != fold]
        roles = [(0, "outer_train", partition.patient_id),
                 (0, "outer_test", outer.loc[outer.outer_fold == fold, "patient_id"])]
        part = inner[inner.outer_fold == fold]
        for k in range(1, 6):
            roles += [(k, "inner_train", part.loc[part.inner_fold != k, "patient_id"]),
                      (k, "inner_val", part.loc[part.inner_fold == k, "patient_id"])]
        for k, role, patients in roles:
            subset = index[index.patient_id.isin(patients)]
            for label, group in subset.groupby("label"):
                rows.append({"outer_fold": fold, "inner_fold": k, "role": role, "class": CLASS_NAMES[label],
                             "patients": group.patient_id.nunique(), "slices": len(group)})
    return pd.DataFrame(rows)


FOLD_DISTRIBUTION = fold_distribution(INDEX, OUTER, INNER)
display(FOLD_DISTRIBUTION[FOLD_DISTRIBUTION.role == "outer_test"])

# %% [markdown]
# ### 08.5 Save Fold Definitions
# Patient assignments plus the sample index fully specify every train/validation/test pair.
# Existing identical definitions are reused; changed input/configuration gets a new directory.

# %%
EXPERIMENT = {"config": CONFIG, "candidates": CANDIDATES, "classes": CLASS_NAMES,
              "data_hash": DATA_HASH, "outer": OUTER.to_dict("records"), "inner": INNER.to_dict("records")}
EXPERIMENT_HASH = signature(EXPERIMENT)
PREPARATION_DIR = RESULT_ROOT / "preparation" / EXPERIMENT_HASH[:12]
PREPARATION_DIR.mkdir(parents=True, exist_ok=True)
INDEX.to_csv(PREPARATION_DIR / "sample_index.csv", index=False)
OUTER.to_csv(PREPARATION_DIR / "outer_patients.csv", index=False)
INNER.to_csv(PREPARATION_DIR / "inner_patients.csv", index=False)
FOLD_DISTRIBUTION.to_csv(PREPARATION_DIR / "fold_distribution.csv", index=False)
DATASET_SUMMARY.to_csv(PREPARATION_DIR / "dataset_summary.csv")
save_json(PREPARATION_DIR / "protocol.json", EXPERIMENT)
save_json(PREPARATION_DIR / "raw_audit.json", RAW_AUDIT)
print("Fold definitions saved:", PREPARATION_DIR.relative_to(PROJECT_ROOT))

# %% [markdown]
# ## 9. Dataset Class
# Each sample returns an MRI, its aligned mask, class label, and identifiers.
# The model receives only the MRI; the GT mask is used only by losses and evaluation.

# %%
class TumorDataset(Dataset):
    def __init__(self, frame, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, position):
        row = self.frame.iloc[position]
        image, mask = preprocess_pair(row.filename)
        if self.augment:
            image, mask = augment_pair(image, mask)
        return {"image": torch.from_numpy(image.copy()).unsqueeze(0),
                "mask": torch.from_numpy(mask.copy()).unsqueeze(0), "label": int(row.label),
                "sample_id": row.sample_id, "patient_id": row.patient_id, "filename": row.filename}

# %% [markdown]
# ## 10. Data Augmentation
# Training only: shared flips and a rotation up to 10 degrees. Images and masks use exactly
# the same geometry. Validation and test loaders never enable augmentation.

# %%
def augment_pair(image, mask):
    if random.random() < 0.5:
        image, mask = np.fliplr(image), np.fliplr(mask)
    if random.random() < 0.5:
        image, mask = np.flipud(image), np.flipud(mask)
    height, width = image.shape
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), random.uniform(-10, 10), 1.0)
    image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return image, mask

# %% [markdown]
# ## 11. DataLoaders
# Batch size is 32. If the final training batch would contain one image, merge it with the
# previous batch (33 images) to support BatchNorm1d without dropping any training sample.
# `num_workers=0` keeps the notebook portable to Windows.

# %%
class TrainingBatches(Sampler):
    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size = size, batch_size
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        order = torch.randperm(self.size, generator=self.generator).tolist()
        batches = [order[i:i + self.batch_size] for i in range(0, self.size, self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size - int(self.size > 1 and self.size % self.batch_size == 1)


def make_loader(frame, training=False, seed=42):
    dataset = TumorDataset(frame, augment=training)
    options = {"num_workers": 0, "pin_memory": DEVICE.type == "cuda"}
    if training:
        if len(frame) < 2:
            raise ValueError("Training requires at least two images.")
        sampler = TrainingBatches(len(frame), CONFIG["batch_size"], seed)
        return DataLoader(dataset, batch_sampler=sampler, **options)
    return DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False, **options)

# %% [markdown]
# ## 12. MAGS-Net Architecture
# Learned scale gates only. Three stages contain 2, 4, and 4 MAG blocks at 32, 96, and 192
# channels. Localization is predicted from image features; GT masks never enter the forward pass.

# %% [markdown]
# ### 12.1 Stem

# %%
def conv_bn_relu(inputs, outputs, kernel=1, stride=1, groups=1):
    return nn.Sequential(nn.Conv2d(inputs, outputs, kernel, stride, kernel // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))


class Stem(nn.Sequential):
    def __init__(self):
        super().__init__(*conv_bn_relu(1, 32, kernel=3, stride=2))

# %% [markdown]
# ### 12.2 MAG Block
# Global and predicted tumor-weighted context produce channel-wise softmax weights for the
# 3×3 and 5×5 depthwise branches. Their weighted fusion uses a residual connection.

# %%
class MAGBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = conv_bn_relu(channels, channels)
        self.dw3 = conv_bn_relu(channels, channels, kernel=3, groups=channels)
        self.dw5 = conv_bn_relu(channels, channels, kernel=5, groups=channels)
        self.localization = nn.Sequential(*conv_bn_relu(channels, channels, kernel=3, groups=channels),
                                          nn.Conv2d(channels, 1, 1))
        self.gate_fc1 = nn.Linear(2 * channels, max(channels // 8, 8))
        self.gate_fc2 = nn.Linear(max(channels // 8, 8), 2 * channels)
        self.fusion = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False),
                                    nn.BatchNorm2d(channels), nn.Dropout2d(0.1))

    def forward(self, x):
        features = self.proj(x)
        branch3, branch5 = self.dw3(features), self.dw5(features)
        localization = self.localization(features)
        mask = localization.sigmoid()
        global_context = F.adaptive_avg_pool2d(features, 1).flatten(1)
        tumor_context = (features * mask).sum((2, 3)) / (mask.sum((2, 3)) + 1e-6)
        context = torch.cat([global_context, tumor_context], dim=1)
        weights = self.gate_fc2(F.relu(self.gate_fc1(context)))
        weights = weights.reshape(x.shape[0], 2, x.shape[1]).softmax(dim=1)
        alpha3, alpha5 = weights[:, 0, :, None, None], weights[:, 1, :, None, None]
        output = F.relu(self.fusion(alpha3 * branch3 + alpha5 * branch5) + x)
        return output, localization, weights.mean(dim=2)

# %% [markdown]
# ### 12.3 MAGS Stages
# Keep the final localization logits from each stage for auxiliary supervision.
# Gate reporting averages channels within each block, then blocks within each stage.

# %%
class MAGSStage(nn.Module):
    def __init__(self, channels, blocks):
        super().__init__()
        self.blocks = nn.ModuleList([MAGBlock(channels) for _ in range(blocks)])

    def forward(self, x):
        gates = []
        for block in self.blocks:
            x, localization, gate = block(x)
            gates.append(gate)
        return x, localization, torch.stack(gates).mean(dim=0)

# %% [markdown]
# ### 12.4 Segmentation Decoder
# Project each stage to 64 channels, fuse skip connections, and restore input resolution.

# %%
class SegmentationDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_s3 = conv_bn_relu(192, 64)
        self.proj_s2 = conv_bn_relu(96, 64)
        self.proj_s1 = conv_bn_relu(32, 64)
        self.fuse2 = nn.Sequential(*conv_bn_relu(64, 64, 3, groups=64), *conv_bn_relu(64, 64))
        self.fuse1 = nn.Sequential(*conv_bn_relu(64, 64, 3, groups=64), *conv_bn_relu(64, 64))
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, s1, s2, s3, size):
        x = F.interpolate(self.proj_s3(s3), size=s2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse2(x + self.proj_s2(s2))
        x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse1(x + self.proj_s1(s1))
        return self.final_conv(F.interpolate(x, size=size, mode="bilinear", align_corners=False))

# %% [markdown]
# ### 12.5 Classification Head
# Combine global features with features weighted by the **predicted** segmentation mask.

# %%
class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, features, segmentation):
        mask = F.interpolate(segmentation.sigmoid(), size=features.shape[-2:], mode="bilinear", align_corners=False)
        global_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        tumor_features = (features * mask).sum((2, 3)) / (mask.sum((2, 3)) + 1e-6)
        joined = torch.cat([global_features, tumor_features], dim=1)
        return self.fc2(self.dropout(F.relu(self.bn_fc(self.fc1(joined)))))

# %% [markdown]
# ### 12.6 Full MAGS-Net

# %%
class MAGSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Stem()
        self.stage1 = MAGSStage(32, 2)
        self.down1 = conv_bn_relu(32, 96, stride=2)
        self.stage2 = MAGSStage(96, 4)
        self.down2 = conv_bn_relu(96, 192, stride=2)
        self.stage3 = MAGSStage(192, 4)
        self.decoder = SegmentationDecoder()
        self.classifier = ClassificationHead()

    def forward(self, images):
        s1, aux1, gate1 = self.stage1(self.stem(images))
        s2, aux2, gate2 = self.stage2(self.down1(s1))
        s3, aux3, gate3 = self.stage3(self.down2(s2))
        segmentation = self.decoder(s1, s2, s3, images.shape[-2:])
        return {"logits": self.classifier(s3, segmentation), "seg_logits": segmentation,
                "aux_masks": [aux1, aux2, aux3], "gates": torch.stack([gate1, gate2, gate3], dim=1)}

# %% [markdown]
# ## 13. Loss Functions
# ### 13.1 Classification Loss

# %%
def classification_loss(logits, labels):
    return F.cross_entropy(logits.float(), labels)

# %% [markdown]
# ### 13.2 Segmentation Loss
# Binary cross-entropy plus soft Dice loss, calculated in float32 under mixed precision.

# %%
def segmentation_loss(logits, mask):
    logits, mask = logits.float(), mask.float()
    probabilities, target = logits.sigmoid().flatten(1), mask.flatten(1)
    dice = (2 * (probabilities * target).sum(1) + 1) / (probabilities.sum(1) + target.sum(1) + 1)
    return F.binary_cross_entropy_with_logits(logits, mask) + 1 - dice.mean()

# %% [markdown]
# ### 13.3 Auxiliary Localization Loss

# %%
def auxiliary_loss(auxiliary, mask):
    losses = [segmentation_loss(logits, F.interpolate(mask, size=logits.shape[-2:], mode="nearest"))
              for logits in auxiliary]
    return torch.stack(losses).mean()

# %% [markdown]
# ### 13.4 Total Multi-Task Loss
# `Loss = Cls + 0.5 × Seg + 0.2 × Aux`. Printed components are unweighted.

# %%
def multitask_loss(outputs, labels, masks, candidate):
    cls = classification_loss(outputs["logits"], labels)
    seg = segmentation_loss(outputs["seg_logits"], masks)
    aux = auxiliary_loss(outputs["aux_masks"], masks)
    total = cls + candidate["lambda_seg"] * seg + candidate["lambda_aux"] * aux
    return {"loss": total, "cls": cls, "seg": seg, "aux": aux}

# %% [markdown]
# ## 14. Training Configuration
# Adam; initial LR 0.001; weight decay 1e-4; StepLR halves LR every 10 epochs.
# Inner fits use at most 50 epochs and patience 10. A strict decrease in validation total
# loss resets patience; ties do not improve. Gradients are clipped to norm 1.0.
# Outer refits use the median best epoch from the selected candidate's five inner fits.
# They train from scratch on **all outer-training patients** for this fixed duration.
# No early-stopping decision uses the outer test set.

# %%
assert CONFIG["outer_folds"] == CONFIG["inner_folds"] == 5
assert 1 <= CONFIG["max_epochs"] <= 50 and CONFIG["patience"] > 0
assert len(set(CONFIG["seeds"])) == len(CONFIG["seeds"]) and CONFIG["batch_size"] >= 2
display(pd.DataFrame(CANDIDATES))
print(f"Planned fits: {5 * len(CONFIG['seeds']) * (5 * len(CANDIDATES) + 1)}")

# %% [markdown]
# ## 15. Training / Validation Functions
# All metrics aggregate the full epoch. Dice/IoU are per-slice binary scores at threshold 0.5.
# Training/validation losses are weighted by batch size, including the final batch.

# %%
def prediction_rows(outputs, batch):
    probabilities = outputs["logits"].float().softmax(1).detach().cpu().numpy()
    predicted = outputs["seg_logits"].float().sigmoid().ge(CONFIG["mask_threshold"]).flatten(1)
    target = batch["mask"].to(predicted.device).gt(0.5).flatten(1)
    intersection = (predicted & target).sum(1).float()
    denominator = predicted.sum(1) + target.sum(1)
    union = denominator - intersection
    dice = torch.where(denominator > 0, 2 * intersection / denominator.clamp_min(1), 1.0)
    iou = torch.where(union > 0, intersection / union.clamp_min(1), 1.0)
    frame = pd.DataFrame({"sample_id": batch["sample_id"], "patient_id": batch["patient_id"],
                          "filename": batch["filename"], "label": batch["label"].numpy(),
                          "prediction": probabilities.argmax(1), "dice": dice.cpu().numpy(), "iou": iou.cpu().numpy(),
                          "tumor_fraction": target.float().mean(1).cpu().numpy()})
    frame[["prob_0", "prob_1", "prob_2"]] = probabilities
    gates = outputs["gates"].detach().float().cpu().numpy()
    for stage in range(3):
        frame[f"stage{stage + 1}_a3"], frame[f"stage{stage + 1}_a5"] = gates[:, stage, 0], gates[:, stage, 1]
    return frame


def score_predictions(frame):
    return {"accuracy": float(accuracy_score(frame.label, frame.prediction)),
            "f1": float(f1_score(frame.label, frame.prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
            "dice": float(frame.dice.mean()), "iou": float(frame.iou.mean())}

# %%
def run_epoch(model, loader, candidate, optimizer=None, scaler=None, mask_dir=None):
    training = optimizer is not None
    model.train(training)
    totals, records = {key: 0.0 for key in ("loss", "cls", "seg", "aux")}, []
    amp = CONFIG["use_amp"] and DEVICE.type == "cuda"
    dtype = torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc="Train" if training else "Evaluate", leave=False):
            images, masks = batch["image"].to(DEVICE), batch["mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, dtype=dtype, enabled=amp):
                outputs = model(images)
                losses = multitask_loss(outputs, labels, masks, candidate)
            if not all(torch.isfinite(value).item() for value in losses.values()):
                raise FloatingPointError("Non-finite multitask loss.")
            if training:
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
            for key, value in losses.items():
                totals[key] += float(value.detach()) * len(labels)
            records.append(prediction_rows(outputs, batch))
            if mask_dir is not None:
                masks_out = outputs["seg_logits"].detach().float().sigmoid().ge(CONFIG["mask_threshold"]).cpu().numpy()
                for name, mask in zip(batch["filename"], masks_out[:, 0]):
                    if not cv2.imwrite(str(mask_dir / name), mask.astype(np.uint8) * 255):
                        raise OSError(f"Could not save predicted mask: {name}")
    predictions = pd.concat(records, ignore_index=True)
    assert len(predictions) == len(loader.dataset) and predictions.sample_id.is_unique
    metrics = {key: value / len(predictions) for key, value in totals.items()}
    metrics.update(score_predictions(predictions))
    metrics["a3"] = float(predictions[["stage1_a3", "stage2_a3", "stage3_a3"]].mul([2, 4, 4]).sum(axis=1).mean() / 10)
    metrics["a5"] = 1.0 - metrics["a3"]
    return metrics, predictions

# %%
def print_epoch(epoch, epochs, lr, train, validation=None, best=None, stale=0):
    print(f"Epoch {epoch:02d}/{epochs} | lr={lr:.6f}")
    for name, metrics in (("Train", train), ("Val", validation)):
        if metrics is not None:
            print(f"{name:5} | Loss={metrics['loss']:.4f} | Cls={metrics['cls']:.4f} | Seg={metrics['seg']:.4f} | "
                  f"Aux={metrics['aux']:.4f} | Acc={metrics['accuracy']:.2%} | F1={metrics['f1']:.4f} | "
                  f"Dice={metrics['dice']:.4f} | IoU={metrics['iou']:.4f}")
    gates = validation if validation is not None else train
    print(f"Gates | a3={gates['a3']:.4f} | a5={gates['a5']:.4f}")
    if best is not None:
        print(f"Best  | Epoch={best['epoch']} | Val Loss={best['loss']:.4f} | Patience={stale}/{CONFIG['patience']}")


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)

# %%
def fit_signature(train_frame, val_frame, candidate, seed, epochs):
    return signature({"experiment": EXPERIMENT_HASH, "candidate": candidate, "seed": seed, "epochs": epochs,
                      "train": sorted(train_frame.sample_id), "val": [] if val_frame is None else sorted(val_frame.sample_id)})


def fit_model(train_frame, val_frame, candidate, seed, epochs, directory):
    run_signature = fit_signature(train_frame, val_frame, candidate, seed, epochs)
    summary_path = directory / "summary.json"
    if summary_path.exists():
        saved = json.loads(summary_path.read_text())
        if saved["signature"] != run_signature or file_hash(directory / saved["checkpoint"]) != saved["checkpoint_hash"]:
            raise ValueError("Saved fit does not match this experiment/checkpoint.")
        return saved
    if directory.exists():
        raise RuntimeError(f"Incomplete fit: {directory}. Start a new study or explicitly remove this incomplete fit before retrying.")
    directory.mkdir(parents=True)
    set_seed(seed)
    model = MAGSNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate["learning_rate"], weight_decay=candidate["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CONFIG["scheduler_step"], CONFIG["scheduler_gamma"])
    scaler = torch.amp.GradScaler("cuda", enabled=CONFIG["use_amp"] and DEVICE.type == "cuda" and not torch.cuda.is_bf16_supported())
    train_loader = make_loader(train_frame, training=True, seed=seed)
    val_loader = make_loader(val_frame) if val_frame is not None else None
    history, best, stale = [], {"epoch": 0, "loss": float("inf")}, 0
    for epoch in range(1, epochs + 1):
        started, lr = time.perf_counter(), optimizer.param_groups[0]["lr"]
        train, _ = run_epoch(model, train_loader, candidate, optimizer, scaler)
        val, _ = run_epoch(model, val_loader, candidate) if val_loader is not None else (None, None)
        improved = val is not None and val["loss"] < best["loss"]
        if improved:
            best, stale = {"epoch": epoch, "loss": val["loss"]}, 0
        elif val is not None:
            stale += 1
        scheduler.step()
        history.append({"epoch": epoch, "lr": lr, "seconds": time.perf_counter() - started,
                        **{f"train_{k}": v for k, v in train.items()},
                        **({f"val_{k}": v for k, v in val.items()} if val is not None else {})})
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "scaler": scaler.state_dict(), "epoch": epoch, "seed": seed, "candidate": candidate,
                      "signature": run_signature, "experiment_hash": EXPERIMENT_HASH}
        atomic_checkpoint(directory / "last.pt", checkpoint)
        if improved:
            atomic_checkpoint(directory / "best.pt", checkpoint)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print_epoch(epoch, epochs, lr, train, val, best if val is not None else None, stale)
        if val is not None and stale >= CONFIG["patience"]:
            break
    checkpoint_name = "best.pt" if val_frame is not None else "last.pt"
    result = {"signature": run_signature, "seed": seed, "epochs_completed": epoch,
              "best_epoch": best["epoch"] if val_frame is not None else epoch,
              "best_val_loss": best["loss"] if val_frame is not None else None,
              "checkpoint": checkpoint_name, "checkpoint_hash": file_hash(directory / checkpoint_name)}
    save_json(summary_path, result)
    return result

# %% [markdown]
# ## 16. Inner-CV Model Selection
# Fit every predefined candidate on all five inner training sets. Select the candidate with
# the smallest **mean of the five best validation total losses** (equal fold weighting;
# candidate order resolves ties). Fix outer training duration to the median of its five best
# epochs. Model weights are never transferred from an inner fold to the outer refit.

# %%
def select_inner_model(outer_fold, seed, directory):
    partition = INNER[INNER.outer_fold == outer_fold]
    results = []
    for candidate_id, candidate in enumerate(CANDIDATES):
        for inner_fold in range(1, 6):
            train_ids = partition.loc[partition.inner_fold != inner_fold, "patient_id"]
            val_ids = partition.loc[partition.inner_fold == inner_fold, "patient_id"]
            train = INDEX[INDEX.patient_id.isin(train_ids)]
            val = INDEX[INDEX.patient_id.isin(val_ids)]
            run_dir = directory / f"candidate_{candidate_id}" / f"inner_{inner_fold}"
            result = fit_model(train, val, candidate, seed, CONFIG["max_epochs"], run_dir)
            results.append({"candidate": candidate_id, "inner_fold": inner_fold, **result})
    table = pd.DataFrame(results)
    means = table.groupby("candidate").best_val_loss.mean()
    winner = int(means.idxmin())
    epochs = int(table.loc[table.candidate == winner, "best_epoch"].median())
    selection = {"candidate_id": winner, "candidate": CANDIDATES[winner], "epochs": epochs,
                 "mean_inner_val_loss": float(means.loc[winner]), "seed": seed, "outer_fold": outer_fold}
    table.to_csv(directory / "inner_results.csv", index=False)
    save_json(directory / "selection.json", selection)
    return selection

# %% [markdown]
# ## 17. Outer-Fold Retraining
# The selected duration is fixed before retraining. `outer_refit/last.pt` is the final outer
# model, not an outer-test-selected checkpoint. The outer test set is absent from this function.

# %%
def retrain_outer(outer_fold, seed, selection, directory):
    train_ids = OUTER.loc[OUTER.outer_fold != outer_fold, "patient_id"]
    train = INDEX[INDEX.patient_id.isin(train_ids)]
    return fit_model(train, None, selection["candidate"], seed, selection["epochs"], directory / "outer_refit")

# %% [markdown]
# ## 18. Outer-Test Evaluation
# Each slice receives one held-out prediction per seed. Save probabilities, Dice/IoU, GT area,
# stage gate weights, and binary predicted masks. Test evaluation never creates an optimizer.

# %%
def evaluate_outer(outer_fold, seed, directory):
    selection = json.loads((directory / "selection.json").read_text())
    run_dir = directory / "outer_refit"
    summary = json.loads((run_dir / "summary.json").read_text())
    checkpoint_path = run_dir / summary["checkpoint"]
    assert file_hash(checkpoint_path) == summary["checkpoint_hash"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["experiment_hash"] == EXPERIMENT_HASH and checkpoint["epoch"] == selection["epochs"]
    assert checkpoint["seed"] == seed and checkpoint["candidate"] == selection["candidate"]
    test_ids = OUTER.loc[OUTER.outer_fold == outer_fold, "patient_id"]
    test = INDEX[INDEX.patient_id.isin(test_ids)]
    train = INDEX[~INDEX.patient_id.isin(test_ids)]
    assert checkpoint["signature"] == fit_signature(train, None, selection["candidate"], seed, selection["epochs"])
    model = MAGSNet().to(DEVICE)
    model.load_state_dict(checkpoint["model"], strict=True)
    mask_dir = directory / "predicted_masks"
    mask_dir.mkdir(exist_ok=True)
    metrics, predictions = run_epoch(model, make_loader(test), selection["candidate"], mask_dir=mask_dir)
    predictions["seed"], predictions["outer_fold"] = seed, outer_fold
    predictions.to_csv(directory / "outer_predictions.csv", index=False)
    save_json(directory / "outer_metrics.json", {"seed": seed, "outer_fold": outer_fold, **metrics})
    save_json(directory / "evaluation.json", {"experiment_hash": EXPERIMENT_HASH, "checkpoint_hash": summary["checkpoint_hash"],
                                             "predictions_hash": file_hash(directory / "outer_predictions.csv")})
    return predictions

# %% [markdown]
# ## 19. Repeat for Outer Folds 1–5
# Set `RUN_TRAINING=True` to train. Leave `RUN_OUTER_TEST=False` until the protocol is fixed.
# Once all 5 outer refits are complete, set `RUN_TRAINING=False`, select `STUDY_DIR`, and
# enable `RUN_OUTER_TEST` to evaluate them. Previously completed fits are reused only when
# their configuration/data/checkpoint signatures match. Incomplete fits require explicit restart.

# %%
def open_study():
    directory = Path(STUDY_DIR) if STUDY_DIR is not None else None
    if directory is not None and not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    if directory is None and RUN_TRAINING:
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        directory = RESULT_ROOT / name
        directory.mkdir(parents=True)
        save_json(directory / "experiment.json", {"experiment_hash": EXPERIMENT_HASH, "protocol": EXPERIMENT,
                                                   "environment": ENVIRONMENT})
    if directory is not None:
        saved = json.loads((directory / "experiment.json").read_text())
        if saved["experiment_hash"] != EXPERIMENT_HASH:
            raise ValueError("This study uses different data, partitions, or settings.")
        print("Study directory:", directory)
    return directory


ACTIVE_STUDY = open_study()
if RUN_TRAINING:
    for seed in CONFIG["seeds"]:
        for outer_fold in range(1, 6):
            directory = ACTIVE_STUDY / f"seed_{seed}" / f"outer_{outer_fold}"
            print(f"\nSEED {seed} | OUTER FOLD {outer_fold}/5")
            selected = select_inner_model(outer_fold, seed, directory)
            retrain_outer(outer_fold, seed, selected, directory)
    save_json(ACTIVE_STUDY / "training_complete.json", {"experiment_hash": EXPERIMENT_HASH, "outer_refits": 5 * len(CONFIG["seeds"])})
else:
    print("Training disabled; no model weights have been updated.")

if RUN_OUTER_TEST:
    if ACTIVE_STUDY is None or not (ACTIVE_STUDY / "training_complete.json").exists():
        raise RuntimeError("Complete every planned outer refit and select its STUDY_DIR before evaluation.")
    # Check all completions before touching any outer test set.
    for seed in CONFIG["seeds"]:
        for fold in range(1, 6):
            assert (ACTIVE_STUDY / f"seed_{seed}" / f"outer_{fold}" / "outer_refit" / "summary.json").is_file()
    for seed in CONFIG["seeds"]:
        for outer_fold in range(1, 6):
            evaluate_outer(outer_fold, seed, ACTIVE_STUDY / f"seed_{seed}" / f"outer_{outer_fold}")

# %% [markdown]
# ## 20. Pool Out-of-Fold Predictions
# Combine the five outer-test folds **within each seed**. Do not average repeated seed rows
# as if they were additional independent patients. Every patient stays in one outer fold.
# Analyses below remain empty until all outer predictions for all requested seeds exist.

# %%
def pool_oof(study):
    if study is None:
        return pd.DataFrame()
    pieces = []
    expected = INDEX.set_index("sample_id").sort_index()
    for seed in CONFIG["seeds"]:
        seed_parts = []
        for fold in range(1, 6):
            directory = study / f"seed_{seed}" / f"outer_{fold}"
            path = directory / "outer_predictions.csv"
            if not (directory / "evaluation.json").exists():
                return pd.DataFrame()
            provenance = json.loads((directory / "evaluation.json").read_text())
            assert provenance["experiment_hash"] == EXPERIMENT_HASH and provenance["predictions_hash"] == file_hash(path)
            summary = json.loads((directory / "outer_refit" / "summary.json").read_text())
            assert provenance["checkpoint_hash"] == file_hash(directory / "outer_refit" / summary["checkpoint"])
            selection = json.loads((directory / "selection.json").read_text())
            train_ids = OUTER.loc[OUTER.outer_fold != fold, "patient_id"]
            train = INDEX[INDEX.patient_id.isin(train_ids)]
            assert summary["signature"] == fit_signature(train, None, selection["candidate"], seed, selection["epochs"])
            rows = pd.read_csv(path, dtype={"sample_id": str, "patient_id": str})
            assert rows.seed.eq(seed).all() and rows.outer_fold.eq(fold).all()
            assert set(rows.patient_id) == set(OUTER.loc[OUTER.outer_fold == fold, "patient_id"])
            seed_parts.append(rows)
        combined = pd.concat(seed_parts, ignore_index=True)
        assert combined.sample_id.is_unique and set(combined.sample_id) == set(expected.index)
        ordered = combined.set_index("sample_id").loc[expected.index]
        assert np.array_equal(ordered[["patient_id", "label"]].to_numpy(), expected[["patient_id", "label"]].to_numpy())
        probabilities = ordered[["prob_0", "prob_1", "prob_2"]].to_numpy()
        assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
        assert np.allclose(ordered[["prob_0", "prob_1", "prob_2"]].sum(axis=1), 1, atol=1e-5)
        assert np.array_equal(ordered.prediction, probabilities.argmax(axis=1))
        assert ordered[["dice", "iou", "tumor_fraction"]].apply(lambda column: column.between(0, 1).all()).all()
        for stage in range(1, 4):
            assert ordered[f"stage{stage}_a3"].between(0, 1).all() and ordered[f"stage{stage}_a5"].between(0, 1).all()
            assert np.allclose(ordered[f"stage{stage}_a3"] + ordered[f"stage{stage}_a5"], 1, atol=1e-5)
        pieces.append(combined)
    return pd.concat(pieces, ignore_index=True)


OOF = pool_oof(ACTIVE_STUDY)
TABLES = {"dataset_summary": DATASET_SUMMARY, "fold_distribution": FOLD_DISTRIBUTION}
FIGURES = {}
if OOF.empty:
    print("OOF results will appear after all outer-test evaluations. No performance is fabricated.")
else:
    print(f"OOF rows: {len(OOF):,} ({len(INDEX):,} per seed)")

# %% [markdown]
# ### Fold Summary and OOF Coverage Audit
# This report reads saved fold definitions, histories, and predictions only. It never trains
# or runs inference. Inner best epochs refer to the selected candidate, using minimum
# validation total multitask loss (first epoch on ties). The final training duration is
# the median of those five epochs, checked against the completed outer-refit history.
# Patient test coverage is counted from distinct patient/fold assignments. Multiple slices
# of one patient in the same test fold are expected and are not duplicate assignments.

# %%
def inner_fold_summary(study, seed, fold, selection):
    partition = INNER[INNER.outer_fold == fold]
    outer_test = set(OUTER.loc[OUTER.outer_fold == fold, "patient_id"])
    rows = []
    for inner_fold in range(1, 6):
        train_ids = set(partition.loc[partition.inner_fold != inner_fold, "patient_id"])
        val_ids = set(partition.loc[partition.inner_fold == inner_fold, "patient_id"])
        train, val = INDEX[INDEX.patient_id.isin(train_ids)], INDEX[INDEX.patient_id.isin(val_ids)]
        path = study / f"seed_{seed}" / f"outer_{fold}" / f"candidate_{selection['candidate_id']}" / f"inner_{inner_fold}"
        saved = json.loads((path / "summary.json").read_text())
        history = pd.read_csv(path / "history.csv", float_precision="round_trip")
        assert saved["signature"] == fit_signature(train, val, selection["candidate"], seed, CONFIG["max_epochs"])
        assert saved["checkpoint_hash"] == file_hash(path / saved["checkpoint"])
        assert history.epoch.tolist() == list(range(1, saved["epochs_completed"] + 1))
        best = history.loc[history.val_loss.idxmin()]
        assert int(best.epoch) == saved["best_epoch"]
        assert np.isclose(best.val_loss, saved["best_val_loss"], rtol=1e-10, atol=1e-12)
        overlap = len(train_ids & val_ids)
        outer_overlap = len((train_ids | val_ids) & outer_test)
        assert overlap == outer_overlap == 0
        rows.append({"seed": seed, "outer_fold": fold, "inner_fold": inner_fold,
                     "candidate_id": selection["candidate_id"], "train_patients": len(train_ids),
                     "val_patients": len(val_ids), "train_slices": len(train), "val_slices": len(val),
                     "patient_overlap": overlap, "overlap_with_outer_test": outer_overlap,
                     "best_epoch": int(best.epoch), "best_validation_total_loss": float(best.val_loss),
                     "completed_epochs": saved["epochs_completed"]})
    return pd.DataFrame(rows)

# %%
def outer_fold_summary(study, seed, fold, selection, inner_rows):
    development_ids = set(OUTER.loc[OUTER.outer_fold != fold, "patient_id"])
    test_ids = set(OUTER.loc[OUTER.outer_fold == fold, "patient_id"])
    development, test = INDEX[INDEX.patient_id.isin(development_ids)], INDEX[INDEX.patient_id.isin(test_ids)]
    path = study / f"seed_{seed}" / f"outer_{fold}" / "outer_refit"
    saved = json.loads((path / "summary.json").read_text())
    history = pd.read_csv(path / "history.csv")
    best_epochs = inner_rows.sort_values("inner_fold").best_epoch.astype(int).tolist()
    median_epoch = int(np.median(best_epochs))
    assert selection["seed"] == seed and selection["outer_fold"] == fold
    assert selection["candidate"] == CANDIDATES[selection["candidate_id"]]
    assert saved["signature"] == fit_signature(development, None, selection["candidate"], seed, selection["epochs"])
    assert saved["checkpoint_hash"] == file_hash(path / saved["checkpoint"])
    assert history.epoch.tolist() == list(range(1, saved["epochs_completed"] + 1))
    assert selection["epochs"] == saved["epochs_completed"] == median_epoch
    overlap = len(development_ids & test_ids)
    assert overlap == 0
    summary = {"seed": seed, "outer_fold": fold, "development_patients": len(development_ids),
               "outer_test_patients": len(test_ids), "development_slices": len(development),
               "outer_test_slices": len(test), "patient_overlap": overlap,
               "selected_candidate_id": selection["candidate_id"], "inner_best_epochs": json.dumps(best_epochs),
               "median_inner_best_epoch": median_epoch, "selected_training_epoch": selection["epochs"],
               "final_outer_training_epochs": saved["epochs_completed"],
               "epoch_selection_rule": "Median of the selected candidate's five inner best-validation-loss epochs"}
    classes = [{"seed": seed, "outer_fold": fold, "label": label, "class": name,
                "patients": test.loc[test.label == label, "patient_id"].nunique(),
                "slices": int(test.label.eq(label).sum())} for label, name in CLASS_NAMES.items()]
    return summary, classes

# %%
def oof_coverage_summary(predictions, seed):
    all_patients = pd.Index(INDEX.patient_id.unique())
    saved_assignments = OUTER[["patient_id", "outer_fold"]]
    planned = saved_assignments.groupby("patient_id").size().reindex(all_patients, fill_value=0)
    row = {"seed": seed, "total_unique_patients": len(all_patients), "total_unique_slices": INDEX.sample_id.nunique(),
           "planned_test_exactly_once": int(planned.eq(1).sum()), "planned_test_more_than_once": int(planned.gt(1).sum()),
           "planned_never_tested": int(planned.eq(0).sum()), "oof_status": "not_available"}
    measured = ["patients_tested_exactly_once", "patients_tested_more_than_once", "patients_never_tested",
                "oof_slices", "oof_unique_slices", "missing_oof_slices", "duplicate_sample_ids_in_oof",
                "duplicate_patient_fold_assignments"]
    row.update({key: pd.NA for key in measured})
    if predictions.empty:
        return row
    rows = predictions[predictions.seed == seed]
    assignments = rows[["patient_id", "outer_fold"]].drop_duplicates()
    observed = assignments.groupby("patient_id").size().reindex(all_patients, fill_value=0)
    row.update({"oof_status": "complete", "patients_tested_exactly_once": int(observed.eq(1).sum()),
                "patients_tested_more_than_once": int(observed.gt(1).sum()), "patients_never_tested": int(observed.eq(0).sum()),
                "oof_slices": len(rows), "oof_unique_slices": rows.sample_id.nunique(),
                "missing_oof_slices": len(set(INDEX.sample_id) - set(rows.sample_id)),
                "duplicate_sample_ids_in_oof": int(rows.sample_id.duplicated().sum()),
                "duplicate_patient_fold_assignments": int(assignments.patient_id.duplicated().sum())})
    assert row["patients_tested_exactly_once"] == len(all_patients) and row["oof_slices"] == len(INDEX)
    assert all(row[key] == 0 for key in ("patients_tested_more_than_once", "patients_never_tested", "missing_oof_slices",
                                       "duplicate_sample_ids_in_oof", "duplicate_patient_fold_assignments"))
    return row


def collect_fold_summary(study):
    check_partitions(INDEX, OUTER, INNER)
    completed = json.loads((study / "training_complete.json").read_text())
    assert completed["experiment_hash"] == EXPERIMENT_HASH
    predictions = pool_oof(study)  # Revalidate saved OOF IDs, labels, folds, and file/checkpoint hashes.
    folds, inner_parts, classes, coverage = [], [], [], []
    for seed in CONFIG["seeds"]:
        for fold in range(1, 6):
            selection = json.loads((study / f"seed_{seed}" / f"outer_{fold}" / "selection.json").read_text())
            inner_rows = inner_fold_summary(study, seed, fold, selection)
            fold_row, class_rows = outer_fold_summary(study, seed, fold, selection, inner_rows)
            folds.append(fold_row)
            inner_parts.append(inner_rows)
            classes.extend(class_rows)
        coverage.append(oof_coverage_summary(predictions, seed))
    return {"fold_summary": pd.DataFrame(folds), "inner_fold_summary": pd.concat(inner_parts, ignore_index=True),
            "outer_test_class_distribution": pd.DataFrame(classes), "oof_coverage_summary": pd.DataFrame(coverage)}

# %%
def format_fold_summary(tables):
    lines = []
    for seed in CONFIG["seeds"]:
        lines.append(f"SEED = {seed}\n")
        for fold in tables["fold_summary"].query("seed == @seed").itertuples(index=False):
            lines += [f"OUTER FOLD {fold.outer_fold}", f"Development patients: {fold.development_patients}",
                      f"Outer-test patients: {fold.outer_test_patients}", f"Development slices: {fold.development_slices}",
                      f"Outer-test slices: {fold.outer_test_slices}", "\nTest class distribution:"]
            classes = tables["outer_test_class_distribution"]
            for _, row in classes[(classes.seed == seed) & (classes.outer_fold == fold.outer_fold)].iterrows():
                lines.append(f"{row['class']}: {int(row.patients)} patients / {int(row.slices)} slices")
            lines.append(f"\nPatient overlap between development and outer-test: {fold.patient_overlap}")
            inner = tables["inner_fold_summary"]
            for row in inner[(inner.seed == seed) & (inner.outer_fold == fold.outer_fold)].itertuples(index=False):
                lines += [f"\nInner Fold {row.inner_fold}:", f"Train patients: {row.train_patients}",
                          f"Val patients: {row.val_patients}", f"Train slices: {row.train_slices}",
                          f"Val slices: {row.val_slices}", f"Overlap: {row.patient_overlap}",
                          f"Overlap with outer-test: {row.overlap_with_outer_test}", f"Best epoch: {row.best_epoch}"]
            lines += [f"\nInner best epochs = {fold.inner_best_epochs}", f"Median = {fold.median_inner_best_epoch}",
                      f"Selected training epoch from Inner CV: {fold.selected_training_epoch}",
                      f"Final outer model training epochs: {fold.final_outer_training_epochs}",
                      f"Selection rule: {fold.epoch_selection_rule}", "-" * 60]
        audit = tables["oof_coverage_summary"].query("seed == @seed").iloc[0]
        lines += [f"\nTotal unique patients = {audit.total_unique_patients}",
                  f"Total unique slices = {audit.total_unique_slices}"]
        if audit.oof_status != "complete":
            lines.append("OOF predictions are not available yet; observed test coverage is not reported.")
            continue
        lines += ["\nPatients appearing as outer-test (observed OOF assignments):",
                  f"exactly once = {audit.patients_tested_exactly_once}", f"more than once = {audit.patients_tested_more_than_once}",
                  f"never tested = {audit.patients_never_tested}", f"\nOOF slices = {audit.oof_slices}",
                  f"Duplicate sample IDs in OOF = {audit.duplicate_sample_ids_in_oof}",
                  f"Duplicate patient/fold assignments = {audit.duplicate_patient_fold_assignments}"]
    return "\n".join(lines)

# %%
# Read saved results and export the fold audit; no training or inference is performed.
if ACTIVE_STUDY is None or not (ACTIVE_STUDY / "training_complete.json").exists():
    print("Select a completed STUDY_DIR to display and save the fold summary.")
else:
    FOLD_SUMMARY_TABLES = collect_fold_summary(ACTIVE_STUDY)
    FOLD_SUMMARY_TEXT = format_fold_summary(FOLD_SUMMARY_TABLES)
    FOLD_SUMMARY_DIR = ACTIVE_STUDY / "analysis" / "fold_summary"
    FOLD_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in FOLD_SUMMARY_TABLES.items():
        table.to_csv(FOLD_SUMMARY_DIR / f"{name}.csv", index=False)
    (FOLD_SUMMARY_DIR / "fold_summary.txt").write_text(FOLD_SUMMARY_TEXT + "\n", encoding="utf-8")
    print(FOLD_SUMMARY_TEXT)
    print("\nFold summary CSV files saved to:", FOLD_SUMMARY_DIR)

# %% [markdown]
# ## 21. Final Classification Analysis
# Per-seed pooled OOF accuracy, macro F1, macro one-vs-rest AUC, and per-class
# precision/recall/F1. Classification always uses argmax over the three class probabilities.
# With one training seed, across-seed SD is not defined and is not reported.

# %%
def classification_tables(frame, level):
    overall, classes = [], []
    for seed, group in frame.groupby("seed"):
        probabilities = group[["prob_0", "prob_1", "prob_2"]].to_numpy()
        overall.append({"seed": seed, "level": level, **score_predictions(group),
                        "auc": roc_auc_score(group.label, probabilities, multi_class="ovr", average="macro", labels=[0, 1, 2])})
        report = classification_report(group.label, group.prediction, labels=[0, 1, 2],
                                       target_names=list(CLASS_NAMES.values()), output_dict=True, zero_division=0)
        classes.extend({"seed": seed, "class": name, **report[name]} for name in CLASS_NAMES.values())
    return pd.DataFrame(overall), pd.DataFrame(classes)


if not OOF.empty:
    TABLES["slice_metrics"], TABLES["slice_classification"] = classification_tables(OOF, "slice")
    TABLES["outer_fold_metrics"] = pd.DataFrame([{"seed": seed, "outer_fold": fold, **score_predictions(group)}
                                               for (seed, fold), group in OOF.groupby(["seed", "outer_fold"])])
    display(TABLES["slice_metrics"])
    if len(CONFIG["seeds"]) > 1:
        TABLES["seed_mean_sd"] = TABLES["slice_metrics"][["accuracy", "f1", "dice", "iou", "auc"]].agg(["mean", "std"]).T
        display(TABLES["seed_mean_sd"])

# %% [markdown]
# ## 22. Final Segmentation Analysis
# Slice-averaged Dice/IoU, GT tumor size by class, and descriptive quality groups:
# Excellent ≥0.75; Moderate 0.50–<0.75; Poor >0–<0.50; Failed =0.
# These thresholds are descriptive, not clinical categories. A predicted and GT empty mask
# would score 1; the integrity audit requires nonempty GT masks in this dataset.

# %%
if not OOF.empty:
    OOF["class"] = OOF.label.map(CLASS_NAMES)
    OOF["quality"] = np.select([OOF.dice >= 0.75, OOF.dice >= 0.5, OOF.dice > 0],
                               ["Excellent", "Moderate", "Poor"], default="Failed")
    TABLES["segmentation_by_class"] = OOF.groupby(["seed", "class"])[["dice", "iou", "tumor_fraction"]].agg(["count", "mean", "std", "median"])
    TABLES["quality_by_class"] = OOF.groupby(["seed", "class", "quality"]).size().rename("slices").reset_index()
    TABLES["tumor_size_by_quality"] = OOF.groupby(["seed", "class", "quality"]).tumor_fraction.agg(["count", "mean", "median", "std"])
    TABLES["poor_vs_excellent_size"] = TABLES["tumor_size_by_quality"].reset_index().query("quality in ['Poor', 'Excellent']")
    display(TABLES["segmentation_by_class"])
    display(TABLES["quality_by_class"])

# %% [markdown]
# ## 23. Patient-Level Analysis
# Average slice probabilities within each patient before argmax. Patient segmentation is the
# mean of that patient's 2D slice Dice/IoU; this is **not 3D volumetric Dice**.

# %%
def patient_predictions(frame):
    means = ["prob_0", "prob_1", "prob_2", "dice", "iou", "tumor_fraction"]
    result = frame.groupby(["seed", "patient_id"], as_index=False).agg(
        label=("label", "first"), outer_fold=("outer_fold", "first"), slices=("sample_id", "size"),
        **{column: (column, "mean") for column in means})
    result["prediction"] = result[["prob_0", "prob_1", "prob_2"]].to_numpy().argmax(1)
    return result


PATIENT_OOF = patient_predictions(OOF) if not OOF.empty else pd.DataFrame()
if not PATIENT_OOF.empty:
    TABLES["patient_metrics"], TABLES["patient_classification"] = classification_tables(PATIENT_OOF, "patient")
    display(TABLES["patient_metrics"])

# %% [markdown]
# ## 24. Statistical Analysis & Confidence Intervals
# Per-seed 95% percentile bootstrap intervals resample **patients**, stratified by class,
# keeping each sampled patient's slices together. Slice metrics weight slices; patient metrics
# weight patients equally. Intervals condition on the fitted OOF models and fixed partition;
# they do not quantify all variability from retraining or repeated CV.
# Seeds and overlapping training folds are not independent patient observations.
# Spearman analyses report exploratory associations without unadjusted slice-level p-values.
# See [SciPy Spearman documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).

# %%
def cluster_bootstrap(frame, statistic, repeats, seed=42):
    rng = np.random.default_rng(seed)
    frame = frame.reset_index(drop=True)
    groups = frame.groupby("patient_id", sort=True).indices
    patients = frame[["patient_id", "label"]].drop_duplicates()
    by_class = [group.patient_id.to_numpy() for _, group in patients.groupby("label")]
    values = []
    for _ in range(repeats):
        sampled = np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in by_class])
        resample = frame.iloc[np.concatenate([groups[patient] for patient in sampled])]
        value = statistic(resample)
        if np.isfinite(value):
            values.append(value)
    low, high = np.quantile(values, [0.025, 0.975]) if len(values) >= 20 else (np.nan, np.nan)
    return {"estimate": statistic(frame), "ci_low": low, "ci_high": high, "valid_resamples": len(values),
            "patients": frame.patient_id.nunique(), "rows": len(frame)}


def spearman_rho(frame, x, y):
    if len(frame) < 3 or frame[x].nunique() < 2 or frame[y].nunique() < 2:
        return np.nan
    return float(spearmanr(frame[x], frame[y]).statistic)

# %%
if not OOF.empty:
    intervals, size_correlations = [], []
    for level, frame in (("slice", OOF), ("patient", PATIENT_OOF)):
        for seed, group in frame.groupby("seed"):
            for metric in ("accuracy", "f1", "dice", "iou"):
                estimate = cluster_bootstrap(group, lambda data: score_predictions(data)[metric], CONFIG["bootstrap_samples"])
                intervals.append({"seed": seed, "level": level, "metric": metric, **estimate})
    for seed, group in OOF.groupby("seed"):
        for label in (None, 0, 1, 2):
            subset = group if label is None else group[group.label == label]
            estimate = cluster_bootstrap(subset, lambda data: spearman_rho(data, "tumor_fraction", "dice"), CONFIG["bootstrap_samples"])
            size_correlations.append({"seed": seed, "class": "All" if label is None else CLASS_NAMES[label], **estimate})
    TABLES["confidence_intervals"] = pd.DataFrame(intervals)
    TABLES["size_dice_spearman"] = pd.DataFrame(size_correlations)
    display(TABLES["confidence_intervals"])
    display(TABLES["size_dice_spearman"])

# %% [markdown]
# ## 25. Gate Analysis
# OOF gates only: stage means, means by tumor class, and Spearman associations of alpha5 with
# tumor size and Dice. Alpha3 = 1 − alpha5, so testing both would duplicate the association.
# Class-specific correlations address potential confounding by tumor class. Correlations are
# descriptive and do not establish that a gate causes better segmentation.

# %%
if not OOF.empty:
    gate_rows = []
    for stage in range(1, 4):
        part = OOF[["seed", "patient_id", "class", "label", "tumor_fraction", "dice"]].copy()
        part["stage"], part["a3"], part["a5"] = stage, OOF[f"stage{stage}_a3"], OOF[f"stage{stage}_a5"]
        gate_rows.append(part)
    GATES = pd.concat(gate_rows, ignore_index=True)
    TABLES["stage_gate_summary"] = GATES.groupby(["seed", "stage"])[["a3", "a5"]].agg(["mean", "std"])
    TABLES["gates_by_class"] = GATES.groupby(["seed", "stage", "class"])[["a3", "a5"]].agg(["mean", "std"])
    correlations = []
    for (seed, stage), group in GATES.groupby(["seed", "stage"]):
        for label in (None, 0, 1, 2):
            subset = group if label is None else group[group.label == label]
            for variable in ("tumor_fraction", "dice"):
                estimate = cluster_bootstrap(subset, lambda data: spearman_rho(data, "a5", variable), CONFIG["bootstrap_samples"])
                correlations.append({"seed": seed, "stage": stage, "class": "All" if label is None else CLASS_NAMES[label],
                                     "variable": variable, **estimate})
    TABLES["gate_correlations"] = pd.DataFrame(correlations)
    for title, key in (("STAGE-LEVEL GATE SUMMARY", "stage_gate_summary"), ("GATES BY TUMOR CLASS", "gates_by_class")):
        print(title)
        display(TABLES[key])
    for title, variable in (("GATE vs TUMOR SIZE", "tumor_fraction"), ("GATE vs SEGMENTATION DICE", "dice")):
        print(title)
        display(TABLES["gate_correlations"].query("variable == @variable"))

# %% [markdown]
# ## 26. Complexity & Efficiency Analysis
# Parameter count uses a fresh untrained model. Optional timing uses batch 1, float32,
# synthetic 224×224 inputs, warm-up, and CUDA synchronization. It excludes file loading and
# preprocessing. Conv/Linear MACs exclude normalization, activations, interpolation, pooling,
# and elementwise gate operations; they are a partial operation count, not total FLOPs.

# %%
@torch.inference_mode()
def model_complexity(benchmark=False):
    model = MAGSNet().to(DEVICE).eval()
    counts, handles = [], []
    def count(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            counts.append(output.numel() * (module.in_channels // module.groups) * np.prod(module.kernel_size))
        elif isinstance(module, nn.Linear):
            counts.append(output.numel() * module.in_features)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(count))
    images = torch.zeros(1, 1, CONFIG["image_size"], CONFIG["image_size"], device=DEVICE)
    model(images)
    for handle in handles:
        handle.remove()
    result = {"parameters": sum(p.numel() for p in model.parameters()), "conv_linear_macs": int(sum(counts)),
              "weight_mib_fp32": sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20,
              "device": str(DEVICE), "batch_size": 1, "precision": "float32"}
    if benchmark:
        for _ in range(10):
            model(images)
        durations = []
        for _ in range(50):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(images)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            durations.append(1000 * (time.perf_counter() - start))
        result["median_ms"] = float(np.median(durations))
        result["images_per_second"] = 1000 / result["median_ms"]
    return pd.DataFrame([result])


TABLES["complexity"] = model_complexity(RUN_BENCHMARK)
display(TABLES["complexity"])

# %% [markdown]
# ## 27. Figures & Tables for Paper
# One panel per seed avoids choosing the best-performing seed after seeing test results.
# Confusion matrices, ROC curves, Dice distributions, size/Dice scatter, gate/size scatter,
# and gate/class distributions use only held-out predictions. PNG (300 dpi) and PDF files
# are saved in the final section. No illustrative performance numbers are inserted.

# %%
def paper_figures(frame):
    figures = {}
    seeds = CONFIG["seeds"]
    for name in ("confusion_matrices", "roc_curves", "dice_by_class", "size_vs_dice", "gate_vs_size", "gate_by_class"):
        figure, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 4), squeeze=False)
        for ax, seed in zip(axes[0], seeds):
            group = frame[frame.seed == seed]
            if name == "confusion_matrices":
                matrix = confusion_matrix(group.label, group.prediction, labels=[0, 1, 2])
                ax.imshow(matrix, cmap="Blues")
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
                ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES.values(), yticklabels=CLASS_NAMES.values(), xlabel="Predicted", ylabel="True")
            elif name == "gate_by_class":
                ax.boxplot([group.loc[group.label == k, "stage3_a5"] for k in CLASS_NAMES], tick_labels=CLASS_NAMES.values(), showfliers=False)
                ax.set_ylabel("Stage 3 alpha5")
            elif name == "dice_by_class":
                ax.boxplot([group.loc[group.label == k, "dice"] for k in CLASS_NAMES], tick_labels=CLASS_NAMES.values(), showfliers=False)
                ax.set(ylabel="Slice Dice", ylim=(0, 1))
            else:
                for label, class_name in CLASS_NAMES.items():
                    subset = group[group.label == label]
                    if name == "roc_curves":
                        fpr, tpr, _ = roc_curve((group.label == label).astype(int), group[f"prob_{label}"])
                        ax.plot(fpr, tpr, label=class_name)
                        ax.set(xlabel="False positive rate", ylabel="True positive rate")
                    else:
                        y = "dice" if name == "size_vs_dice" else "stage3_a5"
                        ax.scatter(subset.tumor_fraction * 100, subset[y], s=7, alpha=0.35, label=class_name)
                        ax.set(xlabel="GT tumor area (%)", ylabel="Dice" if y == "dice" else "Stage 3 alpha5")
                ax.legend(fontsize=8)
            ax.set_title(f"Seed {seed}")
        figure.tight_layout()
        figures[name] = figure
    return figures


if not OOF.empty:
    FIGURES = paper_figures(OOF)
    plt.show()

# %% [markdown]
# ## 28. Save Final Results
# Save fold definitions, full protocol/environment, per-fit checkpoints/history, OOF tables,
# predicted masks, descriptive/statistical tables, and publication figures. Saved patient IDs
# must be pseudonymous before sharing; datasets and generated checkpoints are excluded by
# the accompanying Git ignore template. This notebook does not upload anything to GitHub.

# %%
FINAL_DIR = (ACTIVE_STUDY / "analysis") if ACTIVE_STUDY is not None and not OOF.empty else PREPARATION_DIR
FINAL_DIR.mkdir(parents=True, exist_ok=True)
for name, table in TABLES.items():
    table.to_csv(FINAL_DIR / f"{name}.csv", index=True)
for name, figure in FIGURES.items():
    figure.savefig(FINAL_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(FINAL_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)
if not OOF.empty:
    OOF.to_csv(FINAL_DIR / "oof_slices.csv", index=False)
    PATIENT_OOF.to_csv(FINAL_DIR / "oof_patients.csv", index=False)
save_json(FINAL_DIR / "environment.json", ENVIRONMENT)
save_json(FINAL_DIR / "analysis_status.json", {"experiment_hash": EXPERIMENT_HASH, "oof_available": not OOF.empty,
                                            "training_enabled": RUN_TRAINING, "outer_test_enabled": RUN_OUTER_TEST})
print("Artifacts saved:", FINAL_DIR.relative_to(PROJECT_ROOT))
