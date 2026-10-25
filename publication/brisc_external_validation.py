# %% [markdown]
# # MAGS-Net — BRISC2025 Evaluation
# Frozen inference on the **official test segmentation subset** (860 paired slices).
# Primary prediction: equal-weight probability ensemble of the five saved outer-refit models.
# No training, fine-tuning, augmentation, calibration, threshold tuning, or model selection.
# #
# **Interpretation:** BRISC includes images derived from the Cheng dataset and supplies no
# patient identifiers. This is a cross-dataset assessment with possible source overlap;
# it does **not establish independent patient-level external validation**.
# See [BRISC Methods and Data Records](https://arxiv.org/html/2506.14318v5).
# Slice-bootstrap intervals are exploratory and may be too narrow if slices share patients.

# %% [markdown]
# ## 0. Experiment Configuration
# Run all cells using the `dl` environment. A completed run is reloaded after integrity checks.
# Set `RUN_INFERENCE=False` to require existing predictions. This notebook has no training code.

# %%
SEED = 42
RUN_INFERENCE = False
CONFIG = dict(protocol="magsnet_brisc_test_v1", split="test", expected_slices=860,
              image_size=224, crop_padding=5, batch_size=32, mask_threshold=0.5,
              bootstrap_samples=2000, confidence_level=0.95, seed=SEED,
              ensemble="equal_mean_class_and_segmentation_probabilities", precision="FP32")
CLASS_NAMES = {0: "Meningioma", 1: "Glioma", 2: "Pituitary"}
PROB_COLUMNS = ["prob_meningioma", "prob_glioma", "prob_pituitary"]
PLANES = ["axial", "coronal", "sagittal"]

# %% [markdown]
# ## 1. Imports, Environment and Paths

# %%
import hashlib
import json
import platform
import random
from pathlib import Path, PurePosixPath

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = next(p for p in (Path.cwd(), Path.cwd() / "ZubNet-B", Path.cwd().parent)
                    if (p / "BRISC2025").is_dir())
DATA_ROOT = PROJECT_ROOT / "BRISC2025"
MAGS_STUDY_DIR = PROJECT_ROOT / "results/magsnet_nested_cv/20260916T210025Z_bc27ff71"
SOURCE_PREPARATION = PROJECT_ROOT / "results/magsnet_nested_cv/preparation/04ddbd02826f"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


print("Device:", DEVICE, "| Python:", platform.python_version(), "| Torch:", torch.__version__)
print("Source study:", MAGS_STUDY_DIR)

# %% [markdown]
# ## 2. Paired Test Index and Checkpoint Provenance
# Labels are assigned explicitly; alphabetical folder ordering would be incorrect.
# Masks must point to the matching MRI in the manifest. Original train images are not evaluated.

# %%
def dataset_path(relative):
    path = DATA_ROOT.joinpath(*PurePosixPath(relative.replace("\\", "/")).parts).resolve()
    if not path.is_relative_to(DATA_ROOT.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe dataset path: {relative}")
    return path


def build_index():
    records = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    subset = [r for r in records if r["task"] == "segmentation" and r["split"] == CONFIG["split"]]
    images = {r["relative_path"]: r for r in subset if not r["is_mask"]}
    masks = {r["linked_image"]: r for r in subset if r["is_mask"]}
    assert len(images) == len(masks) == CONFIG["expected_slices"]
    assert len(subset) == 2 * len(images) and images.keys() == masks.keys()
    rows = []
    for relative, image in sorted(images.items()):
        mask = masks[relative]
        for field in ("tumor_label", "plane_label", "split", "width", "height", "sequence"):
            assert image[field] == mask[field], (relative, field)
        label = {v.lower(): k for k, v in CLASS_NAMES.items()}[image["tumor_label"]]
        assert image["plane_label"] in PLANES and image["sequence"].lower() == "t1"
        assert Path(image["filename"]).stem == Path(mask["filename"]).stem
        image_path, mask_path = dataset_path(relative), dataset_path(mask["relative_path"])
        assert file_hash(image_path) == image["sha256"]
        assert file_hash(mask_path) == mask["sha256"]
        rows.append(dict(sample_id=image_path.stem, split=image["split"], true_label=label,
                         tumor_class=CLASS_NAMES[label], plane=image["plane_label"], sequence="T1",
                         image_path=str(image_path.relative_to(DATA_ROOT)),
                         mask_path=str(mask_path.relative_to(DATA_ROOT)),
                         image_sha256=image["sha256"], mask_sha256=mask["sha256"]))
    frame = pd.DataFrame(rows)
    assert not frame.sample_id.duplicated().any() and set(frame.true_label) == {0, 1, 2}
    return frame


INDEX = build_index()
EXPERIMENT = json.loads((MAGS_STUDY_DIR / "experiment.json").read_text())
MAGS_HASH = EXPERIMENT["experiment_hash"]
assert signature(EXPERIMENT["protocol"]) == MAGS_HASH
assert EXPERIMENT["protocol"]["classes"] == {str(k): v for k, v in CLASS_NAMES.items()}
assert EXPERIMENT["protocol"]["config"]["image_size"] == CONFIG["image_size"]
assert EXPERIMENT["protocol"]["config"]["mask_threshold"] == CONFIG["mask_threshold"]
CHECKPOINTS = []
for outer in range(1, 6):
    folder = MAGS_STUDY_DIR / f"seed_42/outer_{outer}/outer_refit"
    summary = json.loads((folder / "summary.json").read_text())
    path = folder / "last.pt"
    assert summary["checkpoint"] == path.name and file_hash(path) == summary["checkpoint_hash"]
    CHECKPOINTS.append(dict(outer_fold=outer, path=str(path), sha256=summary["checkpoint_hash"],
                            signature=summary["signature"], epochs=summary["epochs_completed"]))

PROTOCOL = dict(config=CONFIG, classes=CLASS_NAMES, source_experiment_hash=MAGS_HASH,
                data_hash=signature(INDEX.to_dict("records")), checkpoints=CHECKPOINTS,
                independence="not established: shared source and unavailable patient IDs",
                mask_decoding="0/1 directly; otherwise fixed >127 on 8-bit masks",
                crop_failure_policy="stop; never silently drop or use GT to repair crop")
STUDY_DIR = PROJECT_ROOT / "results/brisc2025_external_validation" / f"test_seed42_{signature(PROTOCOL)[:12]}"
STUDY_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR = STUDY_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)
save_json(STUDY_DIR / "protocol.json", PROTOCOL)
INDEX.to_csv(STUDY_DIR / "test_index.csv", index=False)
pd.DataFrame(CHECKPOINTS).to_csv(STUDY_DIR / "checkpoint_provenance.csv", index=False)
display(pd.crosstab(INDEX.tumor_class, INDEX.plane).reindex(CLASS_NAMES.values()))
print("STUDY_DIR:", STUDY_DIR)

# %% [markdown]
# ## 3. Original MAGS-Net Preprocessing
# This reproduces the preprocessing that created `brainDataset_cropped` in `ZubNet-B.ipynb`:
# per-image min–max uint8 conversion; Gaussian 5×5 blur **for contour detection only**;
# Otsu threshold; erosion once and dilation twice with a 5×5 kernel; largest contour;
# padding=5; joint crop; linear MRI resize and nearest-neighbor mask resize to 224×224.
# The model image is the normalized original crop, not the blurred image. Input is divided
# by 255 exactly as in MAGS-Net. No ImageNet normalization, TTA, or added denoising is used.
# #
# BRISC's nominal PNG masks contain low/high intensity compression artifacts. Decoding them
# with a fixed midpoint threshold restores binary labels before the original preprocessing.
# Using `mask > 0` on these encoded files would label background artifacts as tumor.
# GT is used only for evaluation and crop-retention reporting, never to choose a crop.

# %%
def normalize_to_uint8(image):
    image = image.astype(np.float32)
    low, high = image.min(), image.max()
    image = (image - low) / (high - low) if high > low else np.zeros_like(image)
    return (image * 255.0).clip(0, 255).astype(np.uint8)


def preprocess_pair(image, mask):
    image = normalize_to_uint8(image)
    mask = (mask > 0).astype(np.uint8)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), dtype=np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Brain contour missing; inspect this sample before evaluation.")
    contour = max(contours, key=cv2.contourArea)
    height, width = image.shape
    padding = CONFIG["crop_padding"]
    x1 = max(int(contour[:, :, 0].min()) - padding, 0)
    x2 = min(int(contour[:, :, 0].max()) + padding, width)
    y1 = max(int(contour[:, :, 1].min()) - padding, 0)
    y2 = min(int(contour[:, :, 1].max()) + padding, height)
    assert x2 > x1 and y2 > y1
    crop, target = image[y1:y2, x1:x2], mask[y1:y2, x1:x2]
    size = (CONFIG["image_size"],) * 2
    resized = cv2.resize(crop, size, interpolation=cv2.INTER_LINEAR)
    target_resized = (cv2.resize(target, size, interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    audit = dict(crop_x1=x1, crop_y1=y1, crop_x2=x2, crop_y2=y2,
                 original_gt_pixels=int(mask.sum()), retained_gt_pixels=int(target.sum()),
                 tumor_preserved_ratio=float(target.sum() / max(int(mask.sum()), 1)))
    return resized, target_resized, audit


def read_brisc_pair(row):
    image = cv2.imread(str(dataset_path(row.image_path)), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(dataset_path(row.mask_path)), cv2.IMREAD_GRAYSCALE)
    assert image is not None and mask is not None and image.shape == mask.shape
    assert image.dtype == mask.dtype == np.uint8
    # Decode representation only; this threshold is never selected using model results.
    binary = (mask > (0 if mask.max() <= 1 else 127)).astype(np.uint8)
    assert binary.any(), f"Empty original GT mask: {row.sample_id}"
    return image, binary, int(((mask > 0) & (mask < 128)).sum())


images, masks, raw_thumbnails, audit_rows = [], [], [], []
for row in tqdm(INDEX.itertuples(), total=len(INDEX), desc="Preprocessing paired test slices"):
    image, mask, low_intensity_pixels = read_brisc_pair(row)
    processed, target, audit = preprocess_pair(image, mask)
    images.append(processed)
    masks.append(target)
    raw_thumbnails.append(cv2.resize(normalize_to_uint8(image), (64, 64)))
    audit_rows.append(dict(sample_id=row.sample_id, **audit, low_intensity_mask_pixels=low_intensity_pixels,
                           processed_gt_pixels=int(target.sum()),
                           processed_image_hash=hashlib.sha256(processed.tobytes()).hexdigest()))
IMAGES, MASKS = np.stack(images), np.stack(masks).astype(bool)
RAW_THUMBNAILS = np.stack(raw_thumbnails)
AUDIT = pd.DataFrame(audit_rows)
AUDIT.to_csv(STUDY_DIR / "preprocessing_audit.csv", index=False)
INDEX = INDEX.merge(AUDIT, on="sample_id", validate="one_to_one")
np.savez_compressed(STUDY_DIR / "preprocessed_test.npz", images=IMAGES, masks=MASKS,
                    sample_ids=INDEX.sample_id.to_numpy(dtype=str))
print("MRI / GT:", IMAGES.shape, MASKS.shape)
print("Crops retaining less than 99.9% of GT:", (AUDIT.tumor_preserved_ratio < 0.999).sum())
print("Empty GT masks after cropping:", (~MASKS.any(axis=(1, 2))).sum())
print("No samples excluded or crop coordinates repaired using GT.")

# %% [markdown]
# ## 4. Source-Overlap Audit and Preprocessing Parity
# Exact equality of processed pixels is checked against every source slice. A nearest-image
# screen also compares raw and cropped images at 64×64 under eight rotation/reflection
# transforms using normalized correlation. Scores >=0.995 are **review candidates**, not
# verified identities. JPEG conversion/cropping can hide overlap from exact hashes.
# Even a negative screen cannot establish patient independence. No images are automatically
# excluded based on similarity or model performance. The raw-source pass additionally checks
# that the copied preprocessing recreates every stored MAGS-Net MRI and mask exactly.

# %%
def standardized_vectors(images):
    flat = images.reshape(len(images), -1).astype(np.float32)
    flat -= flat.mean(axis=1, keepdims=True)
    return flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)


def similarity_screen(query, reference):
    reference = torch.from_numpy(standardized_vectors(reference)).to(DEVICE)
    scores, indices, transforms = np.full(len(query), -2.0), np.zeros(len(query), int), np.zeros(len(query), int)
    for transform in range(8):
        transformed = np.rot90(query, transform % 4, axes=(1, 2))
        if transform >= 4:
            transformed = transformed[:, :, ::-1]
        vectors = torch.from_numpy(standardized_vectors(transformed)).to(DEVICE)
        values, nearest = (vectors @ reference.T).max(dim=1)
        values, nearest = values.cpu().numpy(), nearest.cpu().numpy()
        better = values > scores
        scores[better], indices[better], transforms[better] = values[better], nearest[better], transform
    return scores, indices, transforms


SOURCE_INDEX = pd.read_csv(SOURCE_PREPARATION / "sample_index.csv", dtype={"sample_id": str, "patient_id": str})
source_images, source_raw = [], []
parity_count = 0
for row in tqdm(SOURCE_INDEX.itertuples(), total=len(SOURCE_INDEX), desc="Source overlap / preprocessing audit"):
    cropped = cv2.imread(str(PROJECT_ROOT / "brainDataset_cropped/images" / row.filename), 0)
    gt = cv2.imread(str(PROJECT_ROOT / "brainDataset_cropped/masks" / row.filename), 0) > 0
    assert hashlib.sha256(cropped.tobytes()).hexdigest() == row.image_hash
    assert hashlib.sha256((gt.astype(np.uint8) * 255).tobytes()).hexdigest() == row.mask_hash
    source_images.append(cv2.resize(cropped, (64, 64)))
    raw_path = PROJECT_ROOT / "brainDataset" / row.original_mat
    if raw_path.exists():
        with h5py.File(raw_path, "r") as handle:
            raw = np.asarray(handle["cjdata/image"]).squeeze()
            raw_gt = np.asarray(handle["cjdata/tumorMask"]).squeeze()
        reproduced, reproduced_gt, _ = preprocess_pair(raw, raw_gt)
        assert np.array_equal(reproduced, cropped), f"MRI preprocessing mismatch: {row.filename}"
        assert np.array_equal(reproduced_gt > 0, gt), f"Mask preprocessing mismatch: {row.filename}"
        parity_count += 1
        source_raw.append(cv2.resize(normalize_to_uint8(raw), (64, 64)))

cropped_scores, cropped_indices, cropped_transforms = similarity_screen(
    np.stack([cv2.resize(im, (64, 64)) for im in IMAGES]), np.stack(source_images))
OVERLAP = INDEX[["sample_id", "tumor_class", "plane"]].copy()
OVERLAP["processed_exact_source_match"] = INDEX.processed_image_hash.isin(SOURCE_INDEX.image_hash)
OVERLAP["cropped_correlation"] = cropped_scores
OVERLAP["cropped_source_sample_id"] = SOURCE_INDEX.iloc[cropped_indices].sample_id.to_numpy()
OVERLAP["cropped_transform"] = cropped_transforms
if len(source_raw) == len(SOURCE_INDEX):
    raw_scores, raw_indices, raw_transforms = similarity_screen(RAW_THUMBNAILS, np.stack(source_raw))
    OVERLAP["raw_correlation"] = raw_scores
    OVERLAP["raw_source_sample_id"] = SOURCE_INDEX.iloc[raw_indices].sample_id.to_numpy()
    OVERLAP["raw_transform"] = raw_transforms
else:
    print("Raw source audit incomplete; only cropped-image similarity is available.")
score_columns = [c for c in OVERLAP if c.endswith("_correlation")]
OVERLAP["max_correlation"] = OVERLAP[score_columns].max(axis=1)
OVERLAP["review_candidate"] = OVERLAP.max_correlation >= 0.995
OVERLAP.to_csv(STUDY_DIR / "source_overlap_audit.csv", index=False)
OVERLAP.sort_values("max_correlation", ascending=False).head(30).to_csv(
    STUDY_DIR / "source_overlap_review_candidates.csv", index=False)
INDEPENDENCE_AUDIT = dict(source_slices=len(SOURCE_INDEX), exact_matches=int(OVERLAP.processed_exact_source_match.sum()),
                          near_match_candidates=int(OVERLAP.review_candidate.sum()),
                          preprocessing_parity_samples=parity_count, patient_identifiers_available=False,
                          independent_external_validation=False,
                          note="Shared Cheng provenance; image screening cannot rule out patient overlap.")
save_json(STUDY_DIR / "independence_audit.json", INDEPENDENCE_AUDIT)
display(pd.DataFrame([INDEPENDENCE_AUDIT]))

# %% [markdown]
# ## 5. MAGS-Net Architecture
# Exact module definitions and state-dict keys from the trained MAGS-Net notebook.
# The forward pass receives MRI tensors only; all lesion guidance is predicted internally.

# %%
def conv_bn_relu(inputs, outputs, kernel=1, stride=1, groups=1):
    return nn.Sequential(nn.Conv2d(inputs, outputs, kernel, stride, kernel // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))


class Stem(nn.Sequential):
    def __init__(self):
        super().__init__(*conv_bn_relu(1, 32, kernel=3, stride=2))


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
# ## 6. Frozen Models and Per-Slice Metrics
# All five checkpoints must load strictly and match the recorded experiment, seed, epoch,
# and fit signature. `eval()` freezes BatchNorm/dropout behavior, and inference mode disables
# gradients. Dice/IoU use the original binary metric convention (threshold >=0.5; both-empty=1).
# These are evaluation scores, not the smoothed differentiable Dice training loss.

# %%
def load_frozen_model(record):
    payload = torch.load(record["path"], map_location="cpu", weights_only=True)
    assert payload["experiment_hash"] == MAGS_HASH and payload["seed"] == SEED
    assert payload["epoch"] == record["epochs"] and payload["signature"] == record["signature"]
    model = MAGSNet()
    model.load_state_dict(payload["model"], strict=True)
    return model.to(DEVICE).eval().requires_grad_(False)


def model_state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def segmentation_scores(predicted, target):
    axes = (1, 2)
    intersection = (predicted & target).sum(axis=axes)
    denominator = predicted.sum(axis=axes) + target.sum(axis=axes)
    union = denominator - intersection
    dice = np.divide(2 * intersection, denominator, out=np.ones(len(target)), where=denominator > 0)
    iou = np.divide(intersection, union, out=np.ones(len(target)), where=union > 0)
    return dice, iou


def prediction_frame(name, probabilities, predicted_masks):
    frame = INDEX[["sample_id", "split", "true_label", "tumor_class", "plane", "sequence",
                   "tumor_preserved_ratio", "original_gt_pixels", "processed_gt_pixels"]].copy()
    frame.insert(0, "model", name)
    frame["predicted_label"] = probabilities.argmax(axis=1)
    frame[PROB_COLUMNS] = probabilities
    frame["dice"], frame["iou"] = segmentation_scores(predicted_masks, MASKS)
    frame["gt_tumor_fraction"] = MASKS.mean(axis=(1, 2))
    assert not frame.sample_id.duplicated().any() and np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    return frame

# %% [markdown]
# ## 7. Inference and Saved Predictions
# Each model sees all 860 test images. Individual-model results are descriptive sensitivity
# analyses; no model is selected using BRISC. Ensemble segmentation averages sigmoid
# probabilities before thresholding. This is an external prediction CSV, **not an OOF CSV**.

# %%
@torch.inference_mode()
def run_inference():
    class_sum = np.zeros((len(INDEX), 3), np.float32)
    segmentation_sum = np.zeros(MASKS.shape, np.float32)
    state_checks, per_model = [], []
    for record in CHECKPOINTS:
        model = load_frozen_model(record)
        before = model_state_hash(model)
        probabilities = np.empty_like(class_sum)
        predictions = np.zeros_like(MASKS)
        for start in tqdm(range(0, len(INDEX), CONFIG["batch_size"]), desc=f"Outer model {record['outer_fold']}"):
            stop = min(start + CONFIG["batch_size"], len(INDEX))
            batch = torch.from_numpy(IMAGES[start:stop, None].astype(np.float32) / 255.0).to(DEVICE)
            output = model(batch)
            class_probability = output["logits"].float().softmax(dim=1).cpu().numpy()
            segmentation_probability = output["seg_logits"].float().sigmoid()[:, 0].cpu().numpy()
            probabilities[start:stop] = class_probability
            class_sum[start:stop] += class_probability / len(CHECKPOINTS)
            segmentation_sum[start:stop] += segmentation_probability / len(CHECKPOINTS)
            predictions[start:stop] = segmentation_probability >= CONFIG["mask_threshold"]
        after = model_state_hash(model)
        assert before == after and not model.training
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        state_checks.append(dict(outer_fold=record["outer_fold"], state_before=before, state_after=after,
                                  unchanged=True, parameters=sum(p.numel() for p in model.parameters())))
        per_model.append(prediction_frame(f"MAGS-Net outer {record['outer_fold']}", probabilities, predictions))
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    ensemble_masks = segmentation_sum >= CONFIG["mask_threshold"]
    ensemble = prediction_frame("MAGS-Net 5-model ensemble", class_sum, ensemble_masks)
    ensemble.to_csv(STUDY_DIR / "MAGSNet_BRISC_test_predictions.csv", index=False)
    pd.concat(per_model, ignore_index=True).to_csv(STUDY_DIR / "individual_model_predictions.csv", index=False)
    np.save(STUDY_DIR / "ensemble_segmentation_probabilities.npy", segmentation_sum, allow_pickle=False)
    np.savez_compressed(STUDY_DIR / "ensemble_masks.npz", predicted=ensemble_masks,
                        sample_ids=INDEX.sample_id.to_numpy(dtype=str))
    pd.DataFrame(state_checks).to_csv(STUDY_DIR / "frozen_state_verification.csv", index=False)
    files = ["MAGSNet_BRISC_test_predictions.csv", "individual_model_predictions.csv",
             "ensemble_segmentation_probabilities.npy", "ensemble_masks.npz", "frozen_state_verification.csv"]
    save_json(STUDY_DIR / "inference_complete.json", dict(protocol_hash=signature(PROTOCOL),
              rows=len(ensemble), file_hashes={name: file_hash(STUDY_DIR / name) for name in files}))


marker = STUDY_DIR / "inference_complete.json"
if not marker.exists():
    if not RUN_INFERENCE:
        raise RuntimeError("No completed inference found. Set RUN_INFERENCE=True and run this cell.")
    run_inference()
completion = json.loads(marker.read_text())
assert completion["protocol_hash"] == signature(PROTOCOL) and completion["rows"] == len(INDEX)
for filename, expected in completion["file_hashes"].items():
    assert file_hash(STUDY_DIR / filename) == expected, f"Changed result file: {filename}"
PREDICTIONS = pd.read_csv(STUDY_DIR / "MAGSNet_BRISC_test_predictions.csv")
PER_MODEL = pd.read_csv(STUDY_DIR / "individual_model_predictions.csv")
with np.load(STUDY_DIR / "ensemble_masks.npz", allow_pickle=False) as saved:
    assert np.array_equal(saved["sample_ids"], INDEX.sample_id.to_numpy())
    ENSEMBLE_MASKS = saved["predicted"]
assert PREDICTIONS.sample_id.tolist() == INDEX.sample_id.tolist()
assert np.array_equal(PREDICTIONS.true_label, INDEX.true_label)
assert np.allclose(segmentation_scores(ENSEMBLE_MASKS, MASKS)[0], PREDICTIONS.dice)
print("Inference complete:", len(PREDICTIONS), "slices; no weight or BatchNorm updates.")

# %% [markdown]
# ## 8. Overall Classification and Segmentation
# Accuracy and macro-F1 are slice-level; AUC is the unweighted mean of the three one-vs-rest
# AUCs using probability columns in label order 0,1,2. Mean Dice/IoU average per-slice scores.
# There are no patient-level metrics because BRISC has no validated patient identifiers.

# %%
def metric_values(frame):
    y = frame.true_label.to_numpy()
    pred = frame.predicted_label.to_numpy()
    probability = frame[PROB_COLUMNS].to_numpy()
    auc = roc_auc_score(y, probability, labels=[0, 1, 2], multi_class="ovr", average="macro") if len(np.unique(y)) == 3 else np.nan
    return dict(accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0),
                auc_macro_ovr=auc, mean_dice=frame.dice.mean(), mean_iou=frame.iou.mean())


OVERALL = pd.DataFrame([dict(model=PREDICTIONS.model.iloc[0], n_slices=len(PREDICTIONS), **metric_values(PREDICTIONS))])
INDIVIDUAL_SUMMARY = pd.DataFrame([dict(model=name, n_slices=len(frame), **metric_values(frame))
                                   for name, frame in PER_MODEL.groupby("model", sort=True)])
OVERALL.to_csv(STUDY_DIR / "overall_metrics.csv", index=False)
INDIVIDUAL_SUMMARY.to_csv(STUDY_DIR / "individual_model_metrics.csv", index=False)
display(OVERALL)
display(INDIVIDUAL_SUMMARY)

# %% [markdown]
# ## 9. Bootstrap 95% Confidence Intervals
# 2,000 percentile resamples (seed 42), stratified by true tumor class, with replacement.
# Each resample carries a slice's classification probabilities, Dice and IoU together.
# Intervals condition on the observed class composition and the frozen ensemble; they do
# not include training uncertainty. They assume independent slices and are **exploratory**
# because within-patient dependence cannot be modeled. No patient-bootstrap claim is made.
# Subgroup intervals are nominal; no simultaneous-coverage claim is made.

# %%
def bootstrap_metrics(frame, repeats=None):
    repeats = CONFIG["bootstrap_samples"] if repeats is None else repeats
    frame = frame.reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    strata = [np.flatnonzero(frame.true_label.to_numpy() == label) for label in sorted(frame.true_label.unique())]
    samples = []
    for _ in range(repeats):
        selection = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata])
        samples.append(metric_values(frame.iloc[selection]))
    draws = pd.DataFrame(samples)
    alpha = (1 - CONFIG["confidence_level"]) / 2
    return pd.DataFrame([dict(metric=name, estimate=value,
                              ci_low=draws[name].quantile(alpha), ci_high=draws[name].quantile(1 - alpha),
                              n_slices=len(frame), valid_resamples=int(draws[name].notna().sum()),
                              bootstrap_unit="slice", stratification="true tumor class")
                          for name, value in metric_values(frame).items()])


OVERALL_CI = bootstrap_metrics(PREDICTIONS)
OVERALL_CI.to_csv(STUDY_DIR / "overall_bootstrap_ci.csv", index=False)
display(OVERALL_CI)

# %% [markdown]
# ## 10. Tumor-Class and MRI-Plane Results
# Class recall is accuracy **conditional on that true class**; class F1/precision use all
# samples in the group as potential negatives. Each class AUC is one-vs-rest, never computed
# on a positive-only subset. Dice/IoU are restricted to that true tumor class.
# Plane-level macro-F1/AUC compare all three classes within that plane.

# %%
def per_class_values(frame):
    y, pred = frame.true_label.to_numpy(), frame.predicted_label.to_numpy()
    probability = frame[PROB_COLUMNS].to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    rows = []
    for label, name in CLASS_NAMES.items():
        group = frame.loc[frame.true_label == label]
        auc = roc_auc_score(y == label, probability[:, label]) if 0 < len(group) < len(frame) else np.nan
        rows.append(dict(tumor_class=name, n_slices=int(support[label]), precision=precision[label],
                         recall=recall[label], f1=f1[label], auc_ovr=auc,
                         mean_dice=group.dice.mean(), mean_iou=group.iou.mean()))
    return pd.DataFrame(rows)


def class_bootstrap(frame):
    frame = frame.reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    strata = [np.flatnonzero(frame.true_label.to_numpy() == label) for label in range(3)]
    draws = []
    for _ in range(CONFIG["bootstrap_samples"]):
        selection = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata if len(group)])
        draws.append(per_class_values(frame.iloc[selection]))
    draws = pd.concat(draws, ignore_index=True)
    result = per_class_values(frame)
    for metric in ("precision", "recall", "f1", "auc_ovr", "mean_dice", "mean_iou"):
        for suffix, quantile in (("ci_low", 0.025), ("ci_high", 0.975)):
            interval = draws.groupby("tumor_class")[metric].quantile(quantile)
            result[f"{metric}_{suffix}"] = result.tumor_class.map(interval)
    return result


CLASS_RESULTS = class_bootstrap(PREDICTIONS)
CLASS_RESULTS.to_csv(STUDY_DIR / "per_class_metrics_ci.csv", index=False)
PLANE_RESULTS, CLASS_PLANE_RESULTS = [], []
for plane in tqdm(PLANES, desc="Plane and class bootstrap"):
    group = PREDICTIONS.loc[PREDICTIONS.plane == plane]
    PLANE_RESULTS.append(bootstrap_metrics(group).assign(plane=plane))
    CLASS_PLANE_RESULTS.append(class_bootstrap(group).assign(plane=plane))
PLANE_RESULTS = pd.concat(PLANE_RESULTS, ignore_index=True)
CLASS_PLANE_RESULTS = pd.concat(CLASS_PLANE_RESULTS, ignore_index=True)
PLANE_RESULTS.to_csv(STUDY_DIR / "per_plane_metrics_ci.csv", index=False)
CLASS_PLANE_RESULTS.to_csv(STUDY_DIR / "per_class_plane_metrics_ci.csv", index=False)
display(CLASS_RESULTS)
display(PLANE_RESULTS)
display(CLASS_PLANE_RESULTS[["plane", "tumor_class", "n_slices", "recall", "f1", "mean_dice", "mean_iou"]])

# %% [markdown]
# ## 11. Confusion Matrices and ROC Curves
# Overall and per-plane figures use the same fixed label order. ROC coordinates and both
# count/row-normalized confusion matrices are saved as CSV as well as PNG/PDF figures.

# %%
def save_figure(fig, name):
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)


GROUPS = [("overall", PREDICTIONS)] + [(plane, PREDICTIONS.loc[PREDICTIONS.plane == plane]) for plane in PLANES]
roc_rows, confusion_rows = [], []
fig_cm, axes_cm = plt.subplots(2, 4, figsize=(17, 8))
fig_roc, axes_roc = plt.subplots(1, 4, figsize=(18, 4.3))
for column, (name, frame) in enumerate(GROUPS):
    cm = confusion_matrix(frame.true_label, frame.predicted_label, labels=[0, 1, 2])
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    for row in range(3):
        for col in range(3):
            confusion_rows.append(dict(group=name, true_class=CLASS_NAMES[row], predicted_class=CLASS_NAMES[col],
                                       count=int(cm[row, col]), row_fraction=float(normalized[row, col])))
    for number, matrix in enumerate((cm, normalized)):
        ax = axes_cm[number, column]
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if number else None)
        for row in range(3):
            for col in range(3):
                ax.text(col, row, f"{matrix[row, col]:.2f}" if number else str(matrix[row, col]),
                        ha="center", va="center", color="white" if matrix[row, col] > matrix.max() * 0.55 else "black")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES.values(), yticklabels=CLASS_NAMES.values(),
               xlabel="Predicted", ylabel="True", title=f"{name} | {'row fractions' if number else 'counts'}")
        ax.tick_params(axis="x", rotation=30)
    ax = axes_roc[column]
    for label, class_name in CLASS_NAMES.items():
        truth = frame.true_label.to_numpy() == label
        if len(np.unique(truth)) < 2:
            continue
        scores = frame[PROB_COLUMNS[label]].to_numpy()
        fpr, tpr, thresholds = roc_curve(truth, scores)
        ax.plot(fpr, tpr, label=f"{class_name}: {roc_auc_score(truth, scores):.3f}")
        roc_rows.extend(dict(group=name, tumor_class=class_name, fpr=float(x), tpr=float(y),
                             threshold=float(t) if np.isfinite(t) else None) for x, y, t in zip(fpr, tpr, thresholds))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=name, xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8, loc="lower right")
fig_cm.tight_layout()
fig_roc.tight_layout()
save_figure(fig_cm, "confusion_matrices")
save_figure(fig_roc, "roc_curves")
pd.DataFrame(confusion_rows).to_csv(STUDY_DIR / "confusion_matrices.csv", index=False)
pd.DataFrame(roc_rows).to_csv(STUDY_DIR / "roc_coordinates.csv", index=False)

# %% [markdown]
# ## 12. Qualitative Segmentation Examples
# One sample per class × plane, selected with seed 42 **without looking at predictions or
# Dice**. The table records every selected ID. Green=GT contour; red=predicted contour.
# Missing contours (for an empty prediction/processed mask) are not replaced or hidden.

# %%
selected = INDEX.groupby(["true_label", "plane"], group_keys=False).sample(n=1, random_state=SEED)
selected = selected.sort_values(["true_label", "plane"])
selected[["sample_id", "tumor_class", "plane"]].to_csv(STUDY_DIR / "qualitative_sample_ids.csv", index=False)
fig, axes = plt.subplots(len(selected), 4, figsize=(12, 2.6 * len(selected)))
for row_number, (index, row) in enumerate(selected.iterrows()):
    prediction = PREDICTIONS.iloc[index]
    for ax in axes[row_number]:
        ax.axis("off")
    axes[row_number, 0].imshow(IMAGES[index], cmap="gray", vmin=0, vmax=255)
    axes[row_number, 0].set_title(f"{row.tumor_class} | {row.plane}\n{row.sample_id}", fontsize=8)
    axes[row_number, 1].imshow(MASKS[index], cmap="gray", vmin=0, vmax=1)
    axes[row_number, 1].set_title("Ground truth")
    axes[row_number, 2].imshow(ENSEMBLE_MASKS[index], cmap="gray", vmin=0, vmax=1)
    axes[row_number, 2].set_title(f"Predicted | Dice={prediction.dice:.3f}, IoU={prediction.iou:.3f}")
    ax = axes[row_number, 3]
    ax.imshow(IMAGES[index], cmap="gray", vmin=0, vmax=255)
    for mask, color in ((MASKS[index], "lime"), (ENSEMBLE_MASKS[index], "red")):
        if mask.any() and not mask.all():
            ax.contour(mask, levels=[0.5], colors=[color], linewidths=0.8)
    ax.set_title(f"GT green / prediction red\nClass prediction: {CLASS_NAMES[prediction.predicted_label]}", fontsize=9)
fig.tight_layout()
save_figure(fig, "qualitative_segmentation")

# %% [markdown]
# ## 13. Segmentation Distributions and Crop Audit
# Display all outcomes, including failures. Crop-retention audits are descriptive; samples
# are not removed to improve scores. Metrics refer to the 224×224 cropped field of view,
# so any tumor cropped out is explicitly counted in `preprocessing_audit.csv`.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, group_by, names in ((axes[0], "tumor_class", list(CLASS_NAMES.values())), (axes[1], "plane", PLANES)):
    ax.boxplot([PREDICTIONS.loc[PREDICTIONS[group_by] == name, "dice"] for name in names], tick_labels=names)
    ax.set(ylabel="Per-slice Dice", ylim=(-0.03, 1.03), title=f"Dice by {group_by.replace('_', ' ')}")
fig.tight_layout()
save_figure(fig, "dice_distributions")
display(INDEX.groupby(["tumor_class", "plane"]).agg(n_slices=("sample_id", "size"),
        minimum_retention=("tumor_preserved_ratio", "min"), mean_retention=("tumor_preserved_ratio", "mean")))

# %% [markdown]
# ### 13.1 Original Field-of-View Sensitivity Analysis
# Cropping can remove GT tumor pixels. The primary scores above reproduce MAGS-Net's
# cropped-grid convention, including both-empty=1. As a supplementary check, map each
# predicted binary mask back to the original image with nearest-neighbor interpolation and
# fill everything outside the crop with background. Score against the complete original GT.
# This does not alter model inputs or tune predictions. A missed tumor outside the crop now
# counts as a false negative. Pixel-grid differences mean these scores need not equal the
# primary 224×224 scores even when all tumor pixels were retained.

# %%
native_rows = []
for index, row in INDEX.iterrows():
    image, original_gt, _ = read_brisc_pair(row)
    x1, y1, x2, y2 = (int(row[key]) for key in ("crop_x1", "crop_y1", "crop_x2", "crop_y2"))
    restored = np.zeros_like(original_gt, dtype=bool)
    restored[y1:y2, x1:x2] = cv2.resize(ENSEMBLE_MASKS[index].astype(np.uint8), (x2 - x1, y2 - y1),
                                       interpolation=cv2.INTER_NEAREST) > 0
    dice, iou = segmentation_scores(restored[None], original_gt.astype(bool)[None])
    native_rows.append(dict(sample_id=row.sample_id, dice_original_fov=dice[0], iou_original_fov=iou[0]))
NATIVE_RESULTS = PREDICTIONS.merge(pd.DataFrame(native_rows), on="sample_id", validate="one_to_one")
NATIVE_RESULTS.to_csv(STUDY_DIR / "original_fov_predictions.csv", index=False)
native_frame = NATIVE_RESULTS.copy()
native_frame["dice"], native_frame["iou"] = native_frame.dice_original_fov, native_frame.iou_original_fov
NATIVE_CI = bootstrap_metrics(native_frame).query("metric in ['mean_dice', 'mean_iou']").copy()
NATIVE_CI["coordinate_system"] = "original image; zero prediction outside brain crop"
NATIVE_CI.to_csv(STUDY_DIR / "original_fov_segmentation_ci.csv", index=False)
crop_failures = NATIVE_RESULTS.loc[NATIVE_RESULTS.tumor_preserved_ratio < 0.999]
crop_failures.to_csv(STUDY_DIR / "crop_retention_failures.csv", index=False)
display(NATIVE_CI)
display(crop_failures[["sample_id", "tumor_preserved_ratio", "dice", "dice_original_fov"]])

# %% [markdown]
# ### 13.2 Source Similarity Examples for Review
# Show the four strongest **raw-image** candidates, independent of model predictions.
# The BRISC image is rotated/reflected only for this provenance visualization, never for
# model inference. High correlation is a screening result, not a patient-identity label.

# %%
if "raw_correlation" in OVERLAP:
    candidates = OVERLAP.nlargest(4, "raw_correlation")
    fig, axes = plt.subplots(len(candidates), 3, figsize=(10, 3 * len(candidates)))
    for number, row in enumerate(candidates.itertuples()):
        external_row = INDEX.loc[INDEX.sample_id == row.sample_id].iloc[0]
        external, _, _ = read_brisc_pair(external_row)
        source_row = SOURCE_INDEX.loc[SOURCE_INDEX.sample_id == row.raw_source_sample_id].iloc[0]
        with h5py.File(PROJECT_ROOT / "brainDataset" / source_row.original_mat, "r") as handle:
            reference = normalize_to_uint8(np.asarray(handle["cjdata/image"]).squeeze())
        transformed = np.rot90(normalize_to_uint8(external), int(row.raw_transform) % 4)
        if row.raw_transform >= 4:
            transformed = transformed[:, ::-1]
        for ax, image, title in zip(axes[number], (external, transformed, reference),
                (f"BRISC: {row.sample_id}", f"Aligned for review | r={row.raw_correlation:.5f}",
                 f"MAGS source: {source_row.original_mat}")):
            ax.imshow(image, cmap="gray")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    save_figure(fig, "source_overlap_examples")

# %% [markdown]
# ## 14. Save Final Results and Interpretation
# The CSVs contain reusable predictions, class/plane summaries, confidence intervals,
# confusion/ROC data, and audits. Saved probabilities permit inspection without rerunning
# the models. Results apply only to the three tumor classes in the official BRISC test set;
# no claim is made about no-tumor detection, patient-level performance, or independent
# clinical validation. Source overlap must be resolved before calling this an independent
# external cohort in a paper.

# %%
save_json(STUDY_DIR / "environment.json", dict(python=platform.python_version(), torch=torch.__version__,
          numpy=np.__version__, pandas=pd.__version__, opencv=cv2.__version__,
          device=str(DEVICE), gpu=torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None))
save_json(STUDY_DIR / "evaluation_complete.json", dict(protocol_hash=signature(PROTOCOL),
          evaluated_slices=len(PREDICTIONS), duplicate_sample_ids=int(PREDICTIONS.sample_id.duplicated().sum()),
          training_performed=False, fine_tuning_performed=False, inference_models=5,
          primary_model="equal-probability ensemble", bootstrap_unit="slice", bootstrap_samples=CONFIG["bootstrap_samples"],
          independence_audit=INDEPENDENCE_AUDIT))
files = [p for p in STUDY_DIR.rglob("*") if p.is_file() and p.name != "result_inventory.csv"]
pd.DataFrame([dict(file=str(p.relative_to(STUDY_DIR)), bytes=p.stat().st_size, sha256=file_hash(p))
              for p in sorted(files)]).to_csv(STUDY_DIR / "result_inventory.csv", index=False)
print("Saved evaluation:", STUDY_DIR)
print("Interpretation: exploratory cross-dataset evaluation; independent patients are not established.")
display(OVERALL_CI)

# %% [markdown]
# ## 15. BRISC Cohort Curation: Scope and Rules
# **All BRISC samples → potential Cheng overlap → ambiguous samples → source-disjoint
# candidates → final clean external subset → classification + segmentation → 95% CI →
# class-wise / plane-wise results.**
# #
# The inventory starts with all available tumor image/mask pairs. The analysis continues
# to use the **official test split only**, as specified for this experiment; BRISC train
# images are counted but are not added to the evaluation. No source files are deleted.
# #
# These analyses were added after the initial all-test results were available. They are
# **post-hoc, exploratory**. Selection uses image/source audits and documented provenance,
# never predicted labels, confidence, Dice, IoU, or crop tumor-retention outcomes.
# The correlation cutoffs below are screening heuristics, not validated identity tests.
# #
# | Stage | Rule |
# | --- | --- |
# | Potential Cheng overlap | Exact source-image match, correlation >=0.995, or documented Cheng overlap |
# | Ambiguous | Correlation 0.980–0.995, missing source-audit evidence, all members of a duplicate processed-image group, or unresolved review |
# | Source-disjoint candidates | Remaining test images; low similarity is **not proof** of independence |
# | Final clean external subset | Candidate with documented sample-to-source linkage AND source/patient disjointness verified by a named reviewer |
# #
# Keeping duplicate-group members on hold avoids counting repeated copies as independent
# slices. Threshold sensitivity reports counts only; no threshold is selected by accuracy.
# Run sections 0–2 first, then this section onward to reuse saved inference without a model call.

# %%
CURATION_CONFIG = dict(version="brisc_curation_v1", split="test", potential_overlap_threshold=0.995,
                       ambiguous_threshold=0.980, duplicate_policy="hold_all_processed_image_duplicates",
                       bootstrap_samples=2000, confidence_level=0.95, seed=42,
                       final_rule="candidate_plus_documented_source_and_patient_disjointness")
CURATION_ROOT = STUDY_DIR / "curation"
CURATION_ROOT.mkdir(exist_ok=True)
REVIEW_PATH = CURATION_ROOT / "provenance_review.csv"
SAVED_PROTOCOL = json.loads((STUDY_DIR / "protocol.json").read_text())
SAVED_COMPLETION = json.loads((STUDY_DIR / "inference_complete.json").read_text())
assert SAVED_COMPLETION["protocol_hash"] == signature(SAVED_PROTOCOL)
for name, expected in SAVED_COMPLETION["file_hashes"].items():
    assert file_hash(STUDY_DIR / name) == expected, f"Saved inference was changed: {name}"

CURATION_INDEX = pd.read_csv(STUDY_DIR / "test_index.csv")
CURATION_OVERLAP = pd.read_csv(STUDY_DIR / "source_overlap_audit.csv")
CURATION_PREPROCESSING = pd.read_csv(STUDY_DIR / "preprocessing_audit.csv")
CURATION_SOURCE_AUDIT = json.loads((STUDY_DIR / "independence_audit.json").read_text())
assert CURATION_INDEX.sample_id.is_unique and CURATION_INDEX.split.eq("test").all()
assert len(CURATION_INDEX) == SAVED_COMPLETION["rows"]
for audit in (CURATION_OVERLAP, CURATION_PREPROCESSING):
    assert audit.sample_id.is_unique and set(audit.sample_id) == set(CURATION_INDEX.sample_id)
SCREEN = CURATION_INDEX.merge(CURATION_OVERLAP.drop(columns=["tumor_class", "plane"]),
                             on="sample_id", validate="one_to_one")
SCREEN = SCREEN.merge(CURATION_PREPROCESSING[["sample_id", "processed_image_hash"]],
                      on="sample_id", validate="one_to_one")
SCREEN["duplicate_processed_image"] = SCREEN.processed_image_hash.duplicated(keep=False)
REFERENCE_COMPLETE = (CURATION_SOURCE_AUDIT["source_slices"] == 3064
                      and CURATION_SOURCE_AUDIT["preprocessing_parity_samples"] == 3064)
manifest = pd.DataFrame(json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8")))
available = manifest.loc[(manifest.task == "segmentation") & ~manifest.is_mask].copy()
assert available.relative_path.is_unique
INVENTORY = available.groupby(["split", "tumor_label"]).size().rename("n_slices").reset_index()
print("Available segmentation pairs:", len(available), "| Official test:", len(SCREEN))
display(INVENTORY)

# %% [markdown]
# ## 16. Provenance Review and Final-Subset Membership
# `provenance_review.csv` is an editable, prediction-blind review template. Existing entries
# are preserved. Use `pending`, `cheng_overlap`, `unresolved`, or `source_disjoint`.
# A `source_disjoint` claim requires a source dataset, original record ID, traceable evidence,
# reviewer, UTC date, and three explicit verification fields. A different dataset name or a
# low similarity score alone is insufficient. The reviewer must verify the evidence; this
# notebook checks completeness and contradictions, not the truth of an external document.
# Never fill review fields merely to enable the evaluation cells.
# #
# Automated overlap/ambiguity flags cannot be overridden by a positive provenance entry.
# Review conflicts are reported and held out. With no supporting provenance, the final clean
# subset remains empty and its metrics remain unavailable. Candidate metrics are saved under
# a separate, explicitly exploratory name.

# %%
REVIEW_FIELDS = ["sample_id", "decision", "source_dataset", "source_record_id", "evidence_reference",
                 "reviewer", "reviewed_utc", "source_linkage_verified", "non_cheng_source_verified",
                 "patient_disjoint_verified", "notes"]
if not REVIEW_PATH.exists():
    template = pd.DataFrame({column: "" for column in REVIEW_FIELDS}, index=range(len(SCREEN)))
    template["sample_id"] = SCREEN.sample_id.to_numpy()
    template["decision"] = "pending"
    template.to_csv(REVIEW_PATH, index=False)
REVIEW = pd.read_csv(REVIEW_PATH, dtype=str, keep_default_na=False)
assert set(REVIEW_FIELDS).issubset(REVIEW.columns)
assert REVIEW.sample_id.is_unique and set(REVIEW.sample_id) == set(SCREEN.sample_id)
for column in REVIEW_FIELDS:
    REVIEW[column] = REVIEW[column].str.strip()
assert REVIEW.decision.isin(["pending", "cheng_overlap", "unresolved", "source_disjoint"]).all()


def assign_curation_status(screen, reviews, config, reference_complete):
    frame = screen.merge(reviews[REVIEW_FIELDS], on="sample_id", validate="one_to_one")
    text_fields = ["source_dataset", "source_record_id", "evidence_reference", "reviewer", "reviewed_utc"]
    flags = ["source_linkage_verified", "non_cheng_source_verified", "patient_disjoint_verified"]
    valid_date = pd.to_datetime(frame.reviewed_utc, utc=True, errors="coerce").notna()
    for field in flags:
        assert frame[field].str.lower().isin(["", "true", "false"]).all(), field
    documented = frame[text_fields].ne("").all(axis=1) & valid_date
    for field in flags:
        documented &= frame[field].str.lower().eq("true")
    exact = frame.processed_exact_source_match.astype(str).str.lower().eq("true")
    raw = pd.to_numeric(frame.get("raw_correlation", pd.Series(np.nan, index=frame.index)), errors="coerce")
    cropped = pd.to_numeric(frame.cropped_correlation, errors="coerce")
    score = pd.concat([raw, cropped], axis=1).max(axis=1)
    frame["screening_correlation"] = score
    evidence_missing = ~np.isfinite(raw) | ~np.isfinite(cropped) | (not reference_complete)
    potential = exact | score.ge(config["potential_overlap_threshold"]) | frame.decision.eq("cheng_overlap")
    ambiguous = (~potential) & (evidence_missing | score.ge(config["ambiguous_threshold"])
                                | frame.duplicate_processed_image | frame.decision.eq("unresolved"))
    candidate = ~potential & ~ambiguous
    frame["screen_status"] = np.select([potential, ambiguous],
                                      ["potential_cheng_overlap", "ambiguous"], default="source_disjoint_candidate")
    frame["reason"] = np.select(
        [exact, frame.decision.eq("cheng_overlap"), score.ge(config["potential_overlap_threshold"]),
         evidence_missing, frame.duplicate_processed_image, frame.decision.eq("unresolved"),
         score.ge(config["ambiguous_threshold"])],
        ["exact processed source match", "review reports Cheng overlap", "high source correlation",
         "incomplete reference or correlation audit", "duplicate processed-image identity unresolved",
         "review remains unresolved", "intermediate source correlation"], default="low correlation; provenance still required")
    frame["provenance_complete"] = documented
    frame["review_conflict"] = frame.decision.eq("source_disjoint") & (~candidate | ~documented)
    frame["final_clean_member"] = candidate & frame.decision.eq("source_disjoint") & documented
    return frame


CURATED = assign_curation_status(SCREEN, REVIEW, CURATION_CONFIG, REFERENCE_COMPLETE)
CURATION_PROTOCOL = dict(config=CURATION_CONFIG, source_inference_hash=SAVED_COMPLETION["protocol_hash"],
                         source_audit_hash=file_hash(STUDY_DIR / "source_overlap_audit.csv"),
                         index_hash=file_hash(STUDY_DIR / "test_index.csv"),
                         preprocessing_hash=file_hash(STUDY_DIR / "preprocessing_audit.csv"),
                         reference_audit_hash=file_hash(STUDY_DIR / "independence_audit.json"),
                         manifest_hash=file_hash(DATA_ROOT / "manifest.json"),
                         review_hash=file_hash(REVIEW_PATH), analysis_timing="post-hoc exploratory")
CURATION_DIR = CURATION_ROOT / signature(CURATION_PROTOCOL)[:12]
CURATION_DIR.mkdir(exist_ok=True)
save_json(CURATION_DIR / "curation_protocol.json", CURATION_PROTOCOL)
REVIEW.to_csv(CURATION_DIR / "provenance_review_snapshot.csv", index=False)
CURATED.to_csv(CURATION_DIR / "sample_curation_ledger.csv", index=False)
INVENTORY.to_csv(CURATION_DIR / "available_brisc_inventory.csv", index=False)
for status in ("potential_cheng_overlap", "ambiguous", "source_disjoint_candidate"):
    CURATED.loc[CURATED.screen_status == status].to_csv(CURATION_DIR / f"{status}_samples.csv", index=False)
CURATED.loc[CURATED.final_clean_member].to_csv(CURATION_DIR / "final_clean_external_subset.csv", index=False)
CURATED.loc[CURATED.review_conflict].to_csv(CURATION_DIR / "review_conflicts.csv", index=False)
save_json(CURATION_ROOT / "latest_curation.json", dict(directory=CURATION_DIR.name))
print("Curation outputs:", CURATION_DIR)
display(CURATED.groupby(["screen_status", "tumor_class", "plane"]).size().rename("n_slices").reset_index())
print("Final clean subset:", int(CURATED.final_clean_member.sum()), "| Review conflicts:", int(CURATED.review_conflict.sum()))

# %% [markdown]
# ## 17. Cohort Flow Diagram and Threshold Sensitivity
# Counts reconcile exactly: potential overlap + ambiguous + candidates = official test.
# Final-clean membership is a subset of candidates. Unreviewed candidates do not silently
# become clean samples. The chart separates cohort construction from the analysis steps.

# %%
from matplotlib.patches import FancyBboxPatch

N_ALL = len(available)
N_TEST = len(CURATED)
N_OVERLAP = int(CURATED.screen_status.eq("potential_cheng_overlap").sum())
N_AMBIGUOUS = int(CURATED.screen_status.eq("ambiguous").sum())
N_CANDIDATE = int(CURATED.screen_status.eq("source_disjoint_candidate").sum())
N_CLEAN = int(CURATED.final_clean_member.sum())
assert N_OVERLAP + N_AMBIGUOUS + N_CANDIDATE == N_TEST and N_CLEAN <= N_CANDIDATE
FLOW = pd.DataFrame([
    dict(stage="All BRISC samples (available tumor image/mask pairs)", n=N_ALL, remaining=N_ALL),
    dict(stage="Official test split retained", n=N_TEST, remaining=N_TEST),
    dict(stage="Potential Cheng overlap excluded", n=N_OVERLAP, remaining=N_TEST - N_OVERLAP),
    dict(stage="Ambiguous samples held out", n=N_AMBIGUOUS, remaining=N_CANDIDATE),
    dict(stage="Source-disjoint candidates (unverified)", n=N_CANDIDATE, remaining=N_CANDIDATE),
    dict(stage="Final clean external subset (documented review)", n=N_CLEAN, remaining=N_CLEAN)])
FLOW.to_csv(CURATION_DIR / "cohort_flow_counts.csv", index=False)
display(FLOW)
sensitivity = []
for lower in (0.950, 0.970, 0.980, 0.990):
    alternate = dict(CURATION_CONFIG, ambiguous_threshold=lower)
    assigned = assign_curation_status(SCREEN, REVIEW, alternate, REFERENCE_COMPLETE)
    sensitivity.append(dict(ambiguous_threshold=lower, potential_threshold=0.995,
                            potential=int(assigned.screen_status.eq("potential_cheng_overlap").sum()),
                            ambiguous=int(assigned.screen_status.eq("ambiguous").sum()),
                            candidates=int(assigned.screen_status.eq("source_disjoint_candidate").sum())))
pd.DataFrame(sensitivity).to_csv(CURATION_DIR / "screening_threshold_sensitivity.csv", index=False)

nodes = [
    ("All BRISC samples", f"{N_ALL:,} available tumor image/mask pairs", "#e7f0ff"),
    ("Official test split", f"n={N_TEST:,}; train n={N_ALL - N_TEST:,} outside evaluation", "#e7f0ff"),
    ("Potential Cheng overlap", f"Exclude n={N_OVERLAP}; remaining n={N_TEST - N_OVERLAP}", "#fde8e7"),
    ("Ambiguous samples", f"Hold n={N_AMBIGUOUS}; remaining n={N_CANDIDATE}", "#fff1cf"),
    ("Source-disjoint candidates", f"n={N_CANDIDATE}; independence unverified", "#fff1cf"),
    ("Final clean external subset", f"n={N_CLEAN}; pending provenance n={N_CANDIDATE - N_CLEAN}", "#e4f1e8" if N_CLEAN else "#eeeeee"),
    ("Classification + Segmentation", "Evaluate verified subset" if N_CLEAN else "Final-subset evaluation pending", "#eeeeee"),
    ("95% CI", "2,000 slice-bootstrap resamples; exploratory" if N_CLEAN else "Pending a verified nonempty cohort", "#eeeeee"),
    ("Class-wise / plane-wise results", "Missing classes shown as unavailable", "#eeeeee")]
fig, ax = plt.subplots(figsize=(8.2, 14.2))
ax.set(xlim=(0, 1), ylim=(-0.6, len(nodes)))
ax.axis("off")
for number, (title, detail, color) in enumerate(nodes):
    y = len(nodes) - number - 1
    ax.add_patch(FancyBboxPatch((0.07, y), 0.86, 0.73, boxstyle="round,pad=0.02",
                                facecolor=color, edgecolor="#566579", linewidth=1))
    ax.text(0.5, y + 0.47, title, ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.5, y + 0.19, detail, ha="center", va="center", fontsize=10)
    if number < len(nodes) - 1:
        ax.annotate("", xy=(0.5, y - 0.25), xytext=(0.5, y - 0.03), arrowprops=dict(arrowstyle="->", color="#566579"))
ax.text(0.5, -0.35, f"Separate candidate analysis: n={N_CANDIDATE}; not a verified external cohort.",
        ha="center", va="center", fontsize=10)
fig.tight_layout()
for extension in ("png", "pdf"):
    fig.savefig(CURATION_DIR / f"cohort_flow.{extension}", dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)

# %% [markdown]
# ## 18. Subset Classification, Segmentation and Bootstrap 95% CI
# Frozen predictions are selected by sample ID; there is no new training or inference.
# Always save the exploratory candidate analysis separately from the verified final subset.
# An empty final subset writes explicit `not_evaluated` status and NA metrics, not zero scores.
# #
# A three-class macro-F1 or macro-AUC is reported only if all labels 0,1,2 are present.
# Within a single-class cohort, accuracy equals that class's recall. Class precision/F1/OvR
# AUC require both positive and negative cases. Segmentation remains evaluable per slice.
# Absent classes/planes have n=0 and NA results. Small strata and unknown patient dependence
# limit inference even when a percentile interval can be computed.
# #
# Use 2,000 class-stratified slice resamples, seed 42. Each sampled row carries both tasks
# together. Keep cropped-grid and original-field Dice/IoU; do not hide crop failures.

# %%
def subset_point_estimates(frame):
    rows = []
    labels = frame.true_label.to_numpy(dtype=int)
    prediction = frame.predicted_label.to_numpy(dtype=int)
    probability = frame[PROB_COLUMNS].to_numpy(dtype=float)
    count = len(frame)
    present = set(labels)
    three_classes = present == {0, 1, 2}
    metrics = dict(accuracy=float((labels == prediction).mean()) if count else np.nan,
                   macro_f1=f1_score(labels, prediction, labels=[0, 1, 2], average="macro", zero_division=0) if three_classes else np.nan,
                   auc_macro_ovr=roc_auc_score(labels, probability, labels=[0, 1, 2], multi_class="ovr") if three_classes else np.nan,
                   mean_dice=frame.dice.mean(), mean_iou=frame.iou.mean(),
                   mean_dice_original_fov=frame.dice_original_fov.mean(), mean_iou_original_fov=frame.iou_original_fov.mean())
    for name, value in metrics.items():
        reason = "empty subset" if not count else ("requires all three true classes" if name in ("macro_f1", "auc_macro_ovr") and not three_classes else "")
        rows.append(dict(scope="overall", tumor_class="all", metric=name, estimate=value, n_slices=count, unavailable_reason=reason))
    for label, name in CLASS_NAMES.items():
        truth, predicted = labels == label, prediction == label
        positives = int(truth.sum())
        has_both = 0 < positives < count
        tp = int((truth & predicted).sum())
        group = frame.loc[truth]
        values = dict(recall=tp / positives if positives else np.nan,
                      precision=tp / predicted.sum() if has_both and predicted.any() else np.nan,
                      f1=2 * tp / (positives + predicted.sum()) if has_both else np.nan,
                      auc_ovr=roc_auc_score(truth, probability[:, label]) if has_both else np.nan,
                      mean_dice=group.dice.mean(), mean_iou=group.iou.mean(),
                      mean_dice_original_fov=group.dice_original_fov.mean(), mean_iou_original_fov=group.iou_original_fov.mean())
        for metric, value in values.items():
            reason = "class absent" if not positives else ("requires positive and negative cases" if metric in ("precision", "f1", "auc_ovr") and not has_both else "")
            if metric == "precision" and has_both and not predicted.any():
                reason = "no predicted positives"
            rows.append(dict(scope="class", tumor_class=name, metric=metric, estimate=value,
                             n_slices=positives, unavailable_reason=reason))
    return pd.DataFrame(rows)


def subset_bootstrap(frame, repeats=None):
    repeats = CURATION_CONFIG["bootstrap_samples"] if repeats is None else repeats
    result = subset_point_estimates(frame)
    result["ci_low"], result["ci_high"], result["valid_resamples"] = np.nan, np.nan, 0
    if len(frame):
        rng = np.random.default_rng(CURATION_CONFIG["seed"])
        groups = [np.flatnonzero(frame.true_label.to_numpy() == label) for label in sorted(frame.true_label.unique())]
        draws = []
        for _ in range(repeats):
            positions = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
            draws.append(subset_point_estimates(frame.iloc[positions]).estimate.to_numpy())
        draws = np.stack(draws)
        alpha = (1 - CURATION_CONFIG["confidence_level"]) / 2
        for column in range(len(result)):
            finite = draws[:, column][np.isfinite(draws[:, column])]
            result.loc[column, "valid_resamples"] = len(finite)
            if len(finite) and np.isfinite(result.loc[column, "estimate"]):
                result.loc[column, ["ci_low", "ci_high"]] = np.quantile(finite, [alpha, 1 - alpha])
    result["bootstrap_unit"] = "slice"
    result["interpretation"] = "exploratory; patient dependence unknown"
    return result


def evaluate_curated_subset(membership, name):
    folder = CURATION_DIR / name
    folder.mkdir(exist_ok=True)
    saved = pd.read_csv(STUDY_DIR / "original_fov_predictions.csv")
    assert saved.sample_id.is_unique and set(saved.sample_id) == set(CURATION_INDEX.sample_id)
    # Original-field data must preserve the checksum-verified inference columns.
    frozen = pd.read_csv(STUDY_DIR / "MAGSNet_BRISC_test_predictions.csv")
    pd.testing.assert_frame_equal(saved[frozen.columns], frozen, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    frame = saved.merge(membership[["sample_id"]], on="sample_id", validate="one_to_one").sort_values("sample_id").reset_index(drop=True)
    assert len(frame) == len(membership)
    frame.to_csv(folder / "predictions.csv", index=False)
    results = [subset_bootstrap(frame).assign(plane="all")]
    for plane in PLANES:
        results.append(subset_bootstrap(frame.loc[frame.plane == plane]).assign(plane=plane))
    results = pd.concat(results, ignore_index=True)
    results["subset"] = name
    results.to_csv(folder / "metrics_ci.csv", index=False)
    for filename, scope, plane_type in (("overall", "overall", "all"), ("class_wise", "class", "all"),
                                        ("plane_wise", "overall", "planes"), ("class_plane_wise", "class", "planes")):
        selected = results.scope.eq(scope) & (results.plane.eq("all") if plane_type == "all" else results.plane.ne("all"))
        results.loc[selected].to_csv(folder / f"{filename}_metrics_ci.csv", index=False)
    status = dict(subset=name, status="evaluated" if len(frame) else "not_evaluated",
                   n_slices=len(frame), labels_present=sorted(int(x) for x in frame.true_label.unique()),
                   three_class_classification_available=set(frame.true_label) == {0, 1, 2},
                   independent_external_validation_claim=False,
                   reason="documented clean subset empty" if not len(frame) else "post-hoc analysis; review status and missing classes must be reported")
    save_json(folder / "status.json", status)
    return frame, results, status


CANDIDATE_MEMBERS = CURATED.loc[CURATED.screen_status == "source_disjoint_candidate"]
CLEAN_MEMBERS = CURATED.loc[CURATED.final_clean_member]
CANDIDATE_PREDICTIONS, CANDIDATE_METRICS, CANDIDATE_STATUS = evaluate_curated_subset(CANDIDATE_MEMBERS, "candidate_exploratory")
CLEAN_PREDICTIONS, CLEAN_METRICS, CLEAN_STATUS = evaluate_curated_subset(CLEAN_MEMBERS, "final_clean_external")
display(pd.DataFrame([CANDIDATE_STATUS, CLEAN_STATUS]))
display(CANDIDATE_METRICS.query("scope == 'overall' and plane == 'all'")[["metric", "estimate", "ci_low", "ci_high", "unavailable_reason"]])
if len(CLEAN_PREDICTIONS):
    display(CLEAN_METRICS.query("scope == 'overall' and plane == 'all'"))
else:
    print("FINAL CLEAN EXTERNAL SUBSET: 0 verified samples. Final-subset scores and CIs are unavailable.")

# %% [markdown]
# ## 19. Class-Wise / Plane-Wise Tables and Saved Curation Results
# The candidate table is shown separately; it must not be relabeled as a clean external
# validation table. NaN means unavailable, not zero performance. Re-running after documented
# provenance changes creates a new result directory and preserves earlier cohort definitions.
# To review identities, use `sample_curation_ledger.csv` and `provenance_review.csv`.

# %%
SUPPORT = pd.crosstab(CANDIDATE_PREDICTIONS.tumor_class, CANDIDATE_PREDICTIONS.plane).reindex(
    index=list(CLASS_NAMES.values()), columns=PLANES, fill_value=0)
SUPPORT.to_csv(CURATION_DIR / "candidate_class_plane_support.csv")
display(SUPPORT)
display(CANDIDATE_METRICS.query("scope == 'class' and plane == 'all' and metric in ['recall', 'mean_dice', 'mean_iou']")
        [["tumor_class", "metric", "n_slices", "estimate", "ci_low", "ci_high", "unavailable_reason"]])
display(CANDIDATE_METRICS.query("scope == 'overall' and plane != 'all' and metric in ['accuracy', 'mean_dice', 'mean_iou']")
        [["plane", "metric", "n_slices", "estimate", "ci_low", "ci_high"]])
save_json(CURATION_DIR / "curation_complete.json", dict(curation_hash=signature(CURATION_PROTOCOL),
          available_brisc_pairs=N_ALL, official_test=N_TEST, potential_overlap=N_OVERLAP,
          ambiguous=N_AMBIGUOUS, candidates=N_CANDIDATE, final_clean=N_CLEAN,
          candidate_status=CANDIDATE_STATUS, clean_status=CLEAN_STATUS,
          training_performed=False, new_model_inference=False,
          pending_requirement="traceable source and patient-disjointness evidence" if not N_CLEAN else ""))
files = [path for path in CURATION_DIR.rglob("*") if path.is_file() and path.name != "result_inventory.csv"]
pd.DataFrame([dict(file=str(path.relative_to(CURATION_DIR)), bytes=path.stat().st_size, sha256=file_hash(path))
              for path in sorted(files)]).to_csv(CURATION_DIR / "result_inventory.csv", index=False)
print("Saved curation, candidate evaluation, final-subset status and figure:", CURATION_DIR)
