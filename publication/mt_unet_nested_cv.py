# %% [markdown]
# # 03. Lightweight MT-U-Net: Nested Patient-Level Cross-Validation
# A standard U-Net with a bottleneck classification head, trained from scratch on paired
# MRI images and tumor masks. All explanations and code are in English.
# Encoder channels are 32–256 with a 512-channel bottleneck. This narrower U-Net retains
# the standard encoder/decoder structure; report its measured size rather than assuming it matches MAGS-Net.
# No attention, gates, tumor-aware pooling, multiscale branches, or deep supervision.
# Training and outer-test inference are disabled by default.

# %% [markdown]
# ## 0. Experiment Configuration
# One seed, 25 inner fits, and five fresh outer refits: 30 fits in total. Every inner fit
# selects its best epoch by total validation loss. MT-U-Net selects its own durations.

# %%
CONFIG = {
    "protocol": "lightweight_mt_unet_nested_patient_cv_v2", "seed": 42,
    "image_size": 224, "batch_size": 32, "channels": [32, 64, 128, 256, 512],
    "learning_rate": 0.001, "weight_decay": 1e-4, "max_epochs": 50, "patience": 10,
    "scheduler_step": 10, "scheduler_gamma": 0.5, "lambda_seg": 0.5, "dice_smooth": 1.0,
    "mask_threshold": 0.5, "use_amp": True, "gradient_clip": 1.0,
}
MODEL_NAME = "MT-U-Net"
OBJECTIVE = "CrossEntropy + 0.5 * (BCE + Dice)"
CLASS_NAMES = {0: "Meningioma", 1: "Glioma", 2: "Pituitary"}
PROBABILITY_COLUMNS = ["prob_meningioma", "prob_glioma", "prob_pituitary"]
MAGS_STUDY_DIR = "results/magsnet_nested_cv/20260916T210025Z_bc27ff71"
STUDY_DIR = None  # Set your own completed study path for analysis.
RUN_TRAINING = False
RUN_OUTER_TEST = False

# %% [markdown]
# ## 1. Imports and Environment
# Use the Conda `dl` kernel. No pretrained weights, package installation, or downloads.

# %%
import hashlib
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENVIRONMENT = {"python": platform.python_version(), "torch": str(torch.__version__),
               "numpy": np.__version__, "pandas": pd.__version__, "opencv": cv2.__version__,
               "device": str(DEVICE), "gpu": torch.cuda.get_device_name() if DEVICE.type == "cuda" else None}
print(ENVIRONMENT)

def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def signature(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# %% [markdown]
# ## 2. Reproducibility and Paths
# Reset seed 42 before every independent fit. Results are written to a separate MT-U-Net
# study; the MAGS-Net and classification-baseline experiments are read-only.

# %%
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cv2.setNumThreads(0)

set_seed(CONFIG["seed"])
PROJECT_ROOT = next((p for p in (Path.cwd(), Path.cwd() / "ZubNet-B", Path.cwd().parent)
                     if (p / "brainDataset_cropped").is_dir()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Place the notebook beside brainDataset_cropped or update PROJECT_ROOT.")
DATA_ROOT = PROJECT_ROOT / "brainDataset_cropped"
MAGS_STUDY = PROJECT_ROOT / MAGS_STUDY_DIR
RESULT_ROOT = PROJECT_ROOT / "results/mt_unet_nested_cv"

# %% [markdown]
# ## 3. Read the Existing MAGS-Net Fold Definitions
# Load saved outer/inner patient CSVs and confirm exact agreement with experiment.json.
# No splitter is created or called. Do not regenerate patient folds for this baseline.

# %%
def load_saved_folds():
    saved = json.loads((MAGS_STUDY / "experiment.json").read_text())
    protocol = saved["protocol"]
    assert signature(protocol) == saved["experiment_hash"]
    assert protocol["classes"] == {str(k): v for k, v in CLASS_NAMES.items()}
    assert protocol["config"]["outer_folds"] == protocol["config"]["inner_folds"] == 5
    assert protocol["config"]["seeds"] == [CONFIG["seed"]]
    folder = PROJECT_ROOT / "results/magsnet_nested_cv/preparation" / saved["experiment_hash"][:12]
    outer = pd.read_csv(folder / "outer_patients.csv", dtype={"patient_id": str})
    inner = pd.read_csv(folder / "inner_patients.csv", dtype={"patient_id": str})
    assert outer.to_dict("records") == protocol["outer"]
    assert inner.to_dict("records") == protocol["inner"]
    index = pd.read_csv(folder / "sample_index.csv", dtype={"sample_id": str, "patient_id": str, "filename": str})
    return saved, outer, inner, index

MAGS_EXPERIMENT, OUTER, INNER, SAVED_INDEX = load_saved_folds()
MAGS_HASH = MAGS_EXPERIMENT["experiment_hash"]
assert MAGS_EXPERIMENT["protocol"]["config"]["image_size"] == CONFIG["image_size"] == 224
assert MAGS_EXPERIMENT["protocol"]["candidates"][0]["lambda_seg"] == CONFIG["lambda_seg"] == 0.5
print("Saved folds:", MAGS_STUDY_DIR, "| 5 outer folds, each with 5 inner folds")

# %% [markdown]
# ## 4. Dataset Integrity, Class Labels, and Leakage Checks
# Verify the exact master metadata and image/mask bytes used by MAGS-Net. Every mask must
# be nonempty, binary, and aligned. Labels: Meningioma=0, Glioma=1, Pituitary=2.

# %%
def audit_dataset():
    path = DATA_ROOT / "metadata_clean.csv"
    frame = pd.read_csv(path, dtype={"sample_id": str, "patient_id": str, "filename": str})
    frame = frame.sort_values("sample_id").reset_index(drop=True)
    keys = ["sample_id", "patient_id", "label", "filename"]
    assert not frame[keys].isna().any().any() and frame.sample_id.is_unique and frame.filename.is_unique
    assert set(frame.label) == {0, 1, 2} and frame.groupby("patient_id").label.nunique().eq(1).all()
    assert frame[keys].to_dict("records") == SAVED_INDEX[keys].to_dict("records")
    records = []
    for row in frame.itertuples():
        assert Path(row.filename).name == row.filename
        image = cv2.imread(str(DATA_ROOT / "images" / row.filename), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(DATA_ROOT / "masks" / row.filename), cv2.IMREAD_UNCHANGED)
        assert image is not None and mask is not None and image.ndim == 2 and image.shape == mask.shape
        assert image.dtype == np.uint8 and set(np.unique(mask)).issubset({0, 1, 255}) and mask.any()
        records.append({"image_hash": hashlib.sha256(image.tobytes()).hexdigest(),
                        "mask_hash": hashlib.sha256(mask.tobytes()).hexdigest(),
                        "height": image.shape[0], "width": image.shape[1]})
    checked = pd.concat([frame, pd.DataFrame(records)], axis=1)
    columns = keys + ["image_hash", "mask_hash", "height", "width"]
    digest = signature(checked[columns].to_dict("records"))
    assert digest == MAGS_EXPERIMENT["protocol"]["data_hash"], "Dataset changed since MAGS-Net training."
    assert checked.groupby("image_hash").patient_id.nunique().le(1).all()
    return frame, digest

# %%
def check_saved_partitions(index, outer, inner):
    assert outer.patient_id.is_unique and set(outer.patient_id) == set(index.patient_id)
    patient_labels = index.groupby("patient_id").label.first()
    assert np.array_equal(outer.label, outer.patient_id.map(patient_labels))
    assert set(outer.outer_fold) == set(range(1, 6)) and set(inner.outer_fold) == set(range(1, 6))
    for fold in range(1, 6):
        development = set(outer.loc[outer.outer_fold != fold, "patient_id"])
        test = set(outer.loc[outer.outer_fold == fold, "patient_id"])
        part = inner[inner.outer_fold == fold]
        assert part.patient_id.is_unique and set(part.patient_id) == development
        assert np.array_equal(part.label, part.patient_id.map(patient_labels))
        assert set(part.inner_fold) == set(range(1, 6))
        for k in range(1, 6):
            train = set(part.loc[part.inner_fold != k, "patient_id"])
            val = set(part.loc[part.inner_fold == k, "patient_id"])
            assert not (train & val or train & test or val & test)
            assert train | val | test == set(index.patient_id)
            for ids in (train, val, test):
                assert set(index.loc[index.patient_id.isin(ids), "label"]) == {0, 1, 2}

INDEX, DATA_HASH = audit_dataset()
check_saved_partitions(INDEX, OUTER, INNER)
assert len(INDEX) == 3064 and INDEX.patient_id.nunique() == 233
DATASET_SUMMARY = INDEX.groupby("label").agg(patients=("patient_id", "nunique"), slices=("sample_id", "size"))
DATASET_SUMMARY["class"] = DATASET_SUMMARY.index.map(CLASS_NAMES)
display(DATASET_SUMMARY)
print("All 25 inner partitions have zero train/validation/outer-test patient overlap.")

# %% [markdown]
# ## 5. Identical Paired Preprocessing and Augmentation
# Read the same cleaned, brain-cropped grayscale PNGs. Resize images linearly and masks
# with nearest-neighbor interpolation to 224 x 224. Images are divided by 255, masks binarized.
# Training pairs share two independent 50% flips and a uniform rotation from -10 to +10
# degrees, with the same center and zero border as MAGS-Net. No evaluation augmentation.

# %%
def preprocess_pair(filename):
    size = (CONFIG["image_size"], CONFIG["image_size"])
    image = cv2.imread(str(DATA_ROOT / "images" / filename), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(DATA_ROOT / "masks" / filename), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mask = (cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
    return image, mask


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
# ## 6. Paired Dataset and DataLoaders
# Only the MRI enters the model; GT masks enter the loss and segmentation metrics.
# Retain the MAGS-Net batching policy: merge a final singleton into the previous batch
# (33 images when needed), rather than dropping a sample. Windows uses num_workers=0.

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
# ## 7. Standard U-Net Encoder and Decoder
# Each DoubleConv is Conv3x3–BN–ReLU–Conv3x3–BN–ReLU. Four MaxPool2d layers reduce
# 224 to 14 pixels. Decoder up-convolutions are learned ConvTranspose2d(kernel=2, stride=2),
# followed by concatenation with the corresponding encoder skip and another DoubleConv.

# %%
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))

# %% [markdown]
# ## 8. Full MT-U-Net and Bottleneck Classification Head
# Encoder: 32 -> 64 -> 128 -> 256 -> bottleneck 512. Classification: GAP -> FC256 -> ReLU
# -> Dropout(0.5) -> FC3. Decoder: 256 -> 128 -> 64 -> 32 -> one mask logit per pixel.
# Return classification logits, segmentation logits, and sigmoid mask probabilities.
# Losses use seg_logits directly, so BCEWithLogits/Dice never receive a twice-sigmoided mask.
# All layers use fresh PyTorch initialization. No pretrained encoder is used.

# %%
class LightweightMTUNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.enc1 = EncoderBlock(1, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)
        self.bottleneck = DoubleConv(256, 512)
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 32, 32)
        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True),
                                        nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        b = self.bottleneck(x)
        logits = self.classifier(torch.flatten(self.global_pool(b), 1))
        x = self.dec4(b, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        seg_logits = self.seg_head(x)
        return {"logits": logits, "seg_logits": seg_logits, "seg_mask": torch.sigmoid(seg_logits)}


def build_model(name=MODEL_NAME):
    assert name == MODEL_NAME
    assert CONFIG["channels"] == [32, 64, 128, 256, 512]
    return LightweightMTUNet(num_classes=3)


@torch.inference_mode()
def forward_check():
    with torch.random.fork_rng(devices=[]):
        model = build_model().to(DEVICE).eval()
    images = torch.zeros(1, 1, 224, 224, device=DEVICE)
    outputs = model(images)
    assert set(outputs) == {"logits", "seg_logits", "seg_mask"}
    assert outputs["logits"].shape == (1, 3) and outputs["seg_logits"].shape == (1, 1, 224, 224)
    torch.testing.assert_close(outputs["seg_mask"], outputs["seg_logits"].sigmoid())
    assert all(torch.isfinite(value).all() for value in outputs.values())
    return {"classification": list(outputs["logits"].shape), "segmentation": list(outputs["seg_logits"].shape)}


print("Forward-only check:", forward_check())

# %% [markdown]
# ### 8.1 Total and Trainable Parameters
# Count parameters across the complete shared encoder and both heads. FP32 weights use
# four bytes per parameter; exclude buffers, activations, optimizer state, and file overhead.

# %%
def parameter_counts(model, verbose=True):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fp32_size_mib = total_params * 4 / (1024 ** 2)
    if verbose:
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Parameters (M):       {total_params / 1e6:.3f} M")
        print(f"FP32 weights:         {fp32_size_mib:.3f} MiB")
    return {"model": MODEL_NAME, "total_parameters": total_params, "trainable_parameters": trainable_params,
            "parameters_m": total_params / 1e6, "fp32_size_mib": fp32_size_mib}


with torch.random.fork_rng(devices=[]):
    count_model = build_model()
PARAMETER_COUNTS = parameter_counts(count_model)
del count_model

# %% [markdown]
# ## 9. Classification and Segmentation Losses
# L_total = CrossEntropy + 0.5 * (BCEWithLogits + soft Dice loss). Dice uses smooth=1.0,
# per slice, then batch mean, matching the saved MAGS-Net segmentation-loss formula.
# The smoothing value is recorded in the protocol and comparison metadata.
# Calculate all losses in float32 for mixed-precision stability. No auxiliary loss.
# DiceLoss is a soft training loss; the reported Dice metric remains a thresholded overlap score.

# %%
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits.float()).flatten(1)
        targets = targets.float().flatten(1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()


classification_loss_fn = nn.CrossEntropyLoss()
bce_loss_fn = nn.BCEWithLogitsLoss()
dice_loss_fn = DiceLoss(smooth=CONFIG["dice_smooth"])
LAMBDA_SEG = CONFIG["lambda_seg"]


def compute_loss(outputs, labels, masks):
    cls_loss = classification_loss_fn(outputs["logits"].float(), labels)
    bce = bce_loss_fn(outputs["seg_logits"].float(), masks.float())
    dice = dice_loss_fn(outputs["seg_logits"], masks)
    seg_loss = bce + dice
    total_loss = cls_loss + LAMBDA_SEG * seg_loss
    return {"total": total_loss, "classification": cls_loss, "segmentation": seg_loss,
            "bce": bce, "dice_loss": dice}

# %% [markdown]
# ## 10. Record the Fixed Protocol
# Adam, LR=0.001, weight decay=1e-4, batch=32, max=50, patience=10, StepLR(10, 0.5).
# Mixed precision and clipping at gradient norm 1 follow MAGS-Net. Only validation total
# loss selects an inner checkpoint. The outer test set never selects epochs or hyperparameters.

# %%
assert CONFIG["seed"] == 42 and CONFIG["channels"] == [32, 64, 128, 256, 512]
assert CONFIG["max_epochs"] == 50 and CONFIG["patience"] == 10
EXPERIMENT = {"config": CONFIG, "model": MODEL_NAME, "classes": CLASS_NAMES,
              "mags_experiment_hash": MAGS_HASH, "data_hash": DATA_HASH, "torch": str(torch.__version__),
              "objective": OBJECTIVE, "initialization": "scratch", "upsampling": "ConvTranspose2d(2,2)",
              "segmentation_loss": "BCEWithLogits + mean per-slice soft Dice loss; smooth=1.0",
              "epoch_selection": "median of five minimum-validation-total-loss epochs"}
EXPERIMENT_HASH = signature(EXPERIMENT)
PREPARATION_DIR = RESULT_ROOT / "preparation" / EXPERIMENT_HASH[:12]
PREPARATION_DIR.mkdir(parents=True, exist_ok=True)
save_json(PREPARATION_DIR / "protocol.json", EXPERIMENT)
save_json(PREPARATION_DIR / "environment.json", ENVIRONMENT)
OUTER.to_csv(PREPARATION_DIR / "outer_patients.csv", index=False)
INNER.to_csv(PREPARATION_DIR / "inner_patients.csv", index=False)
INDEX.to_csv(PREPARATION_DIR / "sample_index.csv", index=False)
DATASET_SUMMARY.to_csv(PREPARATION_DIR / "dataset_summary.csv")
print("Prepared:", PREPARATION_DIR.relative_to(PROJECT_ROOT), "| planned fits = 30")

# %% [markdown]
# ## 11. Slice Predictions and Metrics
# Binary masks use the fixed threshold 0.5, with the same hard per-slice Dice/IoU as MAGS-Net.
# Empty-empty masks score 1. Tumor fraction is GT area / processed image area, not physical volume.

# %%
PREDICTION_COLUMNS = ["model", "outer_fold", "sample_id", "patient_id", "filename", "true_label",
                      "predicted_label", *PROBABILITY_COLUMNS, "dice", "iou", "tumor_fraction", "seed"]


def prediction_rows(outputs, batch):
    probabilities = outputs["logits"].detach().float().softmax(1).cpu().numpy()
    predicted = outputs["seg_logits"].detach().float().sigmoid().ge(CONFIG["mask_threshold"]).flatten(1)
    target = batch["mask"].to(predicted.device).gt(0.5).flatten(1)
    intersection = (predicted & target).sum(1).float()
    denominator = predicted.sum(1) + target.sum(1)
    union = denominator - intersection
    dice = torch.where(denominator > 0, 2 * intersection / denominator.clamp_min(1), 1.0)
    iou = torch.where(union > 0, intersection / union.clamp_min(1), 1.0)
    rows = pd.DataFrame({"sample_id": batch["sample_id"], "patient_id": batch["patient_id"],
                         "filename": batch["filename"], "true_label": batch["label"].numpy(),
                         "predicted_label": probabilities.argmax(1), "dice": dice.cpu().numpy(),
                         "iou": iou.cpu().numpy(), "tumor_fraction": target.float().mean(1).cpu().numpy()})
    rows[PROBABILITY_COLUMNS] = probabilities
    return rows


def score_predictions(rows):
    return {"accuracy": float(accuracy_score(rows.true_label, rows.predicted_label)),
            "macro_f1": float(f1_score(rows.true_label, rows.predicted_label, labels=[0, 1, 2], average="macro", zero_division=0)),
            "dice": float(rows.dice.mean()), "iou": float(rows.iou.mean())}

# %% [markdown]
# ## 12. Training and Validation Epochs
# Model input is MRI only. Losses are averaged over all images, and metrics over all slice
# records. Predicted masks are saved only during explicit outer-test evaluation.

# %%
def run_epoch(model, loader, optimizer=None, scaler=None, mask_dir=None):
    training = optimizer is not None
    model.train(training)
    loss_names = {"total": "loss", "classification": "cls", "segmentation": "seg", "bce": "bce", "dice_loss": "dice_loss"}
    totals, records = {key: 0.0 for key in loss_names.values()}, []
    amp = CONFIG["use_amp"] and DEVICE.type == "cuda"
    dtype = torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc="Train" if training else "Evaluate", leave=False):
            images, masks, labels = (batch[key].to(DEVICE) for key in ("image", "mask", "label"))
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, dtype=dtype, enabled=amp):
                outputs = model(images)
                losses = compute_loss(outputs, labels, masks)
            if not all(torch.isfinite(value).item() for value in losses.values()):
                raise FloatingPointError("Non-finite multitask loss.")
            if training:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"], error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
            for key, value in losses.items():
                totals[loss_names[key]] += float(value.detach()) * len(labels)
            records.append(prediction_rows(outputs, batch))
            if mask_dir is not None:
                predicted = outputs["seg_logits"].detach().float().sigmoid().ge(CONFIG["mask_threshold"]).cpu().numpy()
                for filename, mask in zip(batch["filename"], predicted[:, 0]):
                    if not cv2.imwrite(str(mask_dir / filename), mask.astype(np.uint8) * 255):
                        raise OSError(f"Could not save predicted mask: {filename}")
    rows = pd.concat(records, ignore_index=True)
    assert len(rows) == len(loader.dataset) and rows.sample_id.is_unique
    return {**{key: value / len(rows) for key, value in totals.items()}, **score_predictions(rows)}, rows


def print_epoch(epoch, epochs, lr, train, val, best, stale):
    print(f"Epoch {epoch:02d}/{epochs} | lr={lr:.6f}")
    for label, values in (("Train", train), ("Val", val)):
        if values is not None:
            print(f"{label:5} | Loss={values['loss']:.4f} | Cls={values['cls']:.4f} | Seg={values['seg']:.4f} | "
                  f"BCE={values['bce']:.4f} | DiceLoss={values['dice_loss']:.4f} | "
                  f"Acc={values['accuracy']:.2%} | F1={values['macro_f1']:.4f} | Dice={values['dice']:.4f} | IoU={values['iou']:.4f}")
    if val is not None:
        print(f"Best  | Epoch={best['epoch']} | Val Loss={best['loss']:.4f} | Patience={stale}/{CONFIG['patience']}")

# %% [markdown]
# ## 13. Independent Fits and Checkpoints
# Strict decrease in validation total loss resets patience; ties retain the earlier epoch.
# Outer refits have no validation/test loader or early stopping. Save best/last inner weights
# and final outer weights. Verified completed fits are reused; incomplete fits require an
# explicit restart. Checkpoints do not support resuming an interrupted optimizer state.

# %%
def fit_signature(name, train, validation, epochs):
    return signature({"experiment": EXPERIMENT_HASH, "model": name, "epochs": int(epochs),
                      "train": sorted(train.sample_id), "validation": [] if validation is None else sorted(validation.sample_id)})


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)

# %%
def fit_model(name, train, validation, epochs, directory):
    digest = fit_signature(name, train, validation, epochs)
    summary_path = directory / "summary.json"
    if summary_path.exists():
        saved = json.loads(summary_path.read_text())
        assert saved["signature"] == digest and saved["checkpoint_hash"] == file_hash(directory / saved["checkpoint"])
        return saved
    if directory.exists():
        raise RuntimeError(f"Incomplete fit: {directory}. Start a new study or explicitly restart this fit.")
    directory.mkdir(parents=True)
    set_seed(CONFIG["seed"])
    model = build_model(name).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CONFIG["scheduler_step"], CONFIG["scheduler_gamma"])
    scaler = torch.amp.GradScaler("cuda", enabled=CONFIG["use_amp"] and DEVICE.type == "cuda" and not torch.cuda.is_bf16_supported())
    train_loader = make_loader(train, training=True)
    validation_loader = make_loader(validation) if validation is not None else None
    best, stale, history = {"epoch": 0, "loss": float("inf")}, 0, []
    for epoch in range(1, epochs + 1):
        started, lr = time.perf_counter(), optimizer.param_groups[0]["lr"]
        train_metrics, _ = run_epoch(model, train_loader, optimizer, scaler)
        val_metrics, _ = run_epoch(model, validation_loader) if validation_loader is not None else (None, None)
        improved = val_metrics is not None and val_metrics["loss"] < best["loss"]
        if improved:
            best, stale = {"epoch": epoch, "loss": val_metrics["loss"]}, 0
        elif val_metrics is not None:
            stale += 1
        scheduler.step()
        history.append({"epoch": epoch, "lr": lr, "seconds": time.perf_counter() - started, "stale_epochs": stale,
                        **{f"train_{k}": v for k, v in train_metrics.items()},
                        **({f"val_{k}": v for k, v in val_metrics.items()} if val_metrics is not None else {})})
        payload = {"model": name, "state_dict": model.state_dict(), "epoch": epoch, "seed": CONFIG["seed"],
                   "signature": digest, "experiment_hash": EXPERIMENT_HASH, "source_mags_hash": MAGS_HASH,
                   "objective": OBJECTIVE, "config": CONFIG, "weights": None}
        save_checkpoint(directory / "last.pt", payload)
        if improved:
            save_checkpoint(directory / "best.pt", payload)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print_epoch(epoch, epochs, lr, train_metrics, val_metrics, best, stale)
        if val_metrics is not None and stale >= CONFIG["patience"]:
            break
    checkpoint = "best.pt" if validation is not None else "last.pt"
    result = {"model": name, "signature": digest, "epochs_completed": epoch,
              "best_epoch": best["epoch"] if validation is not None else epoch,
              "best_val_total_loss": best["loss"] if validation is not None else None,
              "checkpoint": checkpoint, "checkpoint_hash": file_hash(directory / checkpoint)}
    save_json(summary_path, result)
    return result

# %% [markdown]
# ## 14. Five Inner Folds and Fresh Outer Retraining
# For each outer fold, record five best total-loss epochs, take their median, then initialize
# a fresh MT-U-Net and fit all development patients for exactly that many epochs.

# %%
def select_duration(name, outer_fold, directory):
    partition = INNER[INNER.outer_fold == outer_fold]
    rows = []
    for inner_fold in range(1, 6):
        train_ids = partition.loc[partition.inner_fold != inner_fold, "patient_id"]
        val_ids = partition.loc[partition.inner_fold == inner_fold, "patient_id"]
        train = INDEX[INDEX.patient_id.isin(train_ids)]
        validation = INDEX[INDEX.patient_id.isin(val_ids)]
        print(f"\n{name} | OUTER {outer_fold}/5 | INNER {inner_fold}/5")
        result = fit_model(name, train, validation, CONFIG["max_epochs"], directory / f"inner_{inner_fold}")
        rows.append({"outer_fold": outer_fold, "inner_fold": inner_fold, **result})
    table = pd.DataFrame(rows)
    best_epochs = table.best_epoch.astype(int).tolist()
    selected = {"model": name, "outer_fold": outer_fold, "seed": CONFIG["seed"], "inner_best_epochs": best_epochs,
                "epochs": int(np.median(best_epochs)), "selection_rule": "Median of five minimum-validation-total-loss epochs",
                "experiment_hash": EXPERIMENT_HASH}
    table.to_csv(directory / "inner_summary.csv", index=False)
    save_json(directory / "selection.json", selected)
    print("Inner best epochs:", best_epochs, "| Selected outer duration:", selected["epochs"])
    return selected


def refit_outer(name, outer_fold, selected, directory):
    development_ids = OUTER.loc[OUTER.outer_fold != outer_fold, "patient_id"]
    development = INDEX[INDEX.patient_id.isin(development_ids)]
    print(f"\n{name} | OUTER {outer_fold}/5 | FRESH REFIT: {selected['epochs']} epochs")
    return fit_model(name, development, None, selected["epochs"], directory / "outer_refit")

# %% [markdown]
# ## 15. Run All Five Outer Folds (Training Disabled)
# Use STUDY_DIR=None only for a new experiment. Set a matching existing path to reuse
# completed fits. Keep RUN_OUTER_TEST=False during training.

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
        OUTER.to_csv(directory / "outer_patients.csv", index=False)
        INNER.to_csv(directory / "inner_patients.csv", index=False)
    if directory is not None:
        saved = json.loads((directory / "experiment.json").read_text())
        assert saved["experiment_hash"] == signature(saved["protocol"]) == EXPERIMENT_HASH
        print("MT-U-Net study directory:", directory)
    return directory


ACTIVE_STUDY = open_study()
if RUN_TRAINING:
    for fold in range(1, 6):
        directory = ACTIVE_STUDY / f"outer_{fold}"
        selection = select_duration(MODEL_NAME, fold, directory)
        refit_outer(MODEL_NAME, fold, selection, directory)
    save_json(ACTIVE_STUDY / "training_complete.json", {"experiment_hash": EXPERIMENT_HASH,
                                                       "model": MODEL_NAME, "outer_refits": 5})
else:
    print("Training disabled. No weights are updated.")

# %% [markdown]
# ## 16. Prediction and Checkpoint Integrity
# Verify identifiers, labels, fold membership, probabilities, per-slice metrics, selected
# training durations, and checkpoint hashes before using held-out predictions.

# %%
def validate_predictions(frame, expected, model_name, outer_fold=None):
    assert not frame[PREDICTION_COLUMNS].isna().any().any() and frame.sample_id.is_unique
    assert frame.model.eq(model_name).all() and frame.seed.eq(CONFIG["seed"]).all()
    assert set(frame.sample_id) == set(expected.sample_id)
    ordered = frame.set_index("sample_id").loc[expected.sample_id]
    assert np.array_equal(ordered.patient_id.to_numpy(), expected.patient_id.to_numpy())
    assert np.array_equal(ordered.true_label.to_numpy(), expected.label.to_numpy())
    patient_folds = OUTER.set_index("patient_id").outer_fold
    assert np.array_equal(frame.outer_fold.to_numpy(), frame.patient_id.map(patient_folds).to_numpy())
    if outer_fold is not None:
        assert frame.outer_fold.eq(outer_fold).all()
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy()
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    assert np.array_equal(frame.predicted_label, probabilities.argmax(axis=1))
    assert frame.groupby("patient_id").outer_fold.nunique().eq(1).all()
    assert np.array_equal(ordered.filename.to_numpy(), expected.filename.to_numpy())
    metrics = frame[['dice', 'iou', 'tumor_fraction']].to_numpy()
    assert np.isfinite(metrics).all() and ((metrics >= 0) & (metrics <= 1)).all()
    assert np.allclose(frame.dice, 2 * frame.iou / (1 + frame.iou), atol=1e-5)

# %%
def read_outer_checkpoint(name, fold, directory):
    selected = json.loads((directory / "selection.json").read_text())
    saved = json.loads((directory / "outer_refit/summary.json").read_text())
    development_ids = OUTER.loc[OUTER.outer_fold != fold, "patient_id"]
    development = INDEX[INDEX.patient_id.isin(development_ids)]
    assert selected["model"] == saved["model"] == name and selected["outer_fold"] == fold
    assert selected["experiment_hash"] == EXPERIMENT_HASH and selected["seed"] == CONFIG["seed"]
    assert selected["epochs"] == int(np.median(selected["inner_best_epochs"])) == saved["epochs_completed"]
    assert saved["signature"] == fit_signature(name, development, None, selected["epochs"])
    assert len(selected['inner_best_epochs']) == 5
    partition = INNER[INNER.outer_fold == fold]
    for k in range(1, 6):
        inner = json.loads((directory / f'inner_{k}/summary.json').read_text())
        train_ids = partition.loc[partition.inner_fold != k, 'patient_id']
        val_ids = partition.loc[partition.inner_fold == k, 'patient_id']
        train = INDEX[INDEX.patient_id.isin(train_ids)]
        val = INDEX[INDEX.patient_id.isin(val_ids)]
        assert inner['signature'] == fit_signature(name, train, val, CONFIG['max_epochs'])
        assert inner['best_epoch'] == selected['inner_best_epochs'][k - 1]
    checkpoint_path = directory / "outer_refit" / saved["checkpoint"]
    assert saved["checkpoint_hash"] == file_hash(checkpoint_path)
    return selected, saved, checkpoint_path

# %% [markdown]
# ## 17. Outer-Test Evaluation Once per Completed Refit
# All five refits must be complete first. On later runs, verified prediction CSVs and masks
# are reused without inference. No outer-test result changes training or epoch selection.

# %%
def read_cached_predictions(directory, expected, fold, checkpoint_hash):
    metadata = json.loads((directory / "evaluation.json").read_text())
    path = directory / "outer_predictions.csv"
    assert metadata["experiment_hash"] == EXPERIMENT_HASH and metadata["checkpoint_hash"] == checkpoint_hash
    assert metadata["predictions_hash"] == file_hash(path)
    assert set(metadata["mask_hashes"]) == set(expected.filename)
    for filename, digest in metadata["mask_hashes"].items():
        assert file_hash(directory / "predicted_masks" / filename) == digest
    rows = pd.read_csv(path, dtype={"sample_id": str, "patient_id": str, "filename": str})
    validate_predictions(rows, expected, MODEL_NAME, fold)
    return rows


def evaluate_outer(fold, directory):
    selected, summary, checkpoint_path = read_outer_checkpoint(MODEL_NAME, fold, directory)
    test_ids = OUTER.loc[OUTER.outer_fold == fold, "patient_id"]
    test = INDEX[INDEX.patient_id.isin(test_ids)]
    if (directory / "evaluation.json").exists():
        return read_cached_predictions(directory, test, fold, summary["checkpoint_hash"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["model"] == MODEL_NAME and checkpoint["signature"] == summary["signature"]
    assert checkpoint["experiment_hash"] == EXPERIMENT_HASH and checkpoint["epoch"] == selected["epochs"]
    assert checkpoint["objective"] == OBJECTIVE and checkpoint["weights"] is None
    model = build_model().to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    mask_dir = directory / "predicted_masks"
    mask_dir.mkdir(exist_ok=True)
    metrics, rows = run_epoch(model, make_loader(test), mask_dir=mask_dir)
    rows["model"], rows["outer_fold"], rows["seed"] = MODEL_NAME, fold, CONFIG["seed"]
    rows = rows[PREDICTION_COLUMNS]
    validate_predictions(rows, test, MODEL_NAME, fold)
    rows.to_csv(directory / "outer_predictions.csv", index=False)
    pd.DataFrame([{"model": MODEL_NAME, "outer_fold": fold, **metrics}]).to_csv(directory / "outer_metrics.csv", index=False)
    save_json(directory / "evaluation.json", {"experiment_hash": EXPERIMENT_HASH,
        "checkpoint_hash": summary["checkpoint_hash"], "predictions_hash": file_hash(directory / "outer_predictions.csv"),
        "mask_hashes": {filename: file_hash(mask_dir / filename) for filename in test.filename}, "test_metrics": metrics})
    return rows


if RUN_OUTER_TEST:
    if ACTIVE_STUDY is None or not (ACTIVE_STUDY / "training_complete.json").exists():
        raise RuntimeError("Complete all MT-U-Net fits and select STUDY_DIR before outer testing.")
    completed = json.loads((ACTIVE_STUDY / "training_complete.json").read_text())
    assert completed == {"experiment_hash": EXPERIMENT_HASH, "model": MODEL_NAME, "outer_refits": 5}
    for fold in range(1, 6):
        read_outer_checkpoint(MODEL_NAME, fold, ACTIVE_STUDY / f"outer_{fold}")
    for fold in range(1, 6):
        evaluate_outer(fold, ACTIVE_STUDY / f"outer_{fold}")
else:
    print("Outer-test inference disabled.")

# %% [markdown]
# ## 18. Pool OOF Predictions and Read the MAGS-Net Reference
# Each complete model must cover the same 3,064 slices and 233 patients exactly once.
# Read MAGS-Net's saved predictions with provenance checks; do not rerun its inference.

# %%
def load_mt_oof(study):
    if study is None or not all((study / f"outer_{fold}/evaluation.json").exists() for fold in range(1, 6)):
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    parts = []
    for fold in range(1, 6):
        directory = study / f"outer_{fold}"
        _, summary, _ = read_outer_checkpoint(MODEL_NAME, fold, directory)
        expected = INDEX[INDEX.patient_id.isin(OUTER.loc[OUTER.outer_fold == fold, "patient_id"])]
        parts.append(read_cached_predictions(directory, expected, fold, summary["checkpoint_hash"]))
    rows = pd.concat(parts, ignore_index=True)
    validate_predictions(rows, INDEX, MODEL_NAME)
    return rows


def load_mags_reference():
    completion = json.loads((MAGS_STUDY / "training_complete.json").read_text())
    assert completion["experiment_hash"] == MAGS_HASH and completion["outer_refits"] == 5
    parts = []
    for fold in range(1, 6):
        directory = MAGS_STUDY / f"seed_{CONFIG['seed']}" / f"outer_{fold}"
        provenance = json.loads((directory / "evaluation.json").read_text())
        summary = json.loads((directory / "outer_refit/summary.json").read_text())
        selected = json.loads((directory / "selection.json").read_text())
        assert provenance["experiment_hash"] == MAGS_HASH
        assert provenance["predictions_hash"] == file_hash(directory / "outer_predictions.csv")
        assert provenance["checkpoint_hash"] == summary["checkpoint_hash"] == file_hash(directory / "outer_refit" / summary["checkpoint"])
        train_ids = OUTER.loc[OUTER.outer_fold != fold, "patient_id"]
        expected_signature = signature({"experiment": MAGS_HASH, "candidate": selected["candidate"], "seed": CONFIG["seed"],
                                        "epochs": selected["epochs"], "train": sorted(INDEX.loc[INDEX.patient_id.isin(train_ids), "sample_id"]), "val": []})
        assert summary["signature"] == expected_signature
        rows = pd.read_csv(directory / "outer_predictions.csv", dtype={"sample_id": str, "patient_id": str})
        rows = rows.rename(columns={"label": "true_label", "prediction": "predicted_label",
                                    "prob_0": PROBABILITY_COLUMNS[0], "prob_1": PROBABILITY_COLUMNS[1], "prob_2": PROBABILITY_COLUMNS[2]})
        rows["model"] = "MAGS-Net"
        rows = rows[PREDICTION_COLUMNS]
        test_ids = OUTER.loc[OUTER.outer_fold == fold, "patient_id"]
        validate_predictions(rows, INDEX[INDEX.patient_id.isin(test_ids)], "MAGS-Net", fold)
        parts.append(rows)
    pooled = pd.concat(parts, ignore_index=True)
    validate_predictions(pooled, INDEX, "MAGS-Net")
    return pooled

# %%
def oof_coverage(frame, name):
    if frame.empty:
        return {"model": name, "status": "not_evaluated", "oof_rows": pd.NA, "unique_patients": pd.NA,
                "duplicate_samples": pd.NA, "patients_tested_once": pd.NA, "duplicate_patient_fold_assignments": pd.NA}
    validate_predictions(frame, INDEX, name)
    assignments = frame[["patient_id", "outer_fold"]].drop_duplicates()
    counts = assignments.groupby("patient_id").size()
    return {"model": name, "status": "complete", "oof_rows": len(frame), "unique_patients": frame.patient_id.nunique(),
            "duplicate_samples": int(frame.sample_id.duplicated().sum()), "patients_tested_once": int(counts.eq(1).sum()),
            "duplicate_patient_fold_assignments": int(assignments.patient_id.duplicated().sum())}

OOF = load_mt_oof(ACTIVE_STUDY)
MAGS_OOF = load_mags_reference()
OOF_BY_MODEL = {MODEL_NAME: OOF, "MAGS-Net": MAGS_OOF}
COVERAGE = pd.DataFrame([oof_coverage(rows, name) for name, rows in OOF_BY_MODEL.items()])
display(COVERAGE)
if OOF.empty:
    print("MT-U-Net is not evaluated yet. No MT-U-Net performance is reported.")

# %% [markdown]
# ## 19. Slice and Patient Classification; Overall and Per-Class Segmentation
# Patient classification averages slice probabilities before argmax. Patient Dice/IoU are
# mean 2D slice scores, not 3D volumetric metrics. Overall slice Dice/IoU average per-slice
# scores, not pooled pixels. Report all three tumor classes with their support.
# The table labels MAGS-Net's additional auxiliary supervision explicitly.

# %%
def patient_predictions(rows):
    means = PROBABILITY_COLUMNS + ["dice", "iou", "tumor_fraction"]
    patients = rows.groupby(["model", "outer_fold", "patient_id", "seed"], as_index=False).agg(
        true_label=("true_label", "first"), slices=("sample_id", "size"),
        **{column: (column, "mean") for column in means})
    patients["predicted_label"] = patients[PROBABILITY_COLUMNS].to_numpy().argmax(axis=1)
    return patients


def classification_metrics(rows, level):
    name = rows.model.iloc[0]
    return {"model": name, "level": level, "seed": CONFIG["seed"], "samples": len(rows),
            "objective": OBJECTIVE if name == MODEL_NAME else "Cls + 0.5 Seg + 0.2 Aux",
            "dice_loss_smooth": CONFIG["dice_smooth"] if name == MODEL_NAME else 1.0,
            "accuracy": float(accuracy_score(rows.true_label, rows.predicted_label)),
            "macro_f1": float(f1_score(rows.true_label, rows.predicted_label, labels=[0, 1, 2], average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(rows.true_label, rows.predicted_label)),
            "macro_ovr_auc": float(roc_auc_score(rows.true_label, rows[PROBABILITY_COLUMNS], labels=[0, 1, 2], multi_class="ovr", average="macro"))}


def segmentation_summary(rows, level):
    parts = [("Overall", rows)] + [(name, rows[rows.true_label == label]) for label, name in CLASS_NAMES.items()]
    return [{"model": rows.model.iloc[0], "level": level, "class": name, "samples": len(part),
             "dice_mean": float(part.dice.mean()), "dice_std": float(part.dice.std(ddof=1)),
             "iou_mean": float(part.iou.mean()), "iou_std": float(part.iou.std(ddof=1))} for name, part in parts]


PATIENT_OOF, classification, segmentation, reports, per_fold = {}, [], [], [], []
for name, rows in OOF_BY_MODEL.items():
    if rows.empty:
        continue
    PATIENT_OOF[name] = patient_predictions(rows)
    for level, frame in (("slice", rows), ("patient", PATIENT_OOF[name])):
        classification.append(classification_metrics(frame, level))
        segmentation.extend(segmentation_summary(frame, level))
        report = classification_report(frame.true_label, frame.predicted_label, labels=[0, 1, 2],
                                       target_names=list(CLASS_NAMES.values()), output_dict=True, zero_division=0)
        reports.extend({"model": name, "level": level, "class": label, **report[label]} for label in CLASS_NAMES.values())
        for fold, part in frame.groupby("outer_fold"):
            per_fold.append({"outer_fold": fold, **classification_metrics(part, level),
                             "dice_mean": float(part.dice.mean()), "iou_mean": float(part.iou.mean())})
CLASSIFICATION = pd.DataFrame(classification)
SEGMENTATION = pd.DataFrame(segmentation)
display(CLASSIFICATION)
display(SEGMENTATION[SEGMENTATION.level == "slice"])

# %% [markdown]
# ## 20. Parameters, MACs, and FP32 Size
# One synthetic 1 x 1 x 224 x 224 input, FP32 eval mode. Count Conv2d, Linear, and
# ConvTranspose2d multiply-accumulates. The non-overlapping 2x2 up-convolutions are counted
# using input elements * output channels * 2x2. Include the full decoder and both heads.
# As in MAGS-Net, exclude bias additions, normalization, activations, pooling, interpolation,
# and elementwise operations. This is a partial operation count, not total FLOPs.
# FP32 size = parameters * 4 / 2**20 MiB, excluding buffers and checkpoint overhead.

# %%
@torch.inference_mode()
def model_complexity():
    with torch.random.fork_rng(devices=[]):
        model = build_model().cpu().float().eval()
    counts, handles = [], []
    def count(layer, inputs, output):
        if isinstance(layer, nn.ConvTranspose2d):
            assert layer.kernel_size == layer.stride == (2, 2) and layer.padding == layer.output_padding == (0, 0)
            value = inputs[0].numel() * (layer.out_channels // layer.groups) * 4
        elif isinstance(layer, nn.Conv2d):
            value = output.numel() * (layer.in_channels // layer.groups) * np.prod(layer.kernel_size)
        else:
            value = output.numel() * layer.in_features
        counts.append(int(value))
    try:
        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                handles.append(layer.register_forward_hook(count))
        model(torch.zeros(1, 1, 224, 224))
    finally:
        for handle in handles:
            handle.remove()
    parameters = sum(p.numel() for p in model.parameters())
    return {"model": MODEL_NAME, "parameters": parameters, "macs": sum(counts),
            "fp32_size_mib": parameters * 4 / 2**20, "classification": True, "segmentation": True}


reference = pd.read_csv(MAGS_STUDY / "analysis/complexity.csv")
assert len(reference) == 1 and reference.iloc[0].batch_size == 1 and reference.iloc[0].precision == "float32"
reference = reference.iloc[0]
assert np.isclose(reference.weight_mib_fp32, reference.parameters * 4 / 2**20)
COMPLEXITY = pd.DataFrame([model_complexity(), {"model": "MAGS-Net", "parameters": int(reference.parameters),
    "macs": int(reference.conv_linear_macs), "fp32_size_mib": float(reference.weight_mib_fp32),
    "classification": True, "segmentation": True}])
display(COMPLEXITY)

# %% [markdown]
# ## 21. Epoch Selection Summary and Figures
# Display the inner best epochs and verify the selected median against the completed outer
# refit. Plot comparison figures only when both models have complete held-out predictions.

# %%
def epoch_selection_table(study):
    rows = []
    if study is not None:
        for fold in range(1, 6):
            directory = study / f"outer_{fold}"
            if not (directory / "outer_refit/summary.json").exists():
                continue
            selected, saved, _ = read_outer_checkpoint(MODEL_NAME, fold, directory)
            rows.append({"model": MODEL_NAME, "outer_fold": fold, "inner_best_epochs": json.dumps(selected["inner_best_epochs"]),
                         "median_epoch": selected["epochs"], "outer_training_epochs": saved["epochs_completed"],
                         "selection_rule": selected["selection_rule"]})
    return pd.DataFrame(rows)


def comparison_figures(oof_sets):
    figures = {}
    for kind in ("confusion", "roc", "segmentation"):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, (name, rows) in zip(axes, oof_sets.items()):
            if kind == "confusion":
                matrix = confusion_matrix(rows.true_label, rows.predicted_label, labels=[0, 1, 2])
                ax.imshow(matrix, cmap="Blues")
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                                color="white" if matrix[i, j] > matrix.max() / 2 else "black")
                ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES.values(), yticklabels=CLASS_NAMES.values(),
                       xlabel="Predicted", ylabel="True")
            elif kind == "roc":
                for label, class_name in CLASS_NAMES.items():
                    fpr, tpr, _ = roc_curve(rows.true_label.eq(label), rows[PROBABILITY_COLUMNS[label]])
                    ax.plot(fpr, tpr, label=class_name)
                ax.plot([0, 1], [0, 1], "--", color="gray")
                ax.set(xlabel="False positive rate", ylabel="True positive rate")
                ax.legend(fontsize=8)
            else:
                ax.boxplot([rows.loc[rows.true_label == label, "dice"] for label in CLASS_NAMES], tick_labels=list(CLASS_NAMES.values()))
                ax.set(ylabel="Per-slice Dice", ylim=(0, 1.02))
            ax.set_title(name)
        figure.tight_layout()
        figures[kind] = figure
    return figures


EPOCH_SELECTIONS = epoch_selection_table(ACTIVE_STUDY)
if not EPOCH_SELECTIONS.empty:
    display(EPOCH_SELECTIONS)

# %% [markdown]
# ## 22. Save CSVs, Masks, Figures, and Analysis Status
# MT-UNet_oof.csv includes classification probabilities, Dice, IoU, and GT tumor fraction.
# MT-UNet_patient_oof.csv contains averaged probabilities and mean slice Dice/IoU per patient.
# Pending performance is never represented by synthetic rows. Before training, only reference
# metrics, the protocol, and complexity are exported under preparation/<hash>/reference/.

# %%
EXPORT_DIR = ACTIVE_STUDY / "analysis" if ACTIVE_STUDY is not None else PREPARATION_DIR / "reference"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
for name, rows in OOF_BY_MODEL.items():
    if not rows.empty:
        slug = "MT-UNet" if name == MODEL_NAME else "MAGS-Net"
        rows[PREDICTION_COLUMNS].to_csv(EXPORT_DIR / f"{slug}_oof.csv", index=False)
        PATIENT_OOF[name].to_csv(EXPORT_DIR / f"{slug}_patient_oof.csv", index=False)
CLASSIFICATION.to_csv(EXPORT_DIR / "classification_comparison.csv", index=False)
SEGMENTATION.to_csv(EXPORT_DIR / "segmentation_comparison.csv", index=False)
pd.DataFrame(reports).to_csv(EXPORT_DIR / "per_class_classification.csv", index=False)
pd.DataFrame(per_fold).to_csv(EXPORT_DIR / "outer_fold_metrics.csv", index=False)
COVERAGE.to_csv(EXPORT_DIR / "oof_coverage.csv", index=False)
COMPLEXITY.to_csv(EXPORT_DIR / "model_complexity.csv", index=False)
pd.DataFrame([PARAMETER_COUNTS]).to_csv(EXPORT_DIR / "model_parameters.csv", index=False)
if not EPOCH_SELECTIONS.empty:
    EPOCH_SELECTIONS.to_csv(EXPORT_DIR / "epoch_selections.csv", index=False)
if COVERAGE.status.eq("complete").all():
    for name, figure in comparison_figures(OOF_BY_MODEL).items():
        figure.savefig(EXPORT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
        figure.savefig(EXPORT_DIR / f"{name}.pdf", bbox_inches="tight")
        plt.show()
        plt.close(figure)
save_json(EXPORT_DIR / "analysis_status.json", {"experiment_hash": EXPERIMENT_HASH, "source_mags_hash": MAGS_HASH,
    "complete_models": COVERAGE.loc[COVERAGE.status == "complete", "model"].tolist(),
    "training_enabled": RUN_TRAINING, "outer_test_enabled": RUN_OUTER_TEST,
    "dice_loss_smooth": CONFIG["dice_smooth"], "reference_mags_dice_loss_smooth": 1.0,
    "macs_definition": "Conv2d + ConvTranspose2d + Linear MACs; batch=1, input=1x224x224, full multitask model",
    "fp32_size_definition": "Parameters * 4 / 2**20 MiB; excludes buffers and checkpoint overhead"})
print("Results saved:", EXPORT_DIR.relative_to(PROJECT_ROOT))

# %% [markdown]
# ### 22.1 MAGS-Net vs MT-U-Net: Classification, Segmentation, and Complexity
# Run after Sections 17–22. Read the same saved OOF patients; no retraining or new test
# inference. Match sample IDs, patient IDs, labels, outer folds, seeds, and slice counts.
# Patient classification averages slice probabilities before argmax. Mean Dice and Mean
# IoU first average slices within each patient, then weight patients equally. These are
# means of 2D scores, not volumetric scores or pooled-slice means. IoU is averaged directly,
# not derived by transforming the mean Dice. All differences are **MAGS-Net minus MT-U-Net**.
# #
# Use a two-sided exact McNemar test on paired patient correctness. There is one model
# comparison, so no Holm correction is applied. For all four differences, use 10,000
# paired patient-bootstrap draws, seed 42, stratified by true tumor class. Share the
# sampled patient IDs across both models and all metrics. Recompute macro-F1 over classes
# 0, 1, 2 in each sample before subtracting models. Report percentile 95% intervals.
# Accuracy differences use percentage points; macro-F1, Dice, and IoU differences use
# the 0–1 scale. Intervals are pointwise, not simultaneous or multiplicity-adjusted.
# Inference is conditional on the fitted OOF models and fixed folds and excludes
# retraining/partition variability. The comparison includes MAGS-Net's auxiliary supervision.
# #
# Complexity covers each complete multitask model at batch=1, 1x224x224, FP32: count
# Conv2d, ConvTranspose2d, and Linear MACs; exclude normalization, activations, pooling,
# interpolation, and elementwise operations. FP32 size = parameters * 4 / 2**20 MiB,
# excluding buffers, activations, optimizer state, and checkpoint overhead. No timing is inferred.
# #
# References: [exact McNemar](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html),
# [paired bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html),
# and [macro-F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

# %%
from scipy.stats import binomtest

# Analysis settings are separate from the saved training protocol.
STATISTICS_CONFIG = {"resamples": 10_000, "seed": 42, "confidence_level": 0.95}
STATISTICS_MODELS = ["MAGS-Net", "MT-U-Net"]


def align_patient_results(oof_sets):
    patients = {}
    for name in STATISTICS_MODELS:
        validate_predictions(oof_sets[name], INDEX, name)
        frame = patient_predictions(oof_sets[name]).set_index("patient_id").sort_index()
        assert frame.index.is_unique and set(frame.index) == set(INDEX.patient_id)
        patients[name] = frame
    reference = patients["MAGS-Net"]
    for frame in patients.values():
        assert frame.index.equals(reference.index)
        for column in ("true_label", "outer_fold", "seed", "slices"):
            assert np.array_equal(frame[column], reference[column]), f"Patient mismatch: {column}"
    return patients

# %%
def patient_multitask_scores(labels, predictions, dice, iou):
    """Return [accuracy percent, macro-F1, mean patient Dice, mean patient IoU] per model."""
    assert predictions.shape == dice.shape == iou.shape == (len(labels), 2)
    assert np.isin(labels, [0, 1, 2]).all() and np.isin(predictions, [0, 1, 2]).all()
    assert np.isfinite(dice).all() and np.isfinite(iou).all()
    codes = 9 * np.arange(2)[None, :] + 3 * labels[:, None] + predictions
    matrices = np.bincount(codes.astype(int).ravel(), minlength=18).reshape(2, 3, 3)
    true_positive = np.diagonal(matrices, axis1=1, axis2=2)
    denominator = matrices.sum(axis=1) + matrices.sum(axis=2)
    class_f1 = np.divide(2.0 * true_positive, denominator,
                         out=np.zeros_like(true_positive, dtype=float), where=denominator > 0)
    return np.column_stack([100 * true_positive.sum(axis=1) / len(labels), class_f1.mean(axis=1),
                            dice.mean(axis=0), iou.mean(axis=0)])


def paired_patient_bootstrap(labels, predictions, dice, iou, settings):
    assert settings["resamples"] > 0 and 0 < settings["confidence_level"] < 1
    rng = np.random.default_rng(settings["seed"])
    strata = [np.flatnonzero(labels == label) for label in (0, 1, 2)]
    assert all(len(group) > 0 for group in strata)
    estimates = np.empty((settings["resamples"], 4))
    for iteration in range(settings["resamples"]):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata])
        scores = patient_multitask_scores(labels[indices], predictions[indices], dice[indices], iou[indices])
        estimates[iteration] = scores[0] - scores[1]
    tail = (1 - settings["confidence_level"]) / 2
    return np.quantile(estimates, [tail, 1 - tail], axis=0)

# %%
def paired_patient_statistics(oof_sets, settings):
    patients = align_patient_results(oof_sets)
    labels = patients["MAGS-Net"].true_label.to_numpy()
    predictions = np.column_stack([patients[name].predicted_label.to_numpy() for name in STATISTICS_MODELS])
    dice = np.column_stack([patients[name].dice.to_numpy() for name in STATISTICS_MODELS])
    iou = np.column_stack([patients[name].iou.to_numpy() for name in STATISTICS_MODELS])
    scores = patient_multitask_scores(labels, predictions, dice, iou)
    differences = scores[0] - scores[1]
    intervals = paired_patient_bootstrap(labels, predictions, dice, iou, settings)
    correct = predictions == labels[:, None]
    mags, other = correct[:, 0], correct[:, 1]
    b, c = int((mags & ~other).sum()), int((~mags & other).sum())
    row = {"comparison": "MAGS-Net vs MT-U-Net", "reference": "MAGS-Net", "variant": "MT-U-Net",
        "n_patients": len(labels), "mags_correct": int(mags.sum()), "variant_correct": int(other.sum()),
        "both_correct": int((mags & other).sum()), "both_wrong": int((~mags & ~other).sum()),
        "mags_correct_variant_wrong": b, "mags_wrong_variant_correct": c, "discordant_pairs": b + c,
        "mags_patient_accuracy": scores[0, 0] / 100, "variant_patient_accuracy": scores[1, 0] / 100,
        "delta_patient_accuracy_pp": differences[0], "accuracy_ci_low_pp": intervals[0, 0], "accuracy_ci_high_pp": intervals[1, 0],
        "mcnemar_p_raw": float(binomtest(b, b + c, p=0.5, alternative="two-sided").pvalue) if b + c else 1.0}
    for j, metric in enumerate(("macro_f1", "dice", "iou"), start=1):
        score_name = metric if metric == "macro_f1" else f"mean_{metric}"
        row.update({f"mags_patient_{score_name}": scores[0, j], f"variant_patient_{score_name}": scores[1, j],
                    f"delta_patient_{metric}": differences[j], f"{metric}_ci_low": intervals[0, j], f"{metric}_ci_high": intervals[1, j]})
    return pd.DataFrame([row])

# %%
def multitask_comparison_tables(results, complexity):
    assert len(results) == 1 and complexity.model.is_unique
    assert set(complexity.model) == set(STATISTICS_MODELS)
    measured = complexity.set_index("model").loc[STATISTICS_MODELS]
    assert measured[["parameters", "macs", "fp32_size_mib"]].gt(0).all().all()
    assert measured.classification.eq(True).all() and measured.segmentation.eq(True).all()
    np.testing.assert_allclose(measured.fp32_size_mib, measured.parameters * 4 / 2**20)
    result, summary = results.iloc[0], []
    for name, prefix in zip(STATISTICS_MODELS, ("mags", "variant")):
        summary.append({"model": name, "patients": int(result.n_patients),
            "patient_accuracy": result[f"{prefix}_patient_accuracy"], "patient_macro_f1": result[f"{prefix}_patient_macro_f1"],
            "patient_mean_dice": result[f"{prefix}_patient_mean_dice"], "patient_mean_iou": result[f"{prefix}_patient_mean_iou"],
            "parameters": int(measured.loc[name, "parameters"]), "macs": int(measured.loc[name, "macs"]),
            "fp32_size_mib": float(measured.loc[name, "fp32_size_mib"])})
    summary = pd.DataFrame(summary)
    specifications = [("Patient Accuracy (%)", "patient_accuracy", 100, "accuracy_pp", "accuracy_ci_low_pp", "accuracy_ci_high_pp", 2),
                      ("Patient Macro-F1", "patient_macro_f1", 1, "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", 4),
                      ("Mean patient Dice", "patient_mean_dice", 1, "dice", "dice_ci_low", "dice_ci_high", 4),
                      ("Mean patient IoU", "patient_mean_iou", 1, "iou", "iou_ci_low", "iou_ci_high", 4)]
    table = []
    for label, column, scale, delta, low, high, decimals in specifications:
        suffix = " pp" if scale == 100 else ""
        table.append({"Metric": label, "MAGS-Net": f"{scale * summary.iloc[0][column]:.{decimals}f}",
            "MT-U-Net": f"{scale * summary.iloc[1][column]:.{decimals}f}",
            "Δ (MAGS-Net - MT-U-Net)": f"{result[f'delta_patient_{delta}']:+.{decimals}f}{suffix}",
            "95% CI for Δ": f"[{result[low]:+.{decimals}f}, {result[high]:+.{decimals}f}]{suffix}",
            "McNemar exact p": f"{result.mcnemar_p_raw:.8g}" if scale == 100 else "—"})
    for label, column, formatter in (("Parameters", "parameters", "{:,.0f}"), ("MACs", "macs", "{:,.0f}"),
                                     ("FP32 weights (MiB)", "fp32_size_mib", "{:.3f}")):
        table.append({"Metric": label, "MAGS-Net": formatter.format(summary.iloc[0][column]),
            "MT-U-Net": formatter.format(summary.iloc[1][column]), "Δ (MAGS-Net - MT-U-Net)": "—",
            "95% CI for Δ": "—", "McNemar exact p": "—"})
    return summary, pd.DataFrame(table)

# %%
PAIRED_PATIENT_STATISTICS, MULTITASK_SUMMARY, MULTITASK_TABLE = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
STATISTICS_INPUTS = {name: EXPORT_DIR / f"{'MT-UNet' if name == 'MT-U-Net' else name}_oof.csv" for name in STATISTICS_MODELS}
if set(COVERAGE.loc[COVERAGE.status == "complete", "model"]) != set(STATISTICS_MODELS):
    print("Paired statistics pending: complete the saved MAGS-Net and MT-U-Net OOF results first.")
else:
    statistics_oof = {name: pd.read_csv(path, dtype={"sample_id": str, "patient_id": str, "filename": str})
                      for name, path in STATISTICS_INPUTS.items()}
    PAIRED_PATIENT_STATISTICS = paired_patient_statistics(statistics_oof, STATISTICS_CONFIG)
    complexity_path = EXPORT_DIR / "model_complexity.csv"
    MULTITASK_SUMMARY, MULTITASK_TABLE = multitask_comparison_tables(PAIRED_PATIENT_STATISTICS, pd.read_csv(complexity_path))
    PAIRED_PATIENT_STATISTICS.to_csv(EXPORT_DIR / "paired_patient_statistics.csv", index=False)
    MULTITASK_TABLE.iloc[:4].to_csv(EXPORT_DIR / "paired_patient_statistics_table.csv", index=False, encoding="utf-8-sig")
    MULTITASK_SUMMARY.to_csv(EXPORT_DIR / "multitask_comparison_summary.csv", index=False)
    MULTITASK_TABLE.to_csv(EXPORT_DIR / "multitask_comparison_table.csv", index=False, encoding="utf-8-sig")
    patient_classes = statistics_oof["MAGS-Net"].drop_duplicates("patient_id").true_label.value_counts().sort_index()
    save_json(EXPORT_DIR / "paired_patient_statistics_metadata.json", {
        "analysis_version": "patient_multitask_comparison_v2", "experiment_hash": EXPERIMENT_HASH,
        "source_mags_hash": MAGS_HASH, "settings": STATISTICS_CONFIG, "direction": "MAGS-Net minus MT-U-Net",
        "n_patients": int(PAIRED_PATIENT_STATISTICS.n_patients.iloc[0]),
        "patient_counts_by_class": {CLASS_NAMES[int(key)]: int(value) for key, value in patient_classes.items()},
        "accuracy": "Argmax after averaging patient slice probabilities; differences in percentage points",
        "macro_f1": "Mean class 0/1/2 F1, zero_division=0; recomputed in each draw; differences on the 0–1 scale",
        "segmentation": "Mean of per-patient means of slice Dice/IoU; equal weight per patient; direct IoU aggregation; 0–1 scale",
        "mcnemar": "Two-sided exact binomial test of paired correctness; zero discordance gives p=1",
        "correction": "None: one model comparison; bootstrap CIs are pointwise, not simultaneous",
        "bootstrap": "Patient resampling within true class; shared indices for both models and all four metrics; percentile CI",
        "ci_scope": "Conditional on fixed fitted OOF models and folds; excludes retraining/partition variability",
        "complexity_input": "Batch=1, 1x224x224, FP32; full multitask forward",
        "macs_scope": "Conv2d, ConvTranspose2d, Linear only; exclude bias adds, normalization, activation, pooling, interpolation, elementwise operations",
        "fp32_size": "Parameters * 4 / 2**20 MiB; excludes buffers, activations, optimizer, checkpoint overhead",
        "complexity_sha256": file_hash(complexity_path),
        "source_oof_sha256": {path.name: file_hash(path) for path in STATISTICS_INPUTS.values()},
    })
    print(f"MAGS-Net vs MT-U-Net | {int(PAIRED_PATIENT_STATISTICS.n_patients.iloc[0])} paired OOF patients | {STATISTICS_CONFIG['resamples']:,} bootstrap draws | seed {STATISTICS_CONFIG['seed']}")
    display(MULTITASK_TABLE)
    print(f"McNemar exact p = {PAIRED_PATIENT_STATISTICS.mcnemar_p_raw.iloc[0]:.8g}; no Holm correction for one comparison.")
    print("Intervals are pointwise 95% CIs. Complexity uses the full multitask model and the stated MAC counting scope.")
    print("Comparison CSVs and metadata saved:", EXPORT_DIR.relative_to(PROJECT_ROOT))

# %% [markdown]
# ## 23. Run Training Later
# 1. Select Conda `dl` and run from the top with both switches False for preparation only.
# 2. Set RUN_TRAINING=True, RUN_OUTER_TEST=False, STUDY_DIR=None and run from the top.
# 3. Save the printed study path. After all 30 fits finish, set STUDY_DIR to that path,
#    RUN_TRAINING=False and RUN_OUTER_TEST=True; run from the top to evaluate all five folds.
# 4. Later, retain STUDY_DIR and disable both switches to analyze cached predictions.
# Reusing an existing study requires the same configuration and folds. An incomplete fit
# must be explicitly restarted; complete verified fits are reused. Do not change widths,
# thresholds, loss weights, or training settings in response to outer-test results.
