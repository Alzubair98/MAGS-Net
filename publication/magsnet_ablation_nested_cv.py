# %% [markdown]
# # 04. MAGS-Net Gate Ablations: Nested Patient-Level CV
# A0 reuses the completed full MAGS-Net experiment. A1–A4 are new scratch experiments.
# All variants retain the segmentation decoder, classification head, localization heads,
# and the original classification, segmentation, and auxiliary losses.
# Training and new outer-test inference are disabled by default. Use Conda `dl`.

# %% [markdown]
# ## 0. Experiment Configuration
# Same saved patient folds and seed 42. Each new variant performs 25 inner fits and five
# fresh outer refits: **120 new fits**. Inner selection uses validation total multitask loss.

# %%
CONFIG = {
    "protocol": "magsnet_gate_ablation_nested_patient_cv_v1", "seed": 42,
    "image_size": 224, "batch_size": 32, "learning_rate": 0.001, "weight_decay": 1e-4,
    "max_epochs": 50, "patience": 10, "scheduler_step": 10, "scheduler_gamma": 0.5,
    "lambda_seg": 0.5, "lambda_aux": 0.2, "dice_smooth": 1.0,
    "mask_threshold": 0.5, "use_amp": True, "gradient_clip": 1.0,
}
VARIANTS = {
    "A0": {"label": "Full MAGS-Net", "gate_mode": "learned", "lesion_guidance": True},
    "A1": {"label": "Fixed 50/50 gate", "gate_mode": "fixed", "lesion_guidance": False},
    "A2": {"label": "3x3 only", "gate_mode": "3x3", "lesion_guidance": False},
    "A3": {"label": "5x5 only", "gate_mode": "5x5", "lesion_guidance": False},
    "A4": {"label": "No lesion guidance in scale gate", "gate_mode": "learned", "lesion_guidance": False},
}
TRAIN_VARIANTS = ["A1", "A2", "A3", "A4"]
OBJECTIVE = "Cls + 0.5 Seg + 0.2 Aux; Dice smooth=1"
CLASS_NAMES = {0: "Meningioma", 1: "Glioma", 2: "Pituitary"}
PROBABILITY_COLUMNS = ["prob_meningioma", "prob_glioma", "prob_pituitary"]
GATE_COLUMNS = [f"stage{stage}_a{kernel}" for stage in range(1, 4) for kernel in (3, 5)]
MAGS_STUDY_DIR = "results/magsnet_nested_cv/20260916T210025Z_bc27ff71"
STUDY_DIR = None  # Set your own completed study path for analysis.
RUN_TRAINING = False
RUN_OUTER_TEST = False

# %% [markdown]
# ## 1. Imports and Environment
# No pretrained weights, package installation, or downloads. This notebook is self-contained.

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
# Reset seed 42 before each fit. All new results live in a separate ablation study.

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
RESULT_ROOT = PROJECT_ROOT / "results/magsnet_ablation_nested_cv"

# %% [markdown]
# ## 3. Read the Same Saved Outer/Inner Folds
# Load MAGS-Net's existing CSVs and verify exact agreement with its experiment record.
# No splitter is called. The source experiment, dataset bytes, and patient assignments
# identify the reference used by all ablations.

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
source_config = MAGS_EXPERIMENT["protocol"]["config"]
for key in ("image_size", "batch_size", "max_epochs", "patience", "scheduler_step", "scheduler_gamma", "mask_threshold", "use_amp"):
    assert CONFIG[key] == source_config[key]
candidate = MAGS_EXPERIMENT["protocol"]["candidates"]
assert len(candidate) == 1
for key in ("learning_rate", "weight_decay", "lambda_seg", "lambda_aux"):
    assert CONFIG[key] == candidate[0][key]
print("Exact saved fold source:", MAGS_STUDY_DIR)

# %% [markdown]
# ## 4. Dataset Integrity and Patient Leakage Checks
# Same master metadata and paired cleaned/cropped PNGs: 3,064 slices, 233 patients.
# Labels remain Meningioma=0, Glioma=1, Pituitary=2. Verify all image/mask bytes and
# patient separation before any training.

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

# %% [markdown]
# ## 5. Paired Preprocessing and Augmentation
# Identical to MAGS-Net: grayscale image resize to 224x224 with linear interpolation,
# division by 255, binary mask resize with nearest-neighbor interpolation. Training pairs
# share horizontal/vertical flips and a uniform rotation from -10 to +10 degrees.

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
# ## 6. Dataset and DataLoaders
# GT masks are targets, never model inputs or gate descriptors. Retain MAGS-Net's sampler:
# batch 32, merging a final singleton into the previous batch without dropping images.

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
# ## 7. Ablation Definitions
# | ID | Scale gate | Main feature branches | Gate descriptor |
# | --- | --- | --- | --- |
# | A0 | Learned | 3x3 + 5x5 | Global + predicted-lesion-masked |
# | A1 | Fixed 0.5 / 0.5 | 3x3 + 5x5 | None |
# | A2 | Fixed 1 / 0 | 3x3 | None |
# | A3 | Fixed 0 / 1 | 5x5 | None |
# | A4 | Learned | 3x3 + 5x5 | Global only |
# A1–A3 remove the unused gate MLP; A2/A3 also remove the unused main scale branch.
# Localization heads remain present, including their original 3x3 convolutions.
# "5x5 only" therefore describes the main scale branch, not every convolution in the network.

# %%
ABLATION_PLAN = pd.DataFrame([{"variant": key, **value, "new_training": key in TRAIN_VARIANTS,
                             "inner_fits": 25 if key in TRAIN_VARIANTS else 0,
                             "outer_refits": 5 if key in TRAIN_VARIANTS else 0} for key, value in VARIANTS.items()])
display(ABLATION_PLAN)

# %% [markdown]
# ## 8. Stem and MAG Block
# A0 follows the original MAGS-Net operations exactly. A4 computes the scale gate from
# global context alone: its first gate layer takes C inputs rather than 2C, with the same
# hidden width and two scale outputs. Report this small parameter difference explicitly.
# Localization is still computed and returned; the auxiliary targets are unchanged.

# %%
def conv_bn_relu(inputs, outputs, kernel=1, stride=1, groups=1):
    return nn.Sequential(nn.Conv2d(inputs, outputs, kernel, stride, kernel // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))


class Stem(nn.Sequential):
    def __init__(self):
        super().__init__(*conv_bn_relu(1, 32, kernel=3, stride=2))

# %%
class MAGBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate_mode, self.lesion_guidance = "learned", True
        self.proj = conv_bn_relu(channels, channels)
        self.dw3 = conv_bn_relu(channels, channels, kernel=3, groups=channels)
        self.dw5 = conv_bn_relu(channels, channels, kernel=5, groups=channels)
        self.localization = nn.Sequential(*conv_bn_relu(channels, channels, kernel=3, groups=channels), nn.Conv2d(channels, 1, 1))
        self.gate_fc1 = nn.Linear(2 * channels, max(channels // 8, 8))
        self.gate_fc2 = nn.Linear(max(channels // 8, 8), 2 * channels)
        self.fusion = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.Dropout2d(0.1))

    def forward(self, x):
        features = self.proj(x)
        branch3 = self.dw3(features) if self.dw3 is not None else None
        branch5 = self.dw5(features) if self.dw5 is not None else None
        localization = self.localization(features)
        if self.gate_mode == "learned":
            context = F.adaptive_avg_pool2d(features, 1).flatten(1)
            if self.lesion_guidance:
                mask = localization.sigmoid()
                masked = (features * mask).sum((2, 3)) / (mask.sum((2, 3)) + 1e-6)
                context = torch.cat([context, masked], dim=1)
            weights = self.gate_fc2(F.relu(self.gate_fc1(context)))
            weights = weights.reshape(x.shape[0], 2, x.shape[1]).softmax(dim=1)
        else:
            alpha3 = {"fixed": 0.5, "3x3": 1.0, "5x5": 0.0}[self.gate_mode]
            weights = features.new_tensor([alpha3, 1 - alpha3])[None, :, None].expand(x.shape[0], 2, x.shape[1])
        alpha3, alpha5 = weights[:, 0, :, None, None], weights[:, 1, :, None, None]
        mixed = alpha3 * branch3 if branch3 is not None else torch.zeros_like(features)
        if branch5 is not None:
            mixed = mixed + alpha5 * branch5
        output = F.relu(self.fusion(mixed) + x)
        return output, localization, weights.mean(dim=2)

# %% [markdown]
# ## 9. Unchanged Stages, Segmentation Decoder, and Classification Head
# Every variant retains stages [2,4,4] at channels [32,96,192], the same decoder, and the
# same classification head. **A4 removes lesion guidance from scale gates only**;
# the classifier's predicted-mask pooling is intentionally retained.
# Original auxiliary supervision uses the final localization logits of each stage.

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
# ## 10. Construct Each Variant from Scratch
# Construct the original random model first, then apply the ablation. With the same seed,
# unchanged modules receive identical initial values. No trained A0 weights initialize A1–A4.
# A4 keeps the global-input columns and bias of its freshly initialized first gate layer;
# the lesion-input columns are removed. This retains the shared initialization while making
# the gate input genuinely global-only. Gate/branch removals happen before optimizer creation.

# %%
def configure_variant(model, name):
    spec = VARIANTS[name]
    for block in model.modules():
        if not isinstance(block, MAGBlock):
            continue
        block.gate_mode, block.lesion_guidance = spec["gate_mode"], spec["lesion_guidance"]
        if spec["gate_mode"] != "learned":
            block.gate_fc1 = block.gate_fc2 = None
        elif not spec["lesion_guidance"]:
            original = block.gate_fc1
            channels = original.in_features // 2
            with torch.random.fork_rng(devices=[]):
                layer = nn.Linear(channels, original.out_features)
            with torch.no_grad():
                layer.weight.copy_(original.weight[:, :channels])
                layer.bias.copy_(original.bias)
            block.gate_fc1 = layer
        if spec["gate_mode"] == "3x3":
            block.dw5 = None
        elif spec["gate_mode"] == "5x5":
            block.dw3 = None
    return model


def build_model(name):
    assert name in VARIANTS
    return configure_variant(MAGSNet(), name)


@torch.inference_mode()
def forward_checks():
    rows = []
    for name in VARIANTS:
        with torch.random.fork_rng(devices=[]):
            model = build_model(name).to(DEVICE).eval()
        outputs = model(torch.zeros(1, 1, 224, 224, device=DEVICE))
        assert outputs["logits"].shape == (1, 3) and outputs["seg_logits"].shape == (1, 1, 224, 224)
        assert len(outputs["aux_masks"]) == 3 and outputs["gates"].shape == (1, 3, 2)
        assert all(torch.isfinite(outputs[key]).all() for key in ("logits", "seg_logits", "gates"))
        torch.testing.assert_close(outputs["gates"].sum(-1), torch.ones(1, 3, device=DEVICE))
        rows.append({"variant": name, "label": VARIANTS[name]["label"], "forward_check": "passed"})
    return pd.DataFrame(rows)


display(forward_checks())

# %% [markdown]
# ## 11. Identical Multitask Loss for Every Variant
# Total = Cls + 0.5 Seg + 0.2 Aux. Classification is Cross-Entropy. Segmentation is
# BCEWithLogits + mean per-slice soft Dice loss, smooth=1. Auxiliary localization uses
# the same loss at the final block of each stage, averaged across three stages.
# Keeping this original supervision does not add new auxiliary targets to earlier blocks.

# %%
def classification_loss(logits, labels):
    return F.cross_entropy(logits.float(), labels)


def segmentation_loss(logits, mask):
    logits, mask = logits.float(), mask.float()
    probabilities, target = logits.sigmoid().flatten(1), mask.flatten(1)
    dice = (2 * (probabilities * target).sum(1) + 1) / (probabilities.sum(1) + target.sum(1) + 1)
    return F.binary_cross_entropy_with_logits(logits, mask) + 1 - dice.mean()


def auxiliary_loss(auxiliary, mask):
    losses = [segmentation_loss(logits, F.interpolate(mask, size=logits.shape[-2:], mode="nearest"))
              for logits in auxiliary]
    return torch.stack(losses).mean()


def multitask_loss(outputs, labels, masks, candidate):
    cls = classification_loss(outputs["logits"], labels)
    seg = segmentation_loss(outputs["seg_logits"], masks)
    aux = auxiliary_loss(outputs["aux_masks"], masks)
    total = cls + candidate["lambda_seg"] * seg + candidate["lambda_aux"] * aux
    return {"loss": total, "cls": cls, "seg": seg, "aux": aux}

# %% [markdown]
# ## 12. Save the Fixed Ablation Protocol
# Same Adam/StepLR, early stopping, augmentation, precision, clipping, folds, and seed as A0.
# A1–A4 select their own durations. No A0 best epoch is reused to train a new variant.

# %%
assert TRAIN_VARIANTS == ["A1", "A2", "A3", "A4"] and CONFIG["seed"] == 42 and CONFIG["dice_smooth"] == 1.0
EXPERIMENT = {"config": CONFIG, "variants": VARIANTS, "train_variants": TRAIN_VARIANTS, "classes": CLASS_NAMES,
              "objective": OBJECTIVE, "mags_experiment_hash": MAGS_HASH, "data_hash": DATA_HASH,
              "torch": str(torch.__version__), "a4_gate_input": "C global-only; same hidden width; no masked descriptor",
              "initialization": "fresh full-model initialization; retain shared weights; A4 keeps global-input columns",
              "epoch_selection": "median of five minimum-validation-total-loss epochs"}
EXPERIMENT_HASH = signature(EXPERIMENT)
PREPARATION_DIR = RESULT_ROOT / "preparation" / EXPERIMENT_HASH[:12]
PREPARATION_DIR.mkdir(parents=True, exist_ok=True)
save_json(PREPARATION_DIR / "protocol.json", EXPERIMENT)
save_json(PREPARATION_DIR / "environment.json", ENVIRONMENT)
OUTER.to_csv(PREPARATION_DIR / "outer_patients.csv", index=False)
INNER.to_csv(PREPARATION_DIR / "inner_patients.csv", index=False)
INDEX.to_csv(PREPARATION_DIR / "sample_index.csv", index=False)
ABLATION_PLAN.to_csv(PREPARATION_DIR / "ablation_plan.csv", index=False)
print("Planned new fits:", len(TRAIN_VARIANTS) * 30, "| A0 uses its existing results.")

# %% [markdown]
# ## 13. Prediction Schema and Epoch Metrics
# Save class probabilities, per-slice Dice/IoU, GT tumor fraction, and stage-level gates.
# Binary segmentation threshold stays 0.5. Metrics and losses aggregate the entire epoch.

# %%
PREDICTION_COLUMNS = ["model", "outer_fold", "sample_id", "patient_id", "filename", "true_label",
                      "predicted_label", *PROBABILITY_COLUMNS, "dice", "iou", "tumor_fraction", *GATE_COLUMNS, "seed"]

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
    gates = outputs["gates"].detach().float().cpu().numpy()
    for stage in range(3):
        rows[f"stage{stage + 1}_a3"], rows[f"stage{stage + 1}_a5"] = gates[:, stage, 0], gates[:, stage, 1]
    return rows

def score_predictions(rows):
    return {"accuracy": float(accuracy_score(rows.true_label, rows.predicted_label)),
            "macro_f1": float(f1_score(rows.true_label, rows.predicted_label, labels=[0, 1, 2], average="macro", zero_division=0)),
            "dice": float(rows.dice.mean()), "iou": float(rows.iou.mean())}

# %%
def run_epoch(model, loader, optimizer=None, scaler=None, mask_dir=None):
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
                losses = multitask_loss(outputs, labels, masks, CONFIG)
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
                  f"Aux={metrics['aux']:.4f} | Acc={metrics['accuracy']:.2%} | F1={metrics['macro_f1']:.4f} | "
                  f"Dice={metrics['dice']:.4f} | IoU={metrics['iou']:.4f}")
    gates = validation if validation is not None else train
    print(f"Gates | a3={gates['a3']:.4f} | a5={gates['a5']:.4f}")
    if best is not None:
        print(f"Best  | Epoch={best['epoch']} | Val Loss={best['loss']:.4f} | Patience={stale}/{CONFIG['patience']}")

# %% [markdown]
# ## 14. Independent Fits and Checkpoints
# Each fit creates a fresh model, Adam optimizer, and StepLR. Best inner checkpoint means
# the minimum validation total loss; ties retain the earlier epoch and count toward patience.
# Reuse verified completed fits only. Interrupted fits require an explicit restart.

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
    assert name in TRAIN_VARIANTS, "A0 is a saved reference and must not be retrained here."
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
# ## 15. Variant-Specific Inner Selection and Outer Retraining
# Within each outer fold, fit the five inner partitions for this variant, take their median
# best epoch, then refit from scratch on all development patients for that fixed duration.
# The outer refit never receives a validation or test loader.

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
# ## 16. Run A1–A4 (Training Disabled)
# A0 is explicitly excluded from training. STUDY_DIR=None creates a new study only when
# training is enabled; an explicit path reuses matching completed fits.

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
        print("Ablation study directory:", directory)
    return directory

ACTIVE_STUDY = open_study()
if RUN_TRAINING:
    for name in TRAIN_VARIANTS:
        for fold in range(1, 6):
            directory = ACTIVE_STUDY / name / f"outer_{fold}"
            selected = select_duration(name, fold, directory)
            refit_outer(name, fold, selected, directory)
    save_json(ACTIVE_STUDY / "training_complete.json", {"experiment_hash": EXPERIMENT_HASH,
                                                       "variants": TRAIN_VARIANTS, "outer_refits": 20})
else:
    print("Training disabled. No model weights are updated.")

# %% [markdown]
# ## 17. Checkpoint and Prediction Integrity
# Check the variant, experiment, patient assignments, own selected epochs, and file hashes.
# Fixed-gate predictions must also satisfy their known gate values.

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
    gates = frame[GATE_COLUMNS].to_numpy()
    assert np.isfinite(gates).all() and ((gates >= 0) & (gates <= 1)).all()
    for stage in range(1, 4):
        assert np.allclose(frame[f'stage{stage}_a3'] + frame[f'stage{stage}_a5'], 1, atol=0.01)
        if model_name in ('A1', 'A2', 'A3'):
            expected_alpha = {'A1': 0.5, 'A2': 1.0, 'A3': 0.0}[model_name]
            assert np.allclose(frame[f'stage{stage}_a3'], expected_alpha, atol=1e-7)
            assert np.allclose(frame[f'stage{stage}_a5'], 1 - expected_alpha, atol=1e-7)

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
# ## 18. Outer-Test Evaluation and Verified Cache
# Complete all 20 new outer refits before evaluation. Each is evaluated once; future runs
# reuse matching prediction CSVs and masks. Outer performance never changes training settings.

# %%
def read_cached_predictions(name, directory, expected, fold, checkpoint_hash):
    metadata = json.loads((directory / "evaluation.json").read_text())
    path = directory / "outer_predictions.csv"
    assert metadata["experiment_hash"] == EXPERIMENT_HASH and metadata["checkpoint_hash"] == checkpoint_hash
    assert metadata["predictions_hash"] == file_hash(path)
    assert set(metadata["mask_hashes"]) == set(expected.filename)
    for filename, digest in metadata["mask_hashes"].items():
        assert file_hash(directory / "predicted_masks" / filename) == digest
    rows = pd.read_csv(path, dtype={"sample_id": str, "patient_id": str, "filename": str})
    validate_predictions(rows, expected, name, fold)
    return rows

# %%
def evaluate_outer(name, fold, directory):
    selected, summary, checkpoint_path = read_outer_checkpoint(name, fold, directory)
    test_ids = OUTER.loc[OUTER.outer_fold == fold, "patient_id"]
    test = INDEX[INDEX.patient_id.isin(test_ids)]
    if (directory / "evaluation.json").exists():
        return read_cached_predictions(name, directory, test, fold, summary["checkpoint_hash"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["model"] == name and checkpoint["signature"] == summary["signature"]
    assert checkpoint["experiment_hash"] == EXPERIMENT_HASH and checkpoint["epoch"] == selected["epochs"]
    assert checkpoint["objective"] == OBJECTIVE and checkpoint["weights"] is None
    model = build_model(name).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    mask_dir = directory / "predicted_masks"
    mask_dir.mkdir(exist_ok=True)
    metrics, rows = run_epoch(model, make_loader(test), mask_dir=mask_dir)
    rows["model"], rows["outer_fold"], rows["seed"] = name, fold, CONFIG["seed"]
    rows = rows[PREDICTION_COLUMNS]
    validate_predictions(rows, test, name, fold)
    rows.to_csv(directory / "outer_predictions.csv", index=False)
    pd.DataFrame([{"model": name, "outer_fold": fold, **metrics}]).to_csv(directory / "outer_metrics.csv", index=False)
    save_json(directory / "evaluation.json", {"experiment_hash": EXPERIMENT_HASH,
        "checkpoint_hash": summary["checkpoint_hash"], "predictions_hash": file_hash(directory / "outer_predictions.csv"),
        "mask_hashes": {filename: file_hash(mask_dir / filename) for filename in test.filename}, "test_metrics": metrics})
    return rows

if RUN_OUTER_TEST:
    if ACTIVE_STUDY is None or not (ACTIVE_STUDY / "training_complete.json").exists():
        raise RuntimeError("Complete all ablation fits and set STUDY_DIR before outer testing.")
    completed = json.loads((ACTIVE_STUDY / "training_complete.json").read_text())
    assert completed == {"experiment_hash": EXPERIMENT_HASH, "variants": TRAIN_VARIANTS, "outer_refits": 20}
    for name in TRAIN_VARIANTS:
        for fold in range(1, 6):
            read_outer_checkpoint(name, fold, ACTIVE_STUDY / name / f"outer_{fold}")
    for name in TRAIN_VARIANTS:
        for fold in range(1, 6):
            evaluate_outer(name, fold, ACTIVE_STUDY / name / f"outer_{fold}")
else:
    print("New outer-test inference disabled.")

# %% [markdown]
# ## 19. Pool OOF Predictions and Import A0
# A0 comes from the current completed MAGS-Net study, with checkpoint and prediction
# provenance checked. Its epochs are reference metadata only. No A0 inference is rerun.
# Every complete variant must cover 3,064 slices and 233 patients with zero duplicates.

# %%
def load_variant_oof(name, study):
    if study is None or not all((study / name / f"outer_{fold}/evaluation.json").exists() for fold in range(1, 6)):
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    parts = []
    for fold in range(1, 6):
        directory = study / name / f"outer_{fold}"
        _, summary, _ = read_outer_checkpoint(name, fold, directory)
        expected = INDEX[INDEX.patient_id.isin(OUTER.loc[OUTER.outer_fold == fold, "patient_id"])]
        parts.append(read_cached_predictions(name, directory, expected, fold, summary["checkpoint_hash"]))
    rows = pd.concat(parts, ignore_index=True)
    validate_predictions(rows, INDEX, name)
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

A0_OOF = load_mags_reference()
A0_OOF["model"] = "A0"
validate_predictions(A0_OOF, INDEX, "A0")
OOF_BY_MODEL = {"A0": A0_OOF, **{name: load_variant_oof(name, ACTIVE_STUDY) for name in TRAIN_VARIANTS}}
COVERAGE = pd.DataFrame([oof_coverage(rows, name) for name, rows in OOF_BY_MODEL.items()])
display(COVERAGE)

# %% [markdown]
# ## 20. Classification, Segmentation, and Paired Changes from A0
# Slice and patient metrics remain separate. Patient classification averages probabilities
# before argmax. Patient Dice/IoU average 2D slice scores and are not volumetric metrics.
# Report overall/per-class Dice and IoU. Paired slice changes match the same sample IDs.

# %%
def patient_predictions(rows):
    means = PROBABILITY_COLUMNS + GATE_COLUMNS + ["dice", "iou", "tumor_fraction"]
    patients = rows.groupby(["model", "outer_fold", "patient_id", "seed"], as_index=False).agg(
        true_label=("true_label", "first"), slices=("sample_id", "size"),
        **{column: (column, "mean") for column in means})
    patients["predicted_label"] = patients[PROBABILITY_COLUMNS].to_numpy().argmax(axis=1)
    return patients


def classification_metrics(rows, level):
    name = rows.model.iloc[0]
    return {"variant": name, "label": VARIANTS[name]["label"], "level": level, "seed": CONFIG["seed"], "samples": len(rows),
            "accuracy": float(accuracy_score(rows.true_label, rows.predicted_label)),
            "macro_f1": float(f1_score(rows.true_label, rows.predicted_label, labels=[0, 1, 2], average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(rows.true_label, rows.predicted_label)),
            "macro_ovr_auc": float(roc_auc_score(rows.true_label, rows[PROBABILITY_COLUMNS], labels=[0, 1, 2], multi_class="ovr", average="macro"))}


def segmentation_summary(rows, level):
    parts = [("Overall", rows)] + [(name, rows[rows.true_label == label]) for label, name in CLASS_NAMES.items()]
    return [{"variant": rows.model.iloc[0], "label": VARIANTS[rows.model.iloc[0]]["label"], "level": level, "class": name, "samples": len(part),
             "dice_mean": float(part.dice.mean()), "dice_std": float(part.dice.std(ddof=1)),
             "iou_mean": float(part.iou.mean()), "iou_std": float(part.iou.std(ddof=1))} for name, part in parts]

# %%
PATIENT_OOF, classification, segmentation, reports, per_fold, paired = {}, [], [], [], [], []
reference = A0_OOF.set_index("sample_id")
for name, rows in OOF_BY_MODEL.items():
    if rows.empty:
        continue
    PATIENT_OOF[name] = patient_predictions(rows)
    for level, frame in (("slice", rows), ("patient", PATIENT_OOF[name])):
        classification.append(classification_metrics(frame, level))
        segmentation.extend(segmentation_summary(frame, level))
        report = classification_report(frame.true_label, frame.predicted_label, labels=[0, 1, 2],
                                       target_names=list(CLASS_NAMES.values()), output_dict=True, zero_division=0)
        reports.extend({"variant": name, "level": level, "class": label, **report[label]} for label in CLASS_NAMES.values())
        for fold, part in frame.groupby("outer_fold"):
            per_fold.append({"outer_fold": fold, **classification_metrics(part, level),
                             "dice_mean": float(part.dice.mean()), "iou_mean": float(part.iou.mean())})
    if name != "A0":
        aligned = rows.set_index("sample_id").loc[reference.index]
        paired.append(pd.DataFrame({"variant": name, "sample_id": reference.index, "patient_id": reference.patient_id.to_numpy(),
            "dice_change_vs_a0": (aligned.dice - reference.dice).to_numpy(), "iou_change_vs_a0": (aligned.iou - reference.iou).to_numpy(),
            "correct_change_vs_a0": (aligned.predicted_label.eq(aligned.true_label).astype(int)
                                     - reference.predicted_label.eq(reference.true_label).astype(int)).to_numpy()}))
CLASSIFICATION = pd.DataFrame(classification)
SEGMENTATION = pd.DataFrame(segmentation)
display(CLASSIFICATION)
display(SEGMENTATION[SEGMENTATION.level == "slice"])

# %% [markdown]
# ## 21. Gate Summaries by Stage and Tumor Class
# Stage gates average channels within a block, then blocks within a stage. Keep stages
# separate here; the epoch summary weights the three stages by their block counts [2,4,4].

# %%
gate_rows = []
for name, rows in OOF_BY_MODEL.items():
    if rows.empty:
        continue
    for class_name, part in [("Overall", rows)] + [(label, rows[rows.true_label == k]) for k, label in CLASS_NAMES.items()]:
        for stage in range(1, 4):
            gate_rows.append({"variant": name, "class": class_name, "stage": stage, "slices": len(part),
                "a3_mean": float(part[f"stage{stage}_a3"].mean()), "a3_std": float(part[f"stage{stage}_a3"].std(ddof=1)),
                "a5_mean": float(part[f"stage{stage}_a5"].mean()), "a5_std": float(part[f"stage{stage}_a5"].std(ddof=1))})
GATE_SUMMARY = pd.DataFrame(gate_rows)
display(GATE_SUMMARY[GATE_SUMMARY["class"] == "Overall"])

# %% [markdown]
# ## 22. Parameters, Trainable Parameters, MACs, and FP32 Size
# Fresh models, one 1x224x224 input, FP32 eval mode. Count Conv2d/Linear MACs with the
# original MAGS-Net convention; exclude normalization, activations, pooling, interpolation,
# bias adds, and elementwise operations. FP32 size is parameter count * 4 / 2**20 MiB.
# Parameter counts reflect removed unused gate/scale modules in A1–A3 and the narrower
# gate input in A4. Localization modules remain registered in every variant. As in A0,
# only the final localization head of each stage receives direct auxiliary supervision;
# earlier localizers have no loss path once lesion-to-gate guidance is disabled.

# %%
@torch.inference_mode()
def model_complexity(name):
    with torch.random.fork_rng(devices=[]):
        model = build_model(name).cpu().float().eval()
    counts, handles = [], []
    def count(layer, inputs, output):
        products = (layer.in_channels // layer.groups) * np.prod(layer.kernel_size) if isinstance(layer, nn.Conv2d) else layer.in_features
        counts.append(int(output.numel() * products))
    try:
        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                handles.append(layer.register_forward_hook(count))
        model(torch.zeros(1, 1, 224, 224))
    finally:
        for handle in handles:
            handle.remove()
    parameters = sum(p.numel() for p in model.parameters())
    return {"variant": name, "label": VARIANTS[name]["label"], "parameters": parameters,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "macs": sum(counts), "fp32_size_mib": parameters * 4 / 2**20}


COMPLEXITY = pd.DataFrame([model_complexity(name) for name in VARIANTS])
saved_complexity = pd.read_csv(MAGS_STUDY / "analysis/complexity.csv").iloc[0]
assert COMPLEXITY.iloc[0].parameters == saved_complexity.parameters
assert COMPLEXITY.iloc[0].macs == saved_complexity.conv_linear_macs
display(COMPLEXITY)

# %% [markdown]
# ## 23. Selected Epochs and Comparison Figures
# Show each variant's own five inner best epochs and the verified final refit duration.
# Comparison figures are generated only after all A0–A4 OOF sets are complete.

# %%
def epoch_selection_table(study):
    rows = []
    if study is not None:
        for name in TRAIN_VARIANTS:
            for fold in range(1, 6):
                directory = study / name / f"outer_{fold}"
                if not (directory / "outer_refit/summary.json").exists():
                    continue
                selected, saved, _ = read_outer_checkpoint(name, fold, directory)
                rows.append({"variant": name, "outer_fold": fold, "inner_best_epochs": json.dumps(selected["inner_best_epochs"]),
                             "median_epoch": selected["epochs"], "outer_training_epochs": saved["epochs_completed"],
                             "selection_rule": selected["selection_rule"]})
    return pd.DataFrame(rows)


def comparison_figures(oof_sets):
    figures = {}
    for kind in ("confusion", "roc", "segmentation"):
        figure, axes = plt.subplots(2, 3, figsize=(15, 9))
        for ax, (name, rows) in zip(axes.flat, oof_sets.items()):
            if kind == "confusion":
                matrix = confusion_matrix(rows.true_label, rows.predicted_label, labels=[0, 1, 2])
                ax.imshow(matrix, cmap="Blues")
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                                color="white" if matrix[i, j] > matrix.max() / 2 else "black")
                ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES.values(), yticklabels=CLASS_NAMES.values(), xlabel="Predicted", ylabel="True")
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
            ax.set_title(f"{name}: {VARIANTS[name]['label']}")
        axes.flat[-1].set_visible(False)
        figure.tight_layout()
        figures[kind] = figure
    return figures


EPOCH_SELECTIONS = epoch_selection_table(ACTIVE_STUDY)
if not EPOCH_SELECTIONS.empty:
    display(EPOCH_SELECTIONS)

# %% [markdown]
# ## 24. Save Final CSVs, Masks, and Figures
# Per-variant A0_oof.csv ... A4_oof.csv and patient OOF files include classification,
# segmentation, and stage gate results. Pending variants have no invented performance.
# Histories, best/last weights, inner summaries, selections, masks, and per-fold metrics
# remain inside each variant/outer-fold directory. Analysis reuses saved predictions.

# %%
EXPORT_DIR = ACTIVE_STUDY / "analysis" if ACTIVE_STUDY is not None else PREPARATION_DIR / "reference"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
for name, rows in OOF_BY_MODEL.items():
    if not rows.empty:
        rows[PREDICTION_COLUMNS].to_csv(EXPORT_DIR / f"{name}_oof.csv", index=False)
        PATIENT_OOF[name].to_csv(EXPORT_DIR / f"{name}_patient_oof.csv", index=False)
CLASSIFICATION.to_csv(EXPORT_DIR / "classification_comparison.csv", index=False)
SEGMENTATION.to_csv(EXPORT_DIR / "segmentation_comparison.csv", index=False)
pd.DataFrame(reports).to_csv(EXPORT_DIR / "per_class_classification.csv", index=False)
pd.DataFrame(per_fold).to_csv(EXPORT_DIR / "outer_fold_metrics.csv", index=False)
COVERAGE.to_csv(EXPORT_DIR / "oof_coverage.csv", index=False)
COMPLEXITY.to_csv(EXPORT_DIR / "model_complexity.csv", index=False)
GATE_SUMMARY.to_csv(EXPORT_DIR / "gate_summary.csv", index=False)
ABLATION_PLAN.to_csv(EXPORT_DIR / "ablation_plan.csv", index=False)
if paired:
    pd.concat(paired, ignore_index=True).to_csv(EXPORT_DIR / "paired_slice_changes_vs_a0.csv", index=False)
if not EPOCH_SELECTIONS.empty:
    EPOCH_SELECTIONS.to_csv(EXPORT_DIR / "epoch_selections.csv", index=False)
if COVERAGE.status.eq("complete").all():
    for name, figure in comparison_figures(OOF_BY_MODEL).items():
        figure.savefig(EXPORT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
        figure.savefig(EXPORT_DIR / f"{name}.pdf", bbox_inches="tight")
        plt.show()
        plt.close(figure)
save_json(EXPORT_DIR / "analysis_status.json", {"experiment_hash": EXPERIMENT_HASH, "source_a0_hash": MAGS_HASH,
    "complete_variants": COVERAGE.loc[COVERAGE.status == "complete", "model"].tolist(),
    "training_enabled": RUN_TRAINING, "outer_test_enabled": RUN_OUTER_TEST,
    "a4_scope": "Remove predicted-lesion descriptor from scale gate only; retain decoder, classifier, and original auxiliary supervision",
    "macs_definition": "Conv2d and Linear only; batch=1, 1x224x224 input; full multitask forward",
    "fp32_size_definition": "Parameters * 4 / 2**20 MiB, excluding buffers and checkpoint overhead"})
print("Results saved:", EXPORT_DIR.relative_to(PROJECT_ROOT))

# %% [markdown]
# ### 24.1 Paired Patient-Level Statistical Comparisons
# Run after Sections 19–24; this section only reads saved OOF predictions.
# All differences are **A0 minus the ablation**. Patient predictions use mean slice
# probabilities before argmax. Patient Dice is the mean of that patient's slice Dice;
# the cohort mean gives every patient equal weight (not pooled-slice or volumetric Dice).
# #
# - **McNemar:** two-sided exact binomial test of paired patient correctness.
#   No discordant pairs gives p=1. Holm adjusts the four A0-vs-A1–A4 p-values together.
# - **Paired patient bootstrap:** 10,000 resamples, seed 42, stratified by true tumor
#   class to retain the observed class counts. Each draw uses the same patient IDs for
#   all models. Report percentile 95% CIs for the paired mean differences.
# - Dice CIs are **pointwise**, not Holm-adjusted or simultaneous. Holm applies only
#   to the four McNemar tests; Dice CIs are not multiplicity-adjusted significance tests.
#   Uncertainty is conditional on the fitted OOF models and saved folds; this procedure
#   does not estimate variability from retraining or alternative CV partitions.
# #
# Method references: [exact McNemar](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html),
# [Holm adjustment](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html),
# and [paired bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

# %%
from scipy.stats import binomtest

# Analysis settings are separate from CONFIG and never change the training protocol.
STATISTICS_CONFIG = {"resamples": 10_000, "seed": 42, "confidence_level": 0.95, "alpha": 0.05}
STATISTICS_VARIANTS = ["A0", "A1", "A2", "A3", "A4"]


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    assert p_values.ndim == 1 and len(p_values) > 0
    assert np.isfinite(p_values).all() and ((0 <= p_values) & (p_values <= 1)).all()
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(
        p_values[order] * np.arange(len(p_values), 0, -1)))
    return adjusted


def align_patient_results(oof_sets):
    patients = {}
    for name in STATISTICS_VARIANTS:
        validate_predictions(oof_sets[name], INDEX, name)
        frame = patient_predictions(oof_sets[name]).set_index("patient_id").sort_index()
        assert frame.index.is_unique and set(frame.index) == set(INDEX.patient_id)
        patients[name] = frame
    reference = patients["A0"]
    for frame in patients.values():
        assert frame.index.equals(reference.index)
        for column in ("true_label", "outer_fold", "seed", "slices"):
            assert np.array_equal(frame[column], reference[column]), f"Patient mismatch: {column}"
    return patients

# %%
def paired_patient_bootstrap(differences, labels, settings):
    """Resample patient rows jointly for every comparison and metric."""
    assert differences.shape[0] == len(labels) and np.isfinite(differences).all()
    assert settings["resamples"] > 0 and 0 < settings["confidence_level"] < 1
    rng = np.random.default_rng(settings["seed"])
    strata = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    estimates = np.empty((settings["resamples"], *differences.shape[1:]))
    for iteration in range(settings["resamples"]):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata])
        estimates[iteration] = differences[indices].mean(axis=0)
    tail = (1 - settings["confidence_level"]) / 2
    return np.quantile(estimates, [tail, 1 - tail], axis=0)


def paired_patient_statistics(oof_sets, settings):
    patients = align_patient_results(oof_sets)
    labels = patients["A0"].true_label.to_numpy()
    correct = np.column_stack([patients[name].predicted_label.to_numpy() == labels for name in STATISTICS_VARIANTS])
    dice = np.column_stack([patients[name].dice.to_numpy() for name in STATISTICS_VARIANTS])
    accuracy_changes = 100 * (correct[:, :1].astype(float) - correct[:, 1:].astype(float))
    dice_changes = dice[:, :1] - dice[:, 1:]
    differences = np.stack([accuracy_changes, dice_changes], axis=-1)
    intervals = paired_patient_bootstrap(differences, labels, settings)
    rows = []
    for j, name in enumerate(STATISTICS_VARIANTS[1:]):
        a0, other = correct[:, 0], correct[:, j + 1]
        b, c = int((a0 & ~other).sum()), int((~a0 & other).sum())
        rows.append({"comparison": f"A0 vs {name}", "reference": "A0", "variant": name,
            "n_patients": len(labels), "a0_correct": int(a0.sum()), "variant_correct": int(other.sum()),
            "both_correct": int((a0 & other).sum()), "both_wrong": int((~a0 & ~other).sum()),
            "a0_correct_variant_wrong": b, "a0_wrong_variant_correct": c, "discordant_pairs": b + c,
            "a0_patient_accuracy": float(a0.mean()), "variant_patient_accuracy": float(other.mean()),
            "delta_patient_accuracy_pp": float(accuracy_changes[:, j].mean()),
            "accuracy_ci_low_pp": intervals[0, j, 0], "accuracy_ci_high_pp": intervals[1, j, 0],
            "mcnemar_p_raw": float(binomtest(b, b + c, p=0.5, alternative="two-sided").pvalue) if b + c else 1.0,
            "a0_patient_mean_dice": float(dice[:, 0].mean()), "variant_patient_mean_dice": float(dice[:, j + 1].mean()),
            "delta_patient_dice": float(dice_changes[:, j].mean()),
            "dice_ci_low": intervals[0, j, 1], "dice_ci_high": intervals[1, j, 1]})
    results = pd.DataFrame(rows)
    results["mcnemar_p_adj_holm"] = holm_adjust(results.mcnemar_p_raw)
    results["mcnemar_reject_holm"] = results.mcnemar_p_adj_holm <= settings["alpha"]
    return results

# %%
def statistics_paper_table(results):
    return pd.DataFrame({
        "Comparison": results.comparison,
        "Δ Patient Acc": results.delta_patient_accuracy_pp.map(lambda x: f"{x:+.2f} pp"),
        "McNemar p_adj": results.mcnemar_p_adj_holm.map(lambda x: f"{x:.4f}" if x >= 0.0001 else "<0.0001"),
        "Δ Dice": results.delta_patient_dice.map(lambda x: f"{x:+.3f}"),
        "95% CI (Δ Dice)": [f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(results.dice_ci_low, results.dice_ci_high)],
    })


PAIRED_PATIENT_STATISTICS, PAIRED_PATIENT_TABLE = pd.DataFrame(), pd.DataFrame()
STATISTICS_INPUTS = {name: EXPORT_DIR / f"{name}_oof.csv" for name in STATISTICS_VARIANTS}
if set(COVERAGE.loc[COVERAGE.status == "complete", "model"]) != set(STATISTICS_VARIANTS):
    print("Paired statistics pending: complete the saved A0–A4 OOF results first.")
else:
    statistics_oof = {name: pd.read_csv(path, dtype={"sample_id": str, "patient_id": str, "filename": str})
                      for name, path in STATISTICS_INPUTS.items()}
    PAIRED_PATIENT_STATISTICS = paired_patient_statistics(statistics_oof, STATISTICS_CONFIG)
    PAIRED_PATIENT_TABLE = statistics_paper_table(PAIRED_PATIENT_STATISTICS)
    PAIRED_PATIENT_STATISTICS.to_csv(EXPORT_DIR / "paired_patient_statistics.csv", index=False)
    PAIRED_PATIENT_TABLE.to_csv(EXPORT_DIR / "paired_patient_statistics_table.csv", index=False, encoding="utf-8-sig")
    patient_classes = statistics_oof["A0"].drop_duplicates("patient_id").true_label.value_counts().sort_index()
    save_json(EXPORT_DIR / "paired_patient_statistics_metadata.json", {
        "experiment_hash": EXPERIMENT_HASH, "source_a0_hash": MAGS_HASH, "settings": STATISTICS_CONFIG,
        "comparisons": PAIRED_PATIENT_STATISTICS.comparison.tolist(), "direction": "A0 minus ablation",
        "patient_counts_by_class": {CLASS_NAMES[int(k)]: int(v) for k, v in patient_classes.items()},
        "accuracy": "Argmax of each patient's mean slice probabilities; differences in percentage points",
        "dice": "Mean of per-patient mean slice Dice; equal weight per patient; differences on the 0–1 scale",
        "mcnemar": "Two-sided exact binomial test on paired patient correctness; zero discordance gives p=1",
        "correction": "Holm step-down across exactly four McNemar tests: A0 vs A1, A2, A3, A4",
        "bootstrap": "Paired patient resampling within true tumor class; shared draws across models and metrics; percentile CI",
        "ci_scope": "Pointwise, not multiplicity-adjusted; conditional on fitted OOF models and saved folds",
        "source_oof_sha256": {path.name: file_hash(path) for path in STATISTICS_INPUTS.values()},
    })
    print(f"PAIRED PATIENT COMPARISONS | A0 - ablation | {STATISTICS_CONFIG['resamples']:,} paired bootstrap resamples | seed {STATISTICS_CONFIG['seed']}")
    display(PAIRED_PATIENT_TABLE)
    print(f"Holm applies to McNemar p-values only. Dice intervals are pointwise {STATISTICS_CONFIG['confidence_level']:.0%} CIs.")
    print("Statistics CSVs and method metadata saved:", EXPORT_DIR.relative_to(PROJECT_ROOT))

# %% [markdown]
# ## 25. Running the Experiment Later
# 1. Select Conda `dl`; default switches only audit data, run synthetic forward checks,
#    compute complexity, and read the real A0 reference. No training or new test inference.
# 2. When ready, set RUN_TRAINING=True, RUN_OUTER_TEST=False, STUDY_DIR=None and run from the top.
# 3. After all 120 fits finish, set STUDY_DIR to the printed path, RUN_TRAINING=False,
#    RUN_OUTER_TEST=True, then run from the top to evaluate the 20 outer refits.
# 4. For later analysis, retain STUDY_DIR and disable both switches. Verified cached results
#    are reused. A0 and previous experiment files are never modified by this notebook.
