# %% [markdown]
# # 05. ImageNet-Pretrained Classification: Nested Patient-Level CV
# Compare the saved **MAGS-Net (scratch, multitask)** reference against three image-only
# classifiers **fine-tuned from ImageNet weights with Cross-Entropy only**. This compares
# training systems, not an architecture-only ablation with identical supervision.
# This notebook is self-contained and uses the exact saved MAGS-Net patient assignments.
# Training and new outer-test inference are disabled by default.

# %% [markdown]
# ## 0. Experiment Configuration
# Seed 42; 25 inner fits and 5 outer refits for each classifier: **90 new fits** in total.
# MAGS-Net is read from its existing held-out results and is never retrained here.

# %%
CONFIG = {
    "protocol": "pretrained_classification_nested_cv_v1",
    "seed": 42,
    "image_size": 224,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "max_epochs": 50,
    "patience": 10,
    "scheduler_step": 10,
    "scheduler_gamma": 0.5,
    "use_amp": True,
    "gradient_clip": 1.0,
}
MODEL_NAMES = ["ResNet-50", "EfficientNet-B0", "MobileNetV3-Large"]
CLASS_NAMES = {0: "Meningioma", 1: "Glioma", 2: "Pituitary"}
PROBABILITY_COLUMNS = ["prob_meningioma", "prob_glioma", "prob_pituitary"]
MAGS_STUDY_DIR = "results/magsnet_nested_cv/20260916T210025Z_bc27ff71"
PRETRAINED_STUDY_DIR = None  # Set your own completed study path for analysis.
RUN_TRAINING = False
RUN_OUTER_TEST = False

# %% [markdown]
# ## 1. Imports and Environment
# Use Conda `dl`. The default run audits data and checks random architecture shapes only.
# ImageNet weights are loaded only when fitting is enabled; uncached official weights are
# downloaded by TorchVision on the first fit. Download failure stops the fit, with no scratch fallback.

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
import torchvision
from IPython.display import display
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import models
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENVIRONMENT = {"python": platform.python_version(), "torch": str(torch.__version__),
               "torchvision": str(torchvision.__version__), "numpy": np.__version__, "pandas": pd.__version__,
               "opencv": cv2.__version__, "device": str(DEVICE),
               "gpu": torch.cuda.get_device_name() if DEVICE.type == "cuda" else None}
print("Device:", DEVICE, "| TorchVision:", torchvision.__version__)


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
# The training seed is reset for every independent fit. Saved patient folds are never
# regenerated. Model initialization, optimizer, scheduler, and patience state are fresh.

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
RESULT_ROOT = PROJECT_ROOT / "results" / "pretrained_classification_nested_cv"

# %% [markdown]
# ## 3. Load the Same Saved Folds
# Read `experiment.json` and its corresponding preparation CSVs. Their contents must agree.
# There is no splitter call in this notebook. The source MAGS-Net experiment and data hashes
# are recorded in every baseline study and checkpoint.

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
print("Saved fold source:", MAGS_STUDY_DIR)
print("Outer folds:", OUTER.outer_fold.nunique(), "| Inner partitions:", INNER.groupby("outer_fold").inner_fold.nunique().sum())

# %% [markdown]
# ## 4. Verify the Cleaned Dataset and Patient Separation
# Verify the current metadata and image/mask content against MAGS-Net's saved fingerprint.
# Masks are read **only for this provenance audit**. Classifier datasets, forward passes,
# losses, and evaluation use MRI images and class labels only.

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
display(INDEX.groupby("label").agg(patients=("patient_id", "nunique"), slices=("sample_id", "size")))
print("Verified:", len(INDEX), "images |", INDEX.patient_id.nunique(), "patients | patient overlap = 0")

# %% [markdown]
# ## 5. Identical Image Preprocessing and Augmentation
# Read the same cropped grayscale PNGs. Resize with OpenCV linear interpolation to 224×224
# and divide by 255. No new crop, denoising, RGB replication, or ImageNet normalization.
# Training uses the same two independent 50% flips and uniform rotation in [−10°, +10°]
# as MAGS-Net, with the same center, interpolation, and zero border. Evaluation has no augmentation.

# %%
def preprocess_image(filename):
    image = cv2.imread(str(DATA_ROOT / "images" / filename), cv2.IMREAD_GRAYSCALE)
    size = (CONFIG["image_size"], CONFIG["image_size"])
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0


def augment_image(image):
    if random.random() < 0.5:
        image = np.fliplr(image)
    if random.random() < 0.5:
        image = np.flipud(image)
    height, width = image.shape
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), random.uniform(-10, 10), 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


class ClassificationDataset(Dataset):
    def __init__(self, frame, training=False):
        self.frame = frame.reset_index(drop=True)
        self.training = training

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, position):
        row = self.frame.iloc[position]
        image = preprocess_image(row.filename)
        if self.training:
            image = augment_image(image)
        return {"image": torch.from_numpy(image.copy()).unsqueeze(0), "label": int(row.label),
                "sample_id": row.sample_id, "patient_id": row.patient_id}

# %% [markdown]
# ## 6. DataLoaders
# The MAGS-Net sampler policy is retained: shuffle deterministically and merge a final
# singleton into the preceding training batch. This uses every image exactly once per epoch.
# The usual batch size is 32; the merged final batch, when needed, has 33 images.

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


def make_loader(frame, training=False):
    dataset = ClassificationDataset(frame, training)
    options = {"num_workers": 0, "pin_memory": DEVICE.type == "cuda"}
    if training:
        assert len(dataset) >= 2
        sampler = TrainingBatches(len(dataset), CONFIG["batch_size"], CONFIG["seed"])
        return DataLoader(dataset, batch_sampler=sampler, **options)
    return DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False, **options)

# %% [markdown]
# ## 7. ImageNet Initialization: One Input Channel and Three Outputs
# Pin the weight versions instead of using a moving DEFAULT alias. Each independent fit
# starts from the same ImageNet weights and a fresh three-class output layer. All layers,
# including BatchNorm, are fine-tuned; no backbone is frozen.
# #
# To keep MAGS-Net's exact one-channel input and /255 normalization, sum the pretrained
# RGB stem kernels across channels. This preserves the stem response to a grayscale image
# repeated across three channels. Keep the rest of the pretrained backbone unchanged.
# This is a controlled initialization comparison under the MRI protocol; it does not use
# the weights' RGB ImageNet normalization or resize/center-crop transforms, and is not a
# claim about the best independently tuned transfer-learning recipe.
# Official weights: [ResNet-50](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.resnet50.html),
# [EfficientNet-B0](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.efficientnet_b0.html),
# [MobileNetV3-Large](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.mobilenet_v3_large.html).

# %%
MODEL_SPECS = {
    "ResNet-50": (models.resnet50, "conv1", "fc", models.ResNet50_Weights.IMAGENET1K_V2),
    "EfficientNet-B0": (models.efficientnet_b0, "features.0.0", "classifier.1", models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "MobileNetV3-Large": (models.mobilenet_v3_large, "features.0.0", "classifier.3", models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
}
WEIGHT_SOURCES = {name: {"weights": str(spec[3]), "url": spec[3].url} for name, spec in MODEL_SPECS.items()}


def replace_module(model, location, module):
    parent, _, attribute = location.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, attribute, module)


def build_classifier(name, pretrained=True):
    constructor, stem_path, head_path, weights = MODEL_SPECS[name]
    model = constructor(weights=weights) if pretrained else constructor(weights=None, num_classes=3)
    original = model.get_submodule(stem_path)
    assert original.in_channels == 3 and original.groups == 1
    convolution = nn.Conv2d(1, original.out_channels, original.kernel_size, original.stride,
                            original.padding, original.dilation, original.groups,
                            bias=original.bias is not None, padding_mode=original.padding_mode)
    if pretrained:
        with torch.no_grad():
            convolution.weight.copy_(original.weight.sum(dim=1, keepdim=True))
            if original.bias is not None:
                convolution.bias.copy_(original.bias)
        head = nn.Linear(model.get_submodule(head_path).in_features, len(CLASS_NAMES))
        # Match each TorchVision architecture's standard output-layer initialization.
        if name == "EfficientNet-B0":
            nn.init.uniform_(head.weight, -1 / np.sqrt(head.out_features), 1 / np.sqrt(head.out_features))
            nn.init.zeros_(head.bias)
        elif name == "MobileNetV3-Large":
            nn.init.normal_(head.weight, 0, 0.01)
            nn.init.zeros_(head.bias)
        replace_module(model, head_path, head)
    else:
        # Used only for shape checks or before loading a complete trained checkpoint.
        nn.init.kaiming_normal_(convolution.weight, mode="fan_out", nonlinearity="relu")
        if convolution.bias is not None:
            nn.init.zeros_(convolution.bias)
    replace_module(model, stem_path, convolution)
    assert all(parameter.requires_grad for parameter in model.parameters())
    return model

# %%
@torch.inference_mode()
def forward_checks():
    rows = []
    images = torch.zeros(2, 1, CONFIG["image_size"], CONFIG["image_size"], device=DEVICE)
    for name in MODEL_NAMES:
        set_seed(CONFIG["seed"])
        model = build_classifier(name, pretrained=False).to(DEVICE).eval()
        logits = model(images)
        assert logits.shape == (2, 3) and torch.isfinite(logits).all()
        rows.append({"model": name, "planned_weights": WEIGHT_SOURCES[name]["weights"],
                     "input": "1 x 224 x 224", "parameters": sum(p.numel() for p in model.parameters()),
                     "check": "Random shape check only; no pretrained download or evaluation"})
        del model, logits
    return pd.DataFrame(rows)


MODEL_CHECKS = forward_checks()
display(MODEL_CHECKS)

# %% [markdown]
# ## 8. Training Plan and Provenance
# Every model independently chooses its duration using its own five inner-validation losses
# within each outer fold. No MAGS-Net epoch selection is read by the baseline fitting functions.
# The source experiment/data hashes, model list, software versions, and settings identify a study.

# %%
assert CONFIG["seed"] == 42 and CONFIG["image_size"] == 224
assert CONFIG["max_epochs"] == 50 and CONFIG["patience"] == 10
assert len(MODEL_NAMES) == len(set(MODEL_NAMES)) and set(MODEL_NAMES).issubset(MODEL_SPECS)
PRETRAINED_PROTOCOL = {"config": CONFIG, "models": MODEL_NAMES, "classes": CLASS_NAMES,
                     "mags_experiment_hash": MAGS_HASH, "data_hash": DATA_HASH,
                     "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                     "input": "grayscale_224_divide255", "initialization": WEIGHT_SOURCES,
                     "stem_adaptation": "Sum RGB kernels into one grayscale channel", "fine_tuning": "all layers",
                     "objective": "CrossEntropy", "epoch_selection": "median of five inner best CE epochs"}
PRETRAINED_HASH = signature(PRETRAINED_PROTOCOL)
PREPARATION_DIR = RESULT_ROOT / "preparation" / PRETRAINED_HASH[:12]
PREPARATION_DIR.mkdir(parents=True, exist_ok=True)
save_json(PREPARATION_DIR / "protocol.json", PRETRAINED_PROTOCOL)
save_json(PREPARATION_DIR / "environment.json", ENVIRONMENT)
OUTER.to_csv(PREPARATION_DIR / "outer_patients.csv", index=False)
INNER.to_csv(PREPARATION_DIR / "inner_patients.csv", index=False)
MODEL_CHECKS.to_csv(PREPARATION_DIR / "model_checks.csv", index=False)
display(pd.DataFrame({"model": MODEL_NAMES, "inner_fits": 25, "outer_refits": 5, "loss": "Cross-Entropy"}))
print("New training fits:", len(MODEL_NAMES) * 30)

# %% [markdown]
# ## 9. Cross-Entropy Training and Validation
# Adam, StepLR (10, 0.5), and gradient clipping follow the MAGS-Net general protocol.
# Validation selection uses **classification Cross-Entropy only**. No segmentation,
# auxiliary localization, or gate loss is constructed. Epoch loss is averaged over all images.

# %%
def classification_scores(frame):
    return {"accuracy": float(accuracy_score(frame.true_label, frame.predicted_label)),
            "macro_f1": float(f1_score(frame.true_label, frame.predicted_label, labels=[0, 1, 2], average="macro", zero_division=0))}


def run_epoch(model, loader, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, records = 0.0, []
    amp = CONFIG["use_amp"] and DEVICE.type == "cuda"
    dtype = torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc="Train" if training else "Evaluate", leave=False):
            images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, dtype=dtype, enabled=amp):
                logits = model(images)
                loss = F.cross_entropy(logits.float(), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite classification loss.")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"], error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
            loss_sum += float(loss.detach()) * len(labels)
            probabilities = logits.detach().float().softmax(1).cpu().numpy()
            rows = pd.DataFrame({"sample_id": batch["sample_id"], "patient_id": batch["patient_id"],
                                 "true_label": batch["label"].numpy(), "predicted_label": probabilities.argmax(1)})
            rows[PROBABILITY_COLUMNS] = probabilities
            records.append(rows)
    predictions = pd.concat(records, ignore_index=True)
    assert len(predictions) == len(loader.dataset) and predictions.sample_id.is_unique
    return {"loss": loss_sum / len(predictions), **classification_scores(predictions)}, predictions


def print_epoch(epoch, epochs, lr, training, validation, best, stale):
    print(f"Epoch {epoch:02d}/{epochs} | lr={lr:.6f}")
    for label, metrics in (("Train", training), ("Val", validation)):
        if metrics is not None:
            print(f"{label:5} | CE Loss={metrics['loss']:.4f} | Acc={metrics['accuracy']:.2%} | F1={metrics['macro_f1']:.4f}")
    if validation is not None:
        print(f"Best  | Epoch={best['epoch']} | Val CE={best['loss']:.4f} | Patience={stale}/{CONFIG['patience']}")

# %% [markdown]
# ## 10. Independent Fits and Checkpoints
# Inner fits stop after 10 consecutive epochs without a strict decrease in validation CE,
# or after epoch 50. Ties keep the earlier best epoch. Outer refits use a fixed duration and
# receive no validation/test loader. Checkpoints store model weights and provenance; partial
# fits are not automatically resumed. Verified completed fits can be reused in the same study.

# %%
def fit_signature(name, train, validation, epochs):
    return signature({"experiment": PRETRAINED_HASH, "model": name, "epochs": int(epochs),
                      "train": sorted(train.sample_id), "validation": [] if validation is None else sorted(validation.sample_id)})


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def fit_classifier(name, train, validation, epochs, directory):
    digest = fit_signature(name, train, validation, epochs)
    summary_path = directory / "summary.json"
    if summary_path.exists():
        saved = json.loads(summary_path.read_text())
        assert saved["signature"] == digest and saved["checkpoint_hash"] == file_hash(directory / saved["checkpoint"])
        return saved
    if directory.exists():
        raise RuntimeError(f"Incomplete fit: {directory}. Start a new study or explicitly restart this fit.")
    set_seed(CONFIG["seed"])
    model = build_classifier(name, pretrained=True).to(DEVICE)
    # A failed initial weight download should not leave an empty, incomplete fit.
    directory.mkdir(parents=True)
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
                   "signature": digest, "experiment_hash": PRETRAINED_HASH, "source_mags_hash": MAGS_HASH,
                   "objective": "CrossEntropy", "config": CONFIG, "weights": WEIGHT_SOURCES[name]["weights"],
                   "stem_adaptation": "sum_rgb", "fine_tuning": "all_layers"}
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
              "best_val_ce": best["loss"] if validation is not None else None,
              "checkpoint": checkpoint, "checkpoint_hash": file_hash(directory / checkpoint)}
    save_json(summary_path, result)
    return result

# %% [markdown]
# ## 11. Architecture-Specific Inner-CV Selection
# For each architecture and outer fold: fit five fresh models, record each inner best CE
# epoch, and choose their median. The models do **not** inherit MAGS-Net's selected epochs.

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
        result = fit_classifier(name, train, validation, CONFIG["max_epochs"], directory / f"inner_{inner_fold}")
        rows.append({"outer_fold": outer_fold, "inner_fold": inner_fold, **result})
    table = pd.DataFrame(rows)
    best_epochs = table.best_epoch.astype(int).tolist()
    selected = {"model": name, "outer_fold": outer_fold, "seed": CONFIG["seed"], "inner_best_epochs": best_epochs,
                "epochs": int(np.median(best_epochs)), "selection_rule": "Median of five minimum-validation-CE epochs",
                "experiment_hash": PRETRAINED_HASH}
    table.to_csv(directory / "inner_summary.csv", index=False)
    save_json(directory / "selection.json", selected)
    print("Inner best epochs:", best_epochs, "| Selected outer duration:", selected["epochs"])
    return selected


def refit_outer(name, outer_fold, selected, directory):
    development_ids = OUTER.loc[OUTER.outer_fold != outer_fold, "patient_id"]
    development = INDEX[INDEX.patient_id.isin(development_ids)]
    print(f"\n{name} | OUTER {outer_fold}/5 | FRESH REFIT: {selected['epochs']} epochs")
    return fit_classifier(name, development, None, selected["epochs"], directory / "outer_refit")

# %% [markdown]
# ## 12. Run Training (Disabled)
# Set `RUN_TRAINING=True` in section 0 and run from the top when ready. Keep outer testing
# disabled until every planned fit is complete. Save the printed baseline study directory.
# Re-running with `PRETRAINED_STUDY_DIR=None` creates a new study; an explicit matching path
# reuses completed fits. Existing MAGS-Net results are read-only throughout this notebook.

# %%
def open_pretrained_study():
    directory = Path(PRETRAINED_STUDY_DIR) if PRETRAINED_STUDY_DIR is not None else None
    if directory is not None and not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    if directory is None and RUN_TRAINING:
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        directory = RESULT_ROOT / name
        directory.mkdir(parents=True)
        save_json(directory / "experiment.json", {"experiment_hash": PRETRAINED_HASH, "protocol": PRETRAINED_PROTOCOL,
                                                   "environment": ENVIRONMENT, "mags_reference": MAGS_STUDY_DIR})
        OUTER.to_csv(directory / "outer_patients.csv", index=False)
        INNER.to_csv(directory / "inner_patients.csv", index=False)
    if directory is not None:
        saved = json.loads((directory / "experiment.json").read_text())
        assert saved["experiment_hash"] == PRETRAINED_HASH, "Study configuration/folds/software differ."
        print("Pretrained study directory:", directory)
    return directory


ACTIVE_STUDY = open_pretrained_study()
if RUN_TRAINING:
    for name in MODEL_NAMES:
        for fold in range(1, 6):
            directory = ACTIVE_STUDY / name / f"outer_{fold}"
            selected = select_duration(name, fold, directory)
            refit_outer(name, fold, selected, directory)
    save_json(ACTIVE_STUDY / "training_complete.json", {"experiment_hash": PRETRAINED_HASH, "models": MODEL_NAMES,
                                                       "outer_refits": len(MODEL_NAMES) * 5})
else:
    print("Pretrained fine-tuning is disabled. No parameters have been updated.")

# %% [markdown]
# ## 13. OOF Prediction Schema and Integrity
# Each model must provide exactly one held-out prediction per sample. Patient/fold pairs
# are checked against the saved MAGS-Net assignments. Repeated patient IDs across that
# patient's slices are expected; assigning a patient to different outer-test folds is not.

# %%
PREDICTION_COLUMNS = ["model", "outer_fold", "sample_id", "patient_id", "true_label", "predicted_label", *PROBABILITY_COLUMNS, "seed"]


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

# %% [markdown]
# ## 14. Outer-Test Evaluation (Disabled)
# Load each architecture's own completed outer-refit checkpoint. There is no optimizer or
# checkpoint selection on the outer test set. Matching cached predictions are reused.

# %%
def read_outer_checkpoint(name, fold, directory):
    selected = json.loads((directory / "selection.json").read_text())
    saved = json.loads((directory / "outer_refit/summary.json").read_text())
    development_ids = OUTER.loc[OUTER.outer_fold != fold, "patient_id"]
    development = INDEX[INDEX.patient_id.isin(development_ids)]
    assert selected["model"] == saved["model"] == name and selected["outer_fold"] == fold
    assert selected["experiment_hash"] == PRETRAINED_HASH and selected["seed"] == CONFIG["seed"]
    assert selected["epochs"] == int(np.median(selected["inner_best_epochs"])) == saved["epochs_completed"]
    assert saved["signature"] == fit_signature(name, development, None, selected["epochs"])
    checkpoint_path = directory / "outer_refit" / saved["checkpoint"]
    assert saved["checkpoint_hash"] == file_hash(checkpoint_path)
    return selected, saved, checkpoint_path


def evaluate_outer(name, fold, directory):
    selected, saved, checkpoint_path = read_outer_checkpoint(name, fold, directory)
    test_ids = OUTER.loc[OUTER.outer_fold == fold, "patient_id"]
    test = INDEX[INDEX.patient_id.isin(test_ids)]
    prediction_path, metadata_path = directory / "outer_predictions.csv", directory / "evaluation.json"
    if metadata_path.exists():
        cached = json.loads(metadata_path.read_text())
        assert cached["experiment_hash"] == PRETRAINED_HASH and cached["checkpoint_hash"] == saved["checkpoint_hash"]
        assert cached["predictions_hash"] == file_hash(prediction_path)
        rows = pd.read_csv(prediction_path, dtype={"sample_id": str, "patient_id": str})
        validate_predictions(rows, test, name, fold)
        return rows
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["model"] == name and checkpoint["signature"] == saved["signature"]
    assert checkpoint["experiment_hash"] == PRETRAINED_HASH and checkpoint["epoch"] == selected["epochs"]
    assert checkpoint["objective"] == "CrossEntropy" and checkpoint["weights"] == WEIGHT_SOURCES[name]["weights"]
    assert checkpoint["stem_adaptation"] == "sum_rgb" and checkpoint["fine_tuning"] == "all_layers"
    model = build_classifier(name, pretrained=False).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    metrics, rows = run_epoch(model, make_loader(test))
    rows["model"], rows["outer_fold"], rows["seed"] = name, fold, CONFIG["seed"]
    rows = rows[PREDICTION_COLUMNS]
    validate_predictions(rows, test, name, fold)
    rows.to_csv(prediction_path, index=False)
    save_json(metadata_path, {"experiment_hash": PRETRAINED_HASH, "checkpoint_hash": saved["checkpoint_hash"],
                              "predictions_hash": file_hash(prediction_path), "test_metrics": metrics})
    return rows


if RUN_OUTER_TEST:
    if ACTIVE_STUDY is None or not (ACTIVE_STUDY / "training_complete.json").exists():
        raise RuntimeError("Complete all baseline fits and set PRETRAINED_STUDY_DIR before outer-test evaluation.")
    completion = json.loads((ACTIVE_STUDY / "training_complete.json").read_text())
    assert completion["experiment_hash"] == PRETRAINED_HASH and completion["models"] == MODEL_NAMES
    for name in MODEL_NAMES:
        for fold in range(1, 6):
            read_outer_checkpoint(name, fold, ACTIVE_STUDY / name / f"outer_{fold}")
    for name in MODEL_NAMES:
        for fold in range(1, 6):
            evaluate_outer(name, fold, ACTIVE_STUDY / name / f"outer_{fold}")
else:
    print("New outer-test inference is disabled.")

# %% [markdown]
# ## 15. Import the Existing MAGS-Net Reference
# Reuse the five saved outer-test prediction files after verifying their provenance and coverage.
# MAGS-Net used multitask loss and its own inner selections. Those epochs are documented as
# reference metadata only and are never passed to baseline training.

# %%
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


MAGS_OOF = load_mags_reference()
print("Saved MAGS-Net reference:", len(MAGS_OOF), "OOF rows |", MAGS_OOF.patient_id.nunique(), "patients")

# %% [markdown]
# ## 16. Pool and Audit OOF Results per Architecture
# Each complete model must have 3,064 rows, 233 unique patients, and zero duplicate samples
# for the supplied dataset. Counts are computed from the verified index rather than fabricated.
# A model with missing folds remains marked incomplete and is excluded from comparison metrics.

# %%
def load_pretrained_oof(name, study):
    if study is None:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    parts = []
    for fold in range(1, 6):
        directory = study / name / f"outer_{fold}"
        if not (directory / "evaluation.json").exists():
            return pd.DataFrame(columns=PREDICTION_COLUMNS)
        _, summary, _ = read_outer_checkpoint(name, fold, directory)
        metadata = json.loads((directory / "evaluation.json").read_text())
        path = directory / "outer_predictions.csv"
        assert metadata["experiment_hash"] == PRETRAINED_HASH and metadata["checkpoint_hash"] == summary["checkpoint_hash"]
        assert metadata["predictions_hash"] == file_hash(path)
        rows = pd.read_csv(path, dtype={"sample_id": str, "patient_id": str})
        test_ids = OUTER.loc[OUTER.outer_fold == fold, "patient_id"]
        validate_predictions(rows, INDEX[INDEX.patient_id.isin(test_ids)], name, fold)
        parts.append(rows)
    pooled = pd.concat(parts, ignore_index=True)
    validate_predictions(pooled, INDEX, name)
    return pooled


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


OOF_BY_MODEL = {"MAGS-Net": MAGS_OOF}
for name in MODEL_NAMES:
    OOF_BY_MODEL[name] = load_pretrained_oof(name, ACTIVE_STUDY)
COVERAGE = pd.DataFrame([oof_coverage(frame, name) for name, frame in OOF_BY_MODEL.items()])
display(COVERAGE)

# %% [markdown]
# ## 17. Classification Comparison and Model-Specific Durations
# Report slice and patient metrics separately. Patient classification averages slice
# probabilities before argmax, exactly as in MAGS-Net. Never choose a model/epoch using
# these outer-test results. MAGS-Net's additional mask supervision remains explicitly labeled.

# %%
def patient_predictions(frame):
    patients = frame.groupby(["model", "outer_fold", "patient_id", "seed"], as_index=False).agg(
        true_label=("true_label", "first"), slices=("sample_id", "size"),
        **{column: (column, "mean") for column in PROBABILITY_COLUMNS})
    patients["predicted_label"] = patients[PROBABILITY_COLUMNS].to_numpy().argmax(axis=1)
    return patients


def comparison_metrics(frame, level):
    name = frame.model.iloc[0]
    return {"model": name, "level": level, "initialization": "Scratch" if name == "MAGS-Net" else "ImageNet pretrained",
            "weights": "None" if name == "MAGS-Net" else WEIGHT_SOURCES[name]["weights"], "seed": CONFIG["seed"],
            "training_objective": "Multitask (Cls + 0.5 Seg + 0.2 Aux)" if name == "MAGS-Net" else "Cross-Entropy only",
            "samples": len(frame), **classification_scores(frame),
            "balanced_accuracy": float(balanced_accuracy_score(frame.true_label, frame.predicted_label)),
            "macro_ovr_auc": float(roc_auc_score(frame.true_label, frame[PROBABILITY_COLUMNS], multi_class="ovr", average="macro", labels=[0, 1, 2]))}


def selected_epoch_table(study):
    rows = []
    if study is None:
        return pd.DataFrame()
    for name in MODEL_NAMES:
        for fold in range(1, 6):
            directory = study / name / f"outer_{fold}"
            if not (directory / "outer_refit/summary.json").exists():
                continue
            selected, saved, _ = read_outer_checkpoint(name, fold, directory)
            rows.append({"model": name, "outer_fold": fold, "inner_best_epochs": json.dumps(selected["inner_best_epochs"]),
                         "median_epoch": selected["epochs"], "final_outer_epochs": saved["epochs_completed"],
                         "selection_rule": selected["selection_rule"]})
    return pd.DataFrame(rows)


COMPARISON_ROWS, CLASS_REPORTS, PATIENT_OOF = [], [], {}
for name, frame in OOF_BY_MODEL.items():
    if frame.empty:
        continue
    PATIENT_OOF[name] = patient_predictions(frame)
    for level, records in (("slice", frame), ("patient", PATIENT_OOF[name])):
        COMPARISON_ROWS.append(comparison_metrics(records, level))
        report = classification_report(records.true_label, records.predicted_label, labels=[0, 1, 2],
                                       target_names=list(CLASS_NAMES.values()), output_dict=True, zero_division=0)
        CLASS_REPORTS.extend({"model": name, "level": level, "class": label, **report[label]} for label in CLASS_NAMES.values())
COMPARISON = pd.DataFrame(COMPARISON_ROWS)
EPOCH_SELECTIONS = selected_epoch_table(ACTIVE_STUDY)
display(COMPARISON)
if not EPOCH_SELECTIONS.empty:
    display(EPOCH_SELECTIONS)
if COVERAGE.status.ne("complete").any():
    print("Comparison is incomplete: pending baselines have no reported performance.")

# %% [markdown]
# ## 18. Save OOF CSVs, Metrics, and Confusion Matrices
# Save each completed model independently: slice OOF, patient OOF, Accuracy, macro-F1,
# macro one-vs-rest AUC, and count confusion matrices at both levels. No Dice/IoU is computed.
# Pending models have no invented scores. MAGS-Net remains an explicitly labeled reference.

# %%
def save_confusion_matrices(name, slices, patients, directory):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (level, frame) in zip(axes, (("slice", slices), ("patient", patients))):
        matrix = confusion_matrix(frame.true_label, frame.predicted_label, labels=[0, 1, 2])
        pd.DataFrame(matrix, index=list(CLASS_NAMES.values()), columns=list(CLASS_NAMES.values())).rename_axis("True class").to_csv(
            directory / f"{name}_{level}_confusion_matrix.csv")
        ax.imshow(matrix, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES.values(), yticklabels=CLASS_NAMES.values(),
               xlabel="Predicted", ylabel="True", title=f"{name} | {level}")
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"{name}_confusion_matrices.{extension}", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figure)

# %%
EXPORT_DIR = ACTIVE_STUDY / "analysis" if ACTIVE_STUDY is not None else PREPARATION_DIR / "reference"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
for name, frame in OOF_BY_MODEL.items():
    if frame.empty:
        continue
    frame[PREDICTION_COLUMNS].to_csv(EXPORT_DIR / f"{name}_oof.csv", index=False)
    PATIENT_OOF[name].to_csv(EXPORT_DIR / f"{name}_patient_oof.csv", index=False)
    COMPARISON[COMPARISON.model == name].to_csv(EXPORT_DIR / f"{name}_classification_metrics.csv", index=False)
    save_confusion_matrices(name, frame, PATIENT_OOF[name], EXPORT_DIR)
pd.concat([frame for frame in OOF_BY_MODEL.values() if not frame.empty], ignore_index=True).to_csv(EXPORT_DIR / "comparison_oof.csv", index=False)
COVERAGE.to_csv(EXPORT_DIR / "oof_coverage.csv", index=False)
COMPARISON.to_csv(EXPORT_DIR / "classification_comparison.csv", index=False)
pd.DataFrame(CLASS_REPORTS).to_csv(EXPORT_DIR / "per_class_reports.csv", index=False)
if not EPOCH_SELECTIONS.empty:
    EPOCH_SELECTIONS.to_csv(EXPORT_DIR / "epoch_selections.csv", index=False)
save_json(EXPORT_DIR / "analysis_status.json", {"experiment_hash": PRETRAINED_HASH, "source_mags_hash": MAGS_HASH,
    "complete_models": COVERAGE.loc[COVERAGE.status == "complete", "model"].tolist(),
    "training_enabled": RUN_TRAINING, "outer_test_enabled": RUN_OUTER_TEST,
    "planned_initialization": WEIGHT_SOURCES, "input_adaptation": "Sum RGB stem kernels; keep MRI /255 normalization"})
print("Results saved:", EXPORT_DIR.relative_to(PROJECT_ROOT))

# %% [markdown]
# ### 18.1 Paired Patient Classification Statistics
# Run after Sections 16–18. This section only reads saved OOF predictions; no training,
# downloads, or test inference. Patient predictions average slice probabilities before
# argmax. Match patient IDs, labels, folds, seeds, and slice counts across all four models.
# All differences are **MAGS-Net minus the pretrained model**.
# #
# - McNemar uses paired **correct/incorrect** patient outcomes, with a two-sided exact
#   binomial p-value. Zero discordant pairs gives p=1. Holm adjusts the three comparisons
#   with ResNet-50, EfficientNet-B0, and MobileNetV3-Large together (not paper-wide).
# - Use 10,000 paired patient-bootstrap draws, seed 42, stratified by true tumor class.
#   Keep each class's observed patient count, and use the same sampled patient IDs for
#   every model and metric. Report percentile 95% CIs for the paired metric differences.
# - Accuracy differences and CIs are in **percentage points**. Macro-F1 differences are
#   on the **0–1 scale**. Recompute macro-F1 across classes 0, 1, 2 in every bootstrap
#   sample, then subtract models; do not average per-patient F1 or subtract separate CIs.
# - Bootstrap intervals are pointwise, not multiplicity-adjusted. Holm applies only to
#   the three McNemar p-values. Inference is conditional on the fitted OOF models and
#   fixed folds; it excludes retraining and CV-partition variability.
# #
# References: [exact McNemar](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html),
# [Holm](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html),
# [paired bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html),
# and [macro-F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

# %%
from scipy.stats import binomtest

# Analysis settings never change the training protocol or its experiment hash.
STATISTICS_CONFIG = {"resamples": 10_000, "seed": 42, "confidence_level": 0.95, "alpha": 0.05}
STATISTICS_MODELS = ["MAGS-Net", "ResNet-50", "EfficientNet-B0", "MobileNetV3-Large"]


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
def patient_classification_scores(labels, predictions):
    """Compute [accuracy percent, macro-F1] for each column/model efficiently."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    assert predictions.ndim == 2 and len(labels) == len(predictions) > 0
    assert np.isin(labels, [0, 1, 2]).all() and np.isin(predictions, [0, 1, 2]).all()
    n_models = predictions.shape[1]
    # One 3x3 confusion matrix per model, with true labels on rows.
    codes = 9 * np.arange(n_models)[None, :] + 3 * labels[:, None] + predictions
    matrices = np.bincount(codes.astype(int).ravel(), minlength=9 * n_models).reshape(n_models, 3, 3)
    true_positive = np.diagonal(matrices, axis1=1, axis2=2)
    denominator = matrices.sum(axis=1) + matrices.sum(axis=2)
    class_f1 = np.divide(2.0 * true_positive, denominator,
                         out=np.zeros_like(true_positive, dtype=float), where=denominator > 0)
    return np.column_stack([100 * true_positive.sum(axis=1) / len(labels), class_f1.mean(axis=1)])


def paired_classification_bootstrap(labels, predictions, settings):
    assert settings["resamples"] > 0 and 0 < settings["confidence_level"] < 1
    rng = np.random.default_rng(settings["seed"])
    strata = [np.flatnonzero(labels == label) for label in (0, 1, 2)]
    assert all(len(group) > 0 for group in strata)
    estimates = np.empty((settings["resamples"], predictions.shape[1] - 1, 2))
    for iteration in range(settings["resamples"]):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata])
        scores = patient_classification_scores(labels[indices], predictions[indices])
        estimates[iteration] = scores[:1] - scores[1:]
    tail = (1 - settings["confidence_level"]) / 2
    return np.quantile(estimates, [tail, 1 - tail], axis=0)

# %%
def paired_patient_statistics(oof_sets, settings):
    patients = align_patient_results(oof_sets)
    labels = patients["MAGS-Net"].true_label.to_numpy()
    predictions = np.column_stack([patients[name].predicted_label.to_numpy() for name in STATISTICS_MODELS])
    scores = patient_classification_scores(labels, predictions)
    differences = scores[:1] - scores[1:]
    intervals = paired_classification_bootstrap(labels, predictions, settings)
    correct = predictions == labels[:, None]
    rows = []
    for j, name in enumerate(STATISTICS_MODELS[1:]):
        mags, other = correct[:, 0], correct[:, j + 1]
        b, c = int((mags & ~other).sum()), int((~mags & other).sum())
        rows.append({"comparison": f"MAGS-Net vs {name}", "reference": "MAGS-Net", "pretrained_model": name,
            "n_patients": len(labels), "mags_correct": int(mags.sum()), "pretrained_correct": int(other.sum()),
            "both_correct": int((mags & other).sum()), "both_wrong": int((~mags & ~other).sum()),
            "mags_correct_pretrained_wrong": b, "mags_wrong_pretrained_correct": c, "discordant_pairs": b + c,
            "mags_patient_accuracy": scores[0, 0] / 100, "pretrained_patient_accuracy": scores[j + 1, 0] / 100,
            "delta_patient_accuracy_pp": differences[j, 0],
            "accuracy_ci_low_pp": intervals[0, j, 0], "accuracy_ci_high_pp": intervals[1, j, 0],
            "mcnemar_p_raw": float(binomtest(b, b + c, p=0.5, alternative="two-sided").pvalue) if b + c else 1.0,
            "mags_patient_macro_f1": scores[0, 1], "pretrained_patient_macro_f1": scores[j + 1, 1],
            "delta_patient_macro_f1": differences[j, 1],
            "macro_f1_ci_low": intervals[0, j, 1], "macro_f1_ci_high": intervals[1, j, 1]})
    results = pd.DataFrame(rows)
    results["mcnemar_p_adj_holm"] = holm_adjust(results.mcnemar_p_raw)
    results["mcnemar_reject_holm"] = results.mcnemar_p_adj_holm <= settings["alpha"]
    return results

# %%
def statistics_paper_table(results):
    p_format = lambda value: f"{value:.6f}" if value >= 0.000001 else "<0.000001"
    return pd.DataFrame({
        "Comparison": results.comparison,
        "Δ Patient Acc (pp)": results.delta_patient_accuracy_pp.map(lambda value: f"{value:+.2f}"),
        "95% CI (Δ Acc, pp)": [f"[{lo:+.2f}, {hi:+.2f}]" for lo, hi in zip(results.accuracy_ci_low_pp, results.accuracy_ci_high_pp)],
        "McNemar p_raw": results.mcnemar_p_raw.map(p_format),
        "McNemar p_adj (Holm)": results.mcnemar_p_adj_holm.map(p_format),
        "Δ Patient Macro-F1": results.delta_patient_macro_f1.map(lambda value: f"{value:+.4f}"),
        "95% CI (Δ Macro-F1)": [f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in zip(results.macro_f1_ci_low, results.macro_f1_ci_high)],
    })


PAIRED_PATIENT_STATISTICS, PAIRED_PATIENT_TABLE = pd.DataFrame(), pd.DataFrame()
STATISTICS_INPUTS = {name: EXPORT_DIR / f"{name}_oof.csv" for name in STATISTICS_MODELS}
if set(COVERAGE.loc[COVERAGE.status == "complete", "model"]) != set(STATISTICS_MODELS):
    print("Paired statistics pending: complete MAGS-Net and all three pretrained OOF sets first.")
else:
    statistics_oof = {name: pd.read_csv(path, dtype={"sample_id": str, "patient_id": str})
                      for name, path in STATISTICS_INPUTS.items()}
    PAIRED_PATIENT_STATISTICS = paired_patient_statistics(statistics_oof, STATISTICS_CONFIG)
    PAIRED_PATIENT_TABLE = statistics_paper_table(PAIRED_PATIENT_STATISTICS)
    PAIRED_PATIENT_STATISTICS.to_csv(EXPORT_DIR / "paired_patient_statistics.csv", index=False)
    PAIRED_PATIENT_TABLE.to_csv(EXPORT_DIR / "paired_patient_statistics_table.csv", index=False, encoding="utf-8-sig")
    patient_classes = statistics_oof["MAGS-Net"].drop_duplicates("patient_id").true_label.value_counts().sort_index()
    save_json(EXPORT_DIR / "paired_patient_statistics_metadata.json", {
        "experiment_hash": PRETRAINED_HASH, "source_mags_hash": MAGS_HASH, "settings": STATISTICS_CONFIG,
        "comparisons": PAIRED_PATIENT_STATISTICS.comparison.tolist(), "direction": "MAGS-Net minus pretrained model",
        "n_patients": int(PAIRED_PATIENT_STATISTICS.n_patients.iloc[0]),
        "patient_counts_by_class": {CLASS_NAMES[int(key)]: int(value) for key, value in patient_classes.items()},
        "patient_prediction": "Argmax after averaging all slice probabilities within each patient",
        "accuracy_difference_units": "percentage points",
        "macro_f1": "Unweighted mean of class 0/1/2 F1, zero_division=0; recomputed in each bootstrap draw; differences on 0–1 scale",
        "mcnemar": "Two-sided exact binomial test on paired correctness; zero discordant pairs gives p=1",
        "correction": "Holm across these three McNemar comparisons only; not paper-wide",
        "bootstrap": "Patient resampling within true class, shared indices across all models and metrics; percentile CI",
        "ci_scope": "Pointwise, not multiplicity-adjusted; conditional on fitted OOF models and fixed folds",
        "source_oof_sha256": {path.name: file_hash(path) for path in STATISTICS_INPUTS.values()},
    })
    print(f"PAIRED PATIENT CLASSIFICATION | MAGS-Net - pretrained | {STATISTICS_CONFIG['resamples']:,} bootstrap draws | seed {STATISTICS_CONFIG['seed']}")
    display(PAIRED_PATIENT_TABLE)
    print("Holm adjusts the three McNemar tests. Accuracy and macro-F1 intervals are pointwise 95% CIs.")
    print("Statistics saved:", EXPORT_DIR.relative_to(PROJECT_ROOT))

# %% [markdown]
# ## 19. Run Training Later
# 1. Select Conda `dl`. With both switches False, Run All only prepares the experiment,
#    checks architecture shapes, and reads the completed MAGS-Net reference.
# 2. Set RUN_TRAINING=True, RUN_OUTER_TEST=False, PRETRAINED_STUDY_DIR=None, then Run All.
#    First fits download the pinned ImageNet weights if absent from the Torch cache.
#    Complete **75 inner fits + 15 outer refits = 90 fits**; retain the printed study path.
# 3. Set PRETRAINED_STUDY_DIR to that path, RUN_TRAINING=False, RUN_OUTER_TEST=True and Run All.
#    Each outer model is tested once; cached predictions are verified on subsequent runs.
# 4. For saved-result analysis, retain the study path and set both switches False.
#    Evaluation loads complete trained checkpoints and does not download ImageNet weights.
# Scratch baselines, MAGS-Net, MT-U-Net, and ablation studies remain separate and unchanged.
