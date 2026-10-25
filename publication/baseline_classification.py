# %% [markdown]
# # 02. Baseline Classification: Nested Patient-Level CV
# Compare the saved **MAGS-Net (scratch, multitask)** reference against five image-only
# classifiers trained **from scratch with Cross-Entropy only**. This is a comparison of
# training systems, not an architecture-only ablation with identical supervision.
# This notebook is self-contained and uses the exact saved MAGS-Net patient assignments.
# Training and new outer-test inference are disabled by default.

# %% [markdown]
# ## 0. Experiment Configuration
# Seed 42; 25 inner fits and 5 outer refits for each classifier: **150 new fits** in total.
# MAGS-Net is read from its existing held-out results and is never retrained here.

# %%
CONFIG = {
    "protocol": "baseline_classification_nested_cv_v1",
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
MODEL_NAMES = ["ResNet-50", "DenseNet-121", "EfficientNet-B0", "MobileNetV3-Large", "RegNet-Y-400MF"]
CLASS_NAMES = {0: "Meningioma", 1: "Glioma", 2: "Pituitary"}
PROBABILITY_COLUMNS = ["prob_meningioma", "prob_glioma", "prob_pituitary"]
MAGS_STUDY_DIR = "results/magsnet_nested_cv/20260916T210025Z_bc27ff71"
BASELINE_STUDY_DIR = None  # Set your own completed study path for analysis.
RUN_TRAINING = False
RUN_OUTER_TEST = False

# %% [markdown]
# ## 1. Imports and Environment
# Use the Conda `dl` kernel. TorchVision defines the five backbones locally; no weights
# are downloaded. The default run checks data/folds/models and exports the saved reference.

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
RESULT_ROOT = PROJECT_ROOT / "results" / "baseline_classification_nested_cv"

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
# ## 7. Scratch Classifiers: One Input Channel and Three Outputs
# Every constructor explicitly uses `weights=None` and `num_classes=3`. Replace the first
# RGB convolution with a newly initialized one-channel convolution of the same shape/stride.
# Nothing is averaged or copied from ImageNet weights. Keep each backbone's standard layers.
# [TorchVision documents `weights=None` as random initialization.](https://docs.pytorch.org/vision/0.22/models.html)

# %%
MODEL_SPECS = {
    "ResNet-50": (models.resnet50, "conv1"),
    "DenseNet-121": (models.densenet121, "features.conv0"),
    "EfficientNet-B0": (models.efficientnet_b0, "features.0.0"),
    "MobileNetV3-Large": (models.mobilenet_v3_large, "features.0.0"),
    "RegNet-Y-400MF": (models.regnet_y_400mf, "stem.0"),
}


def build_classifier(name):
    constructor, location = MODEL_SPECS[name]
    model = constructor(weights=None, num_classes=3)
    original = model.get_submodule(location)
    convolution = nn.Conv2d(1, original.out_channels, original.kernel_size, original.stride,
                            original.padding, original.dilation, original.groups,
                            bias=original.bias is not None, padding_mode=original.padding_mode)
    mode = "fan_in" if name == "DenseNet-121" else "fan_out"
    nn.init.kaiming_normal_(convolution.weight, mode=mode, nonlinearity="relu")
    if convolution.bias is not None:
        nn.init.zeros_(convolution.bias)
    parent, _, attribute = location.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, attribute, convolution)
    return model


@torch.inference_mode()
def forward_checks():
    rows = []
    images = next(iter(make_loader(INDEX.iloc[:2])))["image"].to(DEVICE)
    for name in MODEL_NAMES:
        set_seed(CONFIG["seed"])
        model = build_classifier(name).to(DEVICE).eval()
        logits = model(images)
        assert logits.shape == (2, 3) and torch.isfinite(logits).all()
        rows.append({"model": name, "initialization": "Scratch (weights=None)", "input": "1 x 224 x 224",
                     "parameters": sum(p.numel() for p in model.parameters()), "forward_check": "passed"})
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
BASELINE_PROTOCOL = {"config": CONFIG, "models": MODEL_NAMES, "classes": CLASS_NAMES,
                     "mags_experiment_hash": MAGS_HASH, "data_hash": DATA_HASH,
                     "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                     "input": "grayscale_224_divide255", "initialization": "weights=None",
                     "objective": "CrossEntropy", "epoch_selection": "median of five inner best CE epochs"}
BASELINE_HASH = signature(BASELINE_PROTOCOL)
PREPARATION_DIR = RESULT_ROOT / "preparation" / BASELINE_HASH[:12]
PREPARATION_DIR.mkdir(parents=True, exist_ok=True)
save_json(PREPARATION_DIR / "protocol.json", BASELINE_PROTOCOL)
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
    return signature({"experiment": BASELINE_HASH, "model": name, "epochs": int(epochs),
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
    directory.mkdir(parents=True)
    set_seed(CONFIG["seed"])
    model = build_classifier(name).to(DEVICE)
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
                   "signature": digest, "experiment_hash": BASELINE_HASH, "source_mags_hash": MAGS_HASH,
                   "objective": "CrossEntropy", "config": CONFIG, "weights": None}
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
                "experiment_hash": BASELINE_HASH}
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
# Re-running with `BASELINE_STUDY_DIR=None` creates a new study; an explicit matching path
# reuses completed fits. Existing MAGS-Net results are read-only throughout this notebook.

# %%
def open_baseline_study():
    directory = Path(BASELINE_STUDY_DIR) if BASELINE_STUDY_DIR is not None else None
    if directory is not None and not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    if directory is None and RUN_TRAINING:
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        directory = RESULT_ROOT / name
        directory.mkdir(parents=True)
        save_json(directory / "experiment.json", {"experiment_hash": BASELINE_HASH, "protocol": BASELINE_PROTOCOL,
                                                   "environment": ENVIRONMENT, "mags_reference": MAGS_STUDY_DIR})
        OUTER.to_csv(directory / "outer_patients.csv", index=False)
        INNER.to_csv(directory / "inner_patients.csv", index=False)
    if directory is not None:
        saved = json.loads((directory / "experiment.json").read_text())
        assert saved["experiment_hash"] == BASELINE_HASH, "Study configuration/folds/software differ."
        print("Baseline study directory:", directory)
    return directory


ACTIVE_STUDY = open_baseline_study()
if RUN_TRAINING:
    for name in MODEL_NAMES:
        for fold in range(1, 6):
            directory = ACTIVE_STUDY / name / f"outer_{fold}"
            selected = select_duration(name, fold, directory)
            refit_outer(name, fold, selected, directory)
    save_json(ACTIVE_STUDY / "training_complete.json", {"experiment_hash": BASELINE_HASH, "models": MODEL_NAMES,
                                                       "outer_refits": len(MODEL_NAMES) * 5})
else:
    print("Baseline training is disabled. No parameters have been updated.")

# %% [markdown]
# ## 13. OOF Prediction Schema and Integrity
# Each model must provide exactly one held-out prediction per sample. Patient/fold pairs
# are checked against the saved MAGS-Net assignments. Repeated patient IDs across that
# patient's slices are expected; assigning a patient to different outer-test folds is not.

# %%


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
    assert selected["experiment_hash"] == BASELINE_HASH and selected["seed"] == CONFIG["seed"]
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
        assert cached["experiment_hash"] == BASELINE_HASH and cached["checkpoint_hash"] == saved["checkpoint_hash"]
        assert cached["predictions_hash"] == file_hash(prediction_path)
        rows = pd.read_csv(prediction_path, dtype={"sample_id": str, "patient_id": str})
        validate_predictions(rows, test, name, fold)
        return rows
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["model"] == name and checkpoint["signature"] == saved["signature"]
    assert checkpoint["experiment_hash"] == BASELINE_HASH and checkpoint["epoch"] == selected["epochs"]
    assert checkpoint["objective"] == "CrossEntropy" and checkpoint["weights"] is None
    model = build_classifier(name).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    metrics, rows = run_epoch(model, make_loader(test))
    rows["model"], rows["outer_fold"], rows["seed"] = name, fold, CONFIG["seed"]
    rows = rows[PREDICTION_COLUMNS]
    validate_predictions(rows, test, name, fold)
    rows.to_csv(prediction_path, index=False)
    save_json(metadata_path, {"experiment_hash": BASELINE_HASH, "checkpoint_hash": saved["checkpoint_hash"],
                              "predictions_hash": file_hash(prediction_path), "test_metrics": metrics})
    return rows


if RUN_OUTER_TEST:
    if ACTIVE_STUDY is None or not (ACTIVE_STUDY / "training_complete.json").exists():
        raise RuntimeError("Complete all baseline fits and set BASELINE_STUDY_DIR before outer-test evaluation.")
    completion = json.loads((ACTIVE_STUDY / "training_complete.json").read_text())
    assert completion["experiment_hash"] == BASELINE_HASH and completion["models"] == MODEL_NAMES
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
def load_baseline_oof(name, study):
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
        assert metadata["experiment_hash"] == BASELINE_HASH and metadata["checkpoint_hash"] == summary["checkpoint_hash"]
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
    OOF_BY_MODEL[name] = load_baseline_oof(name, ACTIVE_STUDY)
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
    return {"model": name, "level": level, "initialization": "Scratch", "seed": CONFIG["seed"],
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
# ## 18. Save Predictions, Tables, and Figures
# Per-model OOF CSVs use the requested columns, plus `seed`. Complete models are exported
# separately and in one combined file. Pending models stay missing, not zero-scored.
# Cross-model figures are created only once all six OOF sets are complete.

# %%
def comparison_figures(oof_sets):
    figures = {}
    for kind in ("confusion", "roc"):
        figure, axes = plt.subplots(2, 3, figsize=(15, 9))
        for ax, (name, frame) in zip(axes.flat, oof_sets.items()):
            if kind == "confusion":
                matrix = confusion_matrix(frame.true_label, frame.predicted_label, labels=[0, 1, 2])
                ax.imshow(matrix, cmap="Blues")
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                                color="white" if matrix[i, j] > matrix.max() / 2 else "black")
                ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES.values(), yticklabels=CLASS_NAMES.values(),
                       xlabel="Predicted", ylabel="True")
            else:
                for label, class_name in CLASS_NAMES.items():
                    fpr, tpr, _ = roc_curve(frame.true_label.eq(label), frame[PROBABILITY_COLUMNS[label]])
                    ax.plot(fpr, tpr, label=class_name)
                ax.plot([0, 1], [0, 1], "--", color="gray")
                ax.set(xlabel="False positive rate", ylabel="True positive rate")
                ax.legend(fontsize=8)
            ax.set_title(name)
        figure.tight_layout()
        figures[kind] = figure
    return figures


EXPORT_DIR = (ACTIVE_STUDY / "analysis") if ACTIVE_STUDY is not None else PREPARATION_DIR / "reference"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
for name, frame in OOF_BY_MODEL.items():
    if not frame.empty:
        frame[PREDICTION_COLUMNS].to_csv(EXPORT_DIR / f"{name}_oof.csv", index=False)
        PATIENT_OOF[name].to_csv(EXPORT_DIR / f"{name}_patient_oof.csv", index=False)
pd.concat([frame for frame in OOF_BY_MODEL.values() if not frame.empty], ignore_index=True).to_csv(EXPORT_DIR / "comparison_oof.csv", index=False)
COVERAGE.to_csv(EXPORT_DIR / "oof_coverage.csv", index=False)
COMPARISON.to_csv(EXPORT_DIR / "classification_comparison.csv", index=False)
pd.DataFrame(CLASS_REPORTS).to_csv(EXPORT_DIR / "per_class_reports.csv", index=False)
if not EPOCH_SELECTIONS.empty:
    EPOCH_SELECTIONS.to_csv(EXPORT_DIR / "epoch_selections.csv", index=False)
if COVERAGE.status.eq("complete").all() and len(OOF_BY_MODEL) == 6:
    for name, figure in comparison_figures(OOF_BY_MODEL).items():
        figure.savefig(EXPORT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
        figure.savefig(EXPORT_DIR / f"{name}.pdf", bbox_inches="tight")
        plt.show()
        plt.close(figure)
save_json(EXPORT_DIR / "analysis_status.json", {"experiment_hash": BASELINE_HASH, "source_mags_hash": MAGS_HASH,
                                              "complete_models": COVERAGE.loc[COVERAGE.status == "complete", "model"].tolist(),
                                              "training_enabled": RUN_TRAINING, "outer_test_enabled": RUN_OUTER_TEST})
print("Results saved:", EXPORT_DIR.relative_to(PROJECT_ROOT))

# %% [markdown]
# ### 18.1 Model Complexity: Parameters, MACs, and FP32 Size
# One image: **1 x 1 x 224 x 224**, evaluation mode, FP32. All models have one input channel
# and three classes. MACs count Conv2d and Linear multiply-accumulates (one MAC per product
# accumulated), including grouped/depthwise convolutions. Bias additions, normalization,
# activations, pooling, interpolation, and elementwise operations are excluded; this is not
# a total FLOP count. FP32 size = all model parameters x 4 / 2**20 MiB, excluding buffers,
# optimizer state, activations, and checkpoint serialization overhead.
# The five baseline architectures are profiled with synthetic input and no training.
# MAGS-Net uses its saved section-26 complexity report with the **same counting convention**:
# the full multitask model, including its decoder, classification head, and learned gates.

# %%
@torch.inference_mode()
def profile_classifier(name):
    counts, handles = [], []

    def count_macs(layer, inputs, output):
        if isinstance(layer, nn.Conv2d):
            products = (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
        else:  # Linear
            products = layer.in_features
        counts.append(output.numel() * products)

    # Isolate initialization so profiling does not change subsequent training RNG state.
    with torch.random.fork_rng(devices=[]):
        model = build_classifier(name).cpu().float().eval()
    try:
        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                handles.append(layer.register_forward_hook(count_macs))
        model(torch.zeros(1, 1, CONFIG["image_size"], CONFIG["image_size"], dtype=torch.float32))
    finally:
        for handle in handles:
            handle.remove()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {"Model": name, "Parameters": parameters, "MACs": int(sum(counts)),
            "FP32 model size (MiB)": parameters * 4 / 2**20}


def mags_complexity_reference():
    path = MAGS_STUDY / "analysis/complexity.csv"
    if not path.is_file():
        raise FileNotFoundError("Run MAGS-Net sections 26 and 28 to save analysis/complexity.csv first.")
    saved = pd.read_csv(path)
    assert len(saved) == 1, "Expected one full-model MAGS-Net complexity record."
    row = saved.iloc[0]
    assert row.batch_size == 1 and row.precision == "float32"
    assert MAGS_EXPERIMENT["protocol"]["config"]["image_size"] == CONFIG["image_size"] == 224
    parameters = int(row.parameters)
    assert np.isclose(row.weight_mib_fp32, parameters * 4 / 2**20)
    return {"Model": "MAGS-Net", "Parameters": parameters, "MACs": int(row.conv_linear_macs),
            "FP32 model size (MiB)": parameters * 4 / 2**20}

# %%
COMPLEXITY_TABLE = pd.DataFrame([profile_classifier(name) for name in MODEL_NAMES]
                               + [mags_complexity_reference()])
assert len(COMPLEXITY_TABLE) == 6 and COMPLEXITY_TABLE.Model.is_unique
assert COMPLEXITY_TABLE[["Parameters", "MACs", "FP32 model size (MiB)"]].gt(0).all().all()
COMPLEXITY_CSV = EXPORT_DIR / "model_complexity.csv"
COMPLEXITY_TABLE.to_csv(COMPLEXITY_CSV, index=False)
save_json(EXPORT_DIR / "model_complexity_metadata.json", {
    "input_shape": [1, 1, 224, 224], "precision": "float32", "mode": "eval",
    "macs": "Conv2d and Linear only; one multiply-accumulate counts as one MAC",
    "excluded_operations": "Bias adds, normalization, activations, pooling, interpolation, elementwise operations",
    "fp32_size": "Parameter count * 4 / 2**20 MiB; excludes buffers, optimizer state, activations, file overhead",
    "mags_scope": "Full multitask model, including decoder, classification head, and learned gates",
    "mags_report": str((MAGS_STUDY / "analysis/complexity.csv").relative_to(PROJECT_ROOT)),
    "mags_report_sha256": file_hash(MAGS_STUDY / "analysis/complexity.csv"),
})
display(COMPLEXITY_TABLE.style.hide(axis="index").format(
    {"Parameters": "{:,}", "MACs": "{:,}", "FP32 model size (MiB)": "{:.3f}"}))
print("Complexity CSV saved:", COMPLEXITY_CSV.relative_to(PROJECT_ROOT))

# %% [markdown]
# ## 19. Running the Experiment Later
# 1. Select the Conda `dl` kernel. The default run is a preparation/forward check only.
# 2. Set `RUN_TRAINING=True`, `RUN_OUTER_TEST=False`, and `BASELINE_STUDY_DIR=None`; run from the top.
# 3. After training finishes, set `BASELINE_STUDY_DIR` to the printed path, disable training,
#    enable outer testing, and run from the top. Each model loads its own final refit.
# 4. For later analysis, keep the study path and disable both switches. Saved predictions
#    are loaded without repeated inference. MAGS-Net's original study is never modified.
# Retain the complete baseline study directory and the referenced MAGS-Net fold definitions.
