"""Frozen settings audited against the completed seed-42 publication experiment."""
import json
import os
from pathlib import Path

import torch

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("MAGSNET_PROJECT_ROOT", APP_DIR.parent)).resolve()
MANIFEST = json.loads((APP_DIR / "checkpoint_manifest.json").read_text(encoding="utf-8"))
APP_NAME = "MAGS-Net Demo"
MODEL_NAME = "MAGS-Net"
CLASS_NAMES = tuple(MANIFEST["classes"][str(i)] for i in range(3))
IMAGE_SIZE = MANIFEST["image_size"]
SEGMENTATION_THRESHOLD = MANIFEST["segmentation_threshold"]
CROP_PADDING = MANIFEST["crop_padding"]
MODEL_SOURCE = PROJECT_ROOT / MANIFEST["model_source"]
STUDY_DIR = Path(os.environ.get("MAGSNET_STUDY_DIR", PROJECT_ROOT / MANIFEST["study_dir"]))
WEIGHTS_DIR = os.environ.get("MAGSNET_WEIGHTS_DIR")
CHECKPOINT_PATHS = tuple(
    Path(WEIGHTS_DIR) / f"outer_{row['outer_fold']}.pt" if WEIGHTS_DIR
    else STUDY_DIR / row["relative_path"] for row in MANIFEST["checkpoints"]
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
DISCLAIMER = (
    "MAGS-Net is a research prototype under development, not a medical device. "
    "It must not be used for clinical diagnosis or treatment decisions."
)
INPUT_LIMITATIONS = (
    "The app does not verify that an upload is a brain MRI and cannot determine whether "
    "a tumor is present. Tumor-free scans and unrelated images (for example, an apple) "
    "may still receive a tumor label and a predicted mask, even with a high model probability."
)
