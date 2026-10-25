"""Restore crop geometry for display; retain predictions on the analyzed grid."""
import cv2
import numpy as np

from magsnet_demo.config import (CLASS_NAMES, DISCLAIMER, IMAGE_SIZE, INPUT_LIMITATIONS,
                                MANIFEST, SEGMENTATION_THRESHOLD)
from magsnet_demo.utils import json_bytes, png_bytes


def restore_to_original(array, prepared, binary=False):
    x1, y1, x2, y2 = prepared.crop_box
    output = np.zeros(prepared.original_gray.shape, np.uint8 if binary else np.float32)
    interpolation = cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR
    crop = cv2.resize(array.astype(output.dtype), (x2 - x1, y2 - y1), interpolation=interpolation)
    output[y1:y2, x1:x2] = crop
    return output


def render_outputs(prepared, result):
    native_mask = restore_to_original(result["mask"], prepared, binary=True).astype(bool)
    native_probability = restore_to_original(result["segmentation_probability"], prepared)
    overlay = prepared.original_rgb.copy()
    tint = np.array([32, 210, 176], dtype=np.float32)
    overlay[native_mask] = np.round(0.65 * overlay[native_mask] + 0.35 * tint).astype(np.uint8)
    heatmap = cv2.applyColorMap((native_probability * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # The original image's area outside the crop was not analyzed by the network.
    x1, y1, x2, y2 = prepared.crop_box
    support = np.zeros(native_mask.shape, bool)
    support[y1:y2, x1:x2] = True
    heatmap[~support] = 0
    return {"mask": native_mask.astype(np.uint8) * 255, "overlay": overlay,
            "probability_map": heatmap, "probability_native": native_probability}


def make_summary(prepared, result, device):
    return {
        "model": "MAGS-Net", "mode": "equal-weight five-outer-fold probability ensemble",
        "models": 5, "seed": MANIFEST["seed"], "precision": "FP32",
        "predicted_class": CLASS_NAMES[result["predicted_label"]],
        "class_probabilities": dict(zip(CLASS_NAMES, map(float, result["probabilities"]))),
        "probability_interpretation": "Model probability, not clinical confidence; not calibrated.",
        "predicted_lesion_area_fraction": result["lesion_area_fraction"],
        "area_fraction_denominator": f"{IMAGE_SIZE}x{IMAGE_SIZE} analyzed crop pixels",
        "predicted_tumor_pixels_analyzed_grid": int(result["mask"].sum()),
        "segmentation_threshold": SEGMENTATION_THRESHOLD,
        "original_dimensions_hw": list(prepared.original_gray.shape),
        "analyzed_dimensions_hw": [IMAGE_SIZE, IMAGE_SIZE],
        "crop_box_xyxy_exclusive": list(prepared.crop_box),
        "display_mapping": "Nearest-neighbor binary mask; bilinear probabilities; outside crop not analyzed.",
        "inference_device": str(device),
        "stage_gates": [{"stage": i + 1, "alpha3": float(g[0]), "alpha5": float(g[1])}
                        for i, g in enumerate(result["gates"])],
        "gate_reduction": "Mean over channels, then blocks within stage, then five models.",
        "experiment_hash": MANIFEST["experiment_hash"],
        "checkpoint_sha256": [r["sha256"] for r in MANIFEST["checkpoints"]],
        "disclaimer": DISCLAIMER,
        "input_limitations": INPUT_LIMITATIONS,
    }


def make_downloads(prepared, result, visuals, device):
    return {"mask": png_bytes(visuals["mask"]), "overlay": png_bytes(visuals["overlay"]),
            "analyzed_mask": png_bytes(result["mask"].astype(np.uint8) * 255),
            "summary": json_bytes(make_summary(prepared, result, device))}
