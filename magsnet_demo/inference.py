"""Frozen FP32 probability ensemble; no training, labels, masks or augmentation."""
import threading
from dataclasses import dataclass, field

import numpy as np
import torch

from magsnet_demo.config import CHECKPOINT_PATHS, IMAGE_SIZE, MANIFEST, MODEL_SOURCE, SEGMENTATION_THRESHOLD
from magsnet_demo.model_adapter import load_model_class
from magsnet_demo.utils import DemoError, file_sha256


@dataclass
class Ensemble:
    models: tuple
    device: torch.device
    lock: threading.RLock = field(default_factory=threading.RLock)


def resource_key():
    """Invalidate Streamlit's model cache if a configured checkpoint changes."""
    return tuple((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                 if p.is_file() else (str(p.resolve()), None, None)
                 for p in (*CHECKPOINT_PATHS, MODEL_SOURCE))


def load_ensemble(device_name="cpu"):
    if device_name not in {"cuda", "cpu"}:
        raise DemoError("Choose CPU or CUDA for inference.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise DemoError("CUDA is unavailable. Select CPU in the sidebar.")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_class = load_model_class()
    models = []
    try:
        for path, record in zip(CHECKPOINT_PATHS, MANIFEST["checkpoints"]):
            if not path.is_file():
                raise DemoError(f"Outer-fold {record['outer_fold']} checkpoint missing. "
                                "See weights/README.md for the five required files.")
            if file_sha256(path) != record["sha256"]:
                raise DemoError(f"Outer-fold {record['outer_fold']} checkpoint failed its SHA-256 check. "
                                "Use the original, unmodified outer-refit checkpoint.")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            expected = {"seed": MANIFEST["seed"], "experiment_hash": MANIFEST["experiment_hash"],
                        "epoch": record["epochs"], "signature": record["signature"]}
            if any(payload.get(key) != value for key, value in expected.items()):
                raise DemoError("Checkpoint metadata does not match the frozen study.")
            model = model_class()
            model.load_state_dict(payload["model"], strict=True)
            if sum(p.numel() for p in model.parameters()) != MANIFEST["parameters_per_model"]:
                raise DemoError("Parameter count does not match the audited architecture.")
            models.append(model.to(device_name).eval().requires_grad_(False))
        if len(models) != 5:
            raise DemoError("The probability ensemble requires all five outer-fold models.")
        return Ensemble(tuple(models), torch.device(device_name))
    except (DemoError, torch.cuda.OutOfMemoryError):
        raise
    except (OSError, RuntimeError, KeyError, ValueError) as exc:
        raise DemoError("Unable to load the frozen model. Verify the source, checkpoint files and dependencies.") from exc


@torch.inference_mode()
def predict(ensemble, tensor):
    if tuple(tensor.shape) != (1, 1, IMAGE_SIZE, IMAGE_SIZE) or tensor.dtype != torch.float32:
        raise DemoError("The model expects one float32 grayscale image of shape 1×1×224×224.")
    if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
        raise DemoError("The model input must contain finite values in [0, 1].")
    with ensemble.lock:
        batch = tensor.to(ensemble.device)
        class_sum = np.zeros(3, np.float32)
        segmentation_sum = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32)
        gates_sum = np.zeros((3, 2), np.float32)
        for model in ensemble.models:
            output = model(batch)
            class_sum += output["logits"].float().softmax(1)[0].cpu().numpy() / 5
            segmentation_sum += output["seg_logits"].float().sigmoid()[0, 0].cpu().numpy() / 5
            # Source model already averages channels, then blocks: [B, stage, scale].
            gates = output["gates"].float().cpu().numpy()
            if gates.shape != (1, 3, 2):
                raise DemoError("Unexpected gate dimensions; inference output cannot be interpreted.")
            gates_sum += gates[0] / 5
        if (not np.isfinite(class_sum).all() or not np.isfinite(segmentation_sum).all()
                or not np.isfinite(gates_sum).all() or not np.isclose(class_sum.sum(), 1, atol=1e-5)
                or not np.allclose(gates_sum.sum(1), 1, atol=1e-5)):
            raise DemoError("The model returned invalid probabilities.")
        mask = segmentation_sum >= SEGMENTATION_THRESHOLD
        return {"probabilities": class_sum, "predicted_label": int(class_sum.argmax()),
                "segmentation_probability": segmentation_sum, "mask": mask,
                "lesion_area_fraction": float(mask.mean()), "gates": gates_sum}
