"""Warm-start batch-one ensemble timing, excluding loading/preprocessing/rendering."""
import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from magsnet_demo.config import APP_DIR
from magsnet_demo.inference import load_ensemble, predict
from magsnet_demo.preprocessing import prepare_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Local de-identified PNG/JPEG export.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    if args.warmup < 1 or args.runs < 2:
        parser.error("Use at least one warm-up and two measured runs.")
    prepared = prepare_image(args.image.read_bytes())
    ensemble = load_ensemble(args.device)

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    for _ in range(args.warmup):
        predict(ensemble, prepared.tensor)
    sync()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(args.runs):
        sync()
        start = time.perf_counter()
        predict(ensemble, prepared.tensor)
        sync()
        timings.append((time.perf_counter() - start) * 1000)
    cpu_name = platform.processor()
    if platform.system() == "Windows":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    peak_cpu = None
    peak_cpu_method = "Not measured: optional Windows psutil peak_wset unavailable."
    try:
        import psutil  # Optional benchmark-only dependency, not required by the app.
        memory = psutil.Process().memory_info()
        if hasattr(memory, "peak_wset"):
            peak_cpu = memory.peak_wset / 1024**2
            peak_cpu_method = "OS process peak working set since startup; includes imports, loading and warm-up, not inference-only."
    except ImportError:
        pass
    report = {
        "device": args.device, "cpu": cpu_name, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "opencv": cv2.__version__, "cpu_threads": torch.get_num_threads(),
        "precision": "FP32", "batch_size": 1, "models": 5, "warmup_runs": args.warmup, "measured_runs": args.runs,
        "input": "Fixed first public BRISC test image; repeated-input implementation benchmark, not accuracy evaluation.",
        "timed_region": "predict(): input transfer, five forwards, probability/gate averaging, CPU output transfer and threshold; excludes loading, image decoding/preprocessing, rendering and downloads.",
        "mean_ms": float(np.mean(timings)), "sd_ms": float(np.std(timings, ddof=1)),
        "median_ms": float(np.median(timings)), "runs_ms": timings,
        "cuda_synchronized_before_and_after_each_run": args.device == "cuda",
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else None,
        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2 if args.device == "cuda" else None,
        "gpu_memory_scope": "PyTorch allocator peak after warm-up reset; includes five resident models and activation tensors; excludes non-PyTorch driver allocations.",
        "peak_cpu_working_set_mib": peak_cpu, "peak_cpu_method": peak_cpu_method,
    }
    (APP_DIR / f"reports/benchmark_{args.device}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "runs_ms"}, indent=2))


if __name__ == "__main__":
    main()
