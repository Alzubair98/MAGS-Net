"""Read-only SARTAJ / BRISC / Cheng image-source audit. No neural network is used.

Run with Python (dl). All writes are restricted to a new audit results directory.
Images are compared across labels. Matching supports shared image identity, not copying
direction or independent patients. Negative retrieval is never proof of independence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits

cv2.setNumThreads(1)
threadpool_limits(limits=1)
SIZE = 256
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASSES = {0: "Meningioma", 1: "Glioma", 2: "Pituitary", 3: "No Tumor"}
TRANSFORMS = ["identity", "rotate_90", "rotate_180", "rotate_270",
              "horizontal_flip", "rotate_90_then_horizontal_flip", "vertical_flip",
              "rotate_270_then_horizontal_flip"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def discover(workspace):
    sartaj, brisc, cheng = [], [], []
    for base, dirs, files in os.walk(workspace):
        root = Path(base)
        dirs[:] = [d for d in dirs if d not in {".git", ".codex", ".agents", "results", "runs", "tmp", "__pycache__"}]
        if len(root.relative_to(workspace).parts) > 3:
            dirs[:] = []
        children = {d.lower(): d for d in dirs}
        if {"training", "testing"}.issubset(children):
            sartaj.append(root)
            dirs[:] = []
        elif "manifest.json" in files and "segmentation_task" in children:
            brisc.append(root)
            dirs[:] = []
        elif sum(name.endswith(".mat") for name in files) > 1000:
            with h5py.File(root / next(name for name in files if name.endswith(".mat")), "r") as handle:
                if "cjdata" in handle and "image" in handle["cjdata"]:
                    cheng.append(root)
            dirs[:] = []
    assert len(sartaj) == len(brisc) == len(cheng) == 1, (sartaj, brisc, cheng)
    project = brisc[0].parent
    audits = sorted((project / "results").rglob("source_overlap_audit.csv"))
    audits = [p for p in audits if (p.parent / "test_index.csv").is_file() and (p.parent / "curation/latest_curation.json").is_file()]
    assert len(audits) == 1, "Resolve multiple previous BRISC studies explicitly."
    prior = audits[0].parent
    pointer = json.loads((prior / "curation/latest_curation.json").read_text())["directory"]
    return dict(workspace=str(workspace), project=str(project), sartaj=str(sartaj[0]), brisc=str(brisc[0]),
                cheng=str(cheng[0]), prior_study=str(prior), prior_curation=str(prior / "curation" / pointer))


def class_name(path):
    name = str(path).lower().replace("-", "_")
    for token, label in (("meningioma", "Meningioma"), ("glioma", "Glioma"),
                          ("pituitary", "Pituitary"), ("no_tumor", "No Tumor"), ("notumor", "No Tumor")):
        if token in name:
            return label
    raise ValueError(f"Unknown class folder: {path}")


def read_image(path):
    path = Path(path)
    if path.suffix == ".mat":
        with h5py.File(path, "r") as handle:
            return np.asarray(handle["cjdata/image"]).squeeze().astype(np.float32)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2:
        raise ValueError(f"Unreadable MRI: {path}")
    return image


def normalized(image):
    image = image.astype(np.float32)
    low, high = np.percentile(image, [1, 99])
    if high - low < 1e-6:
        low, high = float(image.min()), float(image.max())
    if high - low < 1e-6:
        return np.zeros(image.shape, np.uint8)
    return np.round(np.clip((image - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def content_crop(image):
    # Strip nearly constant black/white margins first, with explicit retained-area limits.
    h, w = image.shape
    y1, y2, x1, x2 = 0, h, 0, w
    constant = lambda line: np.mean(line < 8) > .985 or np.mean(line > 247) > .985
    while y1 < h * .2 and constant(image[y1]): y1 += 1
    while y2 > h * .8 and constant(image[y2 - 1]): y2 -= 1
    while x1 < w * .2 and constant(image[:, x1]): x1 += 1
    while x2 > w * .8 and constant(image[:, x2 - 1]): x2 -= 1
    trimmed = image[y1:y2, x1:x2]
    # MRI-derived ROI only; GT masks are never used for provenance matching.
    _, binary = cv2.threshold(cv2.GaussianBlur(trimmed, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if cw * ch >= .35 * trimmed.size:
            xx1, yy1 = max(x - 3, 0), max(y - 3, 0)
            xx2, yy2 = min(x + cw + 3, trimmed.shape[1]), min(y + ch + 3, trimmed.shape[0])
            return trimmed[yy1:yy2, xx1:xx2], (x1 + xx1, y1 + yy1, x1 + xx2, y1 + yy2)
    return trimmed, (x1, y1, x2, y2)


def representations(image):
    full = normalized(image)
    content, bbox = content_crop(full)
    return np.stack([cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for im in (full, content)]), bbox


def transform(image, code):
    result = np.rot90(image, code % 4, axes=(-2, -1))
    return np.ascontiguousarray(result[..., ::-1] if code >= 4 else result)


def phash(image):
    block = cv2.dct(cv2.resize(image.astype(np.float32), (32, 32)))[:8, :8].flatten()[1:]
    return block > np.median(block)


def prepare(paths, out):
    if (out / "preparation_complete.json").is_file():
        print("Using completed inventories and immutable comparison caches.", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    print("Detected paths before image analysis:\n" + json.dumps(paths, indent=2), flush=True)
    save_json(out / "discovered_paths.json", paths)
    sartaj = []
    for path in sorted(Path(paths["sartaj"]).rglob("*")):
        if path.suffix.lower() not in EXTENSIONS: continue
        relative = path.relative_to(paths["sartaj"])
        with Image.open(path) as image:
            width, height = image.size
            file_format = image.format
        gray = read_image(path)
        sartaj.append(dict(sample_id=str(relative).replace("\\", "/"), filepath=str(path), filename=path.name,
                           split=relative.parts[0], tumor_class=class_name(relative.parent), width=width, height=height,
                           file_format=file_format, file_sha256=sha(path),
                           decoded_sha256=hashlib.sha256(str(gray.shape).encode() + gray.tobytes()).hexdigest()))
    sartaj = pd.DataFrame(sartaj)
    sartaj.to_csv(out / "sartaj_inventory.csv", index=False)
    records = json.loads((Path(paths["brisc"]) / "manifest.json").read_text())
    masks = {r["linked_image"]: r for r in records if r["task"] == "segmentation" and r["is_mask"]}
    brisc = []
    for record in records:
        if record["task"] != "segmentation" or record["is_mask"]: continue
        path = Path(paths["brisc"]) / record["relative_path"]
        mask_record = masks[record["relative_path"]]
        mask_path = Path(paths["brisc"]) / mask_record["relative_path"]
        image, mask = read_image(path), read_image(mask_path)
        assert sha(path) == record["sha256"] and sha(mask_path) == mask_record["sha256"]
        valid = image.shape == mask.shape and bool((mask > (0 if mask.max() <= 1 else 127)).any())
        brisc.append(dict(sample_id=path.stem, filepath=str(path), filename=path.name, split=record["split"],
                          tumor_class=record["tumor_label"].capitalize(), plane=record["plane_label"],
                          width=record["width"], height=record["height"], file_format="JPEG", mask_path=str(mask_path),
                          valid_mask=valid, file_sha256=record["sha256"], mask_sha256=mask_record["sha256"],
                          decoded_sha256=hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()))
    brisc = pd.DataFrame(brisc).sort_values("sample_id").reset_index(drop=True)
    old = pd.read_csv(Path(paths["prior_study"]) / "source_overlap_audit.csv", dtype={"raw_source_sample_id": str, "cropped_source_sample_id": str})
    curation = pd.read_csv(Path(paths["prior_curation"]) / "sample_curation_ledger.csv")
    brisc = brisc.merge(old.drop(columns=["tumor_class", "plane"]), on="sample_id", how="left", validate="one_to_one")
    brisc = brisc.merge(curation[["sample_id", "screen_status"]], on="sample_id", how="left", validate="one_to_one")
    brisc.to_csv(out / "brisc_inventory.csv", index=False)
    cheng = []
    for path in sorted(Path(paths["cheng"]).glob("*.mat"), key=lambda p: int(p.stem)):
        with h5py.File(path, "r") as handle:
            data = handle["cjdata"]
            shape = data["image"].shape
            label = int(np.asarray(data["label"]).squeeze()) - 1
            pid = "".join(chr(int(x)) for x in np.asarray(data["PID"]).flatten() if x).strip()
        cheng.append(dict(sample_id=path.stem, filepath=str(path), filename=path.name, tumor_class=CLASSES[label],
                          patient_id=pid, height=shape[0], width=shape[1], file_format="MAT-HDF5", file_sha256=sha(path)))
    cheng = pd.DataFrame(cheng)
    cheng.to_csv(out / "cheng_inventory.csv", index=False)
    test = brisc.loc[brisc.split == "test"].reset_index(drop=True)
    for name, frame in (("sartaj", sartaj), ("brisc_test", test), ("cheng", cheng)):
        frame.to_csv(out / f"{name}_index.csv", index=False)
        cache = np.lib.format.open_memmap(out / f"{name}_representations.npy", mode="w+", dtype=np.uint8, shape=(len(frame), 2, SIZE, SIZE))
        preprocessing = []
        for number, row in enumerate(frame.itertuples()):
            cache[number], bbox = representations(read_image(row.filepath))
            preprocessing.append(dict(sample_id=row.sample_id, content_bbox=json.dumps(tuple(int(x) for x in bbox)),
                                      phash_full="".join(str(int(x)) for x in phash(cache[number, 0])),
                                      phash_content="".join(str(int(x)) for x in phash(cache[number, 1]))))
            if (number + 1) % 500 == 0: print(f"{name}: fingerprinted {number + 1}/{len(frame)}", flush=True)
        cache.flush()
        pd.DataFrame(preprocessing).to_csv(out / f"{name}_fingerprints.csv", index=False)
    counts = []
    for name, frame in (("SARTAJ", sartaj), ("BRISC", brisc)):
        counts.append(frame.groupby(["split", "tumor_class"]).size().rename("n_images").reset_index().assign(dataset=name))
    pd.concat(counts).to_csv(out / "dataset_counts.csv", index=False)
    save_json(out / "preparation_complete.json", dict(sartaj=len(sartaj), sartaj_tumor=int(sartaj.tumor_class.ne("No Tumor").sum()),
              brisc=len(brisc), brisc_test=len(test), cheng=len(cheng), source_files_read_only=True,
              preprocessing="grayscale; 1/99 percentile normalization; full + constant-border/content ROI; 256x256; D4"))
    print(pd.concat(counts).to_string(index=False), flush=True)


def low_vectors(bank):
    small = np.stack([cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA) for image in bank])
    flat = small.reshape(len(small), -1).astype(np.float32)
    flat -= flat.mean(axis=1, keepdims=True)
    flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-8)
    return small, flat


def retrieve(query, reference, query_indices, out_path, top_k=6):
    if out_path.is_file():
        return pd.read_csv(out_path)
    # Torch is used only for dense matrix products and top-k. No nn.Module/optimizer/model.
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scores = torch.full((len(query_indices), len(reference)), -2.0, device=device)
    codes = torch.zeros(scores.shape, dtype=torch.int16, device=device)
    query_vectors = [torch.from_numpy(low_vectors(query[query_indices, rep])[1]).to(device) for rep in range(2)]
    for reference_rep in range(2):
        small, _ = low_vectors(reference[:, reference_rep])
        for t in range(8):
            features = transform(small, t).reshape(len(reference), -1).astype(np.float32)
            features -= features.mean(axis=1, keepdims=True)
            features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
            features = torch.from_numpy(features).to(device)
            for query_rep in range(2):
                current = query_vectors[query_rep] @ features.T
                better = current > scores
                scores = torch.maximum(scores, current)
                codes[better] = query_rep * 16 + reference_rep * 8 + t
    values, nearest = scores.topk(top_k, dim=1)
    selected_codes = codes.gather(1, nearest).cpu().numpy()
    values, nearest = values.cpu().numpy(), nearest.cpu().numpy()
    rows = [dict(query_index=int(q), reference_index=int(nearest[i, k]), rank=k + 1,
                 retrieval_correlation=float(values[i, k]), hypothesis_code=int(selected_codes[i, k]))
            for i, q in enumerate(query_indices) for k in range(top_k)]
    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    del scores, codes, query_vectors
    if device == "cuda": torch.cuda.empty_cache()
    print(f"Retrieved {top_k} candidates per query: {out_path.name} ({len(query_indices)} queries)", flush=True)
    return result


def ncc(a, b, valid=None):
    x, y = (a.ravel(), b.ravel()) if valid is None else (a[valid], b[valid])
    x, y = x.astype(np.float64), y.astype(np.float64)
    x, y = x - x.mean(), y - y.mean()
    return float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))


def pair_metrics(a, b, valid=None):
    a, b = a.astype(np.float32) / 255, b.astype(np.float32) / 255
    valid = np.ones(a.shape, bool) if valid is None else valid.copy()
    valid[:5] = valid[-5:] = False
    valid[:, :5] = valid[:, -5:] = False
    foreground = cv2.dilate(((a > .05) | (b > .05)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    support = foreground & valid
    if support.sum() < 100:
        support = valid
    # Positive linear intensity alignment only. Report its parameters explicitly.
    x, y = b[support], a[support]
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / max(float(((x - x.mean()) ** 2).sum()), 1e-8))
    offset = float(y.mean() - slope * x.mean())
    if not (.5 <= slope <= 2 and abs(offset) <= .2): slope, offset = 1.0, 0.0
    adjusted = np.clip(slope * b + offset, 0, 1)
    au, bu = np.round(a * 255).astype(np.uint8), np.round(adjusted * 255).astype(np.uint8)
    _, ssim_map = cv2.quality.QualitySSIM_compute(au, bu)
    ga = cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
    gb = cv2.magnitude(cv2.Sobel(adjusted, cv2.CV_32F, 1, 0), cv2.Sobel(adjusted, cv2.CV_32F, 0, 1))
    return dict(correlation=ncc(a, adjusted, support), ssim=float(ssim_map[support].mean()),
                ssim_full=float(ssim_map[valid].mean()), mae=float(np.abs(a - adjusted)[support].mean()),
                phash_distance=int(np.count_nonzero(phash(au) != phash(bu))),
                edge_correlation=ncc(ga, gb, support), foreground_fraction=float(support.mean()),
                intensity_slope=slope, intensity_offset=offset)


def hypothesis(a, b, query_rep, ref_rep, t, allow_registration=True):
    target, source = a[query_rep], transform(b[ref_rep], t)
    metrics = pair_metrics(target, source)
    result = dict(**metrics, query_rep=query_rep, reference_rep=ref_rep, transform_code=t,
                   best_transform=TRANSFORMS[t], registration="none", warp_matrix=json.dumps([[1, 0, 0], [0, 1, 0]]),
                   valid_coverage=1.0)
    if allow_registration and .80 < metrics["correlation"] < .9998 and metrics["ssim"] < .995:
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(target.astype(np.float32) / 255, source.astype(np.float32) / 255,
                                           warp, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-5), None, 5)
            scales = np.linalg.svd(warp[:, :2], compute_uv=False)
            if np.linalg.det(warp[:, :2]) > 0 and scales.min() >= .8 and scales.max() <= 1.2 and np.max(np.abs(warp[:, 2])) <= SIZE * .12:
                aligned = cv2.warpAffine(source, warp, (SIZE, SIZE), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                valid = cv2.warpAffine(np.ones(source.shape, np.uint8), warp, (SIZE, SIZE), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
                valid = cv2.erode(valid.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
                if valid.mean() >= .85:
                    aligned_metrics = pair_metrics(target, aligned, valid)
                    if aligned_metrics["ssim"] > metrics["ssim"] and aligned_metrics["correlation"] > metrics["correlation"]:
                        result.update(aligned_metrics, registration="bounded_affine_ECC", warp_matrix=json.dumps(warp.tolist()),
                                      valid_coverage=float(valid.mean()))
        except cv2.error:
            pass
    return result


def verify_pair(a, b, suggested_code=None):
    # All eight orientations and both comparison representations are screened.
    qa = np.stack([low_vectors(a[rep:rep + 1])[1][0] for rep in range(2)])
    variants, details = [], []
    for ref_rep in range(2):
        small = cv2.resize(b[ref_rep], (64, 64), interpolation=cv2.INTER_AREA)
        for t in range(8):
            v = transform(small, t).astype(np.float32).ravel()
            v -= v.mean()
            variants.append(v / max(np.linalg.norm(v), 1e-8))
            details.append((ref_rep, t))
    coarse = qa @ np.stack(variants).T
    order = np.argsort(coarse.ravel())[::-1][:2]
    hypotheses = [(int(k // 16), *details[int(k % 16)]) for k in order]
    if suggested_code is not None:
        hypotheses.insert(0, (int(suggested_code // 16), int((suggested_code % 16) // 8), int(suggested_code % 8)))
    results = [hypothesis(a, b, *h) for h in dict.fromkeys(hypotheses)]
    return max(results, key=lambda x: (x["ssim"] + x["correlation"] + .25 * x["edge_correlation"] - x["mae"]))


def run_verification(name, query, reference, retrieval, out, workers):
    destination = out / f"{name}_verified_pairs.csv"
    if destination.is_file():
        print("Using verified pair cache:", name, flush=True)
        return pd.read_csv(destination)
    tasks = list(retrieval.itertuples(index=False))
    def work(row):
        return dict(query_index=int(row.query_index), reference_index=int(row.reference_index),
                    retrieval_rank=int(row.rank), retrieval_correlation=float(row.retrieval_correlation),
                    **verify_pair(query[int(row.query_index)], reference[int(row.reference_index)], int(row.hypothesis_code)))
    rows = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, result in enumerate(pool.map(work, tasks), 1):
            rows.append(result)
            if n % 100 == 0:
                print(f"{name}: verified {n}/{len(tasks)} pairs ({time.time() - start:.0f}s)", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(destination, index=False)
    return result


def load_caches(out):
    indices = {name: pd.read_csv(out / f"{name}_index.csv", dtype={"sample_id": str, "patient_id": str}) for name in ("sartaj", "brisc_test", "cheng")}
    arrays = {name: np.load(out / f"{name}_representations.npy", mmap_mode="r") for name in indices}
    return indices, arrays


def calibrate(indices, arrays, out, workers):
    if (out / "calibration_pairs.csv").is_file(): return
    rng = np.random.default_rng(42)
    source_ids = rng.choice(len(indices["cheng"]), 80, replace=False)
    # Positives are known modified copies. Hard negatives use a different known Cheng PID.
    reference = arrays["cheng"]
    negative_retrieval = retrieve(reference, reference, source_ids, out / "calibration_retrieval.csv", top_k=128)
    tasks = []
    for number, q in enumerate(source_ids):
        full = read_image(indices["cheng"].iloc[q].filepath)
        full = normalized(full)
        t = number % 8
        edited = transform(full, t)
        if number % 4 == 0:
            h, w = edited.shape
            edited = edited[int(h*.025):int(h*.975), int(w*.025):int(w*.975)]
        edited = cv2.resize(edited, (217 + number % 31, 223 + number % 29))
        edited = np.clip(edited.astype(np.float32) * .83 + 14, 0, 255).astype(np.uint8)
        _, jpeg = cv2.imencode(".jpg", edited, [cv2.IMWRITE_JPEG_QUALITY, 85])
        edited = cv2.imdecode(jpeg, cv2.IMREAD_GRAYSCALE)
        positive, _ = representations(edited)
        candidates = negative_retrieval.loc[negative_retrieval.query_index == q]
        candidates = candidates.loc[indices["cheng"].iloc[candidates.reference_index].patient_id.to_numpy() != indices["cheng"].iloc[q].patient_id]
        assert len(candidates), "Expand hard-negative retrieval."
        n = int(candidates.iloc[0].reference_index)
        tasks.extend([("synthetic_positive", q, q, positive, reference[q]), ("different_patient_hard_negative", q, n, reference[q], reference[n])])
    def work(task):
        kind, q, r, a, b = task
        return dict(kind=kind, query_index=int(q), reference_index=int(r), **verify_pair(a, b))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result = pd.DataFrame(pool.map(work, tasks))
    result.to_csv(out / "calibration_pairs.csv", index=False)
    print("Calibration complete: 80 known transformed positives + 80 different-patient nearest negatives.", flush=True)


def execute_matching(indices, arrays, out, workers):
    calibrate(indices, arrays, out, workers)
    brisc, sartaj, cheng = (indices[n] for n in ("brisc_test", "sartaj", "cheng"))
    # Search every SARTAJ class so label conflicts cannot hide identity matches.
    focus = np.flatnonzero(brisc.screen_status.isin(["source_disjoint_candidate", "ambiguous"]))
    assert len(focus) == 118
    retrieved = retrieve(arrays["brisc_test"], arrays["sartaj"], focus, out / "brisc_sartaj_retrieval.csv")
    run_verification("brisc_sartaj", arrays["brisc_test"], arrays["sartaj"], retrieved, out, workers)
    tumor_indices = np.flatnonzero(sartaj.tumor_class != "No Tumor")
    retrieved = retrieve(arrays["sartaj"], arrays["cheng"], tumor_indices, out / "sartaj_cheng_retrieval.csv")
    run_verification("sartaj_cheng", arrays["sartaj"], arrays["cheng"], retrieved, out, workers)
    # Full retrieval only for the 118 unresolved BRISC images. Previous 742 candidates
    # receive multi-metric verification of their saved source IDs (not a repeated search).
    retrieved = retrieve(arrays["brisc_test"], arrays["cheng"], focus, out / "brisc_cheng_focus_retrieval.csv")
    extra = []
    mapping = {str(row.sample_id): i for i, row in enumerate(cheng.itertuples())}
    for q, row in brisc.iterrows():
        if q in set(focus): continue
        for field in ("raw_source_sample_id", "cropped_source_sample_id"):
            source_id = str(row[field]).removesuffix(".0")
            r = mapping[source_id]
            extra.append(dict(query_index=q, reference_index=r, rank=1, retrieval_correlation=row.max_correlation, hypothesis_code=0))
    retrieval = pd.concat([retrieved, pd.DataFrame(extra)]).drop_duplicates(["query_index", "reference_index"])
    run_verification("brisc_cheng", arrays["brisc_test"], arrays["cheng"], retrieval, out, workers)


def sift_refine(a, b, baseline):
    """Feature-based initialization for modest crops missed by direct ECC.

    No nonrigid registration. Require spatially distributed SIFT inliers, near-unit
    affine scales, near-right-angle rotation, high field coverage, and joint pixel metrics.
    """
    detector = cv2.SIFT_create(nfeatures=600, contrastThreshold=.015)
    matcher = cv2.BFMatcher()
    best = dict(baseline)
    for query_rep, ref_rep in ((0, 0), (1, 1)):
        target = a[query_rep]
        kp_q, desc_q = detector.detectAndCompute(target, None)
        if desc_q is None: continue
        for t in (0, 4):
            source = transform(b[ref_rep], t)
            kp_r, desc_r = detector.detectAndCompute(source, None)
            if desc_r is None: continue
            matches = matcher.knnMatch(desc_r, desc_q, k=2)
            good = [x[0] for x in matches if len(x) == 2 and x[0].distance < .72 * x[1].distance]
            if len(good) < 12: continue
            points_r = np.float32([kp_r[m.queryIdx].pt for m in good])
            points_q = np.float32([kp_q[m.trainIdx].pt for m in good])
            forward, inliers = cv2.estimateAffine2D(points_r, points_q, method=cv2.RANSAC, ransacReprojThreshold=2,
                                                    maxIters=3000, confidence=.999, refineIters=10)
            if forward is None or inliers.sum() < 12 or inliers.mean() < .70: continue
            scales = np.linalg.svd(forward[:, :2], compute_uv=False)
            center = np.array([SIZE / 2, SIZE / 2])
            displacement = np.linalg.norm(forward[:, :2] @ center + forward[:, 2] - center)
            angle = np.degrees(np.arctan2(forward[1, 0], forward[0, 0]))
            right_angle_error = abs((angle + 45) % 90 - 45)
            spatial = min(cv2.contourArea(cv2.convexHull(points[inliers.ravel() > 0])) / (SIZE * SIZE)
                          for points in (points_r, points_q))
            if scales.min() < .80 or scales.max() > 1.20 or np.linalg.det(forward[:, :2]) <= 0 or displacement > .18 * SIZE or right_angle_error > 12 or spatial < .20:
                continue
            warp = cv2.invertAffineTransform(forward).astype(np.float32)
            aligned = cv2.warpAffine(source, warp, (SIZE, SIZE), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            valid = cv2.warpAffine(np.ones(source.shape, np.uint8), warp, (SIZE, SIZE), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
            valid = cv2.erode(valid, np.ones((11, 11), np.uint8)) > 0
            if valid.mean() < .85: continue
            metrics = pair_metrics(target, aligned, valid)
            if metrics["ssim"] > best["ssim"] and metrics["correlation"] > best["correlation"]:
                best.update(metrics, query_rep=query_rep, reference_rep=ref_rep, transform_code=t,
                            best_transform=f"{TRANSFORMS[t]} + affine rotation {angle:.2f} degrees",
                            registration="SIFT_RANSAC_bounded_affine", warp_matrix=json.dumps(warp.tolist()),
                            valid_coverage=float(valid.mean()), sift_inliers=int(inliers.sum()),
                            sift_inlier_fraction=float(inliers.mean()), sift_spatial_coverage=float(spatial))
    return best


def refine_saved(indices, arrays, out, workers):
    for name, qname, rname in (("brisc_sartaj", "brisc_test", "sartaj"),
                                ("sartaj_cheng", "sartaj", "cheng"), ("brisc_cheng", "brisc_test", "cheng")):
        path = out / f"{name}_verified_pairs.csv"
        destination = out / f"{name}_refined_pairs.csv"
        if destination.is_file(): continue
        frame = pd.read_csv(path)
        frame["joint"] = frame.correlation + frame.ssim + .25 * frame.edge_correlation - frame.mae
        best_indices = frame.sort_values("joint").groupby("query_index").tail(2).index
        tasks = [i for i in best_indices if frame.loc[i, "ssim"] < .98]
        def work(i):
            row = frame.loc[i].to_dict()
            improved = sift_refine(arrays[qname][int(row["query_index"])], arrays[rname][int(row["reference_index"])], row)
            return i, improved
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for n, (i, values) in enumerate(pool.map(work, tasks), 1):
                for key, value in values.items(): frame.loc[i, key] = value
                if n % 100 == 0: print(f"{name}: SIFT fallback {n}/{len(tasks)}", flush=True)
        frame.drop(columns="joint").to_csv(destination, index=False)
    destination = out / "calibration_refined.csv"
    if not destination.is_file():
        frame = pd.read_csv(out / "calibration_pairs.csv")
        selected = np.random.default_rng(42).choice(len(indices["cheng"]), 80, replace=False)
        tasks = []
        for i, row in frame.iterrows():
            q, r = int(row.query_index), int(row.reference_index)
            if row.kind == "synthetic_positive":
                number = int(np.flatnonzero(selected == q)[0])
                image = transform(normalized(read_image(indices["cheng"].iloc[q].filepath)), number % 8)
                if number % 4 == 0:
                    h, w = image.shape
                    image = image[int(h*.025):int(h*.975), int(w*.025):int(w*.975)]
                image = cv2.resize(image, (217 + number % 31, 223 + number % 29))
                image = np.clip(image.astype(np.float32)*.83 + 14, 0, 255).astype(np.uint8)
                _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
                a, _ = representations(image)
            else:
                a = arrays["cheng"][q]
            tasks.append((i, a, arrays["cheng"][r], row.to_dict()))
        def work(task):
            i, a, b, row = task
            return i, sift_refine(a, b, row)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, values in pool.map(work, tasks):
                for key, value in values.items(): frame.loc[i, key] = value
        frame.to_csv(destination, index=False)
        print("Refined calibration complete.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stage", choices=["prepare", "match", "refine", "all"], default="all")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    paths = discover(args.workspace.resolve())
    out = args.out or (Path(paths["project"]) / "results/source_provenance_audit" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    assert out.resolve().is_relative_to(Path(paths["project"]).resolve() / "results")
    out.mkdir(parents=True, exist_ok=True)
    print("AUDIT OUTPUT:", out.resolve(), flush=True)
    prepare(paths, out)
    if args.stage != "prepare":
        indices, arrays = load_caches(out)
        if args.stage != "refine": execute_matching(indices, arrays, out, args.workers)
        if args.stage in {"all", "refine"}: refine_saved(indices, arrays, out, args.workers)
    print("Requested stages completed:", out, flush=True)


if __name__ == "__main__":
    main()
