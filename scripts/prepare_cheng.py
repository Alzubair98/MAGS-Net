"""Reconstruct historical Cheng image/mask pairs and check their decoded pixel hashes."""
import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / 'research_reference/results/magsnet_nested_cv/preparation/04ddbd02826f/sample_index.csv'


def reconstruct(row, raw_dir):
    with h5py.File(raw_dir / row['original_mat'], 'r') as handle:
        data = handle['cjdata']
        label = int(np.asarray(data['label']).squeeze()) - 1
        patient = ''.join(chr(int(c)) for c in np.asarray(data['PID']).flatten() if c).strip()
        image = np.asarray(data['image']).squeeze().astype(np.float32)
        mask = (np.asarray(data['tumorMask']).squeeze() > 0).astype(np.uint8)
    if label != int(row['label']) or patient != row['patient_id']:
        raise ValueError(f"Source metadata mismatch: {row['sample_id']}")
    low, high = image.min(), image.max()
    image = (image - low) / (high - low) if high > low else np.zeros_like(image)
    image = (image * 255.0).clip(0, 255).astype(np.uint8)
    x1, y1, x2, y2 = (int(row[k]) for k in ('crop_x1', 'crop_y1', 'crop_x2', 'crop_y2'))
    size = (int(row['width']), int(row['height']))
    image = cv2.resize(image[y1:y2, x1:x2], size, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask[y1:y2, x1:x2], size, interpolation=cv2.INTER_NEAREST) * 255
    for name, array in [('image', image), ('mask', mask)]:
        if hashlib.sha256(array.tobytes()).hexdigest() != row[f'{name}_hash']:
            raise ValueError(f"Pixel fingerprint mismatch: {row['sample_id']} {name}")
    return image, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'brainDataset')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'brainDataset_cropped')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    with INDEX.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    metadata = args.output_dir / 'metadata_clean.csv'
    if not args.verify_only and metadata.exists():
        raise FileExistsError('Output metadata already exists; use --verify-only or a new output directory.')
    # Preflight all image destinations before writing any data.
    if not args.verify_only:
        for row in rows:
            for folder in ('images', 'masks'):
                if (args.output_dir / folder / row['filename']).exists():
                    raise FileExistsError(f"Existing output pair: {row['filename']}")
        for folder in ('images', 'masks'):
            (args.output_dir / folder).mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(rows, 1):
        image, mask = reconstruct(row, args.raw_dir)
        if not args.verify_only:
            for folder, array in [('images', image), ('masks', mask)]:
                if not cv2.imwrite(str(args.output_dir / folder / row['filename']), array):
                    raise OSError(f"Cannot write {row['filename']}")
        if number % 500 == 0:
            print(f'Verified {number}/{len(rows)} pairs', flush=True)
    if not args.verify_only:
        # audit_dataset computes these columns anew; avoid duplicating them in metadata.
        derived = {'image_hash', 'mask_hash', 'height', 'width', 'gt_tumor_pixels', 'tumor_fraction'}
        metadata_fields = [field for field in fields if field not in derived]
        with metadata.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=metadata_fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    print(f'PASS: {len(rows)} exact image/mask pairs; no training or new partitioning.')


if __name__ == '__main__':
    main()
