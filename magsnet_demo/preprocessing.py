"""Image-only reproduction of the original MRI-derived contour pipeline."""
import io
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from magsnet_demo.config import CROP_PADDING, IMAGE_SIZE, MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES
from magsnet_demo.utils import DemoError


@dataclass
class PreparedImage:
    original_rgb: np.ndarray
    original_gray: np.ndarray
    analyzed_uint8: np.ndarray
    tensor: torch.Tensor
    crop_box: tuple  # x1, y1, x2, y2; upper bounds are exclusive, as in the source.


def decode_image(data):
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise DemoError("Upload a PNG or JPEG image no larger than 10 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                if probe.format not in {"PNG", "JPEG"}:
                    raise DemoError("Only PNG, JPG and JPEG images are supported.")
                if getattr(probe, "n_frames", 1) != 1:
                    raise DemoError("Upload one still MRI image, not an animated image.")
                width, height = probe.size
                if min(width, height) < 16 or width * height > MAX_IMAGE_PIXELS:
                    raise DemoError("Image dimensions must be at least 16×16 and at most 16 megapixels.")
                if probe.mode.startswith("I") or probe.mode == "F":
                    raise DemoError("This version accepts 8-bit PNG/JPEG exports. DICOM and 16-bit images are not supported.")
                probe.verify()
        encoded = np.frombuffer(data, np.uint8)
        # Same OpenCV grayscale decoder used by the published evaluation loader.
        gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        color = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if gray is None or color is None or gray.ndim != 2 or gray.size == 0:
            raise DemoError("The image could not be decoded. Export a valid PNG or JPEG and try again.")
        return gray, cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    except DemoError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, cv2.error,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise DemoError("The file is corrupt or is not a supported image.") from exc


def normalize_to_uint8(image):
    image = image.astype(np.float32)
    low, high = image.min(), image.max()
    image = (image - low) / (high - low) if high > low else np.zeros_like(image)
    return (image * 255.0).clip(0, 255).astype(np.uint8)


def preprocess_grayscale(image):
    """No label or mask is accepted. The blurred image is used only to find a contour."""
    image = normalize_to_uint8(image)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), dtype=np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise DemoError("No brain contour was found. Check that this is a readable brain MRI. "
                        "Inference stopped; no alternative crop was substituted.")
    contour = max(contours, key=cv2.contourArea)
    height, width = image.shape
    x1 = max(int(contour[:, :, 0].min()) - CROP_PADDING, 0)
    x2 = min(int(contour[:, :, 0].max()) + CROP_PADDING, width)
    y1 = max(int(contour[:, :, 1].min()) - CROP_PADDING, 0)
    y2 = min(int(contour[:, :, 1].max()) + CROP_PADDING, height)
    if x2 <= x1 or y2 <= y1:
        raise DemoError("The detected crop is empty. Check the MRI image.")
    resized = cv2.resize(image[y1:y2, x1:x2], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return resized, (x1, y1, x2, y2)


def prepare_image(data):
    gray, rgb = decode_image(data)
    analyzed, box = preprocess_grayscale(gray)
    tensor = torch.from_numpy(analyzed.astype(np.float32) / 255.0)[None, None]
    return PreparedImage(rgb, gray, analyzed, tensor, box)
