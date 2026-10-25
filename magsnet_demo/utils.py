"""Small in-memory export and integrity helpers; no upload logging or persistence."""
import hashlib
import io
import json

import numpy as np
from PIL import Image


class DemoError(RuntimeError):
    """A safe, actionable message that may be displayed in the interface."""


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def json_bytes(summary):
    return json.dumps(summary, indent=2, allow_nan=False).encode("utf-8")
