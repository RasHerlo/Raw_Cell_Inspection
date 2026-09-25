"""Reading movie stacks and reference images from disk."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tifffile

TIF_SUFFIXES = (".tif", ".tiff")
IMAGE_SUFFIXES = TIF_SUFFIXES + (".png", ".jpg", ".jpeg", ".bmp")


class StackError(Exception):
    pass


def load_stack(path: str | os.PathLike) -> np.ndarray:
    """Return a (frames, height, width) array; memory-mapped when the tif allows it."""
    path = Path(path)
    try:
        stack = tifffile.memmap(path, mode="r")
    except Exception:
        stack = tifffile.imread(path)
    stack = np.squeeze(stack)
    if stack.ndim != 3:
        raise StackError(
            f"Expected a single-channel time series (frames x height x width), "
            f"got shape {tuple(stack.shape)}."
        )
    if stack.shape[0] < 2:
        raise StackError("The stack has fewer than two frames.")
    return stack


def stack_signature(path: str | os.PathLike, stack: np.ndarray) -> dict:
    path = Path(path)
    return {
        "filename": path.name,
        "shape": tuple(int(s) for s in stack.shape),
        "dtype": str(stack.dtype),
        "file_size": int(path.stat().st_size),
    }


def load_reference_image(path: str | os.PathLike) -> tuple[np.ndarray, str]:
    """Load a 2D reference image. Multi-frame tifs are averaged over their first axis.

    Returns the float32 image and a short note describing what was done.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in TIF_SUFFIXES:
        img = np.squeeze(tifffile.imread(path))
    else:
        img = _read_with_qt(path)
    note = ""
    if img.ndim == 3 and img.shape[-1] in (3, 4) and suffix not in TIF_SUFFIXES:
        img = img[..., :3].mean(axis=-1)
        note = "colour image converted to grey"
    elif img.ndim == 3:
        note = f"{img.shape[0]} frames averaged"
        img = img.mean(axis=0, dtype=np.float64)
    if img.ndim != 2:
        raise StackError(f"Could not interpret image of shape {tuple(img.shape)} as 2D.")
    return np.asarray(img, dtype=np.float32), note


def _read_with_qt(path: Path) -> np.ndarray:
    from PySide6.QtGui import QImage

    qimg = QImage(str(path))
    if qimg.isNull():
        raise StackError(f"Could not read image {path.name}.")
    qimg = qimg.convertToFormat(QImage.Format.Format_Grayscale16)
    w, h = qimg.width(), qimg.height()
    bpl = qimg.bytesPerLine()
    buf = np.frombuffer(qimg.constBits(), dtype=np.uint16, count=h * bpl // 2)
    return buf.reshape(h, bpl // 2)[:, :w].copy()


def iter_frame_chunks(n_frames: int, frame_bytes: int, target_bytes: int = 64 * 2**20):
    """Yield (start, stop) frame ranges of roughly target_bytes each."""
    step = max(1, target_bytes // max(1, frame_bytes))
    for start in range(0, n_frames, step):
        yield start, min(n_frames, start + step)
