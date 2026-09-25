"""Streaming computations over a whole stack: summary images and range heatmaps."""

from __future__ import annotations

from typing import Callable

import numpy as np

from raw_cell_inspection.stack_io import iter_frame_chunks

# progress(fraction) -> False to cancel
Progress = Callable[[float], bool]


class Cancelled(Exception):
    pass


def _frame_bytes(stack: np.ndarray) -> int:
    return int(stack.shape[1] * stack.shape[2] * stack.dtype.itemsize)


def stack_summary(stack: np.ndarray, progress: Progress | None = None) -> dict:
    """One pass: mean image, max image and the mean trace of the whole field of view."""
    n, h, w = stack.shape
    total = np.zeros((h, w), dtype=np.float64)
    peak = np.full((h, w), -np.inf, dtype=np.float64)
    fov_trace = np.zeros(n, dtype=np.float64)
    for start, stop in iter_frame_chunks(n, _frame_bytes(stack)):
        chunk = np.asarray(stack[start:stop])
        total += chunk.sum(axis=0, dtype=np.float64)
        np.maximum(peak, chunk.max(axis=0), out=peak)
        fov_trace[start:stop] = chunk.reshape(stop - start, -1).mean(axis=1, dtype=np.float64)
        if progress is not None and progress(stop / n) is False:
            raise Cancelled()
    return {
        "mean_image": (total / n).astype(np.float32),
        "max_image": peak.astype(np.float32),
        "fov_mean_trace": fov_trace.astype(np.float32),
    }


def frames_in_ranges(n_frames: int, ranges: list[tuple[int, int]]) -> np.ndarray:
    """Boolean frame selector for inclusive (start, end) ranges."""
    sel = np.zeros(n_frames, dtype=bool)
    for start, end in ranges:
        a = max(0, int(min(start, end)))
        b = min(n_frames - 1, int(max(start, end)))
        if b >= a:
            sel[a : b + 1] = True
    return sel


def range_ratio_heatmap(
    stack: np.ndarray, ranges: list[tuple[int, int]], progress: Progress | None = None
) -> np.ndarray:
    """Per pixel: mean signal inside the ranges divided by mean signal in all other frames.

    An unresponsive pixel is ~1. Pixels with a non-positive outside mean are NaN.
    """
    n, h, w = stack.shape
    inside = frames_in_ranges(n, ranges)
    n_in = int(inside.sum())
    n_out = n - n_in
    if n_in == 0:
        raise ValueError("The ranges do not cover any frame of the stack.")
    if n_out == 0:
        raise ValueError("The ranges cover the whole stack; nothing is left to normalise by.")
    sum_in = np.zeros((h, w), dtype=np.float64)
    sum_out = np.zeros((h, w), dtype=np.float64)
    for start, stop in iter_frame_chunks(n, _frame_bytes(stack)):
        chunk = np.asarray(stack[start:stop])
        sel = inside[start:stop]
        if sel.any():
            sum_in += chunk[sel].sum(axis=0, dtype=np.float64)
        if not sel.all():
            sum_out += chunk[~sel].sum(axis=0, dtype=np.float64)
        if progress is not None and progress(stop / n) is False:
            raise Cancelled()
    mean_in = sum_in / n_in
    mean_out = sum_out / n_out
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mean_out > 0, mean_in / mean_out, np.nan)
    return ratio.astype(np.float32)
