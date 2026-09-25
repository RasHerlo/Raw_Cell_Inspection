"""The per-stack pickle document (see DESIGN_LOG.md for the schema)."""

from __future__ import annotations

import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path

from raw_cell_inspection import __version__

SCHEMA_VERSION = 1
PICKLE_SUFFIX = "_rci.pkl"

KIND_SPECIFIC = "specific"
KIND_NONSPECIFIC = "nonspecific"
KIND_PREFIX = {KIND_SPECIFIC: "S", KIND_NONSPECIFIC: "N"}


def pickle_path_for(stack_path: str | os.PathLike) -> Path:
    stack_path = Path(stack_path)
    return stack_path.with_name(stack_path.stem + PICKLE_SUFFIX)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_document(signature: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "app_version": __version__,
        "created": _now(),
        "modified": _now(),
        "stack": dict(signature),
        "fps": None,
        "x_units": "frames",
        "summary": None,  # {'mean_image', 'max_image', 'fov_mean_trace'}
        "reference_image": None,  # {'image', 'source_path', 'source_relpath', 'note'}
        "rois": [],
        "next_roi_number": {KIND_SPECIFIC: 1, KIND_NONSPECIFIC: 1},
        "heatmaps": [],
        "display": {},
        "trace_processing": {},
    }


def load_document(path: str | os.PathLike) -> dict:
    with open(path, "rb") as fh:
        doc = pickle.load(fh)
    if not isinstance(doc, dict) or "schema_version" not in doc:
        raise ValueError(f"{Path(path).name} is not a Raw Cell Inspection file.")
    if doc["schema_version"] > SCHEMA_VERSION:
        raise ValueError(
            f"{Path(path).name} was written by a newer version of the app "
            f"(schema {doc['schema_version']}, this app reads up to {SCHEMA_VERSION})."
        )
    return _upgrade(doc)


def _upgrade(doc: dict) -> dict:
    template = new_document(doc.get("stack", {}))
    for key, value in template.items():
        doc.setdefault(key, value)
    return doc


def save_document(path: str | os.PathLike, doc: dict) -> None:
    """Write atomically, keeping the previous file as <name>.bak."""
    path = Path(path)
    doc["modified"] = _now()
    doc["app_version"] = __version__
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(doc, fh, protocol=pickle.HIGHEST_PROTOCOL)
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    os.replace(tmp, path)


def signature_mismatch(stored: dict, current: dict) -> list[str]:
    """Human-readable differences between the stored and the opened stack."""
    problems = []
    if tuple(stored.get("shape", ())) != tuple(current["shape"]):
        problems.append(f"shape {tuple(stored.get('shape', ()))} -> {tuple(current['shape'])}")
    if stored.get("dtype") != current["dtype"]:
        problems.append(f"data type {stored.get('dtype')} -> {current['dtype']}")
    if stored.get("file_size") != current["file_size"]:
        problems.append("file size changed")
    return problems


def stack_candidates(pkl_path: str | os.PathLike, doc: dict) -> list[Path]:
    """Where to look for an experiment's stack, relative to its pickle first."""
    folder = Path(pkl_path).parent
    info = doc.get("stack", {})
    out = []
    if info.get("relpath"):
        out.append(folder / info["relpath"])
    if info.get("filename"):
        out.append(folder / info["filename"])
    if info.get("path"):
        out.append(Path(info["path"]))
    return out


def relpath_or_none(target: str | os.PathLike, start: str | os.PathLike) -> str | None:
    try:
        return os.path.relpath(target, start)
    except ValueError:  # different drive on Windows
        return None
