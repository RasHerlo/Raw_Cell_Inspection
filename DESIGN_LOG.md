# Raw Cell Inspection — Design Log

## Session 2026-09-24 — Primary GUI (v0.1)

### Intent
Draw ROIs by eye on a reference image or a heatmap, see them on the movie stack, and collect their traces. Trace processing comes later in a separate window that reads the same pickle.

### Agreements

| ID | Agreement |
|----|-----------|
| A1 | **Specific** ROIs encircle cells. **Non-specific** ROIs do not; they show whether the dynamics seen in cells are cell-specific. Any number of each. |
| A2 | Selection is independent per kind: the last selected specific ROI is the upper trace, the last selected non-specific ROI the lower trace. No pairing (yet). |
| A3 | The reference image usually comes from a different recording (other laser), not from the stack. It must have the stack's width × height; multi-frame tifs are averaged. The image itself is stored in the pickle, its path only as a hint. |
| A4 | Draw-view sources: reference image, stack mean, stack max, computed heatmaps. Stack-view sources: movie frame at the cursor, stack mean, stack max. |
| A5 | Drawing = freehand lasso, by eye (no assisted segmentation). Simplified to ≤ 60 vertices, then editable (drag handles / move body) in the draw view only. The stack view selects but does not edit. |
| A6 | A pixel belongs to an ROI when its centre is inside the polygon. Trace = mean raw intensity over those pixels per frame. Overlapping ROIs are allowed; a click selects the smallest ROI under the cursor. |
| A7 | Heatmap ranges are set on the mean trace of the whole field of view. Categories have free names (no presets) and one or more inclusive frame ranges. Metric per pixel: mean inside ranges / mean in all other frames (non-positive denominator → NaN). |
| A8 | Frame rate is entered by the user and saved; the x axis can be frames or seconds (only tick labels change; everything is stored in frames). |
| A9 | Clicking a trace moves the frame cursor; selecting an ROI in either view selects it in both. |
| A10 | One pickle per stack: `<stack stem>_rci.pkl` in the stack's folder, loaded automatically when the stack is opened. Save keeps the previous file as `.bak`. |
| A11 | Toolkit: pyqtgraph + PySide6 (LGPL, fine to bundle). Distributed as a one-file Windows `.exe` on GitHub Releases, built by GitHub Actions on `v*` tags. |
| A12 | Typical stacks are ~512 × 512 × 3000. Stacks are memory-mapped when the tif allows it; whole-stack passes (summary, heatmaps) stream in ~64 MB chunks in a background thread with progress / cancel. |

Measured on a 512 × 512 × 3000 uint16 stack (warm disk cache): open 0.02 s, summary pass 1.2 s, heatmap 0.7 s, one ROI trace 14 ms, so traces follow ROI edits live.

### Layout

```text
+----------------------------------------------------------------------------+
| File | View | Tools (Heatmaps, Trace processing - later) | Help            |
+-------------+-------------------------------+------------------------------+
| Stack       | DRAW VIEW                     | STACK VIEW                   |
|  fps, units | reference / mean / max /      | frame at cursor / mean / max |
| Draw view   |   heatmap                     | ROIs shown, click to select  |
|  source,LUT | ROIs drawn + edited here      |                              |
| Stack view  |                               | [frame slider] [spin] [s]    |
|  source,LUT +-------------------------------+------------------------------+
| ROIs        | Specific trace (selected specific ROI)       -- frame cursor  |
|  draw, list | Non-specific trace (selected non-specific)   -- shared x      |
+-------------+---------------------------------------------------------------+
```

### Pickle schema — v1

Top-level dict:

| Key | Content |
|-----|---------|
| `schema_version` | `1` |
| `app_version`, `created`, `modified` | strings |
| `stack` | `filename`, `shape` (frames, height, width), `dtype`, `file_size` — checked against the opened stack |
| `fps` | float or `None` |
| `x_units` | `"frames"` / `"seconds"` |
| `summary` | `mean_image`, `max_image` (float32 H×W), `fov_mean_trace` (float32, frames) |
| `reference_image` | `None` or `image` (float32 H×W), `source_path`, `source_relpath` (relative to the stack folder, `None` across drives), `note`, `loaded` |
| `rois` | list of ROI dicts (below) |
| `next_roi_number` | `{"specific": n, "nonspecific": n}` for default names S1…, N1… |
| `heatmaps` | list of `name`, `ranges` ([[start, end], …] inclusive frames), `metric` (`"mean_ratio"`), `image` (float32 H×W or `None`), `computed_ranges` (ranges used for `image`; differs from `ranges` → out of date) |
| `display` | `draw_source`, `stack_source`, `draw` / `stack` LUT names |
| `trace_processing` | `{}` reserved for the trace-processing window |

ROI dict: `id` (int, unique), `name` (unique), `kind` (`"specific"` / `"nonspecific"`), `vertices` (float N×2, x = column, y = row, pixel (r, c) spans [c, c+1] × [r, r+1]), `mask` (`bbox` = (r0, r1, c0, c1) and a bool crop), `n_pixels`, `trace` (float32, frames), `created`, `modified`.

Opening rules:
- Signature matches → everything is used as saved.
- Same width × height but frames / size changed → ROIs kept; summary, all traces recomputed; heatmaps reset to "not computed" (ranges clipped to the new length).
- Different width × height → the pickle is not loaded (it survives as `.bak` if you save).

### Parked / next
- Trace-processing window (smoothing, bleach correction, ΔF/F, comparison of specific vs non-specific).
- Possible pairing of specific ↔ non-specific ROIs, if the comparison needs it.
- Polygon (click-to-place) and ellipse drawing tools; undo.
- Registration / resampling when the reference image has a different size from the stack.
- Export of traces (e.g. CSV / .mat).
