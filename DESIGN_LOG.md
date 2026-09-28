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

## Session 2026-09-25 — Experiments, ROI import, view paths, heatmap ranges (v0.2)

| ID | Agreement |
|----|-----------|
| B1 | **File → Open experiment...** opens a `*_rci.pkl` and its stack. The stack is looked for at `stack.relpath` (relative to the pickle), then `stack.filename` next to the pickle, then the absolute `stack.path` hint; if none exists the user locates it. The opened pickle stays the experiment's file (saves go back to it), even if the stack lives elsewhere or has another name. |
| B2 | **File → Load stack...** is the former *Open stack*: stack + its `<stack>_rci.pkl`. |
| B3 | **Load ROIs from pickle...** (button under the draw buttons, also in File) copies `vertices`, `kind` and `name` from another experiment's ROIs. Masks and traces are rebuilt on the current stack at the same pixel coordinates (clipped to the image; ROIs left without pixels are skipped). A size mismatch asks first. With existing ROIs: add or replace. Names are kept unless they clash, then the next free `S#` / `N#` is used, and the name counters move past imported numbers. Each imported ROI records `imported_from` = {`path`, `name`}. |
| B4 | Each view shows the full path of its file under the title (elided in the middle, full path as tooltip): reference image path, or the stack path for stack mean / max / heatmaps and the stack view. |
| B5 | While a heatmap is the draw-view source, its `computed_ranges` (the ranges actually used for the map) are drawn as read-only yellow bands on both trace plots, same style as in the heatmap window. |

Schema stays **v1**; additions are optional keys: `stack.path`, `stack.relpath` (written on save), ROI `imported_from`.

### Parked / next
- Further trace processing: smoothing, bleach correction, ΔF/F.
- Possible pairing of specific ↔ non-specific ROIs, if the comparison needs it.
- Polygon (click-to-place) and ellipse drawing tools; undo.
- Registration / resampling when the reference image has a different size from the stack.
- Export of traces (e.g. CSV / .mat).

## Session 2026-09-28 — Trace processing window

Opened from **Tools → Trace processing...**. Three tabs. Schema stays **v1**; `annotations` and the contents of `trace_processing` are optional and filled in on load.

| ID | Agreement |
|----|-----------|
| C1 | **Raster.** One row per ROI, min–max scaled like suite2p's `tc_norm`. Sort: document order, specific then non-specific, or average-linkage within each kind. Click a row to select that ROI. |
| C2 | **Similarity.** Pearson r of the raw traces (shape, not amplitude — no extra dependency such as scipy). Question answered: is the mean pairwise r inside the specific group higher than inside the non-specific group? p-values reassign ROIs to groups of the same sizes (every assignment when there are ≤ 20 000, otherwise 4000 random ones). Matrices are ordered by average-linkage on distance `1 − r`. |
| C3 | **Annotations** are `{name, ranges}` on the document. Each range is one event; the onset is its first frame. Created in the Z-score tab (default name AirPuff) and shaded on the raster. |
| C4 | **Z-score.** Baseline length and post-stim length, stored in frames, shown in the document's frames/seconds. A trial is kept only when the whole window is inside the recording and the baseline sample SD is non-zero. Per ROI: dotted trial Z-scores, thick mean, grey ± SEM across trials. One comparison plot: each ROI's trial average as a dotted line (red specific, blue non-specific), thick group means, grey ± SEM across ROIs. |
| C5 | Settings remembered in `trace_processing`: `raster_sort`, `raster_lut`, `zscore` (`annotation`, `baseline_frames`, `post_frames`). |

## Session 2026-09-28 — Annotations, clustering, Pearson tab

Supersedes B5 and C1–C4 where they disagree. Schema stays **v1**.

| ID | Agreement |
|----|-----------|
| D1 | Heatmap categories are the annotations. A category with ranges is an annotation even when no heatmap is computed. Older `annotations` lists are folded into `heatmaps` on load. The window title is Annotations. |
| D2 | On/off is display only, stored in `display.shown_annotations`. The same checklist sits at the upper left of the main traces and on the raster. Each category has its own colour. Z-score still uses one selected category; onset is the first frame of each range. |
| D3 | **Similarity** is one hierarchical matrix: Ružička + average, or Euclidean + Ward, both on per-trace min–max. The dendrogram is drawn beside the matrix, leaves lined up with the rows. Order is one tree of all ROIs, or specific then non-specific each with its own tree. The contrast index is the dotted line on the pooled tree (default 0.7 × max merge height); moving it recuts that tree, and every group is outlined, including single ROIs. The step of the control follows the height of the tree. The cut is inactive in the within-kind view. LUT dropdown colours the matrix. |
| D4 | **Pearson** keeps the earlier comparison on its own tab: raw-trace pairwise r, permutation test, two matrices. |
| D5 | Raster sort adds the same four trees (metric × pooled/within-kind) beside document order and kind order. |
| D6 | Z-score no longer edits events. A Groups box lists Specific and Non-specific with every ROI checked; unchecked ids are stored so new ROIs stay checked. Baseline is labelled “Baseline (prior to event)”. Frames/seconds for baseline and post-stim are local to this tab and stored as frames. |
