# Raw Cell Inspection

Draw guided ROIs on 2P images and collect traces from dynamic fluorescence movies (`.tif` stacks).

## Download

Get `RawCellInspection.exe` from the [Releases page](https://github.com/RasHerlo/Raw_Cell_Inspection/releases) and run it. No Python needed. It is a single file, so the first start takes a few seconds while it unpacks.

## Using it

1. Open your data from the **File** menu:
   - **Load stack...** (Ctrl+O): pick a `.tif` movie (frames × height × width, single channel). Its experiment file `<stack>_rci.pkl` next to the stack is loaded if it exists (all saved ROIs, traces, heatmaps and settings), otherwise it is created on the first save. The dialog starts at the last stack you opened.
   - **Open experiment...** (Ctrl+Shift+O): pick an experiment file (`*_rci.pkl`). The dialog starts with the last experiment file selected. Its stack is opened from the same folder (or where it was when last saved); if it can't be found you are asked to locate it, and the experiment stays connected to that pickle.

   The file path of what each view shows is printed under its title (hover to see it in full).
2. **Draw view (left):** pick what to draw on in the dropdown:
   - a **reference image** (**Load reference image...**, e.g. an average recorded with another laser; must have the stack's width × height, multi-frame tifs are averaged),
   - the **stack mean / max**,
   - any computed **heatmap**.
3. **Stack view (right):** the movie frame at the frame cursor (or the stack mean / max), with the same ROIs on top.
   Zoom with the mouse wheel, pan by dragging, right-click for view options; **View → Reset zoom** (Ctrl+0). Both views zoom together unless you untick *Link zoom / pan*.
   Each view has its own LUT (grey, turbo, viridis, magma, jet) and lower / upper display bounds; **Auto** resets them to the 1st-99.5th percentile.
4. **Draw specific** (cells, red) or **Draw non-specific** (regions without cells, blue dashed), then drag an outline on the draw view. Esc cancels. Tick *Stay in draw mode* to draw several in a row.
   **Load ROIs from pickle...** copies the ROI outlines (and their names and types) from another experiment's `*_rci.pkl`, e.g. to reuse the same ROIs on a new recording of the same field of view. Their masks and traces are taken from the current stack. If ROIs already exist you choose to add to or replace them; clashing names get the next free number.
5. Click an ROI in either view (or in the list) to select it. The last selected specific ROI is plotted in the upper trace, the last selected non-specific ROI in the lower trace. In the draw view, drag the selected ROI's handles or body to reshape / move it; its trace updates immediately.
6. Click or drag in the trace plots to move the frame cursor (it drives the stack view). Enter the **frame rate** to switch the x axis to seconds.
7. **Tools → Annotations...** (Ctrl+H): create named categories and add frame ranges on the mean trace of the whole field of view (drag the shaded ranges or type start / end). A category with ranges is an annotation on its own. **Compute heatmap** is optional: each pixel is the mean signal inside the ranges divided by the mean signal in all other frames (unresponsive ≈ 1). Turn categories on or off from the **Annotations** menu at the upper left of the trace plots; each checked category is shaded in its own colour. That switch is display only.
8. **Tools → Trace processing...** (Ctrl+T):
   - **Raster:** every ROI as a row, each trace scaled to its own min–max. Sort by document order, by kind, or by one of the four hierarchical trees (Ružička or Euclidean–Ward, pooled or within kind). Click a row to select that ROI. The same Annotations menu shades the checked categories.
   - **Similarity:** one matrix of every ROI, with the hierarchical tree beside it. Method is Ružička + average, or Euclidean + Ward, both on min–max traces. Order is one tree of all ROIs, or specific then non-specific with a tree inside each kind. In the pooled view the contrast index is the dotted line on the tree; drag it to change how many groups are outlined on the matrix, including single ROIs. It is inactive in the within-kind view. A LUT dropdown colours the matrix.
   - **Pearson:** mean pairwise Pearson correlation of the raw traces within specific ROIs and within non-specific ROIs, a permutation test of whether one group is tighter, and the two correlation matrices ordered by clustering.
   - **Z-score:** pick one annotation as the event. Each range is one event and its first frame is the onset. Baseline (prior to the event) and post-stim length have their own frames/seconds switch. Groups lists every ROI, checked by default; untick some and compute again. Each included ROI shows its trials as dotted Z-scores and their mean ± SEM. A second plot overlays every ROI average — specific and non-specific in their own colours — with each group's mean ± SEM.
9. **File → Save** (Ctrl+S) writes `<stack>_rci.pkl` next to the stack; the previous version is kept as `<stack>_rci.pkl.bak`.

## Development

```powershell
# Use a stock CPython (not Anaconda) for the venv
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv venv_rci
.\venv_rci\Scripts\python.exe -m pip install -e ".[build]" pytest
.\venv_rci\Scripts\python.exe -m raw_cell_inspection          # run the app
.\venv_rci\Scripts\python.exe -m pytest -q tests              # tests
.\venv_rci\Scripts\python.exe tools\make_demo_stack.py        # synthetic stack in demo_data\
```

### Building the exe

```powershell
.\packaging\build.ps1          # -> dist\RawCellInspection.exe
```

The script removes Anaconda from `PATH` for the build; otherwise PyInstaller bundles Anaconda's Qt / MSVC DLLs and the exe fails with *DLL load failed while importing QtCore*. Set `RCI_CONSOLE=1` before building to get an exe that prints tracebacks.

### Releasing

Push a version tag and GitHub Actions (`.github/workflows/release.yml`) runs the tests, builds the exe and attaches it to a GitHub Release:

```powershell
git tag v0.3.0
git push origin v0.3.0
```

Design decisions and the pickle schema are recorded in [DESIGN_LOG.md](DESIGN_LOG.md).
