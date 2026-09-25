# Raw Cell Inspection

Draw guided ROIs on 2P images and collect traces from dynamic fluorescence movies (`.tif` stacks).

## Download

Get `RawCellInspection.exe` from the [Releases page](https://github.com/RasHerlo/Raw_Cell_Inspection/releases) and run it. No Python needed. It is a single file, so the first start takes a few seconds while it unpacks.

## Using it

1. Open your data from the **File** menu:
   - **Load stack...** (Ctrl+O): pick a `.tif` movie (frames × height × width, single channel). Its experiment file `<stack>_rci.pkl` next to the stack is loaded if it exists (all saved ROIs, traces, heatmaps and settings), otherwise it is created on the first save.
   - **Open experiment...** (Ctrl+Shift+O): pick an experiment file (`*_rci.pkl`). Its stack is opened from the same folder (or where it was when last saved); if it can't be found you are asked to locate it, and the experiment stays connected to that pickle.

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
7. **Tools → Heatmaps...** (Ctrl+H): create named categories, add frame ranges on the mean trace of the whole field of view (drag the shaded ranges or type start / end), then **Compute heatmap**. Each pixel is the mean signal inside the ranges divided by the mean signal in all other frames (unresponsive ≈ 1). While a heatmap is shown in the draw view, the ranges it was computed from are shaded yellow on both traces.
8. **File → Save** (Ctrl+S) writes `<stack>_rci.pkl` next to the stack; the previous version is kept as `<stack>_rci.pkl.bak`.

Trace processing will open in its own window in a later version.

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
git tag v0.2.0
git push origin v0.2.0
```

Design decisions and the pickle schema are recorded in [DESIGN_LOG.md](DESIGN_LOG.md).
