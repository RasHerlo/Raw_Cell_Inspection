"""Headless end-to-end run: open, draw, edit, heatmap, save, reopen."""

import os
import time

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pg = pytest.importorskip("pyqtgraph")
from PySide6 import QtWidgets  # noqa: E402

from raw_cell_inspection.store import KIND_NONSPECIFIC, KIND_SPECIFIC, load_document  # noqa: E402


@pytest.fixture(scope="module")
def app():
    pg.setConfigOptions(imageAxisOrder="row-major")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_until(app, cond, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError("timed out")


def square(x, y, r):
    return np.array([[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y + r]], dtype=float)


def test_full_cycle(app, tmp_path):
    from raw_cell_inspection.gui.main_window import MainWindow

    n, s = 60, 32
    stack = np.full((n, s, s), 100, dtype=np.uint16)
    stack[20:30, 8:12, 8:12] = 400
    path = tmp_path / "rec.tif"
    tifffile.imwrite(path, stack)

    win = MainWindow()
    win.show()
    win.open_stack(str(path))
    wait_until(app, lambda: win.doc is not None and win.doc["summary"] is not None and win.stack is not None)

    win.draw_specific_btn.setChecked(True)
    win.draw_panel.sigLassoFinished.emit(square(10, 10, 2))
    win.draw_nonspecific_btn.setChecked(True)
    win.draw_panel.sigLassoFinished.emit(square(25, 25, 3))
    rois = win.doc["rois"]
    assert [r["kind"] for r in rois] == [KIND_SPECIFIC, KIND_NONSPECIFIC]
    assert win.selected == {KIND_SPECIFIC: rois[0]["id"], KIND_NONSPECIFIC: rois[1]["id"]}
    assert rois[0]["trace"][25] == pytest.approx(400)

    win._on_image_clicked(10, 10)
    assert win.active_id == rois[0]["id"]
    win.draw_panel.sigActiveRoiEdited.emit(square(20, 20, 2))
    assert rois[0]["trace"][25] == pytest.approx(100)

    win.fps_spin.setValue(10.0)
    win.units_combo.setCurrentText("seconds")

    win.open_heatmaps()
    ed = win.heatmap_editor
    win.doc["heatmaps"].append({"name": "stim", "ranges": [[20, 29]], "metric": "mean_ratio",
                                "image": None, "computed_ranges": None})
    ed.refresh(select_name="stim")
    ed._compute()
    wait_until(app, lambda: win.doc["heatmaps"][0]["image"] is not None)
    assert win.doc["heatmaps"][0]["image"][9, 9] == pytest.approx(4.0)
    assert win.draw_source.currentData() == "heatmap:stim"

    assert win.save()
    doc = load_document(tmp_path / "rec_rci.pkl")
    assert len(doc["rois"]) == 2 and doc["fps"] == 10.0
    assert doc["heatmaps"][0]["computed_ranges"] == [[20, 29]]

    win2 = MainWindow()
    win2.open_stack(str(path))
    wait_until(app, lambda: win2.doc is not None and win2.stack is not None)
    assert len(win2.doc["rois"]) == 2
    assert win2.draw_source.currentData() == "heatmap:stim"
    win2._on_image_clicked(25, 25)
    assert win2.selected[KIND_NONSPECIFIC] == doc["rois"][1]["id"]
    win.dirty = win2.dirty = False
    win.close()
    win2.close()
