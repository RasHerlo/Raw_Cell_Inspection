"""Headless end-to-end run: open, draw, edit, heatmap, save, reopen."""

import os
import time

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pg = pytest.importorskip("pyqtgraph")
from PySide6 import QtCore, QtWidgets  # noqa: E402

from raw_cell_inspection.gui.main_window import dialog_start, remember_opened  # noqa: E402
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
    assert win.trace_panel._range_items == []
    win.set_annotation_shown("stim", True)
    assert len(win.trace_panel._range_items) == 2

    assert win.save()
    doc = load_document(tmp_path / "rec_rci.pkl")
    assert len(doc["rois"]) == 2 and doc["fps"] == 10.0
    assert doc["heatmaps"][0]["computed_ranges"] == [[20, 29]]
    assert doc["display"]["shown_annotations"] == ["stim"]

    assert win.draw_panel.path_label.toolTip() == str(path)
    assert win.stack_panel.path_label.toolTip() == str(path)
    win.draw_source.setCurrentIndex(win.draw_source.findData("mean"))
    assert len(win.trace_panel._range_items) == 2

    # File > Open experiment, via the pickle
    win2 = MainWindow()
    win2.open_experiment(str(tmp_path / "rec_rci.pkl"))
    wait_until(app, lambda: win2.doc is not None and win2.stack is not None)
    assert len(win2.doc["rois"]) == 2
    assert win2.pkl_path == tmp_path / "rec_rci.pkl"
    assert win2.draw_source.currentData() == "heatmap:stim"
    assert len(win2.trace_panel._range_items) == 2
    win2._on_image_clicked(25, 25)
    assert win2.selected[KIND_NONSPECIFIC] == doc["rois"][1]["id"]

    # Load ROIs from pickle onto another experiment's stack
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = np.full((n, s, s), 50, dtype=np.uint16)
    other[:, 18:22, 18:22] = 70
    tifffile.imwrite(other_dir / "rec2.tif", other)
    win3 = MainWindow()
    win3.open_stack(str(other_dir / "rec2.tif"))
    wait_until(app, lambda: win3.doc is not None and win3.stack is not None)
    win3.draw_specific_btn.setChecked(True)
    win3.draw_panel.sigLassoFinished.emit(square(5, 5, 2))  # takes the name S1
    win3.import_rois(str(tmp_path / "rec_rci.pkl"), mode="add")
    rois3 = win3.doc["rois"]
    assert [r["name"] for r in rois3] == ["S1", "S2", "N1"]
    assert rois3[1]["trace"][0] == pytest.approx(70)  # traces come from the new stack
    np.testing.assert_allclose(rois3[1]["vertices"], doc["rois"][0]["vertices"])
    assert win3.doc["next_roi_number"][KIND_SPECIFIC] == 3
    win3.import_rois(str(tmp_path / "rec_rci.pkl"), mode="replace")
    assert [r["name"] for r in win3.doc["rois"]] == ["S1", "N1"]

    for wdw in (win, win2, win3):
        wdw.dirty = False
        wdw.close()


def test_trace_processing_window(app, tmp_path):
    from raw_cell_inspection.gui.main_window import MainWindow

    n, s = 80, 32
    ramp = np.linspace(100, 180, n)
    stack = np.broadcast_to(ramp[:, None, None], (n, s, s)).copy().astype(np.uint16)
    stack[40:48, 8:12, 8:12] += 80
    path = tmp_path / "rec.tif"
    tifffile.imwrite(path, stack)

    win = MainWindow()
    win.show()
    win.open_stack(str(path))
    wait_until(app, lambda: win.doc is not None and win.doc["summary"] is not None)
    win.keep_drawing.setChecked(True)
    win.draw_specific_btn.setChecked(True)
    win.draw_panel.sigLassoFinished.emit(square(10, 10, 2))
    win.draw_panel.sigLassoFinished.emit(square(14, 18, 2))
    win.draw_nonspecific_btn.setChecked(True)
    win.draw_panel.sigLassoFinished.emit(square(24, 24, 2))
    win.draw_panel.sigLassoFinished.emit(square(20, 26, 2))

    win.open_trace_processing()
    tw = win.trace_window
    assert tw.isVisible()
    assert win.trace_action.isEnabled()
    win.doc["heatmaps"].append({
        "name": "AirPuff",
        "ranges": [[40, 42], [60, 62]],
        "metric": "mean_ratio",
        "image": None,
        "computed_ranges": None,
    })
    win.set_annotation_shown("AirPuff", True)
    tw.refresh()
    assert tw.event_combo.currentData() == "AirPuff"
    assert len(tw._raster_regions) == 2
    assert tw.sort_combo.count() == 6
    tw.sort_combo.setCurrentIndex(4)
    app.processEvents()
    assert len(tw._raster_rows) == 4
    tw.tabs.setCurrentIndex(1)
    app.processEvents()
    assert "Ružička" in tw.cluster_text.text()
    assert tw._tree_curves
    assert tw.cut_spin.singleStep() < tw.cut_spin.maximum() / 20
    tw.cut_spin.setValue(tw.cut_spin.maximum())
    app.processEvents()
    assert len(tw._cluster_view["clusters"]) == 1
    assert len(tw._cluster_outlines) == 1
    tw.cut_spin.setValue(0.0)
    app.processEvents()
    assert len(tw._cluster_view["clusters"]) > 1
    assert len(tw._cluster_outlines) == len(tw._cluster_view["clusters"])
    tw.tabs.setCurrentIndex(2)
    app.processEvents()
    assert "Pearson" in tw.similarity_text.text()
    tw._baseline_frames = 10
    tw._post_frames = 12
    tw._configure_duration_spins()
    tw._compute_zscore()
    assert len(tw._roi_plots) == 4
    assert all(entry["kept"] == 2 for entry in tw._roi_plots)
    assert "AirPuff" in tw.z_status.text()
    nonspecific = tw.roi_tree.topLevelItem(1)
    nonspecific.child(1).setCheckState(0, QtCore.Qt.CheckState.Unchecked)
    tw._compute_zscore()
    assert len(tw._roi_plots) == 3
    saved = win.doc["trace_processing"]["zscore"]
    assert saved["annotation"] == "AirPuff"
    assert saved["baseline_frames"] == 10 and saved["post_frames"] == 12
    assert nonspecific.child(1).data(0, QtCore.Qt.ItemDataRole.UserRole) in saved["excluded_roi_ids"]
    win.dirty = False
    win.close()


def test_open_dialog_starts_at_the_last_experiment(tmp_path):
    experiment = tmp_path / "rec_rci.pkl"
    stack = tmp_path / "rec.tif"
    experiment.write_bytes(b"")
    stack.write_bytes(b"")
    settings = QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.Format.IniFormat)
    remember_opened(settings, experiment=experiment, stack=stack)
    assert dialog_start(settings, "last_experiment", "last_dir") == str(experiment)
    assert dialog_start(settings, "last_stack", "last_dir") == str(stack)
    settings.setValue("last_experiment", str(tmp_path / "missing_rci.pkl"))
    assert dialog_start(settings, "last_experiment", "last_dir") == os.path.join(str(tmp_path), "")
