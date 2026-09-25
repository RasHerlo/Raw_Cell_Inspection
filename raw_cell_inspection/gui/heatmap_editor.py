"""Named heatmap categories: frame ranges set on the FOV mean trace, one map per category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from raw_cell_inspection.analysis import range_ratio_heatmap
from raw_cell_inspection.gui.tasks import run_task
from raw_cell_inspection.gui.trace_panel import apply_time_axis

if TYPE_CHECKING:
    from raw_cell_inspection.gui.main_window import MainWindow

REGION_BRUSH = (255, 200, 0, 50)
REGION_HOVER = (255, 200, 0, 90)


def new_heatmap(name: str) -> dict:
    return {"name": name, "ranges": [], "metric": "mean_ratio", "image": None, "computed_ranges": None}


def heatmap_state(hm: dict) -> str:
    if hm.get("image") is None:
        return "not computed"
    if [list(r) for r in hm["ranges"]] != [list(r) for r in (hm.get("computed_ranges") or [])]:
        return "out of date"
    return "up to date"


class HeatmapEditor(QtWidgets.QWidget):
    def __init__(self, main: MainWindow):
        super().__init__(main, QtCore.Qt.WindowType.Window)
        self.main = main
        self.setWindowTitle("Heatmaps")
        self.resize(1000, 520)

        self.category_list = QtWidgets.QListWidget()
        self.new_btn = QtWidgets.QPushButton("New")
        self.rename_btn = QtWidgets.QPushButton("Rename")
        self.delete_btn = QtWidgets.QPushButton("Delete")
        cat_buttons = QtWidgets.QHBoxLayout()
        for b in (self.new_btn, self.rename_btn, self.delete_btn):
            cat_buttons.addWidget(b)
        cat_box = QtWidgets.QGroupBox("Categories")
        cat_layout = QtWidgets.QVBoxLayout(cat_box)
        cat_layout.addWidget(self.category_list)
        cat_layout.addLayout(cat_buttons)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Start (frame)", "End (frame)", "Seconds"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.add_range_btn = QtWidgets.QPushButton("Add range")
        self.add_range_btn.setToolTip("Adds a range at the current frame cursor; drag its edges on the trace")
        self.remove_range_btn = QtWidgets.QPushButton("Remove range")
        range_buttons = QtWidgets.QHBoxLayout()
        range_buttons.addWidget(self.add_range_btn)
        range_buttons.addWidget(self.remove_range_btn)
        range_box = QtWidgets.QGroupBox("Ranges (inclusive)")
        range_layout = QtWidgets.QVBoxLayout(range_box)
        range_layout.addWidget(self.table)
        range_layout.addLayout(range_buttons)

        self.status = QtWidgets.QLabel()
        self.compute_btn = QtWidgets.QPushButton("Compute heatmap")
        self.compute_btn.setToolTip("Per pixel: mean inside the ranges / mean in all other frames")

        left = QtWidgets.QVBoxLayout()
        left.addWidget(cat_box, 1)
        left.addWidget(range_box, 2)
        left.addWidget(self.status)
        left.addWidget(self.compute_btn)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(360)

        self.plot_widget = pg.PlotWidget()
        self.plot = self.plot_widget.getPlotItem()
        self.plot.setTitle("Mean trace of the whole field of view")
        self.plot.setLabel("left", "Mean F")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.curve = self.plot.plot(pen=pg.mkPen((200, 200, 200), width=1))

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(left_widget)
        layout.addWidget(self.plot_widget, 1)

        self._regions: list[pg.LinearRegionItem] = []
        self._syncing = False

        self.category_list.currentRowChanged.connect(lambda _r: self._show_category())
        self.category_list.itemDoubleClicked.connect(lambda _i: self._rename())
        self.new_btn.clicked.connect(self._new)
        self.rename_btn.clicked.connect(self._rename)
        self.delete_btn.clicked.connect(self._delete)
        self.add_range_btn.clicked.connect(self._add_range)
        self.remove_range_btn.clicked.connect(self._remove_range)
        self.compute_btn.clicked.connect(self._compute)

    # data access -------------------------------------------------------------
    @property
    def heatmaps(self) -> list[dict]:
        return self.main.doc["heatmaps"] if self.main.doc else []

    def current(self) -> dict | None:
        row = self.category_list.currentRow()
        return self.heatmaps[row] if 0 <= row < len(self.heatmaps) else None

    @property
    def n_frames(self) -> int:
        return int(self.main.stack.shape[0]) if self.main.stack is not None else 1

    # refresh -----------------------------------------------------------------
    def refresh(self, select_name: str | None = None) -> None:
        """Rebuild from the main window's document (new stack, fps change, ...)."""
        doc = self.main.doc
        summary = doc.get("summary") if doc else None
        trace = summary["fov_mean_trace"] if summary else None
        if trace is not None:
            self.curve.setData(np.arange(len(trace)), trace)
        else:
            self.curve.setData([], [])
        self.apply_time_axis()
        current = self.current()
        select_name = select_name or (current["name"] if current else None)
        self.category_list.blockSignals(True)
        self.category_list.clear()
        row = 0
        for i, hm in enumerate(self.heatmaps):
            self.category_list.addItem(hm["name"])
            if hm["name"] == select_name:
                row = i
        self.category_list.blockSignals(False)
        if self.heatmaps:
            self.category_list.setCurrentRow(row)
        self._show_category()

    def apply_time_axis(self) -> None:
        doc = self.main.doc or {}
        apply_time_axis(self.plot, doc.get("fps"), doc.get("x_units", "frames"))
        self._refresh_table()

    def _show_category(self) -> None:
        hm = self.current()
        enabled = hm is not None and self.main.stack is not None
        for w in (self.rename_btn, self.delete_btn, self.add_range_btn, self.remove_range_btn, self.compute_btn):
            w.setEnabled(enabled)
        self._rebuild_regions()
        self._refresh_table()
        self._refresh_status()

    def _rebuild_regions(self) -> None:
        for region in self._regions:
            self.plot.removeItem(region)
        self._regions = []
        hm = self.current()
        if hm is None:
            return
        for i, (start, end) in enumerate(hm["ranges"]):
            region = pg.LinearRegionItem(
                values=(start, end),
                brush=pg.mkBrush(*REGION_BRUSH),
                hoverBrush=pg.mkBrush(*REGION_HOVER),
                bounds=(0, self.n_frames - 1),
            )
            region.sigRegionChangeFinished.connect(lambda r, i=i: self._on_region_changed(i, r))
            self.plot.addItem(region)
            self._regions.append(region)

    def _refresh_table(self) -> None:
        hm = self.current()
        self._syncing = True
        self.table.setRowCount(0)
        if hm is not None:
            fps = (self.main.doc or {}).get("fps")
            for i, (start, end) in enumerate(hm["ranges"]):
                self.table.insertRow(i)
                for col, value in enumerate((start, end)):
                    spin = QtWidgets.QSpinBox()
                    spin.setRange(0, self.n_frames - 1)
                    spin.setValue(int(value))
                    spin.valueChanged.connect(lambda _v, i=i: self._on_spin_changed(i))
                    self.table.setCellWidget(i, col, spin)
                secs = f"{start / fps:.2f} - {end / fps:.2f}" if fps else "set fps"
                item = QtWidgets.QTableWidgetItem(secs)
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(i, 2, item)
        self._syncing = False

    def _refresh_status(self) -> None:
        hm = self.current()
        if hm is None:
            self.status.setText("Create a category, add ranges, then compute.")
            return
        n_sel = len(hm["ranges"])
        self.status.setText(f"<b>{hm['name']}</b>: {n_sel} range(s), heatmap {heatmap_state(hm)}")

    # edits -------------------------------------------------------------------
    def _changed(self) -> None:
        self._refresh_status()
        self.main.on_heatmaps_edited()

    def _on_region_changed(self, index: int, region: pg.LinearRegionItem) -> None:
        hm = self.current()
        if hm is None or self._syncing:
            return
        lo, hi = region.getRegion()
        lo = int(round(max(0, lo)))
        hi = int(round(min(self.n_frames - 1, hi)))
        hm["ranges"][index] = [lo, max(lo, hi)]
        self._syncing = True
        region.setRegion((lo, max(lo, hi)))
        self._syncing = False
        self._refresh_table()
        self._changed()

    def _on_spin_changed(self, index: int) -> None:
        hm = self.current()
        if hm is None or self._syncing:
            return
        start = self.table.cellWidget(index, 0).value()
        end = self.table.cellWidget(index, 1).value()
        hm["ranges"][index] = [min(start, end), max(start, end)]
        self._syncing = True
        self._regions[index].setRegion(hm["ranges"][index])
        self._syncing = False
        self._changed()

    def _unique_name(self, title: str, default: str, exclude: dict | None = None) -> str | None:
        name, ok = QtWidgets.QInputDialog.getText(self, title, "Category name:", text=default)
        name = name.strip()
        if not ok or not name:
            return None
        if any(h["name"] == name for h in self.heatmaps if h is not exclude):
            QtWidgets.QMessageBox.warning(self, title, f"A category named '{name}' already exists.")
            return None
        return name

    def _new(self) -> None:
        if self.main.doc is None:
            return
        name = self._unique_name("New category", f"Category {len(self.heatmaps) + 1}")
        if name is None:
            return
        self.heatmaps.append(new_heatmap(name))
        self.refresh(select_name=name)
        self._changed()

    def _rename(self) -> None:
        hm = self.current()
        if hm is None:
            return
        name = self._unique_name("Rename category", hm["name"], exclude=hm)
        if name is None:
            return
        hm["name"] = name
        self.refresh(select_name=name)
        self._changed()

    def _delete(self) -> None:
        hm = self.current()
        if hm is None:
            return
        answer = QtWidgets.QMessageBox.question(self, "Delete category", f"Delete '{hm['name']}' and its heatmap?")
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.heatmaps.remove(hm)
        self.refresh()
        self._changed()

    def _add_range(self) -> None:
        hm = self.current()
        if hm is None:
            return
        start = int(self.main.frame)
        width = max(1, self.n_frames // 20)
        end = min(self.n_frames - 1, start + width)
        if end == start:
            start = max(0, end - width)
        hm["ranges"].append([start, end])
        hm["ranges"].sort()
        self._show_category()
        self._changed()

    def _remove_range(self) -> None:
        hm = self.current()
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()}, reverse=True)
        if hm is None:
            return
        if not rows and hm["ranges"]:
            rows = [len(hm["ranges"]) - 1]
        for r in rows:
            del hm["ranges"][r]
        self._show_category()
        self._changed()

    def _compute(self) -> None:
        hm = self.current()
        stack = self.main.stack
        if hm is None or stack is None:
            return
        if not hm["ranges"]:
            QtWidgets.QMessageBox.information(self, "Heatmap", "Add at least one range first.")
            return
        ranges = [tuple(r) for r in hm["ranges"]]

        def done(image):
            hm["image"] = image
            hm["computed_ranges"] = [list(r) for r in ranges]
            self._refresh_status()
            self.main.on_heatmap_computed(hm["name"])

        run_task(
            self,
            f"Computing heatmap '{hm['name']}'...",
            lambda progress: range_ratio_heatmap(stack, ranges, progress),
            done,
        )
