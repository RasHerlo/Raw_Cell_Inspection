"""Specific / non-specific trace plots sharing a time axis and a movable frame cursor."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from raw_cell_inspection.gui.image_panel import KIND_COLORS
from raw_cell_inspection.store import KIND_NONSPECIFIC, KIND_SPECIFIC

CURSOR_PEN = pg.mkPen((255, 230, 80), width=1.5)
REGION_BRUSH = (255, 200, 0, 50)
REGION_HOVER = (255, 200, 0, 90)
REGION_PEN = (255, 200, 0)

ANNOTATION_COLORS = (
    (255, 176, 0),
    (72, 199, 116),
    (176, 112, 255),
    (255, 112, 72),
    (64, 176, 224),
    (224, 80, 144),
    (160, 200, 64),
    (96, 128, 255),
)


def annotation_color(index: int) -> tuple[int, int, int]:
    return ANNOTATION_COLORS[index % len(ANNOTATION_COLORS)]


def annotation_spans(doc: dict | None) -> list[tuple[int, int, tuple[int, int, int]]]:
    """Inclusive frame ranges of the categories currently switched on."""
    if not doc:
        return []
    shown = set((doc.get("display") or {}).get("shown_annotations") or [])
    spans: list[tuple[int, int, tuple[int, int, int]]] = []
    for index, category in enumerate(doc.get("heatmaps") or []):
        if category.get("name") not in shown:
            continue
        color = annotation_color(index)
        for item in category.get("ranges") or []:
            try:
                start, end = int(item[0]), int(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if end < start:
                start, end = end, start
            spans.append((start, end, color))
    return spans


def fill_annotation_menu(menu: QtWidgets.QMenu, choices: list[tuple[str, bool]], on_toggle) -> None:
    """Checkable category names. ``on_toggle(name, checked)`` fires only for user clicks."""
    menu.clear()
    if not choices:
        empty = menu.addAction("No categories yet")
        empty.setEnabled(False)
        return
    for name, checked in choices:
        action = menu.addAction(name)
        action.setCheckable(True)
        action.blockSignals(True)
        action.setChecked(checked)
        action.blockSignals(False)
        action.toggled.connect(lambda on, name=name: on_toggle(name, on))


def apply_time_axis(plot: pg.PlotItem, fps: float | None, units: str) -> None:
    """Plot coordinates stay in frames; only the tick labels are rescaled."""
    axis = plot.getAxis("bottom")
    if units == "seconds" and fps:
        axis.setScale(1.0 / fps)
        axis.setLabel("Time (s)")
    else:
        axis.setScale(1.0)
        axis.setLabel("Frame")


class TracePanel(QtWidgets.QWidget):
    sigFrameChanged = QtCore.Signal(int)
    sigAnnotationToggled = QtCore.Signal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.annotation_button = QtWidgets.QToolButton()
        self.annotation_button.setText("Annotations")
        self.annotation_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.annotation_menu = QtWidgets.QMenu(self.annotation_button)
        self.annotation_button.setMenu(self.annotation_menu)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 0)
        bar.addWidget(self.annotation_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        bar.addStretch(1)

        self.glw = pg.GraphicsLayoutWidget()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.glw)

        self.plots: dict[str, pg.PlotItem] = {}
        self.curves: dict[str, pg.PlotDataItem] = {}
        self.cursors: dict[str, pg.InfiniteLine] = {}
        for row, kind in enumerate((KIND_SPECIFIC, KIND_NONSPECIFIC)):
            plot = self.glw.addPlot(row=row, col=0)
            plot.showGrid(x=True, y=True, alpha=0.2)
            plot.setLabel("left", "Mean F")
            plot.setMenuEnabled(True)
            curve = plot.plot(pen=pg.mkPen(KIND_COLORS[kind], width=1))
            cursor = pg.InfiniteLine(angle=90, movable=True, pen=CURSOR_PEN)
            cursor.sigPositionChanged.connect(self._on_cursor_moved)
            plot.addItem(cursor, ignoreBounds=True)
            self.plots[kind] = plot
            self.curves[kind] = curve
            self.cursors[kind] = cursor
        self.plots[KIND_NONSPECIFIC].setXLink(self.plots[KIND_SPECIFIC])
        self.glw.scene().sigMouseClicked.connect(self._on_scene_clicked)
        self._range_items: list[tuple[pg.PlotItem, pg.LinearRegionItem]] = []
        self._frame = 0
        self._n_frames = 1
        self._syncing = False
        self.set_trace(KIND_SPECIFIC, None, None)
        self.set_trace(KIND_NONSPECIFIC, None, None)

    def set_n_frames(self, n: int) -> None:
        self._n_frames = max(1, int(n))
        for cursor in self.cursors.values():
            cursor.setBounds((0, self._n_frames - 1))
        self.plots[KIND_SPECIFIC].setXRange(0, self._n_frames - 1, padding=0.01)

    def set_trace(self, kind: str, name: str | None, trace: np.ndarray | None) -> None:
        label = "Specific" if kind == KIND_SPECIFIC else "Non-specific"
        plot = self.plots[kind]
        if trace is None:
            self.curves[kind].setData([], [])
            plot.setTitle(f"{label}: none selected")
            return
        self.curves[kind].setData(np.arange(len(trace)), trace)
        plot.setTitle(f"{label}: {name}")
        plot.enableAutoRange(axis="y")

    def set_annotation_choices(self, choices: list[tuple[str, bool]]) -> None:
        fill_annotation_menu(self.annotation_menu, choices, self.sigAnnotationToggled.emit)

    def set_annotation_spans(self, spans: list[tuple[int, int, tuple[int, int, int]]] | None) -> None:
        """Draw switched-on annotation ranges on both trace plots, one colour per category."""
        for plot, item in self._range_items:
            plot.removeItem(item)
        self._range_items = []
        for start, end, color in spans or []:
            for plot in self.plots.values():
                item = pg.LinearRegionItem(
                    values=(start, end),
                    movable=False,
                    brush=pg.mkBrush(*color, 50),
                    pen=pg.mkPen(color),
                )
                item.setZValue(-10)
                plot.addItem(item, ignoreBounds=True)
                self._range_items.append((plot, item))

    def set_ranges(self, ranges: list | None) -> None:
        """Show read-only frame ranges in the default annotation colour."""
        self.set_annotation_spans([(int(start), int(end), REGION_PEN) for start, end in ranges or []])

    def set_time_axis(self, fps: float | None, units: str) -> None:
        for plot in self.plots.values():
            apply_time_axis(plot, fps, units)

    def set_frame(self, frame: int) -> None:
        self._frame = int(frame)
        self._syncing = True
        for cursor in self.cursors.values():
            cursor.setValue(self._frame)
        self._syncing = False

    def _on_cursor_moved(self, line: pg.InfiniteLine) -> None:
        if self._syncing:
            return
        frame = int(round(line.value()))
        frame = min(max(frame, 0), self._n_frames - 1)
        if frame != self._frame:
            self._frame = frame
            self.set_frame(frame)
            self.sigFrameChanged.emit(frame)

    def _on_scene_clicked(self, ev) -> None:
        if ev.button() != QtCore.Qt.MouseButton.LeftButton or ev.double():
            return
        for plot in self.plots.values():
            vb = plot.getViewBox()
            if vb.sceneBoundingRect().contains(ev.scenePos()):
                x = vb.mapSceneToView(ev.scenePos()).x()
                frame = min(max(int(round(x)), 0), self._n_frames - 1)
                self.set_frame(frame)
                self.sigFrameChanged.emit(frame)
                return
