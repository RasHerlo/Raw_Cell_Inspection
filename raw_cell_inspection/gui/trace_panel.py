"""Specific / non-specific trace plots sharing a time axis and a movable frame cursor."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from raw_cell_inspection.gui.image_panel import KIND_COLORS
from raw_cell_inspection.store import KIND_NONSPECIFIC, KIND_SPECIFIC

CURSOR_PEN = pg.mkPen((255, 230, 80), width=1.5)


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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.glw = pg.GraphicsLayoutWidget()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
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
