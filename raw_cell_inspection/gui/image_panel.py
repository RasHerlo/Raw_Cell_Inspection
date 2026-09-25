"""Image views with ROI overlays, freehand ROI drawing and display (LUT / level) controls."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from raw_cell_inspection.store import KIND_SPECIFIC

LUT_NAMES = ["grey", "turbo", "viridis", "magma", "jet"]

KIND_COLORS = {
    KIND_SPECIFIC: (255, 90, 90),
    "nonspecific": (70, 170, 255),
}
SELECTED_BRIGHTEN = 60


def lookup_table(name: str) -> np.ndarray:
    if name == "grey":
        ramp = np.arange(256, dtype=np.uint8)
        return np.stack([ramp, ramp, ramp], axis=1)
    if name == "jet":
        cmap = pg.ColorMap(
            pos=[0.0, 0.11, 0.34, 0.65, 0.89, 1.0],
            color=[(0, 0, 131), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0), (128, 0, 0)],
        )
        return cmap.getLookupTable(nPts=256)
    return pg.colormap.get(name).getLookupTable(nPts=256)


def _brighten(rgb: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    return tuple(min(255, c + amount) for c in rgb)


class DrawViewBox(pg.ViewBox):
    """ViewBox that turns left-drags into a freehand lasso while draw mode is on."""

    sigLassoFinished = QtCore.Signal(object)  # (N, 2) array of view coordinates
    sigClicked = QtCore.Signal(float, float)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.draw_color: tuple[int, int, int] | None = None
        self.bounds: tuple[int, int] | None = None  # (width, height)
        self._stroke: list[tuple[float, float]] = []
        self._preview = pg.PlotCurveItem(pen=pg.mkPen((255, 255, 0), width=2))
        self._preview.setZValue(30)
        self.addItem(self._preview, ignoreBounds=True)

    @property
    def drawing(self) -> bool:
        return self.draw_color is not None

    def set_draw_color(self, color: tuple[int, int, int] | None) -> None:
        self.draw_color = color
        self._stroke = []
        self._preview.setData([], [])
        if color is not None:
            self._preview.setPen(pg.mkPen(_brighten(color, SELECTED_BRIGHTEN), width=2))

    def _clip(self, p: QtCore.QPointF) -> tuple[float, float]:
        x, y = p.x(), p.y()
        if self.bounds is not None:
            w, h = self.bounds
            x = min(max(x, 0.0), float(w))
            y = min(max(y, 0.0), float(h))
        return x, y

    def mouseDragEvent(self, ev, axis=None):
        if self.drawing and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            x, y = self._clip(self.mapSceneToView(ev.scenePos()))
            if ev.isStart():
                sx, sy = self._clip(self.mapSceneToView(ev.buttonDownScenePos()))
                self._stroke = [(sx, sy), (x, y)]
            elif self._stroke:
                lx, ly = self._stroke[-1]
                if np.hypot(x - lx, y - ly) >= 0.4:
                    self._stroke.append((x, y))
            if self._stroke:
                pts = np.asarray(self._stroke + self._stroke[:1])
                self._preview.setData(pts[:, 0], pts[:, 1])
            if ev.isFinish():
                stroke = np.asarray(self._stroke)
                self._stroke = []
                self._preview.setData([], [])
                if len(stroke) >= 3:
                    self.sigLassoFinished.emit(stroke)
            return
        super().mouseDragEvent(ev, axis)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            p = self.mapSceneToView(ev.scenePos())
            self.sigClicked.emit(p.x(), p.y())
            return
        super().mouseClickEvent(ev)


def _polygon_path(vertices: np.ndarray) -> QtGui.QPainterPath:
    poly = QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in vertices])
    path = QtGui.QPainterPath()
    path.addPolygon(poly)
    path.closeSubpath()
    return path


class ImagePanel(QtWidgets.QWidget):
    """A titled image view showing ROI outlines. Optionally hosts the editable active ROI."""

    sigClicked = QtCore.Signal(float, float)
    sigLassoFinished = QtCore.Signal(object)
    sigActiveRoiEdited = QtCore.Signal(object)  # new vertices
    sigHover = QtCore.Signal(object)  # (x, y) or None

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = QtWidgets.QLabel(title)
        self.title.setStyleSheet("font-weight: bold; padding: 2px;")
        self.glw = pg.GraphicsLayoutWidget()
        self.vb = DrawViewBox(lockAspect=True, invertY=True, enableMenu=True)
        self.glw.addItem(self.vb)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_item.setZValue(0)
        self.vb.addItem(self.image_item)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.title)
        layout.addWidget(self.glw, 1)

        self._overlay_items: list[QtWidgets.QGraphicsItem] = []
        self._active_roi: pg.PolyLineROI | None = None
        self._image: np.ndarray | None = None
        self.show_labels = True

        self.vb.sigClicked.connect(self.sigClicked)
        self.vb.sigLassoFinished.connect(self.sigLassoFinished)
        self.glw.scene().sigMouseMoved.connect(self._on_mouse_moved)

    # image -----------------------------------------------------------------
    def set_title(self, text: str) -> None:
        self.title.setText(text)

    def set_image(self, image: np.ndarray | None, levels: tuple[float, float] | None = None) -> None:
        first = self._image is None or (image is not None and image.shape != self._image.shape)
        self._image = image
        if image is None:
            self.image_item.clear()
            return
        self.vb.bounds = (image.shape[1], image.shape[0])
        self.image_item.setImage(image, autoLevels=levels is None, levels=levels)
        if first:
            self.vb.autoRange(padding=0.01)

    @property
    def image(self) -> np.ndarray | None:
        return self._image

    def set_lut(self, name: str) -> None:
        self.image_item.setLookupTable(lookup_table(name))

    def set_levels(self, lo: float, hi: float) -> None:
        if hi <= lo:
            hi = lo + 1e-6
        self.image_item.setLevels((lo, hi))

    def reset_zoom(self) -> None:
        self.vb.autoRange(padding=0.01)

    def _on_mouse_moved(self, scene_pos) -> None:
        if self._image is None or not self.vb.sceneBoundingRect().contains(scene_pos):
            self.sigHover.emit(None)
            return
        p = self.vb.mapSceneToView(scene_pos)
        self.sigHover.emit((p.x(), p.y()))

    # ROI overlays ------------------------------------------------------------
    def set_rois(self, rois: list[dict], selected_ids: set[int], hidden_id: int | None = None) -> None:
        for item in self._overlay_items:
            self.vb.removeItem(item)
        self._overlay_items = []
        for roi in rois:
            base = KIND_COLORS[roi["kind"]]
            selected = roi["id"] in selected_ids
            color = _brighten(base, SELECTED_BRIGHTEN) if selected else base
            if roi["id"] != hidden_id:
                item = QtWidgets.QGraphicsPathItem(_polygon_path(roi["vertices"]))
                pen = pg.mkPen(color, width=3 if selected else 1.5)
                if roi["kind"] != KIND_SPECIFIC:
                    pen.setStyle(QtCore.Qt.PenStyle.DashLine)
                item.setPen(pen)
                if selected:
                    item.setBrush(pg.mkBrush(*color, 55))
                item.setZValue(11 if selected else 10)
                self.vb.addItem(item, ignoreBounds=True)
                self._overlay_items.append(item)
            if self.show_labels:
                label = pg.TextItem(roi["name"], color=color, anchor=(0.5, 0.5))
                cx, cy = np.asarray(roi["vertices"]).mean(axis=0)
                label.setPos(float(cx), float(cy))
                label.setZValue(12)
                self.vb.addItem(label, ignoreBounds=True)
                self._overlay_items.append(label)

    # editable active ROI -----------------------------------------------------
    def set_active_roi(self, vertices: np.ndarray | None, kind: str | None = None) -> None:
        if self._active_roi is not None:
            self.vb.removeItem(self._active_roi)
            self._active_roi = None
        if vertices is None:
            return
        color = _brighten(KIND_COLORS[kind], SELECTED_BRIGHTEN)
        roi = pg.PolyLineROI(
            [(float(x), float(y)) for x, y in vertices],
            closed=True,
            pen=pg.mkPen(color, width=2),
            hoverPen=pg.mkPen((255, 255, 255), width=2),
            handlePen=pg.mkPen(color),
            handleHoverPen=pg.mkPen((255, 255, 255)),
            movable=True,
            rotatable=False,
            resizable=False,
            removable=False,
        )
        roi.setZValue(20)
        roi.sigRegionChangeFinished.connect(self._on_active_changed)
        roi.sigClicked.connect(self._on_active_clicked)
        self.vb.addItem(roi, ignoreBounds=True)
        self._active_roi = roi

    def _active_vertices(self) -> np.ndarray:
        roi = self._active_roi
        pts = [roi.mapToParent(p) for _, p in roi.getLocalHandlePositions()]
        return np.array([(p.x(), p.y()) for p in pts], dtype=float)

    def _on_active_changed(self, _roi) -> None:
        if self._active_roi is not None:
            self.sigActiveRoiEdited.emit(self._active_vertices())

    def _on_active_clicked(self, _roi, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            p = self.vb.mapSceneToView(ev.scenePos())
            self.sigClicked.emit(p.x(), p.y())

    # drawing -----------------------------------------------------------------
    def set_draw_kind(self, kind: str | None) -> None:
        self.vb.set_draw_color(None if kind is None else KIND_COLORS[kind])
        cursor = QtCore.Qt.CursorShape.CrossCursor if kind else QtCore.Qt.CursorShape.ArrowCursor
        self.glw.viewport().setCursor(cursor)


class DisplayControls(QtWidgets.QGroupBox):
    """LUT dropdown plus lower / upper display bounds, as in s2p_Trace_Curation."""

    sigChanged = QtCore.Signal()
    STEPS = 1000

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.lut = QtWidgets.QComboBox()
        self.lut.addItems(LUT_NAMES)
        self.lower = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.upper = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        for s in (self.lower, self.upper):
            s.setRange(0, self.STEPS)
        self.lower_label = QtWidgets.QLabel()
        self.upper_label = QtWidgets.QLabel()
        for lab in (self.lower_label, self.upper_label):
            lab.setMinimumWidth(60)
            lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.auto_btn = QtWidgets.QPushButton("Auto")
        self.auto_btn.setToolTip("Set bounds to the 1st-99.5th percentile of the image")

        grid = QtWidgets.QGridLayout(self)
        grid.addWidget(QtWidgets.QLabel("LUT"), 0, 0)
        grid.addWidget(self.lut, 0, 1)
        grid.addWidget(self.auto_btn, 0, 2)
        grid.addWidget(QtWidgets.QLabel("Lower"), 1, 0)
        grid.addWidget(self.lower, 1, 1)
        grid.addWidget(self.lower_label, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Upper"), 2, 0)
        grid.addWidget(self.upper, 2, 1)
        grid.addWidget(self.upper_label, 2, 2)

        self._vmin = 0.0
        self._vmax = 1.0
        self._auto_bounds = (0.0, 1.0)
        self.lut.currentTextChanged.connect(lambda _t: self.sigChanged.emit())
        self.lower.valueChanged.connect(self._on_slider)
        self.upper.valueChanged.connect(self._on_slider)
        self.auto_btn.clicked.connect(self.auto)

    def set_data_range(self, image: np.ndarray, keep_levels: bool = False) -> None:
        finite = image[np.isfinite(image)] if image is not None else np.array([])
        if finite.size == 0:
            finite = np.array([0.0, 1.0])
        if finite.size > 400_000:
            finite = finite[:: finite.size // 400_000]
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        lo, hi = np.percentile(finite, [1.0, 99.5])
        old = self.levels()
        self._vmin, self._vmax = vmin, vmax
        self._auto_bounds = (float(lo), float(hi))
        if keep_levels:
            self.set_levels(*old)
        else:
            self.auto()

    def auto(self) -> None:
        self.set_levels(*self._auto_bounds)

    def _to_value(self, step: int) -> float:
        return self._vmin + (self._vmax - self._vmin) * step / self.STEPS

    def _to_step(self, value: float) -> int:
        frac = (value - self._vmin) / (self._vmax - self._vmin)
        return int(round(min(max(frac, 0.0), 1.0) * self.STEPS))

    def set_levels(self, lo: float, hi: float) -> None:
        for s in (self.lower, self.upper):
            s.blockSignals(True)
        self.lower.setValue(self._to_step(lo))
        self.upper.setValue(self._to_step(hi))
        for s in (self.lower, self.upper):
            s.blockSignals(False)
        self._on_slider()

    def _on_slider(self, _v=None) -> None:
        if self.upper.value() <= self.lower.value():
            if self.sender() is self.lower:
                self.upper.blockSignals(True)
                self.upper.setValue(min(self.STEPS, self.lower.value() + 1))
                self.upper.blockSignals(False)
            else:
                self.lower.blockSignals(True)
                self.lower.setValue(max(0, self.upper.value() - 1))
                self.lower.blockSignals(False)
        lo, hi = self.levels()
        self.lower_label.setText(f"{lo:.4g}")
        self.upper_label.setText(f"{hi:.4g}")
        self.sigChanged.emit()

    def levels(self) -> tuple[float, float]:
        return self._to_value(self.lower.value()), self._to_value(self.upper.value())

    def lut_name(self) -> str:
        return self.lut.currentText()

    def state(self) -> dict:
        return {"lut": self.lut_name()}

    def restore(self, state: dict) -> None:
        name = state.get("lut")
        if name in LUT_NAMES:
            self.lut.setCurrentText(name)
