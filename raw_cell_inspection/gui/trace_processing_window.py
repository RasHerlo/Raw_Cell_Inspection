"""Trace processing: raster, within-group similarity, and annotation Z-scores."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from raw_cell_inspection.gui.image_panel import KIND_COLORS, LUT_NAMES, lookup_table
from raw_cell_inspection.gui.trace_panel import annotation_color, annotation_spans, apply_time_axis, fill_annotation_menu
from raw_cell_inspection.store import KIND_NONSPECIFIC, KIND_SPECIFIC
from raw_cell_inspection.trace_processing import (
    METRIC_EUCLIDEAN,
    METRIC_RUZICKA,
    ORDER_POOLED,
    ORDER_SEPARATE,
    SORT_DOCUMENT,
    SORT_KIND,
    compare_trace_groups,
    epoch_x,
    dendrogram_branches,
    event_onsets,
    fit_trace,
    hac_sort_key,
    hierarchical_view,
    recut_view,
    mean_and_sem,
    minmax_trace,
    ordered_rois,
    similarity_conclusion,
    trial_zscores,
)

if TYPE_CHECKING:
    from raw_cell_inspection.gui.main_window import MainWindow

GREY_PEN = pg.mkPen((120, 120, 120), width=1)
GREY_FILL = pg.mkBrush(150, 150, 150, 55)
ONSET_PEN = pg.mkPen((80, 80, 80), width=1, style=QtCore.Qt.PenStyle.DashLine)
KIND_HTML = {KIND_SPECIFIC: "#ff5a5a", KIND_NONSPECIFIC: "#46aaff"}
SORT_LABELS = (
    (SORT_DOCUMENT, "Document order"),
    (SORT_KIND, "Specific, then non-specific"),
    (hac_sort_key(METRIC_RUZICKA, ORDER_POOLED), "Ružička, all ROIs"),
    (hac_sort_key(METRIC_EUCLIDEAN, ORDER_POOLED), "Euclidean–Ward, all ROIs"),
    (hac_sort_key(METRIC_RUZICKA, ORDER_SEPARATE), "Ružička, within kind"),
    (hac_sort_key(METRIC_EUCLIDEAN, ORDER_SEPARATE), "Euclidean–Ward, within kind"),
)
CLUSTER_COLORS = (
    (0, 229, 255),
    (124, 255, 107),
    (255, 213, 74),
    (255, 107, 107),
    (199, 125, 255),
    (79, 195, 247),
    (255, 158, 128),
    (244, 143, 177),
)


def correlation_lut() -> np.ndarray:
    """Blue (−1) – white (0) – red (+1), 256 rows."""
    lut = np.empty((256, 3), dtype=np.uint8)
    for i, t in enumerate(np.linspace(0.0, 1.0, 256)):
        if t < 0.5:
            u = t / 0.5
            lut[i] = (40 + u * 215, 80 + u * 175, 180 + u * 75)
        else:
            u = (t - 0.5) / 0.5
            lut[i] = (255 - u * 55, 255 - u * 205, 255 - u * 205)
    return lut


def _cut_decimals(span: float) -> int:
    """Enough decimals that a step along the tree is not one giant jump."""
    if span <= 0 or not np.isfinite(span):
        return 4
    return int(min(max(4, 3 - int(np.floor(np.log10(span)))), 9))


def apply_epoch_axis(plot: pg.PlotItem, fps: float | None, units: str) -> None:
    """Epoch coordinates stay in frames relative to onset; labels follow the document units."""
    apply_time_axis(plot, fps, units)
    if units == "seconds" and fps:
        plot.setLabel("bottom", "Time from onset (s)")
    else:
        plot.setLabel("bottom", "Frames from onset")


def format_p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def format_r(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{value:.3f}"


class TraceProcessingWindow(QtWidgets.QWidget):
    def __init__(self, main: MainWindow):
        super().__init__(main, QtCore.Qt.WindowType.Window)
        self.main = main
        self.setWindowTitle("Trace processing")
        self.resize(1180, 780)
        self._syncing = False
        self._loading = False
        self._baseline_frames = 30
        self._post_frames = 60
        self._bound_path: str | None = None
        self._raster_signature: tuple | None = None
        self._regions: list[pg.LinearRegionItem] = []
        self._raster_regions: list[pg.LinearRegionItem] = []
        self._cluster_outlines: list[pg.PlotDataItem] = []
        self._tree_curves: list[pg.PlotDataItem] = []
        self._cut_updating = False
        self._z_units = "frames"
        self._raster_rows: list[dict] = []
        self._roi_plots: list[dict] = []
        self._z_stale = False
        self._corr_lut = correlation_lut()

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_raster_tab(), "Raster")
        self.tabs.addTab(self._build_clustering_tab(), "Similarity")
        self.tabs.addTab(self._build_similarity_tab(), "Pearson")
        self.tabs.addTab(self._build_zscore_tab(), "Z-score")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tabs)
        self.tabs.currentChanged.connect(self._on_tab)

    # construction ------------------------------------------------------------
    def _build_raster_tab(self) -> QtWidgets.QWidget:
        self.sort_combo = QtWidgets.QComboBox()
        for key, label in SORT_LABELS:
            self.sort_combo.addItem(label, key)
        self.lut_combo = QtWidgets.QComboBox()
        self.lut_combo.addItems(LUT_NAMES)
        self.lut_combo.setCurrentText("viridis")
        self.raster_ann_button = QtWidgets.QToolButton()
        self.raster_ann_button.setText("Annotations")
        self.raster_ann_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.raster_ann_menu = QtWidgets.QMenu(self.raster_ann_button)
        self.raster_ann_button.setMenu(self.raster_ann_menu)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.raster_ann_button)
        bar.addWidget(QtWidgets.QLabel("Sort"))
        bar.addWidget(self.sort_combo)
        bar.addWidget(QtWidgets.QLabel("LUT"))
        bar.addWidget(self.lut_combo)
        bar.addStretch(1)
        note = QtWidgets.QLabel(
            "Each row is one ROI, scaled to its own minimum and maximum. "
            "The squares on the left mark specific (red) and non-specific (blue). "
            "Click a row to select that ROI."
        )
        note.setWordWrap(True)

        self.kind_plot = pg.PlotWidget()
        self.kind_plot.setFixedWidth(28)
        self.kind_plot.hideAxis("left")
        self.kind_plot.hideAxis("bottom")
        self.kind_plot.setMenuEnabled(False)
        self.kind_plot.setMouseEnabled(x=False, y=False)
        self.kind_scatter = pg.ScatterPlotItem(pxMode=True, symbol="s", size=11)
        self.kind_plot.addItem(self.kind_scatter)

        self.raster_widget = pg.PlotWidget()
        self.raster_plot = self.raster_widget.getPlotItem()
        self.raster_plot.setLabel("left", "ROI")
        self.raster_plot.showGrid(x=True, y=False, alpha=0.15)
        self.raster_plot.invertY(True)
        self.kind_plot.getViewBox().invertY(True)
        self.kind_plot.setYLink(self.raster_plot)
        self.raster_image = pg.ImageItem()
        self.raster_image.setZValue(0)
        self.raster_plot.addItem(self.raster_image)
        self.raster_widget.scene().sigMouseClicked.connect(self._on_raster_clicked)

        plots = QtWidgets.QHBoxLayout()
        plots.setSpacing(0)
        plots.addWidget(self.kind_plot)
        plots.addWidget(self.raster_widget, 1)
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addLayout(bar)
        layout.addWidget(note)
        layout.addLayout(plots, 1)
        self.sort_combo.currentIndexChanged.connect(self._on_raster_option)
        self.lut_combo.currentIndexChanged.connect(self._on_raster_option)
        return page

    def _build_clustering_tab(self) -> QtWidgets.QWidget:
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.addItem("Ružička + average", METRIC_RUZICKA)
        self.metric_combo.addItem("Euclidean + Ward", METRIC_EUCLIDEAN)
        self.order_combo = QtWidgets.QComboBox()
        self.order_combo.addItem("All ROIs, one tree", ORDER_POOLED)
        self.order_combo.addItem("Within kind", ORDER_SEPARATE)
        self.cut_spin = QtWidgets.QDoubleSpinBox()
        self.cut_spin.setDecimals(4)
        self.cut_spin.setRange(0.0, 1.0)
        self.cut_spin.setSingleStep(0.01)
        self.cut_spin.setKeyboardTracking(False)
        self.cut_spin.setToolTip(
            "Distance cut on the pooled tree, the dotted line on the dendrogram. "
            "Lower values make more, smaller groups. Inactive when each kind has its own tree."
        )
        self.matrix_lut = QtWidgets.QComboBox()
        self.matrix_lut.addItems(LUT_NAMES)
        self.matrix_lut.setCurrentText("viridis")
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Method"))
        bar.addWidget(self.metric_combo)
        bar.addWidget(QtWidgets.QLabel("Order"))
        bar.addWidget(self.order_combo)
        bar.addWidget(QtWidgets.QLabel("Contrast index"))
        bar.addWidget(self.cut_spin)
        bar.addWidget(QtWidgets.QLabel("LUT"))
        bar.addWidget(self.matrix_lut)
        bar.addStretch(1)
        self.cluster_text = QtWidgets.QLabel()
        self.cluster_text.setWordWrap(True)
        self.tree_plot = pg.PlotWidget(title="Dendrogram")
        self.tree_plot.setLabel("bottom", "distance")
        self.tree_plot.hideAxis("left")
        self.tree_plot.showGrid(x=True, y=False, alpha=0.2)
        self.tree_plot.invertY(True)
        self.tree_plot.setMenuEnabled(False)
        self.tree_plot.setMinimumWidth(260)
        self.cut_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=True,
            pen=pg.mkPen("#f0e68c", width=2, style=QtCore.Qt.PenStyle.DotLine),
            hoverPen=pg.mkPen("#fff8b0", width=2.5, style=QtCore.Qt.PenStyle.DotLine),
        )
        self.cut_line.setZValue(20)
        self.cut_line.setVisible(False)
        self.tree_plot.addItem(self.cut_line)
        self.cluster_plot = pg.PlotWidget()
        self.cluster_image = pg.ImageItem()
        self.cluster_image.setZValue(0)
        self.cluster_plot.addItem(self.cluster_image)
        self.cluster_plot.setAspectLocked(True)
        self.cluster_plot.invertY(True)
        self.cluster_plot.setMenuEnabled(False)
        self.tree_plot.setYLink(self.cluster_plot)
        plots = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        plots.addWidget(self.tree_plot)
        plots.addWidget(self.cluster_plot)
        plots.setStretchFactor(0, 1)
        plots.setStretchFactor(1, 1)
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addLayout(bar)
        layout.addWidget(self.cluster_text)
        layout.addWidget(plots, 1)
        self.metric_combo.currentIndexChanged.connect(lambda _i: self._clustering_changed(reset_cut=True))
        self.order_combo.currentIndexChanged.connect(lambda _i: self._clustering_changed(reset_cut=False))
        self.cut_spin.valueChanged.connect(self._on_cut_spin)
        self.cut_line.sigPositionChanged.connect(self._on_cut_line)
        self.matrix_lut.currentIndexChanged.connect(lambda _i: self._draw_clustering(recompute=False))
        return page

    def _build_similarity_tab(self) -> QtWidgets.QWidget:
        self.similarity_text = QtWidgets.QLabel()
        self.similarity_text.setWordWrap(True)
        self.similarity_text.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.spec_matrix_plot = pg.PlotWidget()
        self.nonspec_matrix_plot = pg.PlotWidget()
        self.spec_matrix = pg.ImageItem()
        self.nonspec_matrix = pg.ImageItem()
        for plot, image, title in (
            (self.spec_matrix_plot, self.spec_matrix, "Specific"),
            (self.nonspec_matrix_plot, self.nonspec_matrix, "Non-specific"),
        ):
            plot.addItem(image)
            plot.setAspectLocked(True)
            plot.invertY(True)
            plot.setTitle(title)
            plot.setMenuEnabled(False)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.spec_matrix_plot, 1)
        row.addWidget(self.nonspec_matrix_plot, 1)
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addWidget(self.similarity_text)
        layout.addLayout(row, 1)
        return page

    def _build_zscore_tab(self) -> QtWidgets.QWidget:
        self.event_combo = QtWidgets.QComboBox()
        event_box = QtWidgets.QGroupBox("Event annotation")
        event_layout = QtWidgets.QVBoxLayout(event_box)
        event_layout.addWidget(self.event_combo)
        event_hint = QtWidgets.QLabel(
            "Categories come from Tools → Annotations. Each range is one event, "
            "and the onset is its first frame."
        )
        event_hint.setWordWrap(True)
        event_layout.addWidget(event_hint)

        self.roi_tree = QtWidgets.QTreeWidget()
        self.roi_tree.setHeaderHidden(True)
        self.roi_tree.setMinimumHeight(160)
        group_box = QtWidgets.QGroupBox("Groups")
        group_layout = QtWidgets.QVBoxLayout(group_box)
        group_layout.addWidget(self.roi_tree)
        group_hint = QtWidgets.QLabel("Uncheck a ROI to leave it out, then compute again.")
        group_hint.setWordWrap(True)
        group_layout.addWidget(group_hint)

        self.baseline_spin = QtWidgets.QDoubleSpinBox()
        self.post_spin = QtWidgets.QDoubleSpinBox()
        self.z_unit_combo = QtWidgets.QComboBox()
        self.z_unit_combo.addItems(["frames", "seconds"])
        form = QtWidgets.QFormLayout()
        form.addRow("Units", self.z_unit_combo)
        form.addRow("Baseline (prior to event)", self.baseline_spin)
        form.addRow("Post-stim", self.post_spin)
        duration = QtWidgets.QGroupBox("Window")
        duration.setLayout(form)

        self.compute_btn = QtWidgets.QPushButton("Compute Z-scores")
        self.z_status = QtWidgets.QLabel("Choose an annotation, then compute.")
        self.z_status.setWordWrap(True)

        self.ann_plot_widget = pg.PlotWidget()
        self.ann_plot = self.ann_plot_widget.getPlotItem()
        self.ann_plot.setTitle("Mean of the whole field of view")
        self.ann_plot.setLabel("left", "Mean F")
        self.ann_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ann_curve = self.ann_plot.plot(pen=pg.mkPen((200, 200, 200), width=1))

        left = QtWidgets.QVBoxLayout()
        left.addWidget(event_box)
        left.addWidget(group_box, 1)
        left.addWidget(duration)
        left.addWidget(self.compute_btn)
        left.addWidget(self.z_status)
        left.addWidget(self.ann_plot_widget, 1)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(380)

        intro = QtWidgets.QLabel(
            "Each plot is one ROI. Dotted lines are trials, Z-scored to their own baseline. "
            "The thick line is the trial mean; the grey lines and shading are ± SEM across trials."
        )
        intro.setWordWrap(True)
        self.roi_host = QtWidgets.QWidget()
        self.roi_layout = QtWidgets.QVBoxLayout(self.roi_host)
        self.roi_layout.setContentsMargins(0, 0, 0, 0)
        self.roi_layout.addStretch(1)
        self.roi_scroll = QtWidgets.QScrollArea()
        self.roi_scroll.setWidgetResizable(True)
        self.roi_scroll.setWidget(self.roi_host)
        trials = QtWidgets.QWidget()
        trials_layout = QtWidgets.QVBoxLayout(trials)
        trials_layout.setContentsMargins(0, 0, 0, 0)
        trials_layout.addWidget(intro)
        trials_layout.addWidget(self.roi_scroll, 1)

        group_note = QtWidgets.QLabel(
            "Dotted lines are each ROI's trial average, red for specific and blue for non-specific. "
            "Thick lines are the mean across those ROIs; grey is ± SEM across ROIs."
        )
        group_note.setWordWrap(True)
        self.group_widget = pg.PlotWidget()
        self.group_plot = self.group_widget.getPlotItem()
        self.group_plot.setLabel("left", "Z")
        self.group_plot.showGrid(x=True, y=True, alpha=0.2)
        self.group_plot.addLegend()
        self.group_widget.setMinimumHeight(240)
        group = QtWidgets.QWidget()
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.addWidget(group_note)
        group_layout.addWidget(self.group_widget, 1)

        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right.addWidget(trials)
        right.addWidget(group)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)

        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(page)
        layout.addWidget(left_widget)
        layout.addWidget(right, 1)

        self.event_combo.currentIndexChanged.connect(lambda _i: self._on_event_changed())
        self.roi_tree.itemChanged.connect(self._on_roi_item_changed)
        self.compute_btn.clicked.connect(self._compute_zscore)
        self.baseline_spin.valueChanged.connect(self._on_baseline_edited)
        self.post_spin.valueChanged.connect(self._on_post_edited)
        self.z_unit_combo.currentTextChanged.connect(self._on_z_units)
        return page

    # document ----------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        if self.main.stack is None:
            return 1
        return int(self.main.stack.shape[0])

    def _time(self) -> tuple[float | None, str]:
        doc = self.main.doc or {}
        fps = doc.get("fps")
        return (float(fps) if fps else None), doc.get("x_units", "frames")

    def _settings(self) -> dict:
        doc = self.main.doc
        if doc is None:
            return {}
        found = doc.get("trace_processing")
        if not isinstance(found, dict):
            found = {}
            doc["trace_processing"] = found
        return found

    def _store(self, key: str, value) -> None:
        settings = self._settings()
        if settings.get(key) == value:
            return
        settings[key] = value
        if self.main.doc is not None:
            self.main.mark_dirty()

    # refresh -----------------------------------------------------------------
    def refresh(self) -> None:
        """Reload from the open experiment (new stack, or the window just opened)."""
        self._loading = True
        try:
            doc = self.main.doc or {}
            settings = doc.get("trace_processing") if isinstance(doc.get("trace_processing"), dict) else {}
            zscore = settings.get("zscore") if isinstance(settings.get("zscore"), dict) else {}
            self._baseline_frames = int(zscore.get("baseline_frames") or self._baseline_frames)
            self._post_frames = int(zscore.get("post_frames") or self._post_frames)
            self._z_units = zscore.get("units") if zscore.get("units") in ("frames", "seconds") else "frames"
            if self._z_units == "seconds" and not doc.get("fps"):
                self._z_units = "frames"
            self._clamp_windows()
            path = str(self.main.pkl_path) if self.main.pkl_path else None
            if path != self._bound_path:
                self._bound_path = path
                self._raster_signature = None
                self._reset_z_plots()
                self.z_status.setText("Choose an event annotation, then compute.")
            summary = doc.get("summary") if doc else None
            trace = summary.get("fov_mean_trace") if summary else None
            if trace is not None:
                self.ann_curve.setData(np.arange(len(trace)), np.asarray(trace))
            else:
                self.ann_curve.setData([], [])
            self.sort_combo.blockSignals(True)
            sort = settings.get("raster_sort", SORT_KIND)
            if sort == "similarity":
                sort = hac_sort_key(METRIC_RUZICKA, ORDER_SEPARATE)
            index = self.sort_combo.findData(sort)
            self.sort_combo.setCurrentIndex(index if index >= 0 else 1)
            self.sort_combo.blockSignals(False)
            self.lut_combo.blockSignals(True)
            lut = settings.get("raster_lut", "viridis")
            self.lut_combo.setCurrentText(lut if lut in LUT_NAMES else "viridis")
            self.lut_combo.blockSignals(False)
            self.z_unit_combo.blockSignals(True)
            self.z_unit_combo.setCurrentText(self._z_units)
            self.z_unit_combo.blockSignals(False)
            self._load_clustering_controls(settings.get("clustering") if isinstance(settings.get("clustering"), dict) else {})
            self._configure_duration_spins()
            self._fill_event_combo(zscore.get("annotation"))
            self._rebuild_roi_tree()
            self.sync_annotations()
            self._redraw_raster()
            self.apply_time_axis()
            if self.tabs.currentIndex() == 1:
                self._show_clustering()
            elif self.tabs.currentIndex() == 2:
                self._show_similarity()
        finally:
            self._loading = False

    def on_rois_changed(self) -> None:
        self._redraw_raster()
        self._rebuild_roi_tree()
        tab = self.tabs.currentIndex()
        if tab == 1:
            self._show_clustering()
        elif tab == 2:
            self._show_similarity()
        self._mark_z_stale()

    def apply_time_axis(self) -> None:
        fps, units = self._time()
        apply_time_axis(self.raster_plot, fps, units)
        apply_time_axis(self.ann_plot, fps, units)
        z_units = "seconds" if self._using_seconds() else "frames"
        for plot in (self.group_plot, *(entry["plot"] for entry in self._roi_plots)):
            apply_epoch_axis(plot, fps, z_units)
        self._configure_duration_spins()

    def _on_tab(self, index: int) -> None:
        if self.main.doc is None:
            return
        if index == 1:
            self._show_clustering()
        elif index == 2:
            self._show_similarity()

    # raster ------------------------------------------------------------------
    def _on_raster_option(self) -> None:
        if self._syncing or self.main.doc is None:
            return
        self._store("raster_sort", self.sort_combo.currentData())
        self._store("raster_lut", self.lut_combo.currentText())
        self._redraw_raster()

    def _redraw_raster(self) -> None:
        doc = self.main.doc
        rois = doc["rois"] if doc else []
        n_frames = self.n_frames if doc else 1
        mode = self.sort_combo.currentData() or SORT_KIND
        self._raster_rows = ordered_rois(rois, mode, n_frames)
        n = len(self._raster_rows)
        signature = (
            mode,
            n_frames,
            tuple((roi["id"], roi["name"], roi["kind"]) for roi in self._raster_rows),
        )
        if n == 0 or doc is None:
            self.raster_image.setVisible(False)
            self.kind_scatter.setData(x=[], y=[])
            self.raster_plot.getAxis("left").setTicks([])
            self.raster_plot.setTitle("No ROI traces yet")
            self._raster_signature = signature
            self._refresh_raster_regions()
            return
        image = np.vstack([minmax_trace(fit_trace(roi["trace"], n_frames)) for roi in self._raster_rows])
        self.raster_image.setVisible(True)
        self.raster_image.setZValue(0)
        self.raster_image.setImage(image, autoLevels=False)
        self.raster_image.setLevels((0.0, 1.0))
        self.raster_image.setLookupTable(lookup_table(self.lut_combo.currentText()))
        ticks = [(i + 0.5, roi["name"]) for i, roi in enumerate(self._raster_rows)]
        self.raster_plot.getAxis("left").setTicks([ticks])
        self.kind_scatter.setData(
            x=np.zeros(n),
            y=np.arange(n) + 0.5,
            brush=[pg.mkBrush(*KIND_COLORS[roi["kind"]]) for roi in self._raster_rows],
            pen=pg.mkPen(None),
        )
        if signature != self._raster_signature:
            self.raster_plot.setYRange(0, n, padding=0.02)
            self.raster_plot.setXRange(0, n_frames, padding=0.01)
            self.kind_plot.setXRange(-1, 1)
        self._raster_signature = signature
        self.raster_plot.setTitle("Min–max normalised traces")
        self._refresh_raster_regions()

    def _refresh_raster_regions(self) -> None:
        for region in self._raster_regions:
            self.raster_plot.removeItem(region)
        self._raster_regions = []
        for start, end, color in annotation_spans(self.main.doc):
            region = pg.LinearRegionItem(
                values=(start, end), movable=False, brush=pg.mkBrush(*color, 50), pen=pg.mkPen(color)
            )
            region.setZValue(30)
            region.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            self.raster_plot.addItem(region)
            self._raster_regions.append(region)

    def _on_raster_clicked(self, ev) -> None:
        if ev.button() != QtCore.Qt.MouseButton.LeftButton or not self._raster_rows:
            return
        if not self.raster_plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        point = self.raster_plot.vb.mapSceneToView(ev.scenePos())
        row = int(np.floor(point.y()))
        if 0 <= row < len(self._raster_rows):
            self.main.select_roi(self._raster_rows[row]["id"])

    # similarity --------------------------------------------------------------
    def _show_similarity(self) -> None:
        doc = self.main.doc
        if doc is None:
            self.similarity_text.setText("Open a stack first.")
            return
        n_frames = self.n_frames
        specific, spec_names = self._kind_traces(KIND_SPECIFIC, n_frames)
        nonspecific, nonspec_names = self._kind_traces(KIND_NONSPECIFIC, n_frames)
        result = compare_trace_groups(specific, nonspecific, spec_names, nonspec_names)
        self._show_matrix(self.spec_matrix_plot, self.spec_matrix, "Specific", result["specific"])
        self._show_matrix(self.nonspec_matrix_plot, self.nonspec_matrix, "Non-specific", result["nonspecific"])
        self.similarity_text.setText(self._similarity_html(result))

    def _kind_traces(self, kind: str, n_frames: int) -> tuple[np.ndarray, list[str]]:
        rows = [r for r in self.main.doc["rois"] if r.get("kind") == kind and r.get("trace") is not None]
        if not rows:
            return np.zeros((0, n_frames)), []
        stacked = np.vstack([fit_trace(r["trace"], n_frames) for r in rows])
        return stacked, [r["name"] for r in rows]

    def _show_matrix(self, plot: pg.PlotWidget, image: pg.ImageItem, title: str, block: dict) -> None:
        matrix = block["matrix"]
        names = block["names"]
        if matrix.size == 0:
            image.setVisible(False)
            plot.setTitle(f"{title}: no varying traces")
            plot.getAxis("left").setTicks([])
            plot.getAxis("bottom").setTicks([])
            return
        shown = np.array(matrix, dtype=np.float64, copy=True)
        shown[~np.isfinite(shown)] = 0.0
        image.setVisible(True)
        image.setImage(shown, autoLevels=False)
        image.setLevels((-1.0, 1.0))
        image.setLookupTable(self._corr_lut)
        ticks = [(i + 0.5, name) for i, name in enumerate(names)]
        font = QtGui.QFont()
        font.setPointSize(8)
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setTicks([ticks])
            axis.setTickFont(font)
        plot.invertY(True)
        plot.setTitle(f"{title}: mean r = {format_r(block['mean_r'])} ({block['n_pairs']} pairs)")
        plot.setXRange(0, len(names), padding=0.02)
        plot.setYRange(0, len(names), padding=0.02)

    def _similarity_html(self, result: dict) -> str:
        spec, nonspec = result["specific"], result["nonspecific"]
        if result["p_specific_greater"] is None:
            how = "not enough ROIs in both groups to reshuffle"
        elif result["exhaustive"]:
            how = f"all {result['n_assignments']} ways of assigning these ROIs to the two group sizes"
        else:
            how = f"{result['n_assignments']} random reassignments"
        dropped = ""
        if result["dropped"]:
            dropped = f"<br>Left out (flat or incomplete): {', '.join(result['dropped'])}."
        return (
            f"<b>{similarity_conclusion(result)}</b><br>"
            f"Specific mean pairwise r = {format_r(spec['mean_r'])} "
            f"({spec['n_pairs']} pairs, {spec['n_rois']} ROIs). "
            f"Non-specific mean pairwise r = {format_r(nonspec['mean_r'])} "
            f"({nonspec['n_pairs']} pairs, {nonspec['n_rois']} ROIs). "
            f"Difference (specific − non-specific) = {format_r(result['difference'])}.<br>"
            f"One-sided p, specific more similar: {format_p(result['p_specific_greater'])}. "
            f"One-sided p, non-specific more similar: {format_p(result['p_nonspecific_greater'])}. "
            f"Mean r between the two groups: {format_r(result['between_r'])}.<br>"
            "Pearson correlation of the raw traces (shape, not amplitude). "
            "Each matrix is ordered by average-linkage clustering on distance 1 − r, "
            f"so similar traces sit together. The p-values use {how}."
            f"{dropped} Colour is Pearson r from −1 (blue) through 0 (white) to +1 (red)."
        )

    # clustering ----------------------------------------------------------------
    def _load_clustering_controls(self, saved: dict) -> None:
        metric = saved.get("metric") if saved.get("metric") in (METRIC_RUZICKA, METRIC_EUCLIDEAN) else METRIC_RUZICKA
        order = saved.get("order") if saved.get("order") in (ORDER_POOLED, ORDER_SEPARATE) else ORDER_POOLED
        lut = saved.get("lut") if saved.get("lut") in LUT_NAMES else "viridis"
        self.metric_combo.blockSignals(True)
        self.order_combo.blockSignals(True)
        self.matrix_lut.blockSignals(True)
        self.cut_spin.blockSignals(True)
        self.metric_combo.setCurrentIndex(max(0, self.metric_combo.findData(metric)))
        self.order_combo.setCurrentIndex(max(0, self.order_combo.findData(order)))
        self.matrix_lut.setCurrentText(lut)
        self.cut_spin.blockSignals(False)
        self.matrix_lut.blockSignals(False)
        self.order_combo.blockSignals(False)
        self.metric_combo.blockSignals(False)
        self._apply_saved_cut(saved)

    def _apply_saved_cut(self, saved: dict | None = None) -> None:
        saved = saved if saved is not None else self._clustering_settings()
        cuts = saved.get("cuts") if isinstance(saved.get("cuts"), dict) else {}
        metric = self.metric_combo.currentData()
        value = cuts.get(metric)
        self.cut_spin.blockSignals(True)
        if isinstance(value, (int, float)):
            self.cut_spin.setValue(float(value))
        self.cut_spin.blockSignals(False)

    def _clustering_settings(self) -> dict:
        settings = self._settings()
        found = settings.get("clustering")
        return found if isinstance(found, dict) else {}

    def _store_clustering(self, view: dict | None = None) -> None:
        metric = self.metric_combo.currentData()
        payload = dict(self._clustering_settings())
        payload["metric"] = metric
        payload["order"] = self.order_combo.currentData()
        payload["lut"] = self.matrix_lut.currentText()
        cuts = dict(payload.get("cuts") or {})
        if view and view.get("cut") is not None and view.get("order_mode") == ORDER_POOLED:
            cuts[metric] = float(view["cut"])
        payload["cuts"] = cuts
        self._store("clustering", payload)

    def _clustering_changed(self, reset_cut: bool) -> None:
        if self._loading or self.main.doc is None:
            return
        if reset_cut:
            saved = dict(self._clustering_settings())
            cuts = dict(saved.get("cuts") or {})
            cuts.pop(self.metric_combo.currentData(), None)
            saved["cuts"] = cuts
            self._settings()["clustering"] = saved
        self._show_clustering()

    def _show_clustering(self) -> None:
        doc = self.main.doc
        if doc is None:
            self.cluster_text.setText("Open a stack first.")
            return
        rows = [roi for roi in doc["rois"] if roi.get("trace") is not None]
        if not rows:
            self._clear_clustering("Draw some ROIs first.")
            return
        n_frames = self.n_frames
        traces = np.vstack([fit_trace(roi["trace"], n_frames) for roi in rows])
        metric = self.metric_combo.currentData()
        order = self.order_combo.currentData()
        saved_cuts = (self._clustering_settings().get("cuts") or {})
        cut = saved_cuts.get(metric) if order == ORDER_POOLED else None
        view = hierarchical_view(
            traces,
            [roi["name"] for roi in rows],
            [roi["kind"] for roi in rows],
            metric,
            order,
            None if cut is None else float(cut),
        )
        self._draw_clustering(view)
        if not self._loading:
            self._store_clustering(view)

    def _clear_clustering(self, message: str) -> None:
        self.cluster_image.setVisible(False)
        self._clear_tree()
        for item in self._cluster_outlines:
            self.cluster_plot.removeItem(item)
        self._cluster_outlines = []
        self.cluster_plot.getAxis("left").setTicks([])
        self.cluster_plot.getAxis("bottom").setTicks([])
        self.cluster_text.setText(message)
        self.cut_spin.setEnabled(False)
        self.cut_line.setVisible(False)

    def _clear_tree(self) -> None:
        for item in self._tree_curves:
            self.tree_plot.removeItem(item)
        self._tree_curves = []

    def _draw_clustering(self, view: dict | None = None, recompute: bool = True) -> None:
        if view is None:
            if recompute:
                self._show_clustering()
            elif getattr(self, "_cluster_view", None) is not None:
                view = self._cluster_view
            else:
                return
        if view is None:
            return
        self._cluster_view = view
        pooled = view["order_mode"] == ORDER_POOLED
        self._place_cut_controls(view, pooled)
        matrix = view["matrix"]
        for item in self._cluster_outlines:
            self.cluster_plot.removeItem(item)
        self._cluster_outlines = []
        if matrix.size == 0:
            self.cluster_image.setVisible(False)
            self._clear_tree()
            self.cluster_text.setText("No varying traces to cluster.")
            return
        shown = np.array(matrix, dtype=np.float64, copy=True)
        finite = shown[np.isfinite(shown)]
        if view["display"] == "similarity":
            levels = (0.0, 1.0)
        else:
            levels = (0.0, float(finite.max()) if finite.size else 1.0)
        shown[~np.isfinite(shown)] = levels[0]
        self.cluster_image.setVisible(True)
        self.cluster_image.setImage(shown, autoLevels=False)
        self.cluster_image.setLevels(levels)
        self.cluster_image.setLookupTable(lookup_table(self.matrix_lut.currentText()))
        names = view["names"]
        ticks = [(i + 0.5, name) for i, name in enumerate(names)]
        font = QtGui.QFont()
        font.setPointSize(8)
        for axis_name in ("left", "bottom"):
            axis = self.cluster_plot.getAxis(axis_name)
            axis.setTicks([ticks])
            axis.setTickFont(font)
        self.cluster_plot.setXRange(0, len(names), padding=0.02)
        self.cluster_plot.setYRange(0, len(names), padding=0.02)
        self._draw_tree(view)
        self._draw_cluster_marks(view)
        method = "Ružička + average" if view["metric"] == METRIC_RUZICKA else "Euclidean + Ward"
        scale = "similarity (1 − Ružička distance)" if view["display"] == "similarity" else "Euclidean distance"
        if pooled:
            n_groups = len(view["clusters"] or [])
            order_text = (
                f"one tree of every ROI. The dotted line is the contrast index "
                f"({view['cut']:.4g}) and cuts {n_groups} group(s), including single ROIs."
            )
        else:
            order_text = (
                "specific ROIs, then non-specific, each ordered by its own tree. "
                "The contrast index is inactive in this order."
            )
        dropped = f" Left out: {', '.join(view['dropped'])}." if view["dropped"] else ""
        self.cluster_text.setText(
            f"{method} on traces scaled to their own minimum and maximum. "
            f"Rows follow {order_text} Colour is {scale}.{dropped}"
        )

    def _place_cut_controls(self, view: dict, pooled: bool) -> None:
        active = pooled and float(view["max_height"]) > 0
        self.cut_spin.setEnabled(active)
        self.cut_line.setVisible(active)
        self.cut_line.setMovable(active)
        if not active:
            return
        span = float(view["max_height"])
        decimals = _cut_decimals(span)
        step = max(span / 200.0, 10.0 ** (-decimals))
        self._cut_updating = True
        self.cut_spin.blockSignals(True)
        try:
            self.cut_spin.setDecimals(decimals)
            self.cut_spin.setRange(0.0, span)
            self.cut_spin.setSingleStep(step)
            self.cut_spin.setValue(float(view["cut"]))
            self.cut_line.setBounds((0.0, span))
            self.cut_line.setValue(float(view["cut"]))
        finally:
            self.cut_spin.blockSignals(False)
            self._cut_updating = False

    def _draw_tree(self, view: dict) -> None:
        self._clear_tree()
        pen = pg.mkPen((220, 220, 220), width=1)
        pen.setCosmetic(True)
        farthest = 0.0
        for tree in view.get("trees") or []:
            for x, y in dendrogram_branches(tree["linkage"], int(tree["row"])):
                curve = self.tree_plot.plot(x, y, pen=pen)
                curve.setZValue(2)
                self._tree_curves.append(curve)
                if x.size:
                    farthest = max(farthest, float(np.max(x)))
        farthest = max(farthest, float(view.get("max_height") or 0.0), 1e-6)
        self.tree_plot.setXRange(0.0, farthest * 1.05, padding=0.02)

    def _draw_cluster_marks(self, view: dict) -> None:
        for item in self._cluster_outlines:
            self.cluster_plot.removeItem(item)
        self._cluster_outlines = []
        names = view["names"]
        if view["clusters"]:
            for index, span in enumerate(view["clusters"]):
                if not span:
                    continue
                color = CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
                start, end = span[0], span[-1] + 1
                pen = pg.mkPen(color, width=2)
                pen.setCosmetic(True)
                outline = self.cluster_plot.plot(
                    [start, end, end, start, start],
                    [start, start, end, end, start],
                    pen=pen,
                )
                outline.setZValue(8)
                self._cluster_outlines.append(outline)
        elif view["split"]:
            at = float(view["split"])
            line = self.cluster_plot.plot([at, at], [0, len(names)], pen=pg.mkPen((180, 180, 180), width=1))
            line.setZValue(8)
            self._cluster_outlines.append(line)

    def _on_cut_spin(self, value: float) -> None:
        if self._loading or self._cut_updating:
            return
        self._apply_cut(float(value))

    def _on_cut_line(self, _line=None) -> None:
        if self._loading or self._cut_updating:
            return
        self._apply_cut(float(self.cut_line.value()))

    def _apply_cut(self, value: float) -> None:
        view = getattr(self, "_cluster_view", None)
        if not view or view.get("order_mode") != ORDER_POOLED:
            return
        updated = recut_view(view, value)
        self._cluster_view = updated
        self._cut_updating = True
        self.cut_spin.blockSignals(True)
        try:
            self.cut_spin.setValue(float(updated["cut"]))
            self.cut_line.setValue(float(updated["cut"]))
        finally:
            self.cut_spin.blockSignals(False)
            self._cut_updating = False
        self._draw_cluster_marks(updated)
        n_groups = len(updated["clusters"] or [])
        method = "Ružička + average" if updated["metric"] == METRIC_RUZICKA else "Euclidean + Ward"
        scale = "similarity (1 − Ružička distance)" if updated["display"] == "similarity" else "Euclidean distance"
        dropped = f" Left out: {', '.join(updated['dropped'])}." if updated["dropped"] else ""
        self.cluster_text.setText(
            f"{method} on traces scaled to their own minimum and maximum. "
            f"Rows follow one tree of every ROI. The dotted line is the contrast index "
            f"({updated['cut']:.4g}) and cuts {n_groups} group(s), including single ROIs. "
            f"Colour is {scale}.{dropped}"
        )
        if not self._loading:
            self._store_clustering(updated)

    # event annotation and ROI groups -----------------------------------------
    def sync_annotations(self) -> None:
        """Keep the on/off menu and the shaded ranges in step with the main window."""
        if not hasattr(self, "raster_ann_menu"):
            return
        fill_annotation_menu(
            self.raster_ann_menu,
            self.main.annotation_choices() if self.main.doc is not None else [],
            self.main.set_annotation_shown,
        )
        self._refresh_raster_regions()
        self._refresh_event_preview()

    def _categories(self) -> list[dict]:
        doc = self.main.doc
        return list(doc.get("heatmaps") or []) if doc else []

    def _event_category(self) -> dict | None:
        name = self.event_combo.currentData()
        return next((hm for hm in self._categories() if hm.get("name") == name), None)

    def _fill_event_combo(self, select_name: str | None = None) -> None:
        if select_name is None and self.event_combo.count():
            select_name = self.event_combo.currentData()
        self.event_combo.blockSignals(True)
        self.event_combo.clear()
        for hm in self._categories():
            self.event_combo.addItem(hm["name"], hm["name"])
        index = self.event_combo.findData(select_name)
        self.event_combo.setCurrentIndex(index if index >= 0 else 0)
        self.event_combo.blockSignals(False)
        self.compute_btn.setEnabled(self.event_combo.count() > 0 and self.main.stack is not None)
        self._refresh_event_preview()

    def _on_event_changed(self) -> None:
        if self._loading:
            return
        self._refresh_event_preview()
        self._store_zscore()
        self._mark_z_stale()

    def _refresh_event_preview(self) -> None:
        for region in self._regions:
            self.ann_plot.removeItem(region)
        self._regions = []
        ann = self._event_category()
        if ann is None:
            self.ann_plot.setTitle("Mean of the whole field of view")
            return
        index = next((i for i, hm in enumerate(self._categories()) if hm is ann), 0)
        color = annotation_color(index)
        for start, end in ann.get("ranges") or []:
            region = pg.LinearRegionItem(
                values=(int(min(start, end)), int(max(start, end))),
                movable=False,
                brush=pg.mkBrush(*color, 50),
                pen=pg.mkPen(color),
            )
            region.setZValue(-10)
            self.ann_plot.addItem(region)
            self._regions.append(region)
        self.ann_plot.setTitle(f"{ann['name']}: onset is the first frame of each range")

    def _rebuild_roi_tree(self) -> None:
        doc = self.main.doc
        excluded = {int(i) for i in (self._zscore_settings().get("excluded_roi_ids") or [])}
        self.roi_tree.blockSignals(True)
        self.roi_tree.clear()
        rois = doc["rois"] if doc else []
        for kind, label in ((KIND_SPECIFIC, "Specific"), (KIND_NONSPECIFIC, "Non-specific")):
            parent = QtWidgets.QTreeWidgetItem([label])
            parent.setFlags(parent.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            children = [roi for roi in rois if roi.get("kind") == kind]
            for roi in children:
                child = QtWidgets.QTreeWidgetItem([roi["name"]])
                child.setFlags(child.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                child.setData(0, QtCore.Qt.ItemDataRole.UserRole, roi["id"])
                state = QtCore.Qt.CheckState.Unchecked if roi["id"] in excluded else QtCore.Qt.CheckState.Checked
                child.setCheckState(0, state)
                parent.addChild(child)
            self._sync_parent_check(parent)
            self.roi_tree.addTopLevelItem(parent)
            parent.setExpanded(True)
        self.roi_tree.blockSignals(False)

    def _sync_parent_check(self, parent: QtWidgets.QTreeWidgetItem) -> None:
        states = [parent.child(i).checkState(0) for i in range(parent.childCount())]
        if states and all(state == QtCore.Qt.CheckState.Checked for state in states):
            parent.setCheckState(0, QtCore.Qt.CheckState.Checked)
        elif not states or all(state == QtCore.Qt.CheckState.Unchecked for state in states):
            parent.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        else:
            parent.setCheckState(0, QtCore.Qt.CheckState.PartiallyChecked)

    def _on_roi_item_changed(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        if self._loading or self.roi_tree.signalsBlocked():
            return
        self.roi_tree.blockSignals(True)
        if item.parent() is None:
            state = item.checkState(0)
            if state != QtCore.Qt.CheckState.PartiallyChecked:
                for i in range(item.childCount()):
                    item.child(i).setCheckState(0, state)
        else:
            self._sync_parent_check(item.parent())
        self.roi_tree.blockSignals(False)
        self._store_excluded()
        self._mark_z_stale()

    def _checked_roi_ids(self) -> set[int]:
        chosen: set[int] = set()
        root = self.roi_tree.invisibleRootItem()
        for group in range(root.childCount()):
            parent = root.child(group)
            for i in range(parent.childCount()):
                child = parent.child(i)
                if child.checkState(0) == QtCore.Qt.CheckState.Checked:
                    chosen.add(int(child.data(0, QtCore.Qt.ItemDataRole.UserRole)))
        return chosen

    def _store_excluded(self) -> None:
        doc = self.main.doc
        if doc is None:
            return
        checked = self._checked_roi_ids()
        excluded = [int(roi["id"]) for roi in doc["rois"] if roi["id"] not in checked]
        payload = dict(self._zscore_settings())
        payload["excluded_roi_ids"] = excluded
        self._store("zscore", payload)

    def _zscore_settings(self) -> dict:
        settings = self._settings()
        found = settings.get("zscore")
        return found if isinstance(found, dict) else {}


    # Z-score -----------------------------------------------------------------
    def _clamp_windows(self) -> None:
        limit = max(2, self.n_frames - 1)
        self._baseline_frames = min(max(2, int(self._baseline_frames)), limit)
        self._post_frames = min(max(1, int(self._post_frames)), limit)

    def _using_seconds(self) -> bool:
        fps, _units = self._time()
        return bool(self._z_units == "seconds" and fps)

    def _configure_duration_spins(self) -> None:
        fps, _units = self._time()
        seconds = self._using_seconds()
        self._clamp_windows()
        self._syncing = True
        for spin, frames, minimum in (
            (self.baseline_spin, self._baseline_frames, 2),
            (self.post_spin, self._post_frames, 1),
        ):
            spin.blockSignals(True)
            if seconds and fps:
                spin.setDecimals(2)
                spin.setSingleStep(0.1)
                spin.setRange(minimum / fps, max(self.n_frames - 1, minimum) / fps)
                spin.setValue(frames / fps)
            else:
                spin.setDecimals(0)
                spin.setSingleStep(1)
                spin.setRange(minimum, max(self.n_frames - 1, minimum))
                spin.setValue(frames)
            spin.blockSignals(False)
        self._syncing = False

    def _on_z_units(self, text: str) -> None:
        if self._loading:
            return
        fps, _units = self._time()
        chosen = "seconds" if text == "seconds" else "frames"
        if chosen == "seconds" and not fps:
            self.z_unit_combo.blockSignals(True)
            self.z_unit_combo.setCurrentText("frames")
            self.z_unit_combo.blockSignals(False)
            self._z_units = "frames"
            self.z_status.setText("Seconds need a frame rate. Set it on the main window, or stay in frames.")
            return
        self._z_units = chosen
        self.apply_time_axis()
        self._store_zscore()
        self._mark_z_stale()

    def _to_frames(self, value: float, minimum: int) -> int:
        fps, _units = self._time()
        if self._using_seconds() and fps:
            frames = int(round(float(value) * fps))
        else:
            frames = int(round(float(value)))
        return min(max(minimum, frames), max(minimum, self.n_frames - 1))

    def _on_baseline_edited(self, value: float) -> None:
        if self._syncing:
            return
        self._baseline_frames = self._to_frames(value, 2)
        self._store_zscore()
        self._mark_z_stale()

    def _on_post_edited(self, value: float) -> None:
        if self._syncing:
            return
        self._post_frames = self._to_frames(value, 1)
        self._store_zscore()
        self._mark_z_stale()

    def _store_zscore(self) -> None:
        category = self._event_category()
        payload = dict(self._zscore_settings())
        payload["annotation"] = category["name"] if category else None
        payload["baseline_frames"] = int(self._baseline_frames)
        payload["post_frames"] = int(self._post_frames)
        payload["units"] = "seconds" if self._using_seconds() else "frames"
        self._store("zscore", payload)

    def _mark_z_stale(self) -> None:
        if not self._roi_plots:
            return
        self._z_stale = True
        self.z_status.setText(self.z_status.text().split("\n")[0] + "\nTraces or settings changed. Compute again to update the plots.")

    def _reset_z_plots(self) -> None:
        self._clear_roi_plots()
        self.group_plot.clear()
        if self.group_plot.legend is None:
            self.group_plot.addLegend()
        self.group_plot.setLabel("left", "Z")
        self._z_stale = False

    def _clear_roi_plots(self) -> None:
        while self.roi_layout.count() > 1:
            item = self.roi_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._roi_plots = []

    def _compute_zscore(self) -> None:
        ann = self._event_category()
        doc = self.main.doc
        if ann is None or doc is None:
            self.z_status.setText("Choose an event annotation first. Add one under Tools → Annotations.")
            return
        onsets = event_onsets(ann.get("ranges"))
        if not onsets:
            self.z_status.setText(f"{ann['name']} has no ranges yet. Add them under Tools → Annotations.")
            return
        checked = self._checked_roi_ids()
        rois = [r for r in doc["rois"] if r.get("id") in checked and r.get("kind") == KIND_SPECIFIC]
        rois += [r for r in doc["rois"] if r.get("id") in checked and r.get("kind") == KIND_NONSPECIFIC]
        if not rois:
            self.z_status.setText("Tick at least one ROI in Groups, then compute.")
            return
        self._clamp_windows()
        self._store_zscore()
        baseline, post = self._baseline_frames, self._post_frames
        x = epoch_x(baseline, post)
        fps, _units = self._time()
        z_units = "seconds" if self._using_seconds() else "frames"
        self._reset_z_plots()
        _add_onset_line(self.group_plot)

        grouped: dict[str, list[np.ndarray]] = {KIND_SPECIFIC: [], KIND_NONSPECIFIC: []}
        kept_total = 0
        link: pg.PlotWidget | None = None
        for roi in rois:
            if roi.get("trace") is None or roi.get("kind") not in grouped:
                continue
            trials, skipped = trial_zscores(fit_trace(roi["trace"], self.n_frames), onsets, baseline, post)
            plot_widget = pg.PlotWidget()
            plot_widget.setFixedHeight(170)
            plot = plot_widget.getPlotItem()
            plot.setLabel("left", "Z")
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setTitle(_roi_title(roi, trials.shape[0], skipped))
            _add_onset_line(plot)
            if trials.shape[0]:
                color = KIND_COLORS[roi["kind"]]
                for row in trials:
                    _plot_dotted(plot, x, row, color)
                mean, sem = mean_and_sem(trials)
                _plot_band(plot, x, mean, sem, color, band=trials.shape[0] >= 2)
                grouped[roi["kind"]].append(mean)
                kept_total += trials.shape[0]
            if link is None:
                link = plot_widget
            else:
                plot_widget.setXLink(link)
            apply_epoch_axis(plot, fps, z_units)
            plot.setXRange(float(x[0]), float(x[-1]), padding=0.02)
            if trials.shape[0]:
                lo = float(np.min(trials))
                hi = float(np.max(trials))
                pad = 0.08 * max(hi - lo, 1.0)
                plot.setYRange(lo - pad, hi + pad, padding=0)
            self.roi_layout.insertWidget(self.roi_layout.count() - 1, plot_widget)
            self._roi_plots.append({"plot": plot, "kept": int(trials.shape[0]), "name": roi["name"]})

        for kind, means in grouped.items():
            if not means:
                continue
            color = KIND_COLORS[kind]
            label = "Specific" if kind == KIND_SPECIFIC else "Non-specific"
            stacked = np.vstack(means)
            for row in stacked:
                _plot_dotted(self.group_plot, x, row, color)
            mean, sem = mean_and_sem(stacked)
            _plot_band(
                self.group_plot,
                x,
                mean,
                sem,
                color,
                name=f"{label} mean",
                width=3,
                band=stacked.shape[0] >= 2,
            )
        apply_epoch_axis(self.group_plot, fps, z_units)
        self.group_plot.enableAutoRange()
        self.group_plot.setTitle(f"{ann['name']}: specific vs non-specific")
        unit = "s" if self._using_seconds() else "frames"
        base_txt = f"{self.baseline_spin.value():.2f}" if self._using_seconds() else str(baseline)
        post_txt = f"{self.post_spin.value():.2f}" if self._using_seconds() else str(post)
        self._z_stale = False
        self.z_status.setText(
            f"{ann['name']}: {len(onsets)} event(s), baseline {base_txt} {unit}, "
            f"post-stim {post_txt} {unit}. {kept_total} trial trace(s) drawn. "
            "Events that run off the recording, or whose baseline is flat, are left out."
        )


def _roi_title(roi: dict, kept: int, skipped: int) -> str:
    color = KIND_HTML.get(roi["kind"], "#dddddd")
    trial = "trial" if kept == 1 else "trials"
    extra = f", {skipped} left out" if skipped else ""
    return f"<span style='color:{color}'>{roi['name']}</span>  ({kept} {trial}{extra})"


def _add_onset_line(plot: pg.PlotItem) -> None:
    line = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=ONSET_PEN)
    line.setZValue(-2)
    plot.addItem(line)


def _plot_dotted(plot: pg.PlotItem, x: np.ndarray, y: np.ndarray, color: tuple[int, int, int]) -> None:
    curve = plot.plot(x, y, pen=pg.mkPen((*color, 170), width=1, style=QtCore.Qt.PenStyle.DotLine))
    curve.setZValue(2)


def _plot_band(
    plot: pg.PlotItem,
    x: np.ndarray,
    mean: np.ndarray,
    sem: np.ndarray,
    color: tuple[int, int, int],
    name: str | None = None,
    width: float = 2.5,
    band: bool = True,
) -> None:
    if band:
        upper = plot.plot(x, mean + sem, pen=GREY_PEN)
        lower = plot.plot(x, mean - sem, pen=GREY_PEN)
        upper.setZValue(1)
        lower.setZValue(1)
        fill = pg.FillBetweenItem(lower, upper, brush=GREY_FILL)
        fill.setZValue(-8)
        plot.addItem(fill)
    curve = plot.plot(x, mean, pen=pg.mkPen(color, width=width), name=name)
    curve.setZValue(5)
