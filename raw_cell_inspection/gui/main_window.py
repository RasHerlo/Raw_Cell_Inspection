"""Primary window: draw ROIs on a reference image or heatmap, inspect them on the movie stack."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from raw_cell_inspection import __version__
from raw_cell_inspection.analysis import stack_summary
from raw_cell_inspection.gui.heatmap_editor import HeatmapEditor, heatmap_state
from raw_cell_inspection.gui.trace_processing_window import TraceProcessingWindow
from raw_cell_inspection.gui.image_panel import KIND_COLORS, DisplayControls, ImagePanel
from raw_cell_inspection.gui.tasks import run_task
from raw_cell_inspection.gui.trace_panel import TracePanel, annotation_spans
from raw_cell_inspection.masks import (
    clip_vertices,
    contains_point,
    polygon_area,
    polygon_mask,
    roi_trace,
    simplify_stroke,
)
from raw_cell_inspection.stack_io import (
    IMAGE_SUFFIXES,
    TIF_SUFFIXES,
    StackError,
    load_reference_image,
    load_stack,
    stack_signature,
)
from raw_cell_inspection.store import (
    KIND_NONSPECIFIC,
    KIND_PREFIX,
    KIND_SPECIFIC,
    PICKLE_SUFFIX,
    load_document,
    new_document,
    pickle_path_for,
    relpath_or_none,
    save_document,
    signature_mismatch,
    stack_candidates,
)

KIND_LABEL = {KIND_SPECIFIC: "Specific", KIND_NONSPECIFIC: "Non-specific"}
STACK_SOURCES = [("frame", "Movie frame"), ("mean", "Stack mean"), ("max", "Stack max")]


def _file_filter(suffixes) -> str:
    return " ".join(f"*{s}" for s in suffixes)


def dialog_start(settings, *keys: str) -> str:
    """Path for a file dialog: the last file if it is still there, otherwise its folder.

    A directory is returned with a trailing separator so the dialog opens that
    folder instead of treating the last component as a file name.
    """
    for key in keys:
        raw = settings.value(key, "")
        text = raw if isinstance(raw, str) else ""
        if not text:
            continue
        path = Path(text)
        if path.is_file():
            return str(path)
        if path.is_dir():
            return os.path.join(str(path), "")
        if path.parent.is_dir():
            return os.path.join(str(path.parent), "")
    return ""


def remember_opened(settings, *, stack: Path | None = None, experiment: Path | None = None) -> None:
    """Remember the last experiment so the next open dialog starts there.

    Stored with QSettings, which the shipped app keeps per user (the registry
    on Windows), independent of where the program is installed.
    """
    if experiment is not None:
        settings.setValue("last_experiment", str(experiment))
    if stack is not None:
        settings.setValue("last_stack", str(stack))
    folder = experiment.parent if experiment is not None else (stack.parent if stack is not None else None)
    if folder is not None:
        settings.setValue("last_dir", str(folder))
    settings.sync()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1500, 950)
        self.settings = QtCore.QSettings("RasHerlo", "RawCellInspection")

        self.stack_path: Path | None = None
        self.stack: np.ndarray | None = None
        self.doc: dict | None = None
        self.pkl_path: Path | None = None
        self.dirty = False
        self.frame = 0
        self.selected: dict[str, int | None] = {KIND_SPECIFIC: None, KIND_NONSPECIFIC: None}
        self.active_id: int | None = None
        self.draw_kind: str | None = None
        self._frame_sample: np.ndarray | None = None
        self.heatmap_editor: HeatmapEditor | None = None
        self.trace_window: TraceProcessingWindow | None = None

        self._build_ui()
        self._build_menus()
        self._set_enabled(False)
        self._update_title()

    # UI construction ---------------------------------------------------------
    def _build_ui(self) -> None:
        self.draw_panel = ImagePanel("Draw view")
        self.stack_panel = ImagePanel("Stack view")
        self.trace_panel = TracePanel()

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_time = QtWidgets.QLabel()
        self.frame_time.setMinimumWidth(90)
        frame_row = QtWidgets.QHBoxLayout()
        frame_row.setContentsMargins(4, 2, 4, 2)
        frame_row.addWidget(QtWidgets.QLabel("Frame"))
        frame_row.addWidget(self.frame_slider, 1)
        frame_row.addWidget(self.frame_spin)
        frame_row.addWidget(self.frame_time)
        stack_side = QtWidgets.QWidget()
        stack_layout = QtWidgets.QVBoxLayout(stack_side)
        stack_layout.setContentsMargins(0, 0, 0, 0)
        stack_layout.setSpacing(0)
        stack_layout.addWidget(self.stack_panel, 1)
        stack_layout.addLayout(frame_row)

        images = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        images.addWidget(self.draw_panel)
        images.addWidget(stack_side)
        images.setStretchFactor(0, 1)
        images.setStretchFactor(1, 1)
        images.setSizes([10_000, 10_000])

        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right.addWidget(images)
        right.addWidget(self.trace_panel)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        self.vertical_splitter = right

        controls = self._build_controls()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(320)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(scroll)
        layout.addWidget(right, 1)
        self.setCentralWidget(central)
        self.hover_label = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.hover_label)

        self.draw_panel.sigLassoFinished.connect(self._on_lasso)
        self.draw_panel.sigActiveRoiEdited.connect(self._on_active_edited)
        self.draw_panel.sigClicked.connect(self._on_image_clicked)
        self.stack_panel.sigClicked.connect(self._on_image_clicked)
        self.draw_panel.sigHover.connect(lambda p: self._on_hover(self.draw_panel, p))
        self.stack_panel.sigHover.connect(lambda p: self._on_hover(self.stack_panel, p))
        self.trace_panel.sigFrameChanged.connect(self.set_frame)
        self.trace_panel.sigAnnotationToggled.connect(self.set_annotation_shown)
        self.frame_slider.valueChanged.connect(self.set_frame)
        self.frame_spin.valueChanged.connect(self.set_frame)

    def _build_controls(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        stack_box = QtWidgets.QGroupBox("Stack")
        g = QtWidgets.QGridLayout(stack_box)
        self.stack_label = QtWidgets.QLabel("No stack loaded")
        self.stack_label.setWordWrap(True)
        self.fps_spin = QtWidgets.QDoubleSpinBox()
        self.fps_spin.setRange(0.0, 10000.0)
        self.fps_spin.setDecimals(3)
        self.fps_spin.setSpecialValueText("not set")
        self.fps_spin.setSuffix(" Hz")
        self.units_combo = QtWidgets.QComboBox()
        self.units_combo.addItems(["frames", "seconds"])
        g.addWidget(self.stack_label, 0, 0, 1, 2)
        g.addWidget(QtWidgets.QLabel("Frame rate"), 1, 0)
        g.addWidget(self.fps_spin, 1, 1)
        g.addWidget(QtWidgets.QLabel("X axis"), 2, 0)
        g.addWidget(self.units_combo, 2, 1)
        v.addWidget(stack_box)

        draw_box = QtWidgets.QGroupBox("Draw view (left)")
        dv = QtWidgets.QVBoxLayout(draw_box)
        self.draw_source = QtWidgets.QComboBox()
        self.load_ref_btn = QtWidgets.QPushButton("Load reference image...")
        self.draw_display = DisplayControls("Display")
        dv.addWidget(self.draw_source)
        dv.addWidget(self.load_ref_btn)
        dv.addWidget(self.draw_display)
        v.addWidget(draw_box)

        stack_view_box = QtWidgets.QGroupBox("Stack view (right)")
        sv = QtWidgets.QVBoxLayout(stack_view_box)
        self.stack_source = QtWidgets.QComboBox()
        for key, label in STACK_SOURCES:
            self.stack_source.addItem(label, key)
        self.stack_display = DisplayControls("Display")
        self.link_views = QtWidgets.QCheckBox("Link zoom / pan with draw view")
        self.link_views.setChecked(True)
        sv.addWidget(self.stack_source)
        sv.addWidget(self.stack_display)
        sv.addWidget(self.link_views)
        v.addWidget(stack_view_box)

        roi_box = QtWidgets.QGroupBox("ROIs")
        rv = QtWidgets.QVBoxLayout(roi_box)
        draw_row = QtWidgets.QHBoxLayout()
        self.draw_specific_btn = QtWidgets.QPushButton("Draw specific")
        self.draw_nonspecific_btn = QtWidgets.QPushButton("Draw non-specific")
        for btn, kind in ((self.draw_specific_btn, KIND_SPECIFIC), (self.draw_nonspecific_btn, KIND_NONSPECIFIC)):
            btn.setCheckable(True)
            r, gg, b = KIND_COLORS[kind]
            btn.setStyleSheet(f"QPushButton:checked {{ background-color: rgb({r},{gg},{b}); color: black; }}")
            draw_row.addWidget(btn)
        self.import_btn = QtWidgets.QPushButton("Load ROIs from pickle...")
        self.import_btn.setToolTip(
            "Copy the ROI outlines from another experiment's pickle onto this stack; "
            "traces are extracted from this stack"
        )
        self.keep_drawing = QtWidgets.QCheckBox("Stay in draw mode after each ROI")
        self.show_labels = QtWidgets.QCheckBox("Show ROI names on images")
        self.show_labels.setChecked(True)
        self.roi_table = QtWidgets.QTableWidget(0, 3)
        self.roi_table.setHorizontalHeaderLabels(["Name", "Type", "Pixels"])
        self.roi_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.roi_table.verticalHeader().setVisible(False)
        self.roi_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.roi_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.roi_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.roi_table.setMinimumHeight(220)
        edit_row = QtWidgets.QHBoxLayout()
        self.rename_btn = QtWidgets.QPushButton("Rename")
        self.delete_btn = QtWidgets.QPushButton("Delete")
        edit_row.addWidget(self.rename_btn)
        edit_row.addWidget(self.delete_btn)
        hint = QtWidgets.QLabel(
            "Drag to draw (Esc cancels). Click an ROI in either view to select it; "
            "drag its handles or body in the draw view to edit."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        rv.addLayout(draw_row)
        rv.addWidget(self.import_btn)
        rv.addWidget(self.keep_drawing)
        rv.addWidget(self.show_labels)
        rv.addWidget(self.roi_table, 1)
        rv.addLayout(edit_row)
        rv.addWidget(hint)
        v.addWidget(roi_box, 1)

        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        self.units_combo.currentTextChanged.connect(self._on_units_changed)
        self.draw_source.currentIndexChanged.connect(lambda _i: self._on_draw_source_changed())
        self.load_ref_btn.clicked.connect(self.load_reference)
        self.draw_display.sigChanged.connect(lambda: self._apply_display(self.draw_panel, self.draw_display))
        self.stack_source.currentIndexChanged.connect(lambda _i: self._on_stack_source_changed())
        self.stack_display.sigChanged.connect(lambda: self._apply_display(self.stack_panel, self.stack_display))
        self.link_views.toggled.connect(self._on_link_toggled)
        self.draw_specific_btn.toggled.connect(lambda on: self._on_draw_toggled(KIND_SPECIFIC, on))
        self.draw_nonspecific_btn.toggled.connect(lambda on: self._on_draw_toggled(KIND_NONSPECIFIC, on))
        self.show_labels.toggled.connect(self._on_labels_toggled)
        self.roi_table.itemSelectionChanged.connect(self._on_table_selection)
        self.roi_table.itemDoubleClicked.connect(lambda _i: self.rename_active())
        self.import_btn.clicked.connect(lambda: self.import_rois())
        self.rename_btn.clicked.connect(self.rename_active)
        self.delete_btn.clicked.connect(self.delete_active)
        return w

    def _action(self, menu: QtWidgets.QMenu, text: str, slot=None, shortcut=None) -> QtGui.QAction:
        action = QtGui.QAction(text, self)
        if slot is not None:
            action.triggered.connect(lambda _checked=False: slot())
        if shortcut is not None:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        menu.addAction(action)
        return action

    def _build_menus(self) -> None:
        keys = QtGui.QKeySequence.StandardKey
        file_menu = self.menuBar().addMenu("&File")
        self._action(file_menu, "Open &experiment...", self.open_experiment, "Ctrl+Shift+O")
        self._action(file_menu, "&Load stack...", self.open_stack, keys.Open)
        file_menu.addSeparator()
        self.ref_action = self._action(file_menu, "Load &reference image...", self.load_reference)
        self.import_action = self._action(file_menu, "Load ROIs from &pickle...", self.import_rois)
        file_menu.addSeparator()
        self.save_action = self._action(file_menu, "&Save", self.save, keys.Save)
        file_menu.addSeparator()
        self._action(file_menu, "&Quit", self.close, keys.Quit)

        view_menu = self.menuBar().addMenu("&View")
        self._action(view_menu, "Reset zoom", self._reset_zoom, "Ctrl+0")

        tools = self.menuBar().addMenu("&Tools")
        self.heatmap_action = self._action(tools, "&Annotations...", self.open_heatmaps, "Ctrl+H")
        self.trace_action = self._action(tools, "&Trace processing...", self.open_trace_processing, "Ctrl+T")

        help_menu = self.menuBar().addMenu("&Help")
        self._action(help_menu, "About", self._about)

        QtGui.QShortcut(QtGui.QKeySequence("Escape"), self, self._cancel_draw)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Delete, self, self.delete_active)

    def _set_enabled(self, on: bool) -> None:
        for w in (
            self.fps_spin, self.units_combo, self.draw_source, self.load_ref_btn, self.draw_display,
            self.stack_source, self.stack_display, self.draw_specific_btn, self.draw_nonspecific_btn,
            self.roi_table, self.rename_btn, self.delete_btn, self.frame_slider, self.frame_spin,
            self.import_btn,
        ):
            w.setEnabled(on)
        for a in (self.ref_action, self.import_action, self.save_action, self.heatmap_action, self.trace_action):
            a.setEnabled(on)

    # dirty / title -------------------------------------------------------------
    def mark_dirty(self) -> None:
        self.dirty = True
        self._update_title()

    def _update_title(self) -> None:
        name = self.stack_path.name if self.stack_path else "no stack"
        self.setWindowTitle(f"Raw Cell Inspection {__version__} - {name}{' *' if self.dirty else ''}")

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        answer = QtWidgets.QMessageBox.question(
            self,
            "Unsaved changes",
            "Save changes to the ROI file first?",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QtWidgets.QMessageBox.StandardButton.Discard

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        if self._confirm_discard():
            ev.accept()
        else:
            ev.ignore()

    # opening an experiment / a stack ------------------------------------------
    def _ask_stack_path(self, title: str, start: str) -> Path | None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, title, start, f"TIFF stacks ({_file_filter(TIF_SUFFIXES)})"
        )
        return Path(path) if path else None

    def open_experiment(self, pkl_path: str | None = None) -> None:
        """Open an experiment pickle and the stack it belongs to."""
        if not self._confirm_discard():
            return
        if not pkl_path:
            start = dialog_start(self.settings, "last_experiment", "last_dir")
            pkl_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open experiment", start, f"Raw Cell Inspection files (*{PICKLE_SUFFIX});;Pickle files (*.pkl)"
            )
            if not pkl_path:
                return
        pkl_path = Path(pkl_path)
        remember_opened(self.settings, experiment=pkl_path)
        try:
            doc = load_document(pkl_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open experiment", f"Could not read {pkl_path.name}:\n{exc}")
            return
        stack_path = next((p for p in stack_candidates(pkl_path, doc) if p.exists()), None)
        if stack_path is None:
            name = doc["stack"].get("filename", "the stack")
            QtWidgets.QMessageBox.information(
                self,
                "Locate stack",
                f"{name} was not found next to {pkl_path.name}. Please locate the stack of this experiment.",
            )
            stack_path = self._ask_stack_path(f"Locate {name}", str(pkl_path.parent))
            if stack_path is None:
                return
        self.open_stack(str(stack_path), pkl_path=pkl_path, doc=doc, confirm=False)

    def open_stack(
        self,
        path: str | None = None,
        pkl_path: Path | None = None,
        doc: dict | None = None,
        confirm: bool = True,
    ) -> None:
        """Load a stack together with its experiment pickle.

        Without pkl_path the pickle is <stack>_rci.pkl next to the stack (created on first save).
        """
        if confirm and not self._confirm_discard():
            return
        if not path:
            path = self._ask_stack_path("Load stack", dialog_start(self.settings, "last_stack", "last_dir"))
            if path is None:
                return
        path = Path(path)
        pkl = Path(pkl_path) if pkl_path else pickle_path_for(path)
        remember_opened(self.settings, stack=path, experiment=pkl)

        def load(_progress):
            stack = load_stack(path)
            signature = stack_signature(path, stack)
            loaded_doc, doc_error = doc, None
            if loaded_doc is None and pkl.exists():
                try:
                    loaded_doc = load_document(pkl)
                except Exception as exc:
                    doc_error = str(exc)
            return stack, signature, loaded_doc, doc_error

        def loaded(result):
            stack, signature, loaded_doc, doc_error = result
            self._after_load(path, stack, signature, loaded_doc, doc_error, pkl)

        run_task(self, f"Opening {path.name}...", load, loaded, cancellable=False)

    def _after_load(self, path, stack, signature, doc, doc_error, pkl: Path) -> None:
        recompute_traces = False
        if doc_error:
            QtWidgets.QMessageBox.warning(
                self,
                "Could not read ROI file",
                f"{pkl.name} could not be read and was ignored:\n{doc_error}\n\n"
                "Saving will keep the old file as a .bak copy.",
            )
        if doc is not None:
            problems = signature_mismatch(doc["stack"], signature)
            same_fov = tuple(doc["stack"].get("shape", ()))[1:] == tuple(signature["shape"])[1:]
            if problems and not same_fov:
                QtWidgets.QMessageBox.warning(
                    self,
                    "ROI file does not match",
                    f"{pkl.name} was made for a stack with a different image size "
                    f"({'; '.join(problems)}). It was not loaded. Saving will keep it as a .bak copy.",
                )
                doc = None
            elif problems:
                QtWidgets.QMessageBox.information(
                    self,
                    "Stack changed",
                    f"The stack differs from when {pkl.name} was saved ({'; '.join(problems)}).\n"
                    "ROIs are kept; their traces and the summary images are recomputed, "
                    "and heatmaps must be recomputed.",
                )
                doc["stack"] = dict(signature)
                doc["summary"] = None
                n = signature["shape"][0]
                for hm in doc["heatmaps"]:
                    hm["image"] = None
                    hm["computed_ranges"] = None
                    hm["ranges"] = [[min(a, n - 1), min(b, n - 1)] for a, b in hm["ranges"]]
                recompute_traces = True
        is_new = doc is None
        if is_new:
            doc = new_document(signature)

        if doc["summary"] is None or recompute_traces:
            def compute(progress):
                summary = stack_summary(stack, lambda f: progress(f * (0.8 if recompute_traces else 1.0)))
                if recompute_traces:
                    rois = doc["rois"]
                    for i, roi in enumerate(rois):
                        roi["trace"] = roi_trace(stack, roi["mask"])
                        progress(0.8 + 0.2 * (i + 1) / max(1, len(rois)))
                return summary

            def computed(summary):
                doc["summary"] = summary
                self._install(path, stack, doc, pkl, dirty=recompute_traces)

            run_task(self, "Computing mean image and field-of-view trace...", compute, computed)
        else:
            self._install(path, stack, doc, pkl, dirty=False)

    def _install(self, path: Path, stack: np.ndarray, doc: dict, pkl: Path, dirty: bool) -> None:
        self.stack_path = path
        self.stack = stack
        self.doc = doc
        self.pkl_path = pkl
        self.dirty = dirty
        self.selected = {KIND_SPECIFIC: None, KIND_NONSPECIFIC: None}
        self.active_id = None
        self._cancel_draw()
        n = stack.shape[0]
        idx = np.unique(np.linspace(0, n - 1, min(n, 20)).round().astype(int))
        self._frame_sample = np.asarray(stack[idx])

        self._set_enabled(True)
        h, w = stack.shape[1:]
        self.stack_label.setText(
            f"<b>{path.name}</b><br>{n} frames, {w} x {h} px, {stack.dtype}<br>"
            f"Experiment file: {self.pkl_path.name}{'' if self.pkl_path.exists() else ' (new)'}"
        )
        self.stack_label.setToolTip(f"Stack: {path}\nExperiment file: {self.pkl_path}")
        for wdg in (self.frame_slider, self.frame_spin):
            wdg.blockSignals(True)
            wdg.setRange(0, n - 1)
            wdg.setValue(0)
            wdg.blockSignals(False)
        self.frame = 0
        self.trace_panel.set_n_frames(n)

        display = doc.get("display", {})
        self.fps_spin.blockSignals(True)
        self.fps_spin.setValue(float(doc.get("fps") or 0.0))
        self.fps_spin.blockSignals(False)
        self.units_combo.blockSignals(True)
        self.units_combo.setCurrentText(doc.get("x_units", "frames"))
        self.units_combo.blockSignals(False)
        self.draw_display.restore(display.get("draw", {}))
        self.stack_display.restore(display.get("stack", {}))

        self.draw_panel.set_image(None)
        self.stack_panel.set_image(None)
        self._refresh_draw_sources(select=display.get("draw_source"))
        stack_key = display.get("stack_source", "frame")
        self.stack_source.blockSignals(True)
        self.stack_source.setCurrentIndex(max(0, self.stack_source.findData(stack_key)))
        self.stack_source.blockSignals(False)
        self._on_stack_source_changed()
        self._on_link_toggled(self.link_views.isChecked())
        self._apply_time_axis()
        self.set_frame(0, force=True)
        self._refresh_rois()
        self._update_title()
        if self.heatmap_editor is not None:
            self.heatmap_editor.refresh()
        if self.trace_window is not None:
            self.trace_window.refresh()
        self.refresh_annotation_overlays()
        n_rois = len(doc["rois"])
        self.statusBar().showMessage(
            f"Loaded {path.name}" + (f" with {n_rois} saved ROI(s)" if n_rois else ""), 6000
        )

    # saving ------------------------------------------------------------------
    def save(self) -> bool:
        if self.doc is None or self.pkl_path is None:
            return False
        previous = self.doc.get("display") or {}
        self.doc["display"] = {
            "draw_source": self.draw_source.currentData(),
            "stack_source": self.stack_source.currentData(),
            "draw": self.draw_display.state(),
            "stack": self.stack_display.state(),
            "shown_annotations": list(previous.get("shown_annotations") or []),
        }
        self.doc["stack"]["path"] = str(self.stack_path.resolve())
        self.doc["stack"]["relpath"] = relpath_or_none(self.stack_path, self.pkl_path.parent)
        try:
            save_document(self.pkl_path, self.doc)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", f"Could not write {self.pkl_path}:\n{exc}")
            return False
        self.dirty = False
        self._update_title()
        self.statusBar().showMessage(f"Saved {self.pkl_path}", 6000)
        return True

    # reference image ---------------------------------------------------------
    def load_reference(self, path: str | None = None) -> None:
        if self.stack is None:
            return
        if not path:
            start = str(self.stack_path.parent)
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load reference (averaged) image", start, f"Images ({_file_filter(IMAGE_SUFFIXES)})"
            )
            if not path:
                return
        try:
            image, note = load_reference_image(path)
        except (StackError, OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "Reference image", str(exc))
            return
        if image.shape != tuple(self.stack.shape[1:]):
            QtWidgets.QMessageBox.critical(
                self,
                "Reference image",
                f"The image is {image.shape[1]} x {image.shape[0]} px but the stack is "
                f"{self.stack.shape[2]} x {self.stack.shape[1]} px. They must match so the ROIs line up.",
            )
            return
        self.doc["reference_image"] = {
            "image": image,
            "source_path": str(Path(path).resolve()),
            "source_relpath": relpath_or_none(path, self.stack_path.parent),
            "note": note,
            "loaded": datetime.now().isoformat(timespec="seconds"),
        }
        self._refresh_draw_sources(select="reference")
        self.mark_dirty()
        if note:
            self.statusBar().showMessage(f"Reference image loaded ({note})", 6000)

    # display sources ---------------------------------------------------------
    def _refresh_draw_sources(self, select: str | None = None) -> None:
        doc = self.doc
        current = select or self.draw_source.currentData()
        self.draw_source.blockSignals(True)
        self.draw_source.clear()
        ref = doc.get("reference_image")
        if ref is not None:
            self.draw_source.addItem(f"Reference: {Path(ref['source_path']).name}", "reference")
        self.draw_source.addItem("Stack mean", "mean")
        self.draw_source.addItem("Stack max", "max")
        for hm in doc["heatmaps"]:
            if hm.get("image") is not None:
                suffix = "" if heatmap_state(hm) == "up to date" else " (out of date)"
                self.draw_source.addItem(f"Heatmap: {hm['name']}{suffix}", f"heatmap:{hm['name']}")
        index = self.draw_source.findData(current) if current else -1
        if index < 0:
            index = self.draw_source.findData("reference" if ref is not None else "mean")
        self.draw_source.setCurrentIndex(max(0, index))
        self.draw_source.blockSignals(False)
        self._on_draw_source_changed(keep_if_same=True)

    def _draw_source_image(self, key: str) -> np.ndarray | None:
        doc = self.doc
        if key == "reference" and doc.get("reference_image"):
            return doc["reference_image"]["image"]
        if key in ("mean", "max"):
            return doc["summary"][f"{key}_image"]
        heatmap = self._heatmap_for_key(key)
        return heatmap.get("image") if heatmap else None

    def _heatmap_for_key(self, key: str | None) -> dict | None:
        if not key or not key.startswith("heatmap:"):
            return None
        name = key.split(":", 1)[1]
        return next((hm for hm in self.doc["heatmaps"] if hm["name"] == name), None)

    def _on_draw_source_changed(self, keep_if_same: bool = False) -> None:
        if self.doc is None:
            return
        key = self.draw_source.currentData()
        self.draw_panel.set_title(f"Draw view - {self.draw_source.currentText()}", self._draw_source_path(key))
        self.doc.setdefault("display", {})["draw_source"] = key
        image = self._draw_source_image(key)
        if keep_if_same and image is self.draw_panel.image:
            return
        self.draw_panel.set_image(image)
        if image is not None:
            self.draw_display.set_data_range(image)
        self._apply_display(self.draw_panel, self.draw_display)

    def _draw_source_path(self, key: str | None) -> str:
        ref = self.doc.get("reference_image")
        if key == "reference" and ref:
            return ref["source_path"]
        return str(self.stack_path) if self.stack_path else ""

    def _on_stack_source_changed(self) -> None:
        if self.stack is None:
            return
        key = self.stack_source.currentData()
        if key == "frame":
            self.stack_display.set_data_range(self._frame_sample.astype(np.float32))
        else:
            self.stack_display.set_data_range(self.doc["summary"][f"{key}_image"])
        self._update_stack_image()

    def _update_stack_image(self) -> None:
        key = self.stack_source.currentData()
        if key == "frame":
            image = np.asarray(self.stack[self.frame])
            title = f"Stack view - frame {self.frame}"
        else:
            image = self.doc["summary"][f"{key}_image"]
            title = f"Stack view - {self.stack_source.currentText()}"
        self.stack_panel.set_image(image, levels=self.stack_display.levels())
        self._apply_display(self.stack_panel, self.stack_display)
        self.stack_panel.set_title(title, str(self.stack_path))

    def _apply_display(self, panel: ImagePanel, controls: DisplayControls) -> None:
        panel.set_lut(controls.lut_name())
        panel.set_levels(*controls.levels())

    def _on_link_toggled(self, on: bool) -> None:
        vb = self.stack_panel.vb
        vb.setXLink(self.draw_panel.vb if on else None)
        vb.setYLink(self.draw_panel.vb if on else None)

    def _reset_zoom(self) -> None:
        self.draw_panel.reset_zoom()
        self.stack_panel.reset_zoom()

    def _on_hover(self, panel: ImagePanel, pos) -> None:
        image = panel.image
        if pos is None or image is None:
            self.hover_label.setText("")
            return
        c, r = int(np.floor(pos[0])), int(np.floor(pos[1]))
        if 0 <= r < image.shape[0] and 0 <= c < image.shape[1]:
            self.hover_label.setText(f"x={c}  y={r}  value={float(image[r, c]):.4g}")
        else:
            self.hover_label.setText("")

    # time axis / frames ------------------------------------------------------
    def _on_fps_changed(self, value: float) -> None:
        if self.doc is None:
            return
        self.doc["fps"] = float(value) if value > 0 else None
        self._apply_time_axis()
        self.mark_dirty()

    def _on_units_changed(self, units: str) -> None:
        if self.doc is None:
            return
        self.doc["x_units"] = units
        if units == "seconds" and not self.doc.get("fps"):
            self.statusBar().showMessage("Set the frame rate to show the x axis in seconds.", 6000)
        self._apply_time_axis()
        self.mark_dirty()

    def _apply_time_axis(self) -> None:
        self.trace_panel.set_time_axis(self.doc.get("fps"), self.doc.get("x_units", "frames"))
        if self.heatmap_editor is not None:
            self.heatmap_editor.apply_time_axis()
        if self.trace_window is not None:
            self.trace_window.apply_time_axis()
        self._update_frame_label()

    def _update_frame_label(self) -> None:
        fps = self.doc.get("fps") if self.doc else None
        self.frame_time.setText(f"{self.frame / fps:.2f} s" if fps else "")

    def set_frame(self, frame: int, force: bool = False) -> None:
        if self.stack is None:
            return
        frame = int(min(max(frame, 0), self.stack.shape[0] - 1))
        if frame == self.frame and not force:
            return
        self.frame = frame
        for wdg in (self.frame_slider, self.frame_spin):
            wdg.blockSignals(True)
            wdg.setValue(frame)
            wdg.blockSignals(False)
        self.trace_panel.set_frame(frame)
        if self.stack_source.currentData() == "frame":
            self._update_stack_image()
        self._update_frame_label()

    # ROI drawing / editing ---------------------------------------------------
    def _on_draw_toggled(self, kind: str, on: bool) -> None:
        other = self.draw_nonspecific_btn if kind == KIND_SPECIFIC else self.draw_specific_btn
        if on:
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
            self.draw_kind = kind
            self._set_active(None)
            self.statusBar().showMessage(f"Drag on the draw view to outline a {KIND_LABEL[kind].lower()} ROI.")
        elif self.draw_kind == kind:
            self.draw_kind = None
            self.statusBar().clearMessage()
        self.draw_panel.set_draw_kind(self.draw_kind)

    def _cancel_draw(self) -> None:
        for btn in (self.draw_specific_btn, self.draw_nonspecific_btn):
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
        self.draw_kind = None
        self.draw_panel.set_draw_kind(None)

    def _roi_geometry(self, vertices: np.ndarray) -> tuple[np.ndarray, dict | None]:
        h, w = self.stack.shape[1:]
        vertices = clip_vertices(vertices, h, w)
        return vertices, polygon_mask(vertices, h, w)

    def _on_lasso(self, stroke: np.ndarray) -> None:
        if self.draw_kind is None or self.doc is None:
            return
        kind = self.draw_kind
        vertices, mask = self._roi_geometry(simplify_stroke(stroke))
        if mask is None or len(vertices) < 3:
            self.statusBar().showMessage("ROI too small: it does not contain any pixel centre.", 5000)
            return
        numbers = self.doc["next_roi_number"]
        name = f"{KIND_PREFIX[kind]}{numbers[kind]}"
        numbers[kind] += 1
        now = datetime.now().isoformat(timespec="seconds")
        roi = {
            "id": max((r["id"] for r in self.doc["rois"]), default=0) + 1,
            "name": name,
            "kind": kind,
            "vertices": vertices,
            "mask": mask,
            "n_pixels": int(mask["mask"].sum()),
            "trace": roi_trace(self.stack, mask),
            "created": now,
            "modified": now,
        }
        self.doc["rois"].append(roi)
        self.mark_dirty()
        if self.keep_drawing.isChecked():
            self.selected[kind] = roi["id"]
            self._refresh_rois()
        else:
            self._cancel_draw()
            self.select_roi(roi["id"])
        self.statusBar().showMessage(f"Added {name} ({roi['n_pixels']} px)", 4000)

    def _on_active_edited(self, vertices: np.ndarray) -> None:
        roi = self._roi_by_id(self.active_id)
        if roi is None:
            return
        vertices, mask = self._roi_geometry(vertices)
        if mask is None:
            self.statusBar().showMessage("Edit rejected: the ROI would contain no pixels.", 5000)
            self._set_active(roi["id"])
            return
        roi["vertices"] = vertices
        roi["mask"] = mask
        roi["n_pixels"] = int(mask["mask"].sum())
        roi["trace"] = roi_trace(self.stack, mask)
        roi["modified"] = datetime.now().isoformat(timespec="seconds")
        self.mark_dirty()
        self._refresh_rois(rebuild_active=False)

    # ROI selection -----------------------------------------------------------
    def _roi_by_id(self, roi_id: int | None) -> dict | None:
        if roi_id is None or self.doc is None:
            return None
        return next((r for r in self.doc["rois"] if r["id"] == roi_id), None)

    def _on_image_clicked(self, x: float, y: float) -> None:
        if self.doc is None or self.draw_kind is not None:
            return
        hits = [r for r in self.doc["rois"] if contains_point(r["vertices"], x, y)]
        if hits:
            smallest = min(hits, key=lambda r: polygon_area(r["vertices"]))
            self.select_roi(smallest["id"])
        else:
            self._set_active(None)
            self._refresh_rois()

    def select_roi(self, roi_id: int) -> None:
        roi = self._roi_by_id(roi_id)
        if roi is None:
            return
        self.selected[roi["kind"]] = roi_id
        self._set_active(roi_id)
        self._refresh_rois(rebuild_active=False)

    def _set_active(self, roi_id: int | None) -> None:
        self.active_id = roi_id
        roi = self._roi_by_id(roi_id)
        if roi is None:
            self.active_id = None
            self.draw_panel.set_active_roi(None)
        else:
            self.draw_panel.set_active_roi(roi["vertices"], roi["kind"])

    def _on_table_selection(self) -> None:
        rows = self.roi_table.selectionModel().selectedRows()
        if not rows:
            return
        roi_id = self.roi_table.item(rows[0].row(), 0).data(QtCore.Qt.ItemDataRole.UserRole)
        if roi_id != self.active_id:
            self._cancel_draw()
            self.select_roi(roi_id)

    def _on_labels_toggled(self, on: bool) -> None:
        self.draw_panel.show_labels = on
        self.stack_panel.show_labels = on
        self._refresh_rois(rebuild_active=False)

    def _refresh_rois(self, rebuild_active: bool = True) -> None:
        rois = self.doc["rois"] if self.doc else []
        for kind, roi_id in self.selected.items():
            if self._roi_by_id(roi_id) is None:
                self.selected[kind] = None
        if rebuild_active:
            self._set_active(self.active_id)
        selected_ids = {i for i in self.selected.values() if i is not None}
        self.draw_panel.set_rois(rois, selected_ids, hidden_id=self.active_id)
        self.stack_panel.set_rois(rois, selected_ids)
        self._refresh_table()
        for kind in (KIND_SPECIFIC, KIND_NONSPECIFIC):
            roi = self._roi_by_id(self.selected[kind])
            self.trace_panel.set_trace(kind, roi["name"] if roi else None, roi["trace"] if roi else None)
        self.rename_btn.setEnabled(self.active_id is not None)
        self.delete_btn.setEnabled(self.active_id is not None)
        if self.trace_window is not None and self.trace_window.isVisible():
            self.trace_window.on_rois_changed()

    def _refresh_table(self) -> None:
        rois = self.doc["rois"] if self.doc else []
        self.roi_table.blockSignals(True)
        self.roi_table.setRowCount(len(rois))
        active_row = -1
        for row, roi in enumerate(rois):
            color = QtGui.QColor(*KIND_COLORS[roi["kind"]])
            values = (roi["name"], KIND_LABEL[roi["kind"]], str(roi["n_pixels"]))
            for col, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, roi["id"])
                if col == 1:
                    item.setForeground(color)
                font = item.font()
                font.setBold(roi["id"] in self.selected.values())
                item.setFont(font)
                self.roi_table.setItem(row, col, item)
            if roi["id"] == self.active_id:
                active_row = row
        if active_row >= 0:
            self.roi_table.selectRow(active_row)
        else:
            self.roi_table.clearSelection()
        self.roi_table.blockSignals(False)

    def rename_active(self) -> None:
        roi = self._roi_by_id(self.active_id)
        if roi is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename ROI", "Name:", text=roi["name"])
        name = name.strip()
        if not ok or not name or name == roi["name"]:
            return
        if any(r["name"] == name for r in self.doc["rois"]):
            QtWidgets.QMessageBox.warning(self, "Rename ROI", f"An ROI named '{name}' already exists.")
            return
        roi["name"] = name
        self.mark_dirty()
        self._refresh_rois(rebuild_active=False)

    def delete_active(self) -> None:
        roi = self._roi_by_id(self.active_id)
        if roi is None:
            return
        answer = QtWidgets.QMessageBox.question(self, "Delete ROI", f"Delete {roi['name']}?")
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.doc["rois"].remove(roi)
        self._set_active(None)
        self.mark_dirty()
        self._refresh_rois()

    def import_rois(self, path: str | None = None, mode: str | None = None) -> None:
        """Copy ROI outlines from another experiment's pickle; masks and traces come from this stack.

        mode: "add" or "replace" existing ROIs (asked when None and ROIs exist).
        """
        if self.doc is None:
            return
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load ROIs from pickle", self.settings.value("last_roi_dir", ""),
                f"Raw Cell Inspection files (*{PICKLE_SUFFIX});;Pickle files (*.pkl)",
            )
            if not path:
                return
        path = Path(path)
        self.settings.setValue("last_roi_dir", str(path.parent))
        if self.pkl_path is not None and path.resolve() == self.pkl_path.resolve():
            QtWidgets.QMessageBox.information(self, "Load ROIs", "That is this experiment's own file.")
            return
        try:
            source = load_document(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load ROIs", f"Could not read {path.name}:\n{exc}")
            return
        src_rois = source.get("rois", [])
        if not src_rois:
            QtWidgets.QMessageBox.information(self, "Load ROIs", f"{path.name} contains no ROIs.")
            return

        h, w = self.stack.shape[1:]
        src_shape = tuple(source.get("stack", {}).get("shape", ()))[1:]
        if src_shape and src_shape != (h, w) and mode is None:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Different image size",
                f"The ROIs were drawn on {src_shape[1]} x {src_shape[0]} px images, this stack is "
                f"{w} x {h} px. Import anyway? ROIs are placed at the same pixel coordinates and "
                "clipped at the image border.",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        if mode is None and self.doc["rois"]:
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Load ROIs")
            box.setText(f"This experiment already has {len(self.doc['rois'])} ROI(s).")
            add = box.addButton("Add to them", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            replace = box.addButton("Replace them", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            box.exec()
            clicked = box.clickedButton()
            mode = "add" if clicked is add else "replace" if clicked is replace else None
            if mode is None:
                return
        if mode == "replace":
            self.doc["rois"] = []
            self.doc["next_roi_number"] = {KIND_SPECIFIC: 1, KIND_NONSPECIFIC: 1}
            self.selected = {KIND_SPECIFIC: None, KIND_NONSPECIFIC: None}
            self._set_active(None)

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            added, skipped = self._add_imported_rois(src_rois, path)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.mark_dirty()
        self._refresh_rois()
        message = f"Loaded {added} ROI(s) from {path.name}"
        if skipped:
            message += f"; {skipped} skipped (no pixels inside this image)"
        self.statusBar().showMessage(message, 8000)

    def _add_imported_rois(self, src_rois: list[dict], source_path: Path) -> tuple[int, int]:
        rois = self.doc["rois"]
        numbers = self.doc["next_roi_number"]
        names = {r["name"] for r in rois}
        next_id = max((r["id"] for r in rois), default=0) + 1
        now = datetime.now().isoformat(timespec="seconds")
        added = skipped = 0
        for src in src_rois:
            kind = src.get("kind", KIND_SPECIFIC)
            if kind not in KIND_PREFIX:
                kind = KIND_SPECIFIC
            vertices, mask = self._roi_geometry(np.asarray(src["vertices"], dtype=float))
            if mask is None:
                skipped += 1
                continue
            name = src.get("name") or ""
            match = re.fullmatch(rf"{KIND_PREFIX[kind]}(\d+)", name)
            if match:
                numbers[kind] = max(numbers[kind], int(match.group(1)) + 1)
            if not name or name in names:
                while f"{KIND_PREFIX[kind]}{numbers[kind]}" in names:
                    numbers[kind] += 1
                name = f"{KIND_PREFIX[kind]}{numbers[kind]}"
                numbers[kind] += 1
            names.add(name)
            rois.append({
                "id": next_id,
                "name": name,
                "kind": kind,
                "vertices": vertices,
                "mask": mask,
                "n_pixels": int(mask["mask"].sum()),
                "trace": roi_trace(self.stack, mask),
                "created": now,
                "modified": now,
                "imported_from": {"path": str(source_path.resolve()), "name": src.get("name")},
            })
            next_id += 1
            added += 1
        return added, skipped

    # heatmaps ----------------------------------------------------------------
    def open_heatmaps(self) -> None:
        if self.doc is None:
            return
        if self.heatmap_editor is None:
            self.heatmap_editor = HeatmapEditor(self)
        self.heatmap_editor.refresh()
        self.heatmap_editor.show()
        self.heatmap_editor.raise_()
        self.heatmap_editor.activateWindow()

    def open_trace_processing(self) -> None:
        if self.doc is None:
            return
        if self.trace_window is None:
            self.trace_window = TraceProcessingWindow(self)
        self.trace_window.refresh()
        self.trace_window.show()
        self.trace_window.raise_()
        self.trace_window.activateWindow()

    def on_heatmaps_edited(self) -> None:
        self.mark_dirty()
        self._refresh_draw_sources()
        self.refresh_annotation_overlays()

    def annotation_choices(self) -> list[tuple[str, bool]]:
        if self.doc is None:
            return []
        shown = set((self.doc.get("display") or {}).get("shown_annotations") or [])
        return [(hm["name"], hm["name"] in shown) for hm in self.doc.get("heatmaps") or []]

    def set_annotation_shown(self, name: str, shown: bool) -> None:
        if self.doc is None:
            return
        display = self.doc.setdefault("display", {})
        names = [str(item) for item in (display.get("shown_annotations") or [])]
        if shown and name not in names:
            names.append(name)
        elif not shown and name in names:
            names.remove(name)
        else:
            return
        display["shown_annotations"] = names
        self.mark_dirty()
        self.refresh_annotation_overlays()

    def refresh_annotation_overlays(self) -> None:
        if self.doc is None:
            return
        self.trace_panel.set_annotation_choices(self.annotation_choices())
        self.trace_panel.set_annotation_spans(annotation_spans(self.doc))
        if self.trace_window is not None:
            self.trace_window.sync_annotations()

    def on_heatmap_computed(self, name: str) -> None:
        self.mark_dirty()
        self._refresh_draw_sources(select=f"heatmap:{name}")

    def _about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About Raw Cell Inspection",
            f"Raw Cell Inspection {__version__}<br>"
            "Draw guided ROIs on .tif movie stacks and collect their traces.<br>"
            '<a href="https://github.com/RasHerlo/Raw_Cell_Inspection">github.com/RasHerlo/Raw_Cell_Inspection</a>',
        )
