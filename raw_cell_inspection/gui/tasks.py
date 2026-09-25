"""Run slow stack passes in a background thread behind a cancellable progress dialog."""

from __future__ import annotations

import traceback
from typing import Callable

from PySide6 import QtCore, QtWidgets

from raw_cell_inspection.analysis import Cancelled


class _Worker(QtCore.QThread):
    progressed = QtCore.Signal(int)
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, fn: Callable, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    def _progress(self, fraction: float) -> bool:
        self.progressed.emit(int(round(fraction * 1000)))
        return not self._cancel

    def run(self) -> None:
        try:
            result = self._fn(self._progress)
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:  # reported to the user in the GUI thread
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")
        else:
            self.succeeded.emit(result)


def run_task(
    parent: QtWidgets.QWidget,
    label: str,
    fn: Callable,
    on_done: Callable[[object], None],
    on_error: Callable[[str], None] | None = None,
    cancellable: bool = True,
) -> None:
    """fn(progress) runs off the GUI thread; progress(fraction) returns False when cancelled."""
    dialog = QtWidgets.QProgressDialog(label, "Cancel", 0, 1000, parent)
    if not cancellable:
        dialog.setCancelButton(None)
    dialog.setWindowTitle("Working")
    dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    dialog.setMinimumDuration(300)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setValue(0)

    worker = _Worker(fn, parent)
    worker.progressed.connect(dialog.setValue)
    dialog.canceled.connect(worker.request_cancel)
    worker.finished.connect(worker.deleteLater)

    def finish():
        dialog.close()
        dialog.deleteLater()

    def done(result):
        finish()
        on_done(result)

    def error(msg):
        finish()
        if on_error is not None:
            on_error(msg)
        else:
            QtWidgets.QMessageBox.critical(parent, "Error", msg.split("\n\n")[0])

    worker.succeeded.connect(done)
    worker.failed.connect(error)
    worker.cancelled.connect(finish)
    worker.start()
