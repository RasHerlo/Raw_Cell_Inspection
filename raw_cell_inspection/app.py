"""Application entry point."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    import pyqtgraph as pg
    from PySide6 import QtWidgets

    from raw_cell_inspection.gui.main_window import MainWindow

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName("Raw Cell Inspection")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    if len(argv) > 1:
        window.open_stack(argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
