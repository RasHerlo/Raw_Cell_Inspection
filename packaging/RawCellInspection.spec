# PyInstaller spec for the one-file Windows build:
#     pyinstaller packaging/RawCellInspection.spec --noconfirm
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent

# pyqtgraph picks its Qt-binding-specific UI templates at runtime.
hidden = collect_submodules(
    "pyqtgraph",
    filter=lambda name: not name.startswith(("pyqtgraph.examples", "pyqtgraph.opengl", "pyqtgraph.jupyter")),
)

a = Analysis(
    [str(ROOT / "packaging" / "launch.py")],
    pathex=[str(ROOT)],
    hiddenimports=hidden,
    excludes=["PyQt5", "PyQt6", "PySide2", "tkinter", "matplotlib", "IPython", "OpenGL", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="RawCellInspection",
    # RCI_CONSOLE=1 gives a build that prints tracebacks, for debugging.
    console=bool(os.environ.get("RCI_CONSOLE")),
    upx=False,
)
