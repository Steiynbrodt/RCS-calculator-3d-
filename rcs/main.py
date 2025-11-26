"""Entry point for launching the PyQt RCS GUI."""

from __future__ import annotations

import ctypes.util
import importlib.util


def _ensure_pyqt5() -> None:
    """Validate that PyQt5 is installed before importing GUI modules."""

    if importlib.util.find_spec("PyQt5") is None:
        msg = (
            "PyQt5 is required to launch the GUI. "
            "Install it with `pip install PyQt5` and try again."
        )
        raise SystemExit(msg)

    if ctypes.util.find_library("GL") is None:
        msg = (
            "System OpenGL libraries (libGL) are missing. "
            "Install a package such as `libgl1`/`mesa-libGL` from your OS package "
            "manager before launching the GUI."
        )
        raise SystemExit(msg)


def main() -> None:
    _ensure_pyqt5()

    from .gui import run_app

    run_app()


if __name__ == "__main__":
    main()
