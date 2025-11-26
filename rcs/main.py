"""Entry point for launching the PyQt RCS GUI."""

from __future__ import annotations

import importlib


def _ensure_pyqt5() -> None:
    """Validate that PyQt5 is installed before importing GUI modules."""

    try:
        importlib.import_module("PyQt5")
    except ImportError as exc:  # pragma: no cover - startup guard
        hint = "PyQt5 is required to launch the GUI. Install it with `pip install PyQt5`."

        if "libGL" in str(exc):
            hint += (
                " The import error references `libGL`, which usually means your system "
                "OpenGL libraries are missing. Install a package such as `libgl1`/"
                "`mesa-libGL` from your OS package manager before launching the GUI."
            )

        raise SystemExit(f"{hint}\nOriginal error: {exc}")


def main() -> None:
    _ensure_pyqt5()

    from .gui import run_app

    run_app()


if __name__ == "__main__":
    main()
