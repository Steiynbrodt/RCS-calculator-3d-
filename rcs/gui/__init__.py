"""PyQt5 GUI package for the RCS simulator."""

from __future__ import annotations


def run_app() -> None:
    """Launch the PyQt5 application.

    Importing PyQt5-heavy modules lazily avoids ImportError during simple
    package import when the dependency is missing. This provides a clearer
    message when users run ``python RCS.py`` without installing requirements.
    """

    try:
        from .main_window import run_app as _run_app
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyQt5 is required to run the GUI. Install dependencies with "
            "'pip install -r requirements.txt'."
        ) from exc

    _run_app()


__all__ = ["run_app"]
