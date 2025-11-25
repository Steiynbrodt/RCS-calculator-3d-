"""PyQt5 GUI package for the RCS simulator."""

from __future__ import annotations


def run_app() -> None:
    """Launch the PyQt5 application with clearer dependency errors.

    Importing PyQt5-heavy modules lazily avoids ImportError during simple
    package import when the dependency is missing. This provides a clearer
    message when users run ``python RCS.py`` without installing requirements
    or required system libraries such as ``libGL``.
    """

    try:
        from .main_window import run_app as _run_app
    except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
        help_text = [
            "PyQt5 is required to run the GUI. Install dependencies with 'pip install -r requirements.txt'.",
            "If PyQt5 is already installed, ensure required Qt/OpenGL system libraries (e.g., libGL) are present.",
        ]

        if "libGL.so.1" in str(exc):
            help_text.append(
                "A system OpenGL library is missing. Install libGL (e.g., 'apt-get install -y libgl1') on Debian/Ubuntu-based systems."
            )

        help_text.append(f"Original error: {exc}")
        raise ImportError("\n".join(help_text)) from exc

    _run_app()


__all__ = ["run_app"]
