"""Entry point for launching the PyQt RCS GUI."""

from __future__ import annotations

from .gui import run_app


def main() -> None:
    try:
        run_app()
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(exc)
        print("Install required packages with 'pip install -r requirements.txt'.")
        return


if __name__ == "__main__":
    main()
