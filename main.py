"""Entry point for launching the Radar RCS GUI."""

from __future__ import annotations

import tkinter as tk

from .gui import RadarGUI


def main() -> None:
    root = tk.Tk()
    RadarGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
