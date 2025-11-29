"""PyQt5 GUI package for the RCS simulator."""
from .main_window import MainWindow, run_app

        # Cached grids / RCS for fast plotting
self._az_grid: Optional[np.ndarray] = None          # azimuth grid in radians
self._el_grid: Optional[np.ndarray] = None          # elevation grid in radians
self._az_deg_grid: Optional[np.ndarray] = None      # azimuth grid in degrees (heatmap)
self._el_deg_grid: Optional[np.ndarray] = None      # elevation grid in degrees (heatmap)

self._rcs_lin: Optional[np.ndarray] = None          # shape (F, E, A), linear σ
self._rcs_norm: Optional[np.ndarray] = None         # normalized per frequency
self._rcs_radius: Optional[np.ndarray] = None       # radius field for 3D plot

__all__ = ["MainWindow", "run_app"]
