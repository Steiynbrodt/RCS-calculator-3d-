"""Mathematical helpers for geometry and coordinate transforms."""

from __future__ import annotations

import numpy as np


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return a combined rotation matrix (yaw, pitch, roll in degrees)."""
    y, p, r = np.radians([yaw, pitch, roll])

    # Aerospace convention: yaw (z), pitch (y), roll (x)
    r_yaw = np.array(
        [[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]]
    )
    r_pitch = np.array(
        [[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]]
    )
    r_roll = np.array(
        [[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]]
    )

    return r_yaw @ r_pitch @ r_roll


def direction_grid(azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    """Compute Cartesian direction vectors for azimuth/elevation grids.

    Parameters
    ----------
    azimuth_deg:
        1D array of azimuth angles in degrees.
    elevation_deg:
        1D array of elevation angles in degrees.

    Returns
    -------
    np.ndarray
        Array of shape (len(elevation_deg), len(azimuth_deg), 3) with unit
        vectors pointing toward each azimuth/elevation direction.
    """
    az_grid, el_grid = np.meshgrid(np.radians(azimuth_deg), np.radians(elevation_deg))
    dirs = np.stack(
        (
            np.cos(el_grid) * np.cos(az_grid),
            np.cos(el_grid) * np.sin(az_grid),
            np.sin(el_grid),
        ),
        axis=-1,
    )
    return dirs


def spherical_directions(
    az_steps: int,
    el_steps: int,
    *,
    center_az: float = 0.0,
    center_el: float = 0.0,
    beam_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create azimuth/elevation grids and corresponding direction vectors.

    When ``beam_width`` is provided, the sampling grid is narrowed around the
    specified center angles. This makes it possible to approximate a radar's
    field of view instead of tracing a full sphere, which is significantly
    faster for targeted perspectives.
    """

    az_full = beam_width is None or beam_width >= 360
    el_full = beam_width is None or beam_width >= 180

    if az_full:
        az = np.linspace(0, 360, az_steps, endpoint=False)
    else:
        half = beam_width / 2
        az = np.linspace(center_az - half, center_az + half, az_steps, endpoint=False)
        az = np.mod(az, 360)

    if el_full:
        el = np.linspace(-90, 90, el_steps)
    else:
        half = min(beam_width / 2, 90)
        el_low = np.clip(center_el - half, -90, 90)
        el_high = np.clip(center_el + half, -90, 90)
        el = np.linspace(el_low, el_high, el_steps)
    az_grid, el_grid = np.meshgrid(az, el)
    dirs = np.stack(
        (
            np.cos(np.radians(el_grid)) * np.cos(np.radians(az_grid)),
            np.cos(np.radians(el_grid)) * np.sin(np.radians(az_grid)),
            np.sin(np.radians(el_grid)),
        ),
        axis=-1,
    )
    return az, el, dirs


def frequency_loss(freq_ghz: float, min_loss: float = 0.2) -> float:
    """Calculate frequency-dependent attenuation per reflection."""
    return max(0.8 - 0.02 * (freq_ghz - 1), min_loss)


__all__ = ["rotation_matrix", "spherical_directions", "frequency_loss", "direction_grid"]
