"""Physical-optics style facet summation inspired by PyPOFacets."""

from __future__ import annotations

import numpy as np
import trimesh


def facet_rcs(
    mesh: trimesh.Trimesh,
    reflectivity: float,
    frequency_hz: float,
    directions: np.ndarray,
    *,
    tx_direction: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate RCS by coherently summing illuminated facet fields.

    Parameters
    ----------
    mesh:
        Target mesh with pre-computed face normals, areas and triangle vertices.
    reflectivity:
        Scalar reflectivity applied uniformly to all facets.
    frequency_hz:
        Radar carrier frequency in Hertz.
    directions:
        Array of shape ``(M, 3)`` whose rows point from target to radar
        (receive directions). In the monostatic case these also set the
        incidence direction.
    tx_direction:
        Optional unit vector for the transmitter direction. When provided the
        incidence direction is fixed to ``-tx_direction`` while ``directions``
        describe receive look angles (bistatic preview).

    Notes
    -----
    * Scalar, isotropic reflectivity is assumed (no shadowing or multiple
      scattering).
    * The complex field from each illuminated facet is proportional to
      ``area * cos(theta) / lambda`` with a spatial phase term ``exp(j k r)``.
    * No edge diffraction or masking is included; this is a first-order PO
      approximation intended for interactive use.
    """

    wavelength = 3e8 / frequency_hz
    k = 2.0 * np.pi / wavelength

    normals = mesh.face_normals
    areas = mesh.area_faces
    centers = mesh.triangles.mean(axis=1)

    rx_dirs = np.asarray(directions, dtype=float)
    rx_dirs /= np.linalg.norm(rx_dirs, axis=1, keepdims=True)

    if tx_direction is None:
        incidence_dirs = -rx_dirs
    else:
        tx_dir = np.asarray(tx_direction, dtype=float)
        tx_dir /= np.linalg.norm(tx_dir)
        incidence_dirs = np.repeat((-tx_dir)[None, :], len(rx_dirs), axis=0)

    cos_theta = normals @ incidence_dirs.T  # shape (F, M)
    illumination = np.clip(cos_theta, 0.0, None)

    phase = np.exp(1j * k * (centers @ rx_dirs.T))

    field = reflectivity * (areas[:, None] / wavelength) * illumination * phase
    sigma = np.abs(field.sum(axis=0)) ** 2
    return np.maximum(sigma, 1e-20)


__all__ = ["facet_rcs"]
