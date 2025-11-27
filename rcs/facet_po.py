"""Physical-optics style facet summation inspired by PyPOFacets.

The implementation uses field summation across illuminated facets rather than
energy addition to preserve phase information. It remains a single-bounce,
scalar model without shadowing or diffraction.
"""

from __future__ import annotations

import numpy as np
import trimesh


def facet_rcs(
    mesh: trimesh.Trimesh,
    reflectivity: float,
    frequency_hz: float,
    directions: np.ndarray,
) -> np.ndarray:
    """Estimate (bi/static) RCS by summing complex facet responses.

    Parameters
    ----------
    mesh:
        Target mesh with per-facet normals and areas.
    reflectivity:
        Scalar reflectivity coefficient applied to every facet (0–1).
    frequency_hz:
        Radar frequency in Hz.
    directions:
        Array of unit vectors pointing from target towards the receiver for each
        look direction (shape: ``[N, 3]``).

    Notes
    -----
    * Incident direction is assumed to be ``-k_hat`` for monostatic views.
    * No multiple scattering, occlusion, or edge diffraction are modelled.
    * Amplitude is proportional to facet area and cosine of the incidence angle
      with a ``1/lambda`` scaling to preserve wavelength dependence.
    """

    c = 3e8
    wavelength = c / max(frequency_hz, 1e-9)
    k = 2.0 * np.pi / wavelength

    normals = mesh.face_normals  # (F, 3)
    areas = mesh.area_faces  # (F,)
    centers = mesh.triangles.mean(axis=1)  # (F, 3)

    rcs = np.zeros(len(directions), dtype=float)

    for idx, k_hat in enumerate(directions):
        inc_dir = -k_hat
        cos_theta = normals @ inc_dir
        illuminated = cos_theta > 0.0
        if not np.any(illuminated):
            continue
        cos_vals = cos_theta[illuminated]
        phase = centers[illuminated] @ k_hat
        field = reflectivity * areas[illuminated] * cos_vals / wavelength
        field *= np.exp(1j * k * phase)
        total_field = np.sum(field)
        rcs[idx] = np.abs(total_field) ** 2

    return np.maximum(rcs, 1e-20)


__all__ = ["facet_rcs"]
