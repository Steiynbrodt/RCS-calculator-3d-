"""Physical-optics style facet summation inspired by PyPOFacets."""

from __future__ import annotations

import numpy as np
import trimesh

from .physics import wavenumber_from_frequency


def facet_rcs(
    mesh: trimesh.Trimesh,
    reflectivity: float,
    frequency_hz: float,
    directions: np.ndarray,
) -> np.ndarray:
    """Estimate monostatic RCS using a coherent facet-based PO approximation.

    The model treats every illuminated triangle as an independent flat patch and
    coherently sums their complex field contributions. Shadowing, multiple
    scattering, diffraction, and edge effects are neglected to keep the
    computation lightweight. A unit-amplitude proportionality constant is used
    (no absolute calibration).

    Parameters
    ----------
    mesh:
        Target mesh with ``face_normals`` and ``area_faces`` populated.
    reflectivity:
        Scalar amplitude multiplier representing average material
        reflectivity/PEC-ness. Values in ``[0, 1]`` are expected.
    frequency_hz:
        Operating frequency in Hertz.
    directions:
        Array of shape ``(N, 3)`` giving unit vectors pointing **from radar to
        target** for each look angle.

    Returns
    -------
    np.ndarray
        Linear RCS values (``|field|^2``) per look direction.
    """

    k = wavenumber_from_frequency(frequency_hz)
    normals = mesh.face_normals  # (F, 3)
    areas = mesh.area_faces.reshape(-1, 1)  # (F, 1)
    centers = mesh.triangles_center  # (F, 3)

    # Radar-to-target direction -> target-to-radar for phase accumulation.
    k_hat = -directions  # (N, 3)

    # Cosine of incidence angle for each facet vs direction (F, N)
    cos_theta = normals @ (-k_hat).T
    illuminated = cos_theta > 0.0
    cos_theta = np.where(illuminated, cos_theta, 0.0)

    # Phase term per facet/look using patch centers.
    phase = k * (centers @ k_hat.T)
    field = reflectivity * areas * cos_theta * np.exp(1j * phase)
    total_field = field.sum(axis=0)
    rcs_lin = np.abs(total_field) ** 2
    return np.maximum(rcs_lin, 1e-20)


__all__ = ["facet_rcs"]


if __name__ == "__main__":
    # Lightweight sanity check: PEC square plate shows strongest return at broadside.
    square = trimesh.Trimesh(
        vertices=[
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        faces=[[0, 1, 2], [0, 2, 3]],
    )
    dirs = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.5, 0.8660254],
        [0.0, 0.707, 0.707],
    ])
    rcs_vals = facet_rcs(square, 1.0, 10e9, dirs)
    print("Flat-plate monotonic check:", rcs_vals)
    assert rcs_vals[0] >= rcs_vals[1] >= rcs_vals[2]
