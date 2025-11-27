"""Physical-optics style facet summation inspired by PyPOFacets.

The implementation uses field summation across illuminated facets and adds
heuristic edge diffraction and corner enhancements. It remains a single-bounce
scalar model that ignores multiple scattering and full UTD behaviour but keeps
phase information, optional shadowing, and wavelength scaling intact.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .diffraction import build_sharp_edges, corner_field, edge_diffraction_field
from .physics import build_ray_intersector

# Toggle expensive shadowing and diffraction logic if required.
SHADOWING_ENABLED = True
EDGE_DIFFRACTION_ENABLED = True


# ---------------------------------------------------------------------------

def _shadow_mask(
    mesh: trimesh.Trimesh, centers: np.ndarray, directions: np.ndarray, face_indices: np.ndarray
) -> np.ndarray:
    """Return a boolean mask indicating which facets are visible to the radar."""

    if not hasattr(mesh, "ray"):
        mesh.ray = build_ray_intersector(mesh)

    origins = centers + directions * (mesh.bounding_sphere.primitive.radius * 4.0 + 1.0)
    try:
        hit_faces = mesh.ray.intersects_first(origins, -directions)
    except Exception:
        return np.ones(len(centers), dtype=bool)

    visible = hit_faces == face_indices
    visible[np.isnan(hit_faces)] = False
    visible[hit_faces < 0] = False
    return visible


def facet_rcs(
    mesh: trimesh.Trimesh,
    reflectivity: float,
    freq_hz: float,
    directions: np.ndarray,
) -> np.ndarray:
    """Estimate (bi/static) RCS by summing complex facet responses.

    This is a simplified PO-inspired model with optional shadowing and heuristic
    edge/corner contributions. No multiple scattering or full diffraction is
    modelled; the aim is to retain phase-driven lobes/nulls and richer
    phenomenology than pure specular returns.

    Parameters
    ----------
    mesh:
        Target mesh with per-facet normals and areas.
    reflectivity:
        Scalar reflectivity coefficient applied to every facet (0–1).
    freq_hz:
        Radar frequency in Hz.
    directions:
        Array of unit vectors pointing from target towards the receiver for each
        look direction (shape: ``[N, 3]`` or a single ``(3,)`` vector).

    Returns
    -------
    np.ndarray
        Linear RCS (m^2) per look direction.
    """

    directions_arr = np.asarray(directions, dtype=float)
    if directions_arr.size == 0:
        return np.zeros(0, dtype=float)
    if directions_arr.ndim == 1:
        if directions_arr.size != 3:
            raise ValueError(
                "directions must be a sequence of 3D unit vectors shaped (N, 3)"
            )
        directions_arr = directions_arr.reshape(1, 3)
    elif directions_arr.shape[-1] != 3:
        raise ValueError(
            "directions must be a sequence of 3D unit vectors shaped (N, 3)"
        )

    if mesh is None or len(mesh.faces) == 0:
        return np.zeros(len(directions_arr), dtype=float)

    c = 3e8
    wavelength = c / max(freq_hz, 1e-9)
    k = 2.0 * np.pi / wavelength

    normals = mesh.face_normals  # (F, 3)
    areas = mesh.area_faces  # (F,)
    centers = mesh.triangles.mean(axis=1)  # (F, 3)
    edge_catalog = build_sharp_edges(mesh) if EDGE_DIFFRACTION_ENABLED else []

    rcs = np.zeros(len(directions_arr), dtype=float)

    for idx, k_hat in enumerate(directions_arr):
        inc_dir = -k_hat
        cos_theta = normals @ inc_dir
        illuminated = cos_theta > 0.0
        if not np.any(illuminated):
            continue

        illum_indices = np.where(illuminated)[0]
        local_dirs = np.repeat(inc_dir.reshape(1, 3), len(illum_indices), axis=0)
        visible_mask = np.ones(len(illum_indices), dtype=bool)
        if SHADOWING_ENABLED:
            visible_mask = _shadow_mask(
                mesh, centers[illum_indices], local_dirs, illum_indices
            )
        if not np.any(visible_mask):
            facet_field = 0.0 + 0.0j
        else:
            idx_visible = illum_indices[visible_mask]
            cos_vals = cos_theta[idx_visible]
            phase = centers[idx_visible] @ k_hat
            facet_field = (
                reflectivity
                * areas[idx_visible]
                * cos_vals
                * np.exp(1j * k * phase)
            ).sum()

        corner = corner_field(
            k_hat=k_hat,
            normals=normals,
            areas=areas,
            centers=centers,
            k=k,
            illuminated_mask=illuminated,
        )

        edge_field = 0.0 + 0.0j
        if EDGE_DIFFRACTION_ENABLED and len(edge_catalog) > 0:
            edge_field = edge_diffraction_field(
                edge_catalog, k_hat, k, mesh if SHADOWING_ENABLED else None
            )

        total_field = facet_field + edge_field + corner
        rcs[idx] = np.abs(total_field) ** 2

    return np.maximum(rcs, 1e-20)


__all__ = ["facet_rcs"]
