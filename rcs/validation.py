"""Lightweight validation helpers for quick sanity checks.

These routines are not formal unit tests but provide reproducible, small
scenarios to visually compare ray-traced and facet-PO RCS as well as the
micro-Doppler model with/without rotating sub-meshes.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .rcs_engine import Material, RCSEngine, SimulationSettings
from .nctr import simulate_nctr_signature


def _pec_material() -> Material:
    return Material(
        name="PEC",
        epsilon_real=1.0,
        epsilon_imag=0.0,
        conductivity=1e7,
        reflectivity=0.98,
        reflectivity_h=None,
        reflectivity_v=None,
    )


def compare_flat_plate(size: float = 1.0, freq_hz: float = 10e9) -> dict[str, np.ndarray]:
    """Compare ray and facet-PO responses for a PEC flat plate.

    Returns a dictionary with azimuth angles in degrees and RCS in dBsm for each
    method at 0 deg elevation. A broadside peak and off-boresight roll-off
    indicate a healthy simulation chain.
    """

    mesh = trimesh.creation.box(extents=(size, size, 0.02))
    engine = RCSEngine()
    settings_ray = SimulationSettings(
        band="X",
        polarization="H",
        max_reflections=2,
        method="ray",
        frequency_hz=freq_hz,
        elevation_start=0.0,
        elevation_stop=0.0,
        elevation_step=1.0,
        azimuth_step=2.0,
    )
    settings_facet = SimulationSettings(
        band="X",
        polarization="H",
        max_reflections=1,
        method="facet_po",
        frequency_hz=freq_hz,
        elevation_start=0.0,
        elevation_stop=0.0,
        elevation_step=1.0,
        azimuth_step=2.0,
    )
    mat = _pec_material()

    ray_res = engine.compute(mesh, mat, settings_ray)
    facet_res = engine.compute(mesh, mat, settings_facet)

    az = ray_res.azimuth_deg
    ray_db = ray_res.rcs_dbsm[0, 0]
    facet_db = facet_res.rcs_dbsm[0, 0]

    return {"az_deg": az, "ray_dbsm": ray_db, "facet_dbsm": facet_db}


def compare_box(freq_hz: float = 10e9) -> dict[str, np.ndarray]:
    """Contrast ray and facet-PO patterns for a simple PEC cube."""

    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    engine = RCSEngine()
    settings = SimulationSettings(
        band="X",
        polarization="H",
        max_reflections=2,
        method="ray",
        frequency_hz=freq_hz,
        elevation_start=0.0,
        elevation_stop=0.0,
        elevation_step=1.0,
        azimuth_step=5.0,
    )
    mat = _pec_material()
    ray_res = engine.compute(mesh, mat, settings)

    settings_po = settings
    settings_po = SimulationSettings(**settings_po.__dict__)
    settings_po.method = "facet_po"
    facet_res = engine.compute(mesh, mat, settings_po)

    return {
        "az_deg": ray_res.azimuth_deg,
        "ray_dbsm": ray_res.rcs_dbsm[0, 0],
        "facet_dbsm": facet_res.rcs_dbsm[0, 0],
    }


def compare_nctr_rotors(mesh: trimesh.Trimesh | None = None) -> dict[str, np.ndarray]:
    """Compare whole-body spin against a rotating sub-component."""

    if mesh is None:
        mesh = trimesh.creation.box(extents=(1.0, 0.5, 0.5))

    rot_indices = np.arange(len(mesh.vertices) // 2)
    times, freqs, base_spec, _ = simulate_nctr_signature(mesh, {"reflectivity": 1.0}, 10.0)
    _, _, group_spec, _, _, _, _ = simulate_nctr_signature(
        mesh,
        {"reflectivity": 1.0},
        10.0,
        rotating_groups={"rotor": {"indices": rot_indices, "rpm": 1200.0, "axis": [0, 0, 1]}},
        return_range_doppler=True,
    )

    return {"times": times, "freqs": freqs, "baseline": base_spec, "grouped": group_spec}


__all__ = ["compare_flat_plate", "compare_box", "compare_nctr_rotors"]
