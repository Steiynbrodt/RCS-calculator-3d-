"""Quick visual validation helpers for the simplified RCS/NCTR models."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from .facet_po import facet_rcs
from .rcs_engine import RCSEngine, Material as _Material, SimulationSettings, to_dbsm
from .math_utils import rotation_matrix


def _plate(size: float = 1.0) -> trimesh.Trimesh:
    """Return a simple square plate in the XY plane centred on the origin."""

    half = size / 2.0
    vertices = np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _box(size: float = 1.0) -> trimesh.Trimesh:
    """Create a simple PEC cube for quick checks."""

    return trimesh.creation.box((size, size, size))


def engine_material(mat_dict: dict) -> _Material:
    """Utility to convert a material dictionary into ``Material`` dataclass."""

    return _Material(
        name=mat_dict.get("name", "material"),
        epsilon_real=mat_dict.get("epsilon_real", 1.0),
        epsilon_imag=mat_dict.get("epsilon_imag", 0.0),
        conductivity=mat_dict.get("conductivity", 0.0),
        reflectivity=mat_dict.get("reflectivity", 1.0),
    )


def compare_plate_patterns(freq_hz: float = 10e9) -> None:
    """Plot ray-based vs facet PO RCS of a flat plate at fixed elevation."""

    engine = RCSEngine()
    mesh = _plate(1.0)
    settings = SimulationSettings(
        band="X",
        polarization="H",
        max_reflections=2,
        frequency_hz=freq_hz,
        azimuth_step=5.0,
        elevation_start=0.0,
        elevation_stop=0.0,
        elevation_step=1.0,
    )
    material = dict(name="PEC", epsilon_real=1.0, epsilon_imag=0.0, conductivity=1e7, reflectivity=1.0)
    result = engine.compute(mesh, engine_material(material), settings)
    az = result.azimuth_deg
    rcs_ray = result.rcs_dbsm[0, 0]

    dirs = np.stack((np.cos(np.radians(az)), np.sin(np.radians(az)), np.zeros_like(az)), axis=1)
    rcs_po = to_dbsm(facet_rcs(mesh, 1.0, freq_hz, dirs))

    plt.figure()
    plt.plot(az, rcs_ray, label="Ray (coherent)")
    plt.plot(az, rcs_po, label="Facet PO")
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("RCS (dBsm)")
    plt.title("Flat plate broadside check")
    plt.legend()
    plt.grid(True)
    plt.show()


def compare_box_patterns(freq_hz: float = 10e9) -> None:
    """Compare ray and facet patterns for a cube."""

    engine = RCSEngine()
    mesh = _box(1.0)
    settings = SimulationSettings(
        band="X",
        polarization="H",
        max_reflections=2,
        frequency_hz=freq_hz,
        azimuth_step=5.0,
        elevation_start=0.0,
        elevation_stop=0.0,
        elevation_step=1.0,
    )
    material = dict(name="PEC", epsilon_real=1.0, epsilon_imag=0.0, conductivity=1e7, reflectivity=1.0)
    result = engine.compute(mesh, engine_material(material), settings)
    az = result.azimuth_deg
    rcs_ray = result.rcs_dbsm[0, 0]

    dirs = np.stack((np.cos(np.radians(az)), np.sin(np.radians(az)), np.zeros_like(az)), axis=1)
    rcs_po = to_dbsm(facet_rcs(mesh, 1.0, freq_hz, dirs))

    plt.figure()
    plt.plot(az, rcs_ray, label="Ray (coherent)")
    plt.plot(az, rcs_po, label="Facet PO")
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("RCS (dBsm)")
    plt.title("Cube pattern sanity check")
    plt.legend()
    plt.grid(True)
    plt.show()


def nctr_rotor_comparison() -> None:
    """Show spectrogram difference between rigid spin and rotating sub-mesh."""

    base = trimesh.creation.cylinder(radius=0.5, height=1.0, sections=16)
    blade = trimesh.creation.box((0.8, 0.05, 0.02))
    blade.apply_translation((0.4, 0.0, 0.0))
    blades = trimesh.util.concatenate(
        [blade.copy().apply_rotation(matrix=rotation_matrix(0, 0, ang)) for ang in (0, 90, 180, 270)]
    )
    mesh = trimesh.util.concatenate([base, blades])

    rotating = {
        "blades": {
            "indices": np.arange(len(base.vertices), len(mesh.vertices)),
            "rpm": 3000.0,
            "axis": np.array([0.0, 0.0, 1.0]),
        }
    }
    from .nctr import simulate_nctr_signature

    t_all, f_all, spec_all, _ = simulate_nctr_signature(mesh, {"reflectivity": 1.0}, 10.0, pulses=256)
    t_rot, f_rot, spec_rot, _ = simulate_nctr_signature(
        mesh,
        {"reflectivity": 1.0},
        10.0,
        pulses=256,
        rotating_groups=rotating,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    im0 = axes[0].imshow(
        spec_all, aspect="auto", origin="lower", extent=[t_all[0], t_all[-1], f_all[0], f_all[-1]]
    )
    axes[0].set_title("Rigid body spin")
    im1 = axes[1].imshow(
        spec_rot, aspect="auto", origin="lower", extent=[t_rot[0], t_rot[-1], f_rot[0], f_rot[-1]]
    )
    axes[1].set_title("Rotating sub-mesh")
    for ax in axes:
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Doppler (Hz)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    plt.tight_layout()
    plt.show()


__all__ = ["compare_plate_patterns", "compare_box_patterns", "nctr_rotor_comparison"]
