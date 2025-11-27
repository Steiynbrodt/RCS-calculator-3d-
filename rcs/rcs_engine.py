"""Core RCS simulation routines and data containers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable, Sequence

import numpy as np
import trimesh

from .facet_po import facet_rcs
from .math_utils import direction_grid, frequency_loss
from .physics import (
    MIN_ENERGY,
    build_ray_intersector,
    monostatic_phase,
    wavenumber_from_frequency,
)


BAND_DEFAULTS = {
    "L": (1.0e9, 2.0e9),
    "S": (2.0e9, 4.0e9),
    "C": (4.0e9, 8.0e9),
    "X": (8.0e9, 12.0e9),
}


@dataclass(slots=True)
class FrequencySweep:
    """Frequency sweep definition."""

    start_hz: float
    stop_hz: float
    steps: int

    def as_array(self) -> np.ndarray:
        return np.linspace(self.start_hz, self.stop_hz, self.steps)


@dataclass(slots=True)
class SimulationSettings:
    """Parameters for an RCS simulation run."""

    band: str
    polarization: str
    max_reflections: int
    method: str = "ray"  # "ray" | "facet_po"
    frequency_hz: float | None = None
    sweep: FrequencySweep | None = None
    azimuth_start: float = 0.0
    azimuth_stop: float = 360.0
    azimuth_step: float = 5.0
    elevation_start: float = -90.0
    elevation_stop: float = 90.0
    elevation_step: float = 5.0
    elevation_slice_deg: float | None = None
    azimuth_slice_deg: float | None = None
    target_speed_mps: float = 0.0
    radar_profile: str | None = None

    def frequencies(self) -> np.ndarray:
        if self.frequency_hz is not None:
            return np.array([self.frequency_hz])
        if self.sweep is not None:
            return self.sweep.as_array()
        default = BAND_DEFAULTS.get(self.band, BAND_DEFAULTS["S"])
        return np.array([np.mean(default)])

    def azimuths(self) -> np.ndarray:
        return np.arange(self.azimuth_start, self.azimuth_stop + 1e-6, self.azimuth_step)

    def elevations(self) -> np.ndarray:
        return np.arange(self.elevation_start, self.elevation_stop + 1e-6, self.elevation_step)


@dataclass(slots=True)
class Material:
    """Material properties used by the simplified RCS engine."""

    name: str
    epsilon_real: float
    epsilon_imag: float
    conductivity: float
    reflectivity: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class SimulationResult:
    """Container for RCS outputs."""

    band: str
    polarization: str
    frequencies_hz: np.ndarray
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    rcs_dbsm: np.ndarray  # shape (f, el, az)
    target_speed_mps: float
    radar_profile: str | None = None
    doppler_hz: np.ndarray | None = None

    def slice_for_elevation(self, elevation: float) -> tuple[np.ndarray, np.ndarray]:
        idx = int(np.argmin(np.abs(self.elevation_deg - elevation)))
        return self.azimuth_deg, self.rcs_dbsm[:, idx, :]


class RCSEngine:
    """High-level interface for computing RCS."""

    def __init__(self) -> None:
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def reset(self) -> None:
        self._stop_requested = False

    def compute(
        self,
        mesh: trimesh.Trimesh,
        material: Material,
        settings: SimulationSettings,
        *,
        progress: callable | None = None,
    ) -> SimulationResult:
        if mesh is None:
            raise ValueError("A mesh must be provided for simulation.")
        if not hasattr(mesh, "ray"):
            mesh.ray = build_ray_intersector(mesh)

        freqs = settings.frequencies()
        az = settings.azimuths()
        el = settings.elevations()
        dirs = direction_grid(az, el)

        distance = mesh.bounding_sphere.primitive.radius * 6.0 + 1.0
        origins = mesh.bounding_sphere.center + distance * (-dirs.reshape(-1, 3))
        directions = dirs.reshape(-1, 3)

        rcs_all = np.zeros((len(freqs), len(el), len(az)), dtype=float)
        doppler_all = np.zeros(len(freqs), dtype=float) if settings.target_speed_mps else None

        for fi, freq_hz in enumerate(freqs):
            if self._stop_requested:
                break
            freq_ghz = freq_hz / 1e9
            if doppler_all is not None:
                doppler_all[fi] = 2 * settings.target_speed_mps * freq_hz / 3e8
            loss_per_reflection = frequency_loss(freq_ghz)
            k = wavenumber_from_frequency(freq_hz)
            if settings.method == "facet_po":
                rcs_lin = facet_rcs(mesh, material.reflectivity, freq_hz, directions)
            else:
                rcs_lin = np.zeros(len(directions), dtype=float)
                for idx, (origin, direction) in enumerate(zip(origins, directions)):
                    if self._stop_requested:
                        break
                    energy = 1.0
                    path_length = 0.0
                    field = 0.0j
                    ray_origin = origin
                    ray_dir = direction
                    for _ in range(settings.max_reflections):
                        locs, _, tri_idx = mesh.ray.intersects_location(
                            np.array([ray_origin]), np.array([ray_dir]), multiple_hits=False
                        )
                        if len(locs) == 0:
                            break
                        hit = locs[0]
                        face_index = tri_idx[0]
                        normal = mesh.face_normals[face_index]
                        reflect_dir = ray_dir - 2 * np.dot(ray_dir, normal) * normal
                        reflect_dir /= np.linalg.norm(reflect_dir)

                        segment_len = np.linalg.norm(hit - ray_origin)
                        path_length += segment_len
                        alignment = np.dot(reflect_dir, -ray_dir)
                        if alignment > 0.95:
                            cos_theta = np.dot(normal, -ray_dir)
                            amp = energy * mesh.area_faces[face_index] * cos_theta
                            phase = monostatic_phase(k, path_length)
                            field += amp * np.exp(1j * phase)

                        energy *= material.reflectivity * loss_per_reflection
                        if energy < MIN_ENERGY:
                            break

                        ray_origin = hit + 1e-4 * reflect_dir
                        ray_dir = reflect_dir
                    rcs_lin[idx] = max(np.abs(field) ** 2, 1e-20)

            rcs_db = 10 * np.log10(rcs_lin.reshape(len(el), len(az)))
            rcs_all[fi] = rcs_db
            if progress:
                progress(int((fi + 1) / len(freqs) * 100))

        self._stop_requested = False
        return SimulationResult(
            band=settings.band,
            polarization=settings.polarization,
            frequencies_hz=freqs,
            azimuth_deg=az,
            elevation_deg=el,
            rcs_dbsm=rcs_all,
            target_speed_mps=settings.target_speed_mps,
            radar_profile=settings.radar_profile,
            doppler_hz=doppler_all,
        )


def _sanity_compare_methods() -> None:
    """Run a quick comparison between ray and facet PO outputs on a box."""

    import trimesh

    engine = RCSEngine()
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    material = Material("PEC", 1.0, 0.0, 1e7, 1.0)
    settings = SimulationSettings(
        band="X",
        polarization="HH",
        max_reflections=2,
        method="ray",
        frequency_hz=10e9,
        azimuth_start=0,
        azimuth_stop=90,
        azimuth_step=30,
        elevation_start=-30,
        elevation_stop=30,
        elevation_step=30,
    )
    ray_res = engine.compute(mesh, material, settings)
    settings_po = replace(settings, method="facet_po")
    facet_res = engine.compute(mesh, material, settings_po)
    print("Ray (dBsm):\n", ray_res.rcs_dbsm[0])
    print("Facet PO (dBsm):\n", facet_res.rcs_dbsm[0])


if __name__ == "__main__":
    _sanity_compare_methods()


__all__ = [
    "BAND_DEFAULTS",
    "FrequencySweep",
    "SimulationSettings",
    "Material",
    "SimulationResult",
    "RCSEngine",
]
