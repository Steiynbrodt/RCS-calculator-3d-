"""Core RCS simulation routines and data containers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np
import trimesh

from .math_utils import direction_grid
from .physics import MIN_ENERGY, build_ray_intersector, frequency_loss


BAND_DEFAULTS = {
    "L": (1.0e9, 2.0e9),
    "S": (2.0e9, 4.0e9),
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

        for fi, freq_hz in enumerate(freqs):
            if self._stop_requested:
                break
            freq_ghz = freq_hz / 1e9
            loss_per_reflection = frequency_loss(freq_ghz)
            rcs_lin = np.zeros(len(directions), dtype=float)

            for idx, (origin, direction) in enumerate(zip(origins, directions)):
                if self._stop_requested:
                    break
                energy = 1.0
                contribution = 0.0
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

                    alignment = np.dot(reflect_dir, -ray_dir)
                    if alignment > 0.95:
                        contribution += energy * mesh.area_faces[face_index] * (np.dot(normal, -ray_dir)) ** 2

                    energy *= material.reflectivity * loss_per_reflection
                    if energy < MIN_ENERGY:
                        break

                    ray_origin = hit + 1e-4 * reflect_dir
                    ray_dir = reflect_dir
                rcs_lin[idx] = max(contribution, 1e-10)

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
        )


__all__ = [
    "BAND_DEFAULTS",
    "FrequencySweep",
    "SimulationSettings",
    "Material",
    "SimulationResult",
    "RCSEngine",
]
