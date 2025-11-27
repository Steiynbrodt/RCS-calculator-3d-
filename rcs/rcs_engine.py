"""Core RCS simulation routines and data containers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import trimesh

from .facet_po import facet_rcs
from .math_utils import direction_grid, frequency_loss
from .physics import MIN_ENERGY, build_ray_intersector


BAND_DEFAULTS = {
    "L": (1.0e9, 2.0e9),
    "S": (2.0e9, 4.0e9),
    "C": (4.0e9, 8.0e9),
    "X": (8.0e9, 12.0e9),
}


def compute_wavelength(freq_hz: float) -> float:
    """Return the free-space wavelength [m] for the supplied frequency."""

    return 3e8 / freq_hz


def compute_wavenumber(freq_hz: float) -> float:
    """Return the free-space wavenumber ``k = 2*pi/lambda`` [rad/m]."""

    lambda_ = compute_wavelength(freq_hz)
    return 2.0 * np.pi / lambda_


def monostatic_phase(k: float, path_length: float) -> complex:
    """Return the monostatic propagation phase for a two-way path.

    Parameters
    ----------
    k:
        Free-space wavenumber (``2*pi/lambda``) in rad/m.
    path_length:
        One-way geometric path length from the radar to the current
        interaction point in metres.
    """

    return k * 2.0 * path_length


def to_dbsm(rcs_lin: np.ndarray) -> np.ndarray:
    """Convert linear RCS values (m^2) to dBsm with floor protection."""

    return 10 * np.log10(np.maximum(rcs_lin, 1e-20))


def _select_reflectivity(material: Material, polarization: str) -> float:
    """Return the effective scalar reflectivity for the requested polarisation."""

    pol = polarization.lower()
    if pol.startswith("h") and material.reflectivity_h is not None:
        return material.reflectivity_h
    if pol.startswith("v") and material.reflectivity_v is not None:
        return material.reflectivity_v
    return material.reflectivity


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
    tx_yaw_deg: float | None = None
    tx_elev_deg: float | None = None

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
    """Material properties used by the simplified RCS engine.

    Reflectivity may be provided as a scalar or per-polarisation pair
    (``reflectivity_h`` and ``reflectivity_v``). When polarisation-specific
    values are omitted, ``reflectivity`` is used for both H and V.
    """

    name: str
    epsilon_real: float
    epsilon_imag: float
    conductivity: float
    reflectivity: float
    reflectivity_h: float | None = None
    reflectivity_v: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class SimulationResult:
    """Container for RCS outputs.

    RCS values are stored in dBsm (decibels relative to 1 m²). Frequencies are
    expressed in Hertz and the ``band`` field records the band preset used to
    pick default GHz ranges when a specific frequency is not provided.
    """

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
    """High-level interface for computing RCS.

    The engine keeps public inputs/outputs backwards compatible with the GUI
    while internally using coherent field summation for the ray tracer and a
    wavelength-aware facet physical optics path. All RCS values are linear in
    m² until converted to ``rcs_dbsm`` in the returned :class:`SimulationResult`.
    """

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

        # Bistatic support: if a dedicated transmitter direction is provided
        # use it for illumination while the az/el grid represents receive look
        # angles. In the monostatic default, tx_dir equals the receive direction.
        tx_dir = None
        if settings.tx_yaw_deg is not None and settings.tx_elev_deg is not None:
            yaw_rad = np.radians(settings.tx_yaw_deg)
            el_rad = np.radians(settings.tx_elev_deg)
            tx_dir = np.array(
                [
                    np.cos(el_rad) * np.cos(yaw_rad),
                    np.cos(el_rad) * np.sin(yaw_rad),
                    np.sin(el_rad),
                ]
            )

        rx_dirs = dirs.reshape(-1, 3)

        distance = mesh.bounding_sphere.primitive.radius * 6.0 + 1.0
        if tx_dir is None:
            origins = mesh.bounding_sphere.center + distance * (-rx_dirs)
            ray_dirs = rx_dirs
        else:
            origins = np.repeat(
                mesh.bounding_sphere.center + distance * (-tx_dir)[None, :],
                len(rx_dirs),
                axis=0,
            )
            ray_dirs = np.repeat(tx_dir[None, :], len(rx_dirs), axis=0)

        rcs_all = np.zeros((len(freqs), len(el), len(az)), dtype=float)
        doppler_all = np.zeros(len(freqs), dtype=float) if settings.target_speed_mps else None

        reflectivity = _select_reflectivity(material, settings.polarization)

        for fi, freq_hz in enumerate(freqs):
            if self._stop_requested:
                break
            freq_ghz = freq_hz / 1e9
            if doppler_all is not None:
                doppler_all[fi] = 2 * settings.target_speed_mps * freq_hz / 3e8
            loss_per_reflection = frequency_loss(freq_ghz)
            lambda_ = compute_wavelength(freq_hz)
            k = compute_wavenumber(freq_hz)
            if settings.method == "facet_po":
                rcs_lin = facet_rcs(
                    mesh,
                    reflectivity,
                    freq_hz,
                    rx_dirs,
                    tx_direction=tx_dir,
                )
            else:
                rcs_lin = np.zeros(len(rx_dirs), dtype=float)
                for idx, (origin, ray_direction, rx_dir) in enumerate(
                    zip(origins, ray_dirs, rx_dirs)
                ):
                    if self._stop_requested:
                        break
                    energy = 1.0
                    field = 0.0 + 0.0j
                    path_length = 0.0
                    ray_origin = origin
                    ray_dir = ray_direction
                    for _ in range(settings.max_reflections):
                        locs, _, tri_idx = mesh.ray.intersects_location(
                            np.array([ray_origin]), np.array([ray_dir]), multiple_hits=False
                        )
                        if len(locs) == 0:
                            break
                        hit = locs[0]
                        face_index = tri_idx[0]
                        segment_len = np.linalg.norm(hit - ray_origin)
                        path_length += segment_len
                        normal = mesh.face_normals[face_index]
                        reflect_dir = ray_dir - 2 * np.dot(ray_dir, normal) * normal
                        reflect_dir /= np.linalg.norm(reflect_dir)

                        alignment = np.dot(reflect_dir, -rx_dir)
                        cos_theta = np.dot(normal, -ray_dir)
                        if alignment > 0.95 and cos_theta > 0:
                            amp = (energy * mesh.area_faces[face_index] * cos_theta) / lambda_
                            field += amp * np.exp(1j * monostatic_phase(k, path_length))

                        energy *= reflectivity * loss_per_reflection
                        if energy < MIN_ENERGY:
                            break

                        ray_origin = hit + 1e-4 * reflect_dir
                        ray_dir = reflect_dir
                    rcs_lin[idx] = max(np.abs(field) ** 2, 1e-20)

            rcs_db = to_dbsm(rcs_lin.reshape(len(el), len(az)))
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


__all__ = [
    "BAND_DEFAULTS",
    "FrequencySweep",
    "SimulationSettings",
    "Material",
    "SimulationResult",
    "RCSEngine",
    "compute_wavelength",
    "compute_wavenumber",
    "to_dbsm",
]
