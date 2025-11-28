"""Core RCS simulation routines and data containers.

The simplified engine keeps the public API stable for the GUI while adding
lightweight physics-inspired effects such as coherent field summation,
dual-polarization reflectivity selection, and optional bistatic geometry.
All distances are in metres and frequencies in Hz unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
import traceback
from typing import Iterable, Sequence
import concurrent.futures

import numpy as np
import trimesh

from .diffraction import build_sharp_edges, corner_field, edge_diffraction_field
from .facet_po import facet_rcs
from .math_utils import direction_grid, frequency_loss
from .physics import MIN_ENERGY, build_ray_intersector


BAND_DEFAULTS = {
    "L": (1.0e9, 2.0e9),
    "S": (2.0e9, 4.0e9),
    "C": (4.0e9, 8.0e9),
    "X": (8.0e9, 12.0e9),
}


def to_dbsm(rcs_lin: np.ndarray) -> np.ndarray:
    """Convert linear RCS (m^2) to decibel-square-metres.

    Values are clipped to ``1e-20`` to avoid log singularities.
    """

    return 10.0 * np.log10(np.maximum(rcs_lin, 1e-20))


@dataclass
class FrequencySweep:
    """Frequency sweep definition."""

    start_hz: float
    stop_hz: float
    steps: int

    def as_array(self) -> np.ndarray:
        return np.linspace(self.start_hz, self.stop_hz, self.steps)


@dataclass
class SimulationSettings:
    """Parameters for an RCS simulation run."""

    band: str
    polarization: str
    max_reflections: int
    method: str = "ray"  # "ray" | "facet_po"
    engines: List["EngineMount"] = field(default_factory=list)
    propellers: List["Propeller"] = field(default_factory=list)
    frequency_hz: Optional[float] = None
    sweep: Optional[FrequencySweep] = None
    azimuth_start: float = 0.0
    azimuth_stop: float = 360.0
    azimuth_step: float = 5.0
    elevation_start: float = -90.0
    elevation_stop: float = 90.0
    elevation_step: float = 5.0
    elevation_slice_deg: Optional[float] = None
    azimuth_slice_deg: Optional[float] = None
    target_speed_mps: float = 0.0
    radar_profile: Optional[str] = None
    tx_yaw_deg: Optional[float] = None
    tx_elev_deg: Optional[float] = None
    max_workers: Optional[int] = None

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


@dataclass
class Material:
    """Material properties used by the simplified RCS engine.

    ``reflectivity_h``/``reflectivity_v`` allow basic HH/VV differences; if
    omitted the scalar ``reflectivity`` is used for all channels.
    """

    name: str
    epsilon_real: float
    epsilon_imag: float
    conductivity: float
    reflectivity: float
    reflectivity_h: Optional[float] = None
    reflectivity_v: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimulationResult:
    """Container for RCS outputs.

    Notes
    -----
    ``rcs_dbsm`` expresses radar cross section in decibel-square-metres
    (dBsm) relative to 1 m^2. Frequencies are expressed in Hz even when band
    presets are provided in gigahertz.
    """

    band: str
    polarization: str
    frequencies_hz: np.ndarray
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    rcs_dbsm: np.ndarray  # shape (f, el, az)
    target_speed_mps: float
    radar_profile: Optional[str] = None
    doppler_hz: Optional[np.ndarray] = None

    def slice_for_elevation(self, elevation: float) -> Tuple[np.ndarray, np.ndarray]:
        idx = int(np.argmin(np.abs(self.elevation_deg - elevation)))
        return self.azimuth_deg, self.rcs_dbsm[:, idx, :]


@dataclass
class EngineMount:
    """Simple powerplant cavity model that boosts RCS when illuminated on-axis."""

    x: float
    y: float
    z: float
    radius_m: float = 0.5
    length_m: float = 1.0
    yaw_deg: float = 0.0

    def axis(self) -> np.ndarray:
        yaw = np.radians(self.yaw_deg)
        return np.array([np.cos(yaw), np.sin(yaw), 0.0])


@dataclass
class Propeller:
    """Parametric propeller disk representation with RPM and blade density."""

    x: float
    y: float
    z: float
    radius_m: float = 1.0
    blade_count: int = 3
    rpm: float = 1200.0
    yaw_deg: float = 0.0

    def axis(self) -> np.ndarray:
        yaw = np.radians(self.yaw_deg)
        return np.array([np.cos(yaw), np.sin(yaw), 0.0])


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
        progress: Optional[Callable[[int], None]] = None,
    ) -> SimulationResult:
        if mesh is None:
            raise ValueError("A mesh must be provided for simulation.")
        # Only build the ray intersector when the ray-tracing engine is used; the
        # facet-based path can run without optional ray backends installed.
        if settings.method != "facet_po":
            mesh.ray = build_ray_intersector(mesh)

        freqs = settings.frequencies()
        az = settings.azimuths()
        el = settings.elevations()
        dirs = direction_grid(az, el)

        distance = mesh.bounding_sphere.primitive.radius * 6.0 + 1.0
        directions = dirs.reshape(-1, 3)
        sphere_center = mesh.bounding_sphere.center

        if settings.tx_yaw_deg is not None and settings.tx_elev_deg is not None:
            tx_origin = sphere_center + distance * (
                -self._direction_from_angles(settings.tx_yaw_deg, settings.tx_elev_deg)
            )
            origin_center = tx_origin
            origin_distance = 0.0
        else:
            tx_origin = None
            origin_center = sphere_center
            origin_distance = distance

        bundle_offsets = np.linspace(-0.6, 0.6, 3) * mesh.bounding_sphere.primitive.radius
        bundle_grid = np.array([(ox, oy) for ox in bundle_offsets for oy in bundle_offsets])

        rcs_all = np.zeros((len(freqs), len(el), len(az)), dtype=float)
        doppler_all = np.zeros(len(freqs), dtype=float) if settings.target_speed_mps else None
        centers = mesh.triangles.mean(axis=1)
        edges = build_sharp_edges(mesh)
        reflectivity = self._select_reflectivity(material, settings.polarization)

        try:
            for fi, freq_hz in enumerate(freqs):
                if self._stop_requested:
                    break

                freq_ghz = freq_hz / 1e9
                wavelength = self._compute_wavelength(freq_hz)
                k = 2 * np.pi / wavelength
                if doppler_all is not None:
                    doppler_all[fi] = 2 * settings.target_speed_mps * freq_hz / 3e8
                loss_per_reflection = frequency_loss(freq_ghz)

                if settings.method == "facet_po":
                    rcs_lin = facet_rcs(mesh, reflectivity, freq_hz, directions)
                else:
                    rcs_lin = np.zeros(len(directions), dtype=float)
                    workers = settings.max_workers or (os.cpu_count() or 1)
                    chunk = max(64, len(directions) // max(workers, 1) // 2)
                    ray_intersector = mesh.ray
                    area_faces = mesh.area_faces
                    face_normals = mesh.face_normals
                    stop_flag = lambda: self._stop_requested  # noqa: E731
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                        futures: dict[concurrent.futures.Future[np.ndarray], slice] = {}
                        for start in range(0, len(directions), chunk):
                            end = min(len(directions), start + chunk)
                            dir_chunk = directions[start:end]
                            futures[
                                executor.submit(
                                    self._trace_direction_block,
                                    mesh,
                                    dir_chunk,
                                    bundle_grid,
                                    settings.max_reflections,
                                    reflectivity,
                                    loss_per_reflection,
                                    k,
                                    edges,
                                    centers,
                                    stop_flag,
                                    origin_center,
                                    origin_distance,
                                    ray_intersector,
                                    area_faces,
                                    face_normals,
                                )
                            ] = slice(start, end)

                        self._drain_trace_futures(futures, rcs_lin)

                    if self._stop_requested:
                        break

                rcs_lin = self._apply_powerplant_signatures(
                    directions, rcs_lin, reflectivity, settings
                )

                rcs_db = 10 * np.log10(rcs_lin.reshape(len(el), len(az)))
                rcs_all[fi] = rcs_db
                if progress:
                    progress(int((fi + 1) / len(freqs) * 100))
        finally:
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

    # ------------------------------------------------------------------
    def _drain_trace_futures(
        self,
        futures: dict[concurrent.futures.Future[np.ndarray], slice],
        rcs_lin: np.ndarray,
    ) -> None:
        """Collect results from ray-tracing workers with clear failure context."""

        pending = set(futures.keys())
        try:
            for fut in concurrent.futures.as_completed(pending):
                pending.remove(fut)
                sl = futures[fut]
                try:
                    rcs_lin[sl] = fut.result()
                except concurrent.futures.CancelledError as exc:
                    raise RuntimeError(
                        "Ray tracing was cancelled before completion. "
                        "Ensure ray dependencies are installed or retry with the 'facet_po' method."
                    ) from exc
                except Exception as exc:  # pragma: no cover - defensive guard for worker failures
                    formatted = "".join(traceback.format_exception(exc)).strip()
                    raise RuntimeError(
                        "Ray tracing failed during batch execution. "
                        "Ensure ray dependencies are installed or switch to the 'facet_po' method. "
                        f"Worker slice {sl} error: {exc}.\n{formatted}"
                    ) from exc
                if self._stop_requested:
                    break
        finally:
            if self._stop_requested:
                for fut in pending:
                    fut.cancel()

    # ------------------------------------------------------------------
    def _apply_powerplant_signatures(
        self,
        directions: np.ndarray,
        rcs_lin: np.ndarray,
        reflectivity: float,
        settings: SimulationSettings,
    ) -> np.ndarray:
        """Blend simplified engine intake and propeller disk responses into RCS."""

        if not settings.engines and not settings.propellers:
            return rcs_lin

        enhanced = rcs_lin.copy()

        if settings.engines:
            for engine in settings.engines:
                axis = engine.axis()
                alignment = np.clip(directions @ axis, 0.0, 1.0)
                area = np.pi * engine.radius_m**2
                cavity_gain = 1.5 + (engine.length_m / max(engine.radius_m, 1e-3))
                enhanced += reflectivity * area * cavity_gain * alignment**2

        if settings.propellers:
            for prop in settings.propellers:
                axis = prop.axis()
                alignment = np.clip(directions @ axis, 0.0, 1.0)
                disk_area = np.pi * prop.radius_m**2
                blade_fill = min(1.0, prop.blade_count * 0.15)
                tip_speed = 2 * np.pi * prop.radius_m * prop.rpm / 60.0
                doppler_gain = 1.0 + min(tip_speed / 200.0, 3.0) * 0.25
                enhanced += reflectivity * disk_area * blade_fill * alignment * doppler_gain

        return enhanced

    def _trace_direction_block(
        self,
        mesh: trimesh.Trimesh,
        directions: np.ndarray,
        bundle_grid: np.ndarray,
        max_reflections: int,
        reflectivity: float,
        loss_per_reflection: float,
        k: float,
        edges: tuple,
        centers: np.ndarray,
        stop_cb,
        origin_center: np.ndarray,
        origin_distance: float,
        ray_intersector,
        area_faces: np.ndarray,
        face_normals: np.ndarray,
    ) -> np.ndarray:
        """Trace a batch of directions and return their linear RCS contributions."""

        results = np.zeros(len(directions), dtype=float)
        for idx, direction in enumerate(directions):
            if stop_cb():
                break

            origin = origin_center - origin_distance * direction
            specular_sum = 0.0j
            basis_u, basis_v = self._orthonormal_basis(direction)
            bundle_origins = origin + bundle_grid[:, 0:1] * basis_u + bundle_grid[:, 1:2] * basis_v
            for ray_origin in bundle_origins:
                energy = 1.0
                path_length = 0.0
                ray_dir = direction
                for _ in range(max_reflections):
                    try:
                        locs, _, tri_idx = ray_intersector.intersects_location(
                            ray_origin[np.newaxis, :], ray_dir[np.newaxis, :], multiple_hits=False
                        )
                    except Exception:
                        # If the backend fails (e.g., missing optional acceleration modules),
                        # treat this path as non-contributing instead of crashing the worker.
                        break
                    if len(locs) == 0:
                        break
                    hit = locs[0]
                    face_index = tri_idx[0]
                    normal = face_normals[face_index]
                    reflect_dir = ray_dir - 2 * np.dot(ray_dir, normal) * normal
                    reflect_dir /= np.linalg.norm(reflect_dir)

                    path_length += float(np.linalg.norm(hit - ray_origin))

                    alignment = np.dot(reflect_dir, -ray_dir)
                    if alignment > 0.95:
                        area_term = area_faces[face_index] * (np.dot(normal, -ray_dir)) ** 2
                        total_path = self._path_to_receiver(path_length, hit, origin, direction)
                        phase = self._monostatic_phase(k, total_path)
                        specular_sum += energy * area_term * np.exp(1j * phase)

                    energy *= reflectivity * loss_per_reflection
                    if energy < MIN_ENERGY:
                        break

                    ray_origin = hit + 1e-4 * reflect_dir
                    ray_dir = reflect_dir

            k_hat = direction / (np.linalg.norm(direction) + 1e-12)
            illum_mask = (mesh.face_normals @ -k_hat) > 0.0
            edge_term = edge_diffraction_field(edges, k_hat, k, mesh)
            corner_term = corner_field(
                k_hat,
                mesh.face_normals,
                mesh.area_faces,
                centers,
                k,
                illuminated_mask=illum_mask,
            )
            diffraction_power = reflectivity * np.abs(edge_term + corner_term) ** 2

            averaged_specular = specular_sum / len(bundle_grid)
            specular_power = np.abs(averaged_specular) ** 2
            results[idx] = max(specular_power + diffraction_power, 1e-12)

        return results

    @staticmethod
    def _orthonormal_basis(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return two unit vectors spanning the plane perpendicular to *direction*."""

        w = direction / (np.linalg.norm(direction) + 1e-12)
        # Pick an arbitrary vector not parallel to w
        if abs(w[2]) < 0.9:
            helper = np.array([0.0, 0.0, 1.0])
        else:
            helper = np.array([0.0, 1.0, 0.0])
        u = np.cross(helper, w)
        u /= np.linalg.norm(u) + 1e-12
        v = np.cross(w, u)
        v /= np.linalg.norm(v) + 1e-12
        return u, v

    @staticmethod
    def _compute_wavelength(freq_hz: float) -> float:
        """Return the free-space wavelength (metres) for ``freq_hz``."""

        c = 3e8
        return c / max(freq_hz, 1e-9)

    @staticmethod
    def _monostatic_phase(k: float, total_path_length: float) -> float:
        """Compute the propagation phase for a given total path length.

        Parameters
        ----------
        k:
            Wavenumber ``2*pi/lambda``.
        total_path_length:
            Total travelled path length in metres (already including outbound
            and inbound paths where appropriate).
        """

        return k * total_path_length

    @staticmethod
    def _direction_from_angles(yaw_deg: float, elev_deg: float) -> np.ndarray:
        """Convert azimuth/elevation angles in degrees to a unit vector."""

        az = np.radians(yaw_deg)
        el = np.radians(elev_deg)
        return np.array(
            [
                np.cos(el) * np.cos(az),
                np.cos(el) * np.sin(az),
                np.sin(el),
            ]
        )

    @staticmethod
    def _path_to_receiver(
        path_length: float, hit: np.ndarray, rx_origin: np.ndarray, rx_dir: np.ndarray
    ) -> float:
        """Approximate total Tx-to-hit plus hit-to-Rx distance for phase.

        The ray tracing loop accumulates ``path_length`` along the transmit
        path. The additional leg to the receiver is approximated using the
        straight-line distance to the receiver origin placed on the far-field
        sphere along ``rx_dir``.
        """

        del rx_dir  # Receiver orientation is implicit in ``rx_origin``.
        return path_length + float(np.linalg.norm(hit - rx_origin))

    @staticmethod
    def _select_reflectivity(material: Material, polarization: str) -> float:
        """Choose an effective scalar reflectivity for the desired polarization.

        This is a simplified HH/VV selector. If polarization-specific values
        are not provided the broadband ``reflectivity`` is used. Cross/Co-pol
        flags collapse to averages rather than a full scattering matrix.
        """

        pol = polarization.upper()
        refl_h = material.reflectivity_h if material.reflectivity_h is not None else material.reflectivity
        refl_v = material.reflectivity_v if material.reflectivity_v is not None else material.reflectivity

        if pol.startswith("H"):
            return refl_h
        if pol.startswith("V"):
            return refl_v
        if "CROSS" in pol or pol.startswith("X"):
            return 0.5 * (refl_h + refl_v) * 0.5
        return 0.5 * (refl_h + refl_v)


__all__ = [
    "BAND_DEFAULTS",
    "FrequencySweep",
    "SimulationSettings",
    "Material",
    "EngineMount",
    "Propeller",
    "SimulationResult",
    "RCSEngine",
    "to_dbsm",
]
