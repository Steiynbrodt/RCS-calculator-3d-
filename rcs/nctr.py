"""Simplified NCTR (Non-Cooperative Target Recognition) helpers.

The routines in this module generate micro-Doppler and range–Doppler
signatures from a 3D mesh that can be used as lightweight NCTR inputs.
They operate purely on the mesh geometry and do not require additional
dependencies beyond NumPy.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .math_utils import rotation_matrix


def _vertex_weights(mesh: trimesh.Trimesh) -> np.ndarray:
    """Approximate per-vertex scattering weights using face areas."""

    if not len(mesh.faces):
        return np.ones(len(mesh.vertices))

    weights = np.zeros(len(mesh.vertices))
    areas = mesh.area_faces

    for idx, faces in enumerate(mesh.vertex_faces):
        valid = faces[faces != -1]
        if len(valid):
            weights[idx] = float(np.mean(areas[valid]))

    if np.all(weights == 0):
        return np.ones(len(mesh.vertices))

    weights /= np.max(weights)
    return weights


def _spin_matrix(angle_rad: float) -> np.ndarray:
    """Rotation matrix for spinning around the z-axis by ``angle_rad``."""

    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _axis_angle_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Return rotation matrix for rotating ``angle_rad`` about ``axis``."""

    axis = np.asarray(axis, dtype=float)
    if np.linalg.norm(axis) == 0:
        return np.eye(3)
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    one_c = 1 - c
    return np.array(
        [
            [c + ux * ux * one_c, ux * uy * one_c - uz * s, ux * uz * one_c + uy * s],
            [uy * ux * one_c + uz * s, c + uy * uy * one_c, uy * uz * one_c - ux * s],
            [uz * ux * one_c - uy * s, uz * uy * one_c + ux * s, c + uz * uz * one_c],
        ]
    )


def simulate_nctr_signature(
    mesh: trimesh.Trimesh,
    material: dict,
    freq_ghz: float,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    rpm: float = 120.0,
    prf: float = 1800.0,
    pulses: int = 256,
    window: int = 64,
    hop: int = 16,
    rotating_groups: dict[str, dict] | None = None,
    return_range_doppler: bool = False,
    range_bin_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic micro-Doppler (and optional range–Doppler) signature.

    Parameters
    ----------
    mesh:
        Target mesh. Geometry is treated as a cloud of scattering centres.
    material:
        Mapping containing at least ``reflectivity`` used as a scalar weight.
    freq_ghz:
        Radar carrier frequency in GHz.
    yaw, pitch, roll:
        Initial orientation of the target in degrees.
    rpm:
        Default rigid-body spin rate in revolutions per minute. Only applied
        when ``rotating_groups`` is ``None``.
    prf:
        Pulse repetition frequency in Hertz.
    pulses:
        Number of slow-time pulses to synthesise.
    window:
        STFT window length (pulses).
    hop:
        Hop size between adjacent STFT windows (pulses).
    rotating_groups:
        Optional dictionary defining independently rotating vertex groups. Each
        entry should contain ``indices`` (vertex indices), ``rpm`` and ``axis``
        fields. When omitted the entire mesh spins about +z as before.
    return_range_doppler:
        If ``True``, additionally compute a coarse range–Doppler map using
        per-pulse range binning. The base return signature remains unchanged.
    range_bin_width:
        Optional range bin width in metres. If ``None`` a wavelength/2 bin is
        used for convenience.

    Returns
    -------
    times, freqs, spectrogram, envelope
        Default outputs for legacy callers. ``spectrogram`` is in dB relative
        to the coherent sum magnitude.
    If ``return_range_doppler`` is ``True`` the function returns an additional
    tuple ``(range_bins, doppler_freqs, range_doppler)`` describing the coarse
    range–Doppler map (in dB).

    Notes
    -----
    * This is a geometric micro-Doppler synthesiser; it does not model clutter
      or multipath.
    * Noise is injected as small complex Gaussian jitter to keep the output
      visually realistic.
    * Units: ranges are in metres, Doppler in Hertz, time axes in seconds.
    """

    if mesh is None or not hasattr(mesh, "vertices"):
        raise ValueError("Kein gültiges 3D-Mesh geladen.")

    if pulses < window:
        raise ValueError("Die Fenstergröße darf nicht größer als die Pulsanzahl sein.")

    wavelength = 0.3 / freq_ghz
    verts = mesh.vertices - mesh.centroid
    base_rotation = rotation_matrix(yaw, pitch, roll)
    verts = (base_rotation @ verts.T).T

    look_dir = np.array([1.0, 0.0, 0.0])
    look_dir /= np.linalg.norm(look_dir)

    omega = 2 * np.pi * rpm / 60.0
    dt = 1.0 / prf
    weights = _vertex_weights(mesh) * material.get("reflectivity", 1.0)

    signal = np.zeros(pulses, dtype=complex)
    per_bin_slowtime: list[np.ndarray] | None = [] if return_range_doppler else None

    base_min_range: float | None = None

    for idx in range(pulses):
        t = idx * dt
        current = verts.copy()

        if rotating_groups:
            static_mask = np.ones(len(current), dtype=bool)
            for cfg in rotating_groups.values():
                indices = np.asarray(cfg.get("indices", []), dtype=int)
                rpm_group = float(cfg.get("rpm", rpm))
                axis = np.asarray(cfg.get("axis", [0.0, 0.0, 1.0]), dtype=float)
                angle = 2 * np.pi * rpm_group / 60.0 * t
                rot = _axis_angle_matrix(axis, angle)
                current[indices] = (rot @ current[indices].T).T
                static_mask[indices] = False
            # Small wobble on static parts to avoid perfectly stationary echoes
            if np.any(static_mask):
                wobble = 0.001 * np.sin(2 * np.pi * 0.5 * t)
                current[static_mask] += wobble
        else:
            spin = _spin_matrix(omega * t)
            current = (spin @ current.T).T

        ranges = current @ look_dir
        radial_vel = np.cross(np.array([0.0, 0.0, omega]), current) @ look_dir

        phase = 4 * np.pi / wavelength * ranges
        doppler_phase = 2 * np.pi * (2 * radial_vel / wavelength) * t

        pulse_contrib = weights * np.exp(1j * (phase + doppler_phase))
        signal[idx] = np.sum(pulse_contrib)

        if per_bin_slowtime is not None:
            bin_width = wavelength / 2 if range_bin_width is None else range_bin_width
            if base_min_range is None:
                base_min_range = float(np.min(ranges))
            min_r = base_min_range
            bin_indices = ((ranges - min_r) / bin_width).astype(int)
            max_bin = int(np.max(bin_indices)) + 1
            if len(per_bin_slowtime) < max_bin:
                per_bin_slowtime.extend(
                    [np.zeros(pulses, dtype=complex) for _ in range(max_bin - len(per_bin_slowtime))]
                )
            for i_pt, b_idx in enumerate(bin_indices):
                per_bin_slowtime[b_idx][idx] += pulse_contrib[i_pt]

    signal += (np.random.normal(0, 0.02, size=pulses) + 1j * np.random.normal(0, 0.02, size=pulses))

    window_func = np.hanning(window)
    specs: list[np.ndarray] = []
    times: list[float] = []

    for start in range(0, pulses - window + 1, hop):
        segment = signal[start : start + window] * window_func
        fft_vals = np.fft.fftshift(np.fft.fft(segment))
        power = 20 * np.log10(np.abs(fft_vals) + 1e-6)
        specs.append(power)
        times.append((start + window / 2) * dt)

    spectrogram = np.array(specs).T
    freqs = np.fft.fftshift(np.fft.fftfreq(window, d=dt))
    envelope = 20 * np.log10(np.abs(signal) + 1e-6)

    if per_bin_slowtime is None:
        return np.array(times), freqs, spectrogram, envelope

    range_bins = np.arange(len(per_bin_slowtime)) * (wavelength / 2 if range_bin_width is None else range_bin_width)
    rd_cubes: list[np.ndarray] = []
    for slowtime in per_bin_slowtime:
        bin_specs: list[np.ndarray] = []
        for start in range(0, pulses - window + 1, hop):
            segment = slowtime[start : start + window] * window_func
            fft_vals = np.fft.fftshift(np.fft.fft(segment))
            bin_specs.append(20 * np.log10(np.abs(fft_vals) + 1e-6))
        rd_cubes.append(np.array(bin_specs).T)

    range_doppler = np.stack(rd_cubes, axis=0)
    return (
        np.array(times),
        freqs,
        spectrogram,
        envelope,
        range_bins,
        freqs,
        range_doppler,
    )


__all__ = ["simulate_nctr_signature"]
