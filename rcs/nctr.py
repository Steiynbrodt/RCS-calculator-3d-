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


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotation matrix for an arbitrary axis using Rodrigues' formula."""

    axis = axis / (np.linalg.norm(axis) + 1e-12)
    ux, uy, uz = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    r = np.array(
        [
            [c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
            [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s],
            [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
        ],
        dtype=float,
    )
    return r


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate simplified NCTR (micro-Doppler) signatures for a mesh.

    The model synthesises complex returns from weighted mesh vertices and can
    optionally treat subsets of vertices as independently rotating sub-assemblies
    (e.g., propellers or fans). It is intentionally lightweight: no full-wave
    EM, clutter, antenna patterns, or receiver effects are modelled.

    Parameters
    ----------
    mesh:
        Triangular mesh of the target. Vertices are assumed to be in metres.
    material:
        Dictionary containing at least ``"reflectivity"`` as an amplitude
        multiplier.
    freq_ghz:
        Carrier frequency in GHz.
    yaw, pitch, roll:
        Orientation of the target in degrees (aerospace convention).
    rpm:
        Global spin rate in revolutions per minute used when no rotating groups
        are supplied.
    prf:
        Pulse repetition frequency in Hertz.
    pulses:
        Number of slow-time pulses to synthesise.
    window:
        Window length for the short-time Fourier transform (STFT).
    hop:
        Hop size between adjacent STFT windows.
    rotating_groups:
        Optional mapping of group names to dictionaries describing vertex
        indices, ``rpm``, and rotation ``axis`` (3-vector). Vertices not
        included in any group remain static.
    return_range_doppler:
        If ``True`` also return a coarse range–Doppler cube per range bin.

    Returns
    -------
    tuple
        ``times, freqs, spectrogram, envelope`` when ``return_range_doppler`` is
        False. If ``True``, two additional arrays are returned:
        ``range_bins`` (bin centres) and ``range_doppler`` with shape
        ``(num_bins, window, num_frames)``.
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
    range_bin_size = 3e8 / (2.0 * prf)
    range_span = np.linalg.norm(verts, axis=1).max() * 2.0
    base_ranges = verts @ look_dir
    min_range = base_ranges.min() - range_span * 0.1
    max_range = base_ranges.max() + range_span * 0.1
    num_bins = int(np.clip(np.ceil((max_range - min_range) / range_bin_size) + 1, 16, 256))
    range_bins = min_range + np.arange(num_bins) * range_bin_size
    rd_cube = np.zeros((num_bins, pulses), dtype=complex)

    if rotating_groups:
        group_data: list[tuple[np.ndarray, float, np.ndarray]] = []
        for params in rotating_groups.values():
            indices = np.asarray(params.get("indices", []), dtype=int)
            if indices.size == 0:
                continue
            axis = np.asarray(params.get("axis", [0.0, 0.0, 1.0]), dtype=float)
            group_rpm = float(params.get("rpm", rpm))
            group_omega = 2 * np.pi * group_rpm / 60.0
            group_data.append((indices, group_omega, axis))
    else:
        group_data = []

    for idx in range(pulses):
        t = idx * dt
        current = verts.copy()
        radial_vel = np.zeros(len(current))

        if group_data:
            for indices, omega_val, axis in group_data:
                rot = _axis_angle_rotation(axis, omega_val * t)
                rotated = (rot @ current[indices].T).T
                current[indices] = rotated
                omega_vec = omega_val * (axis / (np.linalg.norm(axis) + 1e-12))
                radial_vel[indices] = (np.cross(omega_vec, rotated) @ look_dir)
        else:
            spin = _spin_matrix(omega * t)
            current = (spin @ current.T).T
            omega_vec = np.array([0.0, 0.0, omega])
            radial_vel = (np.cross(omega_vec, current) @ look_dir)

        ranges = current @ look_dir
        phase = 4 * np.pi / wavelength * ranges
        doppler_phase = 2 * np.pi * (2 * radial_vel / wavelength) * t

        contribution = weights * np.exp(1j * (phase + doppler_phase))
        signal[idx] = np.sum(contribution)

        bin_idx = ((ranges - min_range) / range_bin_size).astype(int)
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)
        np.add.at(rd_cube[:, idx], bin_idx, contribution)

    signal += (np.random.normal(0, 0.02, size=pulses) + 1j * np.random.normal(0, 0.02, size=pulses))

    window_func = np.hanning(window)
    specs: list[np.ndarray] = []
    times: list[float] = []
    rd_slices: list[np.ndarray] = []

    for start in range(0, pulses - window + 1, hop):
        segment = signal[start : start + window] * window_func
        fft_vals = np.fft.fftshift(np.fft.fft(segment))
        power = 20 * np.log10(np.abs(fft_vals) + 1e-6)
        specs.append(power)
        times.append((start + window / 2) * dt)

        if return_range_doppler:
            rd_segment = rd_cube[:, start : start + window] * window_func
            rd_fft = np.fft.fftshift(np.fft.fft(rd_segment, axis=1), axes=1)
            rd_power = 20 * np.log10(np.abs(rd_fft) + 1e-6)
            rd_slices.append(rd_power)

    spectrogram = np.array(specs).T
    freqs = np.fft.fftshift(np.fft.fftfreq(window, d=dt))
    envelope = 20 * np.log10(np.abs(signal) + 1e-6)

    if return_range_doppler:
        range_doppler = np.stack(rd_slices, axis=-1) if rd_slices else np.empty((num_bins, window, 0))
        return np.array(times), freqs, spectrogram, envelope, range_bins, range_doppler

    return np.array(times), freqs, spectrogram, envelope


__all__ = ["simulate_nctr_signature"]


if __name__ == "__main__":
    # Compare whole-body spin vs rotor-only spin for a small synthetic mesh.
    mesh = trimesh.creation.box(extents=(1.0, 0.3, 0.3))
    mat = {"reflectivity": 1.0}
    times, freqs, spec_all, _ = simulate_nctr_signature(mesh, mat, 10.0, rpm=3000.0)
    rotor_group = {"fan": {"indices": np.array([0, 1, 2, 3]), "rpm": 8000.0, "axis": [0, 0, 1]}}
    _, _, spec_rotor, _ = simulate_nctr_signature(
        mesh, mat, 10.0, rpm=0.0, rotating_groups=rotor_group
    )
    contrast = float(np.mean(np.abs(spec_all - spec_rotor)))
    print("Spectrogram contrast between modes:", contrast)
