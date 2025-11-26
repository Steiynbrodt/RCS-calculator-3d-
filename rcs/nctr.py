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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a micro-Doppler signature for the supplied mesh.

    The simulation uses a rigid-body spin around the z-axis to emulate
    compressor/rotor modulation. The returned spectrogram is expressed in
    dB and can be used for previewing or exporting simplified NCTR cues.
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

    for idx in range(pulses):
        t = idx * dt
        spin = _spin_matrix(omega * t)
        current = (spin @ verts.T).T

        ranges = current @ look_dir
        radial_vel = np.cross(np.array([0.0, 0.0, omega]), current) @ look_dir

        phase = 4 * np.pi / wavelength * ranges
        doppler_phase = 2 * np.pi * (2 * radial_vel / wavelength) * t

        signal[idx] = np.sum(weights * np.exp(1j * (phase + doppler_phase)))

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

    return np.array(times), freqs, spectrogram, envelope


__all__ = ["simulate_nctr_signature"]
