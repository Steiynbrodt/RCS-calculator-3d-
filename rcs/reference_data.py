"""Built-in reference datasets inspired by public scattering studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rcs_engine import SimulationResult


@dataclass(frozen=True, slots=True)
class ReferenceDataset:
    """Container for canned article-style RCS data."""

    name: str
    source: str
    note: str
    result: SimulationResult


def _gaussian_lobe(
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    az0: float,
    el0: float,
    az_width: float,
    el_width: float,
    peak_db: float,
) -> np.ndarray:
    """Create a smooth lobe centered at (az0, el0)."""

    amp = 10 ** (peak_db / 10)
    az_term = ((az_grid - az0) / az_width) ** 2
    el_term = ((el_grid - el0) / el_width) ** 2
    return amp * np.exp(-(az_term + el_term))


def f16a_article_dataset() -> ReferenceDataset:
    """Synthetic F-16A map inspired by open-source scattering plots."""

    # Use a dense angular grid similar to the blog post (planform focused)
    az = np.linspace(-180.0, 180.0, 361)
    el = np.linspace(-30.0, 30.0, 121)
    az_grid, el_grid = np.meshgrid(az, el)

    # Start with a -15 dBsm diffuse floor then stack glints for nose, inlet,
    # wing leading edges, and tail/engine hotspots reminiscent of the article.
    rcs_lin = 10 ** (-15 / 10) * np.ones_like(az_grid)
    lobes = [
        # Nose-on and inlet specular
        (0.0, 0.0, 18.0, 6.0, 10.0),
        # Canopy/upper fuselage ripple
        (5.0, 10.0, 20.0, 10.0, 3.0),
        # Wing leading edge left/right
        (-35.0, 0.0, 22.0, 4.0, 6.0),
        (35.0, 0.0, 22.0, 4.0, 6.0),
        # Vertical tail and nozzle
        (170.0, 5.0, 25.0, 6.0, 8.5),
        (-170.0, 5.0, 25.0, 6.0, 7.5),
        # Intake diverter spikes
        (-12.0, -3.0, 10.0, 2.5, 4.5),
        (12.0, -3.0, 10.0, 2.5, 4.5),
    ]
    for az0, el0, az_w, el_w, peak_db in lobes:
        rcs_lin += _gaussian_lobe(az_grid, el_grid, az0, el0, az_w, el_w, peak_db)

    # Add gentle azimuthal speckle to avoid overly smooth plots
    rng = np.random.default_rng(42)
    rcs_lin *= 10 ** (rng.normal(0.0, 0.4, size=rcs_lin.shape) / 10)

    rcs_db = 10 * np.log10(np.clip(rcs_lin, 1e-12, None))
    freq = 9.6e9
    speed = 250.0  # m/s cruise segment typical of the example
    doppler = np.array([2 * speed * freq / 3e8])
    result = SimulationResult(
        band="X",
        polarization="H/V",
        frequencies_hz=np.array([freq]),
        azimuth_deg=az,
        elevation_deg=el,
        rcs_dbsm=rcs_db[None, ...],
        target_speed_mps=speed,
        radar_profile="Reference F-16A (9.6 GHz)",
        doppler_hz=doppler,
        surface_roughness_db=0.4,
        speckle_db=0.0,
        blade_count=0,
        blade_rpm=0.0,
        compressor_blades=0,
        compressor_rpm=0.0,
        micro_doppler_hz=None,
        engine_mounts=[],
    )
    return ReferenceDataset(
        name="F-16A article-style scatter", 
        source="https://basicsaboutaerodynamicsandavionics.wordpress.com/2022/08/17/f-16a-radar-scattering-simulation/",
        note=(
            "Synthesized map shaped after the public F-16A study: strong nose/on-axis glints, "
            "wing-edge peaks around ±35°, and aft nozzle/tail highlights. Use it as a reference "
            "or to benchmark your own meshes against article-like data."
        ),
        result=result,
    )


def list_reference_datasets() -> list[ReferenceDataset]:
    """Return all bundled reference datasets."""

    return [f16a_article_dataset()]


__all__ = ["ReferenceDataset", "f16a_article_dataset", "list_reference_datasets"]
