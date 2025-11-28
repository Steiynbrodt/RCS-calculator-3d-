"""Reference radar presets for quick configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class RadarProfile:
    """Minimal radar description used to pre-fill simulation settings."""

    name: str
    band: str
    frequency_ghz: Optional[float] = None
    polarization: str = "H"
    sweep_start_ghz: Optional[float] = None
    sweep_stop_ghz: Optional[float] = None
    sweep_steps: Optional[int] = None
    max_reflections: Optional[int] = None
    default_speed_mps: Optional[float] = None
    note: Optional[str] = None


RADAR_PROFILES: Dict[str, RadarProfile] = {
    "Custom (manual)": RadarProfile(name="Custom (manual)", band="S", frequency_ghz=None),
    "S-300 30N6 Flap Lid (X-band 9.2 GHz)": RadarProfile(
        name="S-300 30N6 Flap Lid (X-band 9.2 GHz)", band="X", frequency_ghz=9.2, polarization="H/V", max_reflections=4
    ),
    "S-400 92N6E Grave Stone (X-band 10 GHz)": RadarProfile(
        name="S-400 92N6E Grave Stone (X-band 10 GHz)", band="X", frequency_ghz=10.0, polarization="H/V", max_reflections=5
    ),
    "S-400 91N6E Big Bird (S-band 3.2 GHz)": RadarProfile(
        name="S-400 91N6E Big Bird (S-band 3.2 GHz)", band="S", frequency_ghz=3.2, polarization="H/V"
    ),
    "Buk-M2 9S36 Fire Dome (X-band 9.5 GHz)": RadarProfile(
        name="Buk-M2 9S36 Fire Dome (X-band 9.5 GHz)", band="X", frequency_ghz=9.5, polarization="H"
    ),
    "Patriot AN/MPQ-65 (C-band 4.0 GHz)": RadarProfile(
        name="Patriot AN/MPQ-65 (C-band 4.0 GHz)", band="C", frequency_ghz=4.0, polarization="H", max_reflections=4
    ),
    "NASAMS AN/MPQ-64 Sentinel (X-band 9.5 GHz)": RadarProfile(
        name="NASAMS AN/MPQ-64 Sentinel (X-band 9.5 GHz)", band="X", frequency_ghz=9.5, polarization="H"
    ),
    "MEADS MFCR (X-band 10 GHz sweep)": RadarProfile(
        name="MEADS MFCR (X-band 10 GHz sweep)",
        band="X",
        frequency_ghz=None,
        polarization="H",
        sweep_start_ghz=8.5,
        sweep_stop_ghz=11.5,
        sweep_steps=6,
        max_reflections=4,
    ),
    "IRIS-T SLM TRML-4D (S-band 3.5 GHz)": RadarProfile(
        name="IRIS-T SLM TRML-4D (S-band 3.5 GHz)", band="S", frequency_ghz=3.5, polarization="H"
    ),
}


__all__ = ["RadarProfile", "RADAR_PROFILES"]
