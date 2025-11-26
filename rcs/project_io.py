"""Project save/load helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .rcs_engine import EngineMount, FrequencySweep, SimulationSettings


@dataclass(slots=True)
class ProjectState:
    mesh_path: str | None
    settings: SimulationSettings
    material_name: str


def save_project(path: str | Path, state: ProjectState) -> None:
    settings_dict = asdict(state.settings)
    if state.settings.sweep:
        settings_dict["sweep"] = asdict(state.settings.sweep)
    payload = {
        "mesh_path": state.mesh_path,
        "material_name": state.material_name,
        "settings": settings_dict,
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_project(path: str | Path) -> ProjectState:
    data = json.loads(Path(path).read_text())
    settings_data = data["settings"]
    sweep_data = settings_data.pop("sweep", None)
    sweep_obj = None
    if sweep_data:
        sweep_obj = FrequencySweep(sweep_data["start_hz"], sweep_data["stop_hz"], sweep_data["steps"])
    settings_data.setdefault("surface_roughness_db", 0.0)
    settings_data.setdefault("speckle_db", 0.0)
    settings_data.setdefault("random_seed", None)
    settings_data.setdefault("blade_count", 0)
    settings_data.setdefault("blade_rpm", 0.0)
    settings_data.setdefault("compressor_blades", 0)
    settings_data.setdefault("compressor_rpm", 0.0)
    mounts = settings_data.pop("engine_mounts", []) or []
    settings_data["engine_mounts"] = [EngineMount(**m) if isinstance(m, dict) else m for m in mounts]
    settings = SimulationSettings(**settings_data)
    settings.sweep = sweep_obj
    return ProjectState(mesh_path=data.get("mesh_path"), settings=settings, material_name=data.get("material_name", ""))


__all__ = ["ProjectState", "save_project", "load_project"]
