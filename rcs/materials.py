"""Material database utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Dict, Iterable, List

from .rcs_engine import Material


DEFAULT_MATERIALS: list[Material] = [
    Material("Aluminium", epsilon_real=1.0, epsilon_imag=0.0, conductivity=3.5e7, reflectivity=0.99),
    Material("Stahl", epsilon_real=1.0, epsilon_imag=0.0, conductivity=1e7, reflectivity=0.98),
    Material("Carbon", epsilon_real=5.0, epsilon_imag=0.5, conductivity=1e4, reflectivity=0.6),
    Material("PVC", epsilon_real=3.0, epsilon_imag=0.2, conductivity=1e-15, reflectivity=0.3),
    Material("Gummi (RAM)", epsilon_real=10.0, epsilon_imag=3.0, conductivity=1e-5, reflectivity=0.1),
    Material("Titan", epsilon_real=1.0, epsilon_imag=0.0, conductivity=2.4e6, reflectivity=0.95),
]


class MaterialDatabase:
    """A simple JSON-backed material database."""

    def __init__(self, path: str | Path = "materials.json") -> None:
        self.path = Path(path)
        self.materials: Dict[str, Material] = {m.name: m for m in DEFAULT_MATERIALS}
        if self.path.exists():
            self.load()

    def load(self) -> None:
        data = json.loads(self.path.read_text()) if self.path.exists() else []
        materials: Dict[str, Material] = {}
        for entry in data:
            materials[entry["name"]] = Material(
                entry["name"],
                epsilon_real=entry.get("epsilon_real", 1.0),
                epsilon_imag=entry.get("epsilon_imag", 0.0),
                conductivity=entry.get("conductivity", 0.0),
                reflectivity=entry.get("reflectivity", 0.5),
            )
        if materials:
            self.materials = materials

    def save(self) -> None:
        payload = [asdict(mat) for mat in self.materials.values()]
        self.path.write_text(json.dumps(payload, indent=2))

    def add_material(self, material: Material) -> None:
        self.materials[material.name] = material
        self.save()

    def update_material(self, name: str, **kwargs) -> None:
        if name not in self.materials:
            raise KeyError(f"Material '{name}' not found")
        mat = self.materials[name]
        updated = Material(
            kwargs.get("name", mat.name),
            kwargs.get("epsilon_real", mat.epsilon_real),
            kwargs.get("epsilon_imag", mat.epsilon_imag),
            kwargs.get("conductivity", mat.conductivity),
            kwargs.get("reflectivity", mat.reflectivity),
        )
        if name != updated.name and updated.name in self.materials:
            raise ValueError("A material with the new name already exists.")
        if name != updated.name:
            del self.materials[name]
        self.materials[updated.name] = updated
        self.save()

    def delete_material(self, name: str) -> None:
        if name in self.materials:
            del self.materials[name]
            self.save()

    def get(self, name: str) -> Material:
        return self.materials[name]

    def names(self) -> list[str]:
        return list(self.materials.keys())

    def as_list(self) -> List[Material]:
        return list(self.materials.values())


__all__ = ["MaterialDatabase", "DEFAULT_MATERIALS"]
