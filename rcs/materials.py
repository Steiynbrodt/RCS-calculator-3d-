"""Material definitions and helper database wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from .rcs_engine import Material


@dataclass(frozen=True, slots=True)
class _LegacyMaterial:
    sigma: float
    eps_r: float
    tan_delta: float
    reflectivity: float


MATERIAL_DB: dict[str, _LegacyMaterial] = {
    "Aluminium": _LegacyMaterial(sigma=3.5e7, eps_r=1.0, tan_delta=0.0, reflectivity=0.99),
    "Stahl": _LegacyMaterial(sigma=1e7, eps_r=1.0, tan_delta=0.0, reflectivity=0.98),
    "Carbon": _LegacyMaterial(sigma=1e4, eps_r=5.0, tan_delta=0.05, reflectivity=0.6),
    "PVC": _LegacyMaterial(sigma=1e-15, eps_r=3.0, tan_delta=0.02, reflectivity=0.3),
    "Gummi (RAM)": _LegacyMaterial(sigma=1e-5, eps_r=10.0, tan_delta=0.3, reflectivity=0.1),
    "Titan": _LegacyMaterial(sigma=2.4e6, eps_r=1.0, tan_delta=0.0, reflectivity=0.95),
    "Kupfer": _LegacyMaterial(sigma=5.8e7, eps_r=1.0, tan_delta=0.0, reflectivity=0.99),
    "Gold": _LegacyMaterial(sigma=4.1e7, eps_r=1.0, tan_delta=0.0, reflectivity=0.995),
    "Graphit": _LegacyMaterial(sigma=1e3, eps_r=12.0, tan_delta=0.1, reflectivity=0.5),
    "RAM – Iron Ball Paint": _LegacyMaterial(sigma=1e-4, eps_r=10.0, tan_delta=0.15, reflectivity=0.25),
    "RAM – Carbon Nanotube": _LegacyMaterial(sigma=1e2, eps_r=12.0, tan_delta=0.1, reflectivity=0.1),
    "RAM – Conductive Polymer": _LegacyMaterial(sigma=1e0, eps_r=5.0, tan_delta=0.3, reflectivity=0.2),
    "RAM – Magnetic Ferrite": _LegacyMaterial(sigma=1e-2, eps_r=13.0, tan_delta=0.5, reflectivity=0.15),
    "RAM – Foam Layer": _LegacyMaterial(sigma=1e-6, eps_r=1.5, tan_delta=0.05, reflectivity=0.3),
    "RAM – Dallenbach Layer": _LegacyMaterial(sigma=1e-5, eps_r=7.0, tan_delta=0.2, reflectivity=0.05),
    "RAM – Jaumann Layer": _LegacyMaterial(sigma=1e-4, eps_r=9.0, tan_delta=0.4, reflectivity=0.05),
    "MagRAM": _LegacyMaterial(sigma=1e-4, eps_r=15.0, tan_delta=0.25, reflectivity=0.05),
    "Metamaterial RAM": _LegacyMaterial(sigma=1e-7, eps_r=25.0, tan_delta=0.4, reflectivity=0.01),
    "Carbon Nanotube Foam": _LegacyMaterial(sigma=1e1, eps_r=7.5, tan_delta=0.03, reflectivity=0.15),
    "Spray-on Polymer RAM": _LegacyMaterial(sigma=5e-3, eps_r=9.0, tan_delta=0.12, reflectivity=0.2),
    "Stainless Steel": _LegacyMaterial(sigma=1.4e6, eps_r=1.0, tan_delta=0.0, reflectivity=0.92),
    "Painted Aluminium": _LegacyMaterial(sigma=3.0e7, eps_r=2.5, tan_delta=0.01, reflectivity=0.93),
    "Carbon Fiber Composite": _LegacyMaterial(sigma=5e3, eps_r=8.0, tan_delta=0.08, reflectivity=0.55),
    "Glass Fiber Composite": _LegacyMaterial(sigma=1e-12, eps_r=4.5, tan_delta=0.01, reflectivity=0.25),
    "Honeycomb Sandwich": _LegacyMaterial(sigma=1e-3, eps_r=2.0, tan_delta=0.02, reflectivity=0.35),
    "Conductive Paint": _LegacyMaterial(sigma=8e3, eps_r=5.0, tan_delta=0.05, reflectivity=0.65),
    "Sea Water": _LegacyMaterial(sigma=4.0, eps_r=80.0, tan_delta=0.1, reflectivity=0.65),
    "Wet Soil": _LegacyMaterial(sigma=0.02, eps_r=20.0, tan_delta=0.12, reflectivity=0.4),
    "Dry Soil": _LegacyMaterial(sigma=0.005, eps_r=6.0, tan_delta=0.08, reflectivity=0.35),
    "Concrete": _LegacyMaterial(sigma=1e-3, eps_r=5.0, tan_delta=0.02, reflectivity=0.35),
    "Glass": _LegacyMaterial(sigma=1e-13, eps_r=6.5, tan_delta=0.005, reflectivity=0.2),
    "Ice": _LegacyMaterial(sigma=1e-6, eps_r=3.2, tan_delta=0.005, reflectivity=0.3),
    "Wood": _LegacyMaterial(sigma=1e-8, eps_r=2.2, tan_delta=0.05, reflectivity=0.25),
    "Fuel (JP-8)": _LegacyMaterial(sigma=5e-5, eps_r=2.1, tan_delta=0.02, reflectivity=0.15),
}


class MaterialDatabase:
    """Simple material store with editable entries."""

    def __init__(self) -> None:
        self.materials: dict[str, Material] = {
            name: self._convert_legacy(name, legacy)
            for name, legacy in MATERIAL_DB.items()
        }

    @staticmethod
    def _convert_legacy(name: str, legacy: _LegacyMaterial) -> Material:
        epsilon_imag = legacy.eps_r * legacy.tan_delta
        return Material(
            name=name,
            epsilon_real=legacy.eps_r,
            epsilon_imag=epsilon_imag,
            conductivity=legacy.sigma,
            reflectivity=legacy.reflectivity,
        )

    # Public API used by the GUI
    def names(self) -> list[str]:
        return list(self.materials.keys())

    def as_list(self) -> list[Material]:
        return list(self.materials.values())

    def get(self, name: str) -> Material:
        return self.materials[name]

    def add_material(self, material: Material) -> None:
        self.materials[material.name] = material

    def update_material(self, name: str, **kwargs) -> None:
        if name not in self.materials:
            raise KeyError(f"Unknown material '{name}'")
        updated = Material(**kwargs)
        if updated.name != name:
            self.materials.pop(name, None)
        self.materials[updated.name] = updated

    def delete_material(self, name: str) -> None:
        self.materials.pop(name, None)


__all__ = ["MATERIAL_DB", "MaterialDatabase"]
