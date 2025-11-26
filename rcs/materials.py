"""Material definitions for radar cross-section simulations."""

from __future__ import annotations

MATERIAL_DB = {
    "Aluminium": {"sigma": 3.5e7, "eps_r": 1.0, "tan_delta": 0.0, "reflectivity": 0.99},
    "Stahl": {"sigma": 1e7, "eps_r": 1.0, "tan_delta": 0.0, "reflectivity": 0.98},
    "Carbon": {"sigma": 1e4, "eps_r": 5.0, "tan_delta": 0.05, "reflectivity": 0.6},
    "PVC": {"sigma": 1e-15, "eps_r": 3.0, "tan_delta": 0.02, "reflectivity": 0.3},
    "Gummi (RAM)": {"sigma": 1e-5, "eps_r": 10.0, "tan_delta": 0.3, "reflectivity": 0.1},
    "Titan": {"sigma": 2.4e6, "eps_r": 1.0, "tan_delta": 0.0, "reflectivity": 0.95},
    "Kupfer": {"sigma": 5.8e7, "eps_r": 1.0, "tan_delta": 0.0, "reflectivity": 0.99},
    "Gold": {"sigma": 4.1e7, "eps_r": 1.0, "tan_delta": 0.0, "reflectivity": 0.995},
    "Graphit": {"sigma": 1e3, "eps_r": 12.0, "tan_delta": 0.1, "reflectivity": 0.5},
    "RAM – Iron Ball Paint": {"sigma": 1e-4, "eps_r": 10.0, "tan_delta": 0.15, "reflectivity": 0.25},
    "RAM – Carbon Nanotube": {"sigma": 1e2, "eps_r": 12.0, "tan_delta": 0.1, "reflectivity": 0.1},
    "RAM – Conductive Polymer": {"sigma": 1e0, "eps_r": 5.0, "tan_delta": 0.3, "reflectivity": 0.2},
    "RAM – Magnetic Ferrite": {"sigma": 1e-2, "eps_r": 13.0, "tan_delta": 0.5, "reflectivity": 0.15},
    "RAM – Foam Layer": {"sigma": 1e-6, "eps_r": 1.5, "tan_delta": 0.05, "reflectivity": 0.3},
    "RAM – Dallenbach Layer": {"sigma": 1e-5, "eps_r": 7.0, "tan_delta": 0.2, "reflectivity": 0.05},
    "RAM – Jaumann Layer": {"sigma": 1e-4, "eps_r": 9.0, "tan_delta": 0.4, "reflectivity": 0.05},
    "MagRAM": {"sigma": 1e-4, "eps_r": 15.0, "tan_delta": 0.25, "reflectivity": 0.05},
    "Metamaterial RAM": {"sigma": 1e-7, "eps_r": 25.0, "tan_delta": 0.4, "reflectivity": 0.01},
    "Carbon Nanotube Foam": {"sigma": 1e1, "eps_r": 7.5, "tan_delta": 0.03, "reflectivity": 0.15},
    "Spray-on Polymer RAM": {"sigma": 5e-3, "eps_r": 9.0, "tan_delta": 0.12, "reflectivity": 0.2},
}

__all__ = ["MATERIAL_DB"]
