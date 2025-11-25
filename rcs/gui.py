"""Tkinter-based GUI for RCS simulation and visualization."""

from __future__ import annotations

import csv
import os
import threading
import traceback
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tkinter import filedialog, messagebox, ttk

from .materials import MATERIAL_DB
from .math_utils import rotation_matrix
from .physics import build_ray_intersector, robust_freq_sweep, simulate_rcs
from .presets import RADAR_PRESETS


def _homogeneous_rotation(rotation: np.ndarray) -> np.ndarray:
    """Create a 4x4 homogeneous transform from a 3x3 rotation."""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


class RadarGUI:
    """Encapsulated GUI controller."""

    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Radar RCS Simulation")
        master.geometry("800x600")
        master.minsize(700, 500)
        master.configure(bg="#222831")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#222831", foreground="#eeeeee", font=("Segoe UI", 11))
        style.configure("TButton", font=("Segoe UI", 11), padding=6)
        style.configure("TCheckbutton", background="#222831", foreground="#eeeeee", font=("Segoe UI", 11))
        style.configure("TMenubutton", font=("Segoe UI", 11))

        self.main_frame = ttk.Frame(master, padding=20, style="TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(tuple(range(20)), weight=1)

        self._build_controls()

        self.mesh: trimesh.Trimesh | None = None
        self.last_rcs: np.ndarray | None = None
        self.last_az: np.ndarray | None = None
        self.last_el: np.ndarray | None = None

    # ------------------------------------------------------------------
    # UI construction
    def _build_controls(self) -> None:
        ttk.Label(self.main_frame, text="3D-Modell (STL, OBJ, GLB):").grid(row=0, column=0, sticky="e", pady=4)
        self.file_label = ttk.Label(self.main_frame, text="Keine Datei", style="TLabel")
        self.file_label.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(self.main_frame, text="Laden", command=self.load_file).grid(row=0, column=2, padx=5, pady=4)

        ttk.Label(self.main_frame, text="Material:").grid(row=1, column=0, sticky="e", pady=4)
        self.material_var = tk.StringVar(self.master, value="Aluminium")
        material_menu = ttk.OptionMenu(self.main_frame, self.material_var, "Aluminium", *MATERIAL_DB.keys())
        material_menu.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(self.main_frame, text="Radarprofil:").grid(row=2, column=0, sticky="e", pady=4)
        self.preset_var = tk.StringVar(self.master, value="NATO – X-Band (10 GHz)")
        preset_menu = ttk.OptionMenu(
            self.main_frame,
            self.preset_var,
            "NATO – X-Band (10 GHz)",
            *RADAR_PRESETS.keys(),
            command=self.apply_preset,
        )
        preset_menu.grid(row=2, column=1, sticky="ew", pady=4)

        ttk.Label(self.main_frame, text="Frequenz (MHz):").grid(row=3, column=0, sticky="e", pady=4)
        self.freq_scale = tk.Scale(self.main_frame, from_=100, to=20000, orient=tk.HORIZONTAL, bg="#393e46", fg="#eeeeee", highlightbackground="#222831")
        self.freq_scale.set(10000)
        self.freq_scale.grid(row=3, column=1, sticky="ew", pady=4)

        ttk.Label(self.main_frame, text="Max. Reflexionen:").grid(row=4, column=0, sticky="e", pady=4)
        self.refl_slider = tk.Scale(self.main_frame, from_=1, to=10, orient=tk.HORIZONTAL, bg="#393e46", fg="#eeeeee", highlightbackground="#222831")
        self.refl_slider.set(3)
        self.refl_slider.grid(row=4, column=1, sticky="ew", pady=4)

        ttk.Label(self.main_frame, text="Elevation-Schnitt (°):").grid(row=5, column=0, sticky="e", pady=4)
        self.el_slider = tk.Scale(self.main_frame, from_=-90, to=90, orient=tk.HORIZONTAL, bg="#393e46", fg="#eeeeee", highlightbackground="#222831")
        self.el_slider.set(0)
        self.el_slider.grid(row=5, column=1, sticky="ew", pady=4)

        ttk.Label(self.main_frame, text="Yaw (°):").grid(row=6, column=0, sticky="e", pady=4)
        self.yaw = tk.Scale(self.main_frame, from_=-180, to=180, orient=tk.HORIZONTAL, bg="#393e46", fg="#eeeeee", highlightbackground="#222831")
        self.yaw.set(0)
        self.yaw.grid(row=6, column=1, sticky="ew", pady=4)

        ttk.Label(self.main_frame, text="Pitch (°):").grid(row=7, column=0, sticky="e", pady=4)
        self.pitch = tk.Scale(self.main_frame, from_=-90, to=90, orient=tk.HORIZONTAL, bg="#393e46", fg="#eeeeee", highlightbackground="#222831")
        self.pitch.set(0)
        self.pitch.grid(row=7, column=1, sticky="ew", pady=4)

        ttk.Label(self.main_frame, text="Roll (°):").grid(row=8, column=0, sticky="e", pady=4)
        self.roll = tk.Scale(self.main_frame, from_=-180, to=180, orient=tk.HORIZONTAL, bg="#393e46", fg="#eeeeee", highlightbackground="#222831")
        self.roll.set(0)
        self.roll.grid(row=8, column=1, sticky="ew", pady=4)

        ttk.Label(self.main_frame, text="Radar-Azimuth (°):").grid(row=9, column=0, sticky="e", pady=4)
        self.radar_az = tk.Scale(self.main_frame, from_=0, to=360, orient=tk.HORIZONTAL, bg="#393e46", fg="#eeeeee", highlightbackground="#222831")
        self.radar_az.set(0)
        self.radar_az.grid(row=9, column=1, sticky="ew", pady=4)

        ttk.Label(self.main_frame, text="Radar-Elevation (°):").grid(row=10, column=0, sticky="e", pady=4)
        self.radar_el = tk.Scale(self.main_frame, from_=-90, to=90, orient=tk.HORIZONTAL, bg="#393e46", fg="#eeeeee", highlightbackground="#222831")
        self.radar_el.set(0)
        self.radar_el.grid(row=10, column=1, sticky="ew", pady=4)

        ttk.Label(self.main_frame, text="Radar-Sichtfeld (°):").grid(row=11, column=0, sticky="e", pady=4)
        self.beamwidth_var = tk.DoubleVar(self.master, value=360.0)
        self.beamwidth = tk.Scale(
            self.main_frame,
            from_=20,
            to=360,
            orient=tk.HORIZONTAL,
            resolution=5,
            variable=self.beamwidth_var,
            bg="#393e46",
            fg="#eeeeee",
            highlightbackground="#222831",
        )
        self.beamwidth.grid(row=11, column=1, sticky="ew", pady=4)

        ttk.Label(self.main_frame, text="Azimut-Auflösung:").grid(row=12, column=0, sticky="e", pady=4)
        self.az_res_var = tk.IntVar(self.master, value=360)
        ttk.Spinbox(self.main_frame, from_=30, to=720, increment=30, textvariable=self.az_res_var, width=8).grid(row=12, column=1, sticky="w", pady=4)

        ttk.Label(self.main_frame, text="Elevations-Auflösung:").grid(row=13, column=0, sticky="e", pady=4)
        self.el_res_var = tk.IntVar(self.master, value=181)
        ttk.Spinbox(self.main_frame, from_=45, to=181, increment=9, textvariable=self.el_res_var, width=8).grid(row=13, column=1, sticky="w", pady=4)

        ttk.Label(self.main_frame, text="Radar-Entfernung (x Radius):").grid(row=14, column=0, sticky="e", pady=4)
        self.range_var = tk.DoubleVar(self.master, value=6.0)
        self.range_scale = ttk.Scale(
            self.main_frame,
            from_=2.0,
            to=20.0,
            orient=tk.HORIZONTAL,
            variable=self.range_var,
            style="TScale",
        )
        self.range_scale.grid(row=14, column=1, sticky="ew", pady=4)

        self.show_model = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.main_frame, text="Modell anzeigen", variable=self.show_model, style="TCheckbutton"
        ).grid(row=15, column=0, sticky="w", pady=4)

        ttk.Button(self.main_frame, text="Simulation starten", command=self.run_simulation).grid(row=16, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Button(
            self.main_frame,
            text="Frequenz-Sweep",
            command=lambda: robust_freq_sweep(
                self.mesh,
                MATERIAL_DB[self.material_var.get()],
                self.refl_slider.get(),
                lambda func: self.master.after(0, func),
            ),
        ).grid(row=18, column=0, sticky="ew", pady=4)
        ttk.Button(self.main_frame, text="Heatmap exportieren", command=self.export_heatmap).grid(row=18, column=1, sticky="ew", pady=4)

        self.live_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.main_frame,
            text="Live Simulation (experimentell)",
            variable=self.live_mode,
            style="TCheckbutton",
            command=self.toggle_live_mode,
        ).grid(row=17, column=0, columnspan=2, sticky="w", pady=4)

    # ------------------------------------------------------------------
    # Event handlers
    def apply_preset(self, selection: str) -> None:
        if selection in RADAR_PRESETS:
            self.freq_scale.set(RADAR_PRESETS[selection])

    def load_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("3D-Dateien", "*.stl *.obj *.glb *.gltf")])
        if not path:
            return
        try:
            mesh = trimesh.load_mesh(path, force="mesh")

            if not isinstance(mesh, trimesh.Trimesh):
                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
                else:
                    raise ValueError("Unbekannter Mesh-Typ geladen.")

            mesh.remove_unreferenced_vertices()
            mesh.update_faces(mesh.nondegenerate_faces())

            if len(mesh.faces) > 50000:
                mesh = mesh.simplify_quadratic_decimation(50000)

            mesh.ray = build_ray_intersector(mesh)

            self.mesh = mesh
            self.file_label.config(text=os.path.basename(path))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Fehler beim Laden", f"Das Modell konnte nicht geladen werden:\n{exc}")
            self.mesh = None
            self.file_label.config(text="Keine Datei")

    def toggle_live_mode(self) -> None:
        callbacks: list[tuple[tk.Scale, str]] = [
            (self.freq_scale, "<ButtonRelease-1>"),
            (self.refl_slider, "<ButtonRelease-1>"),
            (self.el_slider, "<ButtonRelease-1>"),
            (self.yaw, "<ButtonRelease-1>"),
            (self.pitch, "<ButtonRelease-1>"),
            (self.roll, "<ButtonRelease-1>"),
            (self.radar_az, "<ButtonRelease-1>"),
            (self.radar_el, "<ButtonRelease-1>"),
            (self.beamwidth, "<ButtonRelease-1>"),
        ]
        if self.live_mode.get():
            for widget, event in callbacks:
                widget.bind(event, self.live_update)
            self.material_var.trace_add("write", lambda *_: self.live_update())
            self.preset_var.trace_add("write", lambda *_: self.live_update())
            self.az_res_var.trace_add("write", lambda *_: self.live_update())
            self.el_res_var.trace_add("write", lambda *_: self.live_update())
            self.range_var.trace_add("write", lambda *_: self.live_update())
        else:
            for widget, event in callbacks:
                widget.unbind(event)

    def live_update(self, event=None) -> None:  # noqa: ANN001
        if self.mesh is None or not self.live_mode.get():
            return

        def worker() -> None:
            try:
                material = MATERIAL_DB[self.material_var.get()]
                freq = self.freq_scale.get() / 1000
                refl = self.refl_slider.get()
                rotated = self._rotated_mesh()
                az, el, rcs = simulate_rcs(
                    rotated,
                    material,
                    refl,
                    freq,
                    az_steps=self.az_res_var.get(),
                    el_steps=self.el_res_var.get(),
                    look_az=self.radar_az.get(),
                    look_el=self.radar_el.get(),
                    beam_width=self.beamwidth_var.get(),
                    range_factor=self.range_var.get(),
                    max_workers=os.cpu_count(),
                )
                self.last_rcs = rcs
                self.last_az = az
                self.last_el = el
                self.master.after(0, self.plot_2d)
                self.master.after(0, lambda: self.plot_3d(rotated))
                export_dir = Path.home() / "Documents" / "RCS_Live_Exports"
                export_dir.mkdir(parents=True, exist_ok=True)
                self.export_csv(export_dir, live=True)
                self.export_heatmap(export_dir, live=True)
            except Exception:
                traceback.print_exc()

        threading.Thread(target=worker, daemon=True).start()

    def run_simulation(self) -> None:
        if self.mesh is None:
            messagebox.showwarning("Kein Modell", "Kein Modell geladen.")
            return
        try:
            material = MATERIAL_DB[self.material_var.get()]
            freq = self.freq_scale.get() / 1000
            refl = self.refl_slider.get()
            rotated = self._rotated_mesh()
            az, el, rcs = simulate_rcs(
                rotated,
                material,
                refl,
                freq,
                az_steps=self.az_res_var.get(),
                el_steps=self.el_res_var.get(),
                look_az=self.radar_az.get(),
                look_el=self.radar_el.get(),
                beam_width=self.beamwidth_var.get(),
                range_factor=self.range_var.get(),
                max_workers=os.cpu_count(),
            )
            self.last_rcs = rcs
            self.last_az = az
            self.last_el = el
            self.plot_2d()
            self.plot_3d(rotated)
            self.export_csv()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Fehler bei Simulation", f"Fehler während der Simulation:\n{exc}")

    # ------------------------------------------------------------------
    # Visualization helpers
    def _rotated_mesh(self) -> trimesh.Trimesh:
        rotation = rotation_matrix(self.yaw.get(), self.pitch.get(), self.roll.get())
        rotated = self.mesh.copy()
        rotated.apply_transform(_homogeneous_rotation(rotation))
        rotated.ray = build_ray_intersector(rotated)
        return rotated

    def plot_2d(self) -> None:
        if self.last_rcs is None:
            return
        idx = int(np.argmin(np.abs(self.last_el - self.el_slider.get())))
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.plot(np.radians(self.last_az), self.last_rcs[idx], label=f"Elevation {self.last_el[idx]}°")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title("2D Radar Cross Section")
        ax.legend()
        try:
            img = mpimg.imread("aircraft_shadow.png")
            axins = fig.add_axes([0.3, 0.3, 0.4, 0.4], polar=True, zorder=-1)
            axins.imshow(img, extent=[0, 2 * np.pi, 0, np.max(self.last_rcs)], aspect="auto", alpha=0.3)
            axins.axis("off")
        except Exception:
            pass
        plt.show()

    def plot_3d(self, mesh: trimesh.Trimesh) -> None:
        if self.last_rcs is None:
            return
        az_rad = np.radians(self.last_az)
        el_rad = np.radians(self.last_el)
        az_grid, el_grid = np.meshgrid(az_rad, el_rad)
        r_raw = 10 ** (self.last_rcs / 10.0)
        model_radius = np.linalg.norm(mesh.bounding_box.extents) / 2
        r = r_raw / np.max(r_raw) * model_radius * 4
        x = r * np.cos(el_grid) * np.cos(az_grid)
        y = r * np.cos(el_grid) * np.sin(az_grid)
        z = r * np.sin(el_grid)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            x,
            y,
            z,
            facecolors=plt.cm.viridis((r_raw - r_raw.min()) / (r_raw.max() - r_raw.min())),
            rstride=1,
            cstride=1,
            alpha=0.5,
        )
        mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        mappable.set_array(r_raw)
        fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="RCS (linear)")
        if self.show_model.get():
            ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces], facecolor="gray", edgecolor="k", alpha=0.2))
        ax.set_box_aspect([1, 1, 1])
        ax.set_title("3D RCS Plot")
        plt.show()

    # ------------------------------------------------------------------
    # Export helpers
    def export_csv(self, export_dir: Path | None = None, live: bool = False) -> None:
        if self.last_rcs is None:
            return
        if export_dir is None and not live:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Speichern unter..."
            )
            if not file_path:
                return
        elif export_dir is not None:
            file_path = os.path.join(export_dir, "rcs_export.csv")
        else:
            file_path = "rcs_export.csv"
        try:
            with open(file_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Azimuth (°)", "Elevation (°)", "RCS (dBsm)"])
                for i, el_val in enumerate(self.last_el):
                    for j, az_val in enumerate(self.last_az):
                        writer.writerow([az_val, el_val, self.last_rcs[i, j]])
            if not live:
                messagebox.showinfo("Export erfolgreich", f"RCS-Daten wurden als {os.path.basename(file_path)} gespeichert.")
        except Exception as exc:  # noqa: BLE001
            if not live:
                messagebox.showerror("Fehler beim Export", f"Fehler beim Exportieren der CSV:\n{exc}")

    def export_heatmap(self, export_dir: Path | None = None, live: bool = False) -> None:
        if self.last_rcs is None:
            return

        def worker() -> None:
            try:
                if export_dir is None and not live:
                    file_path = filedialog.asksaveasfilename(
                        defaultextension=".png", filetypes=[("PNG files", "*.png")], title="Speichern unter..."
                    )
                    if not file_path:
                        return
                elif export_dir is not None:
                    file_path = os.path.join(export_dir, "rcs_heatmap.png")
                else:
                    file_path = "rcs_heatmap.png"
                plt.imshow(self.last_rcs, extent=[0, 360, -90, 90], aspect="auto", cmap="hot", origin="lower")
                plt.colorbar(label="RCS (dBsm)")
                plt.title("RCS Heatmap")
                plt.xlabel("Azimuth")
                plt.ylabel("Elevation")
                plt.savefig(file_path)
                plt.close()
                if not live:
                    self.master.after(
                        0, lambda: messagebox.showinfo("Export erfolgreich", f"Heatmap wurde als {os.path.basename(file_path)} gespeichert.")
                    )
            except Exception as exc:  # noqa: BLE001
                if not live:
                    self.master.after(0, lambda e=exc: messagebox.showerror("Fehler beim Export", f"Fehler beim Exportieren der Heatmap:\n{e}"))

        threading.Thread(target=worker, daemon=True).start()


__all__ = ["RadarGUI"]
