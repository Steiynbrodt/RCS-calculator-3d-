# Radar RCS Simulator – Vollständiger Code (GUI + Visualisierung + Export)

import numpy as np
import trimesh
import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import csv
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Materialdatenbank ---
material_db = {
    "Aluminium":   {"sigma": 3.5e7,  "eps_r": 1.0, "tan_delta": 0.0, "reflectivity": 0.99},
    "Stahl":       {"sigma": 1e7,    "eps_r": 1.0, "tan_delta": 0.0, "reflectivity": 0.98},
    "Carbon":      {"sigma": 1e4,    "eps_r": 5.0, "tan_delta": 0.05, "reflectivity": 0.6},
    "PVC":         {"sigma": 1e-15,  "eps_r": 3.0, "tan_delta": 0.02, "reflectivity": 0.3},
    "Gummi (RAM)": {"sigma": 1e-5,   "eps_r": 10.0, "tan_delta": 0.3, "reflectivity": 0.1},
    "Titan":       {"sigma": 2.4e6,  "eps_r": 1.0, "tan_delta": 0.0, "reflectivity": 0.95},
    "Kupfer":      {"sigma": 5.8e7,  "eps_r": 1.0, "tan_delta": 0.0, "reflectivity": 0.99},
    "Gold":        {"sigma": 4.1e7,  "eps_r": 1.0, "tan_delta": 0.0, "reflectivity": 0.995},
    "Graphit":     {"sigma": 1e3,    "eps_r": 12.0, "tan_delta": 0.1, "reflectivity": 0.5}
}

# --- Radar Presets ---
radar_presets = {
    "NATO – L-Band (1.3 GHz)": 1300,
    "NATO – X-Band (10 GHz)": 10000,
    "Russland – S-Band (3.5 GHz)": 3500,
    "Russland – L-Band (1.4 GHz)": 1400,
    "F-14 AWG-9 Radar (5 GHz)": 5000,
    "F-15 AN/APG-63 (10.2 GHz)": 10200,
    "F-16 AN/APG-68 (10.5 GHz)": 10500,
    "F-22 AN/APG-77 (12 GHz)": 12000,
    "Su-35 Irbis-E (10 GHz)": 10000,
    "S-400 91N6E Big Bird (6 GHz)": 6000
}

# --- Hilfsfunktionen ---
def sph2cart(az, el):
    az = np.radians(az)
    el = np.radians(el)
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return np.stack((x, y, z), axis=-1)

def rotation_matrix(yaw, pitch, roll):
    y, p, r = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]).dot(np.array([[np.cos(p), -np.sin(p), 0], [np.sin(p), np.cos(p), 0], [0, 0, 1]]))
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry

def frequency_dependent_loss(freq_ghz):
    return max(0.8 - 0.02 * (freq_ghz - 1), 0.2)

def realistic_rcs_with_noise(rcs_val):
    return rcs_val + np.random.normal(0, 2)

# --- Raytracing ---
def trace_ray(mesh, origin, direction, max_depth, reflectivity, freq_ghz):
    try:
        rays = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    except Exception:
        rays = mesh.ray
    energy = 1.0
    rcs = 0.0
    loss_per_reflection = frequency_dependent_loss(freq_ghz)
    for _ in range(max_depth):
        locs, _, tri_idx = rays.intersects_location([origin], [direction], multiple_hits=False)
        if len(locs) == 0:
            break
        hit = locs[0]
        normal = mesh.face_normals[tri_idx[0]]
        reflect_dir = direction - 2 * np.dot(direction, normal) * normal
        reflect_dir /= np.linalg.norm(reflect_dir)
        if np.dot(reflect_dir, -direction) > 0.95:
            contrib = energy * mesh.area_faces[tri_idx[0]] * (np.dot(normal, -direction))**2
            rcs += contrib
        energy *= reflectivity * loss_per_reflection
        origin = hit + 1e-4 * reflect_dir
        direction = reflect_dir
    return rcs

# --- Simulation ---
def simulate_rcs(mesh, material, max_reflections, freq_ghz, az_steps=360, el_steps=181):
    az = np.linspace(0, 360, az_steps)
    el = np.linspace(-90, 90, el_steps)
    az_grid, el_grid = np.meshgrid(az, el)
    dirs = sph2cart(az_grid, el_grid)
    rcs_db = np.zeros_like(az_grid)
    for i in range(dirs.shape[0]):
        for j in range(dirs.shape[1]):
            origin = mesh.bounding_sphere.center + 100 * (-dirs[i, j])
            rcs_lin = trace_ray(mesh, origin, dirs[i, j], max_reflections, material["reflectivity"], freq_ghz)
            rcs_db[i, j] = realistic_rcs_with_noise(10 * np.log10(rcs_lin + 1e-6))
    return az, el, rcs_db

# --- GUI, Visualisierung & Export ---

class RadarGUI:
    def __init__(self, master):
        self.master = master
        master.title("Radar RCS Simulation")

        # Datei & Materialwahl
        tk.Label(master, text="3D-Modell (STL, OBJ, GLB):").grid(row=0, column=0)
        self.file_label = tk.Label(master, text="Keine Datei")
        self.file_label.grid(row=0, column=1)
        tk.Button(master, text="Laden", command=self.load_file).grid(row=0, column=2)

        tk.Label(master, text="Material:").grid(row=1, column=0)
        self.material_var = tk.StringVar(master)
        self.material_var.set("Aluminium")
        ttk.OptionMenu(master, self.material_var, "Aluminium", *material_db.keys()).grid(row=1, column=1)

        tk.Label(master, text="Radarprofil:").grid(row=2, column=0)
        self.preset_var = tk.StringVar(master)
        self.preset_var.set("NATO – X-Band (10 GHz)")
        ttk.OptionMenu(master, self.preset_var, "NATO – X-Band (10 GHz)", *radar_presets.keys(), command=self.apply_preset).grid(row=2, column=1)

        # Parameter
        tk.Label(master, text="Frequenz (MHz):").grid(row=3, column=0)
        self.freq_scale = tk.Scale(master, from_=100, to=20000, orient=tk.HORIZONTAL)
        self.freq_scale.set(10000)
        self.freq_scale.grid(row=3, column=1)

        tk.Label(master, text="Max. Reflexionen:").grid(row=4, column=0)
        self.refl_slider = tk.Scale(master, from_=1, to=10, orient=tk.HORIZONTAL)
        self.refl_slider.set(3)
        self.refl_slider.grid(row=4, column=1)

        tk.Label(master, text="Elevation-Schnitt (°):").grid(row=5, column=0)
        self.el_slider = tk.Scale(master, from_=-90, to=90, orient=tk.HORIZONTAL)
        self.el_slider.set(0)
        self.el_slider.grid(row=5, column=1)

        # Rotation
        tk.Label(master, text="Yaw (°):").grid(row=6, column=0)
        self.yaw = tk.Scale(master, from_=-180, to=180, orient=tk.HORIZONTAL)
        self.yaw.set(0)
        self.yaw.grid(row=6, column=1)

        tk.Label(master, text="Pitch (°):").grid(row=7, column=0)
        self.pitch = tk.Scale(master, from_=-90, to=90, orient=tk.HORIZONTAL)
        self.pitch.set(0)
        self.pitch.grid(row=7, column=1)

        tk.Label(master, text="Roll (°):").grid(row=8, column=0)
        self.roll = tk.Scale(master, from_=-180, to=180, orient=tk.HORIZONTAL)
        self.roll.set(0)
        self.roll.grid(row=8, column=1)

        # Optionen
        self.show_model = tk.BooleanVar(value=True)
        self.show_rays = tk.BooleanVar(value=False)
        tk.Checkbutton(master, text="Modell anzeigen", variable=self.show_model).grid(row=9, column=0)
        tk.Checkbutton(master, text="Rayvisualisierung", variable=self.show_rays).grid(row=9, column=1)

        # Start + Export
        tk.Button(master, text="Simulation starten", command=self.run_simulation).grid(row=10, column=0, columnspan=2)
        tk.Button(master, text="Frequenz-Sweep", command=self.plot_freq_sweep).grid(row=11, column=0)
        tk.Button(master, text="Heatmap exportieren", command=self.export_heatmap).grid(row=11, column=1)

        self.mesh = None
        self.last_rcs = None
        self.last_az = None
        self.last_el = None

    def apply_preset(self, selection):
        if selection in radar_presets:
            self.freq_scale.set(radar_presets[selection])

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("3D-Dateien", "*.stl *.obj *.glb *.gltf")])
        if path:
            self.mesh = trimesh.load_mesh(path, force='mesh')
            self.file_label.config(text=os.path.basename(path))

    def run_simulation(self):
        if self.mesh is None:
            print("Kein Modell geladen")
            return
        material = material_db[self.material_var.get()]
        freq = self.freq_scale.get() / 1000
        refl = self.refl_slider.get()
        rot = rotation_matrix(self.yaw.get(), self.pitch.get(), self.roll.get())
        rotated = self.mesh.copy()
        rotated.apply_transform(np.vstack((np.hstack((rot, [[0], [0], [0]])), [0, 0, 0, 1])))
        az, el, rcs = simulate_rcs(rotated, material, refl, freq)
        self.last_rcs = rcs
        self.last_az = az
        self.last_el = el
        self.plot_2d()
        self.plot_3d(rotated)
        self.export_csv()

    def plot_2d(self):
        if self.last_rcs is None:
            return
        idx = np.argmin(np.abs(self.last_el - self.el_slider.get()))
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.plot(np.radians(self.last_az), self.last_rcs[idx], label=f'Elevation {self.last_el[idx]}°')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title("2D Radar Cross Section")
        ax.legend()
        try:
            img = mpimg.imread("aircraft_shadow.png")
            axins = fig.add_axes([0.3, 0.3, 0.4, 0.4], polar=True, zorder=-1)
            axins.imshow(img, extent=[0, 2*np.pi, 0, np.max(self.last_rcs)], aspect='auto', alpha=0.3)
            axins.axis('off')
        except:
            pass
        plt.show()

    def plot_3d(self, mesh):
        if self.last_rcs is None:
            return
        az_rad = np.radians(self.last_az)
        el_rad = np.radians(self.last_el)
        az_grid, el_grid = np.meshgrid(az_rad, el_rad)
        r = 10 ** (self.last_rcs / 10.0)
        x = r * np.cos(el_grid) * np.cos(az_grid)
        y = r * np.cos(el_grid) * np.sin(az_grid)
        z = r * np.sin(el_grid)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, facecolors=plt.cm.viridis((r - r.min()) / (r.max() - r.min())), rstride=1, cstride=1, alpha=0.9)
        if self.show_model.get():
            ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces], facecolor='gray', edgecolor='k', alpha=0.2))
        ax.set_box_aspect([1, 1, 1])
        ax.set_title("3D RCS Plot")
        plt.show()

    def export_csv(self):
        with open("rcs_export.csv", mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Azimuth (°)', 'Elevation (°)', 'RCS (dBsm)'])
            for i in range(len(self.last_el)):
                for j in range(len(self.last_az)):
                    writer.writerow([self.last_az[j], self.last_el[i], self.last_rcs[i, j]])

    def export_heatmap(self):
        if self.last_rcs is None:
            return
        plt.imshow(self.last_rcs, extent=[0, 360, -90, 90], aspect='auto', cmap='hot', origin='lower')
        plt.colorbar(label="RCS (dBsm)")
        plt.title("RCS Heatmap")
        plt.xlabel("Azimuth")
        plt.ylabel("Elevation")
        plt.savefig("rcs_heatmap.png")
        plt.close()
        print("Heatmap exportiert")

    def plot_freq_sweep(self):
        if self.mesh is None:
            return
        material = material_db[self.material_var.get()]
        refl = self.refl_slider.get()
        freqs = np.linspace(1, 35, 30)
        values = []
        for f in freqs:
            _, _, rcs = simulate_rcs(self.mesh, material, refl, f)
            mid_el = rcs.shape[0] // 2
            avg_rcs = np.mean(rcs[mid_el])
            values.append(avg_rcs)
        plt.plot(freqs, values)
        plt.xlabel("Frequenz (GHz)")
        plt.ylabel("RCS (dBsm)")
        plt.title("Frequenz-Sweep")
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    root = tk.Tk()
    gui = RadarGUI(root)
    root.mainloop()
