import numpy as np
import trimesh
import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import csv

# Materialdatenbank (Radar-relevant)
material_db = {
    "Aluminium":   {"sigma": 3.5e7,  "eps_r": 1.0, "tan_delta": 0.0, "reflectivity": 0.99},
    "Stahl":       {"sigma": 1e7,    "eps_r": 1.0, "tan_delta": 0.0, "reflectivity": 0.98},
    "Carbon":      {"sigma": 1e4,    "eps_r": 5.0, "tan_delta": 0.05, "reflectivity": 0.6},
    "PVC":         {"sigma": 1e-15,  "eps_r": 3.0, "tan_delta": 0.02, "reflectivity": 0.3},
    "Gummi (RAM)": {"sigma": 1e-5,   "eps_r": 10.0, "tan_delta": 0.3, "reflectivity": 0.1},
}

# Radar-Frequenz-Presets (GHz)
radar_presets = {
    "NATO – L-Band (1.3 GHz)": 1300,
    "NATO – X-Band (10 GHz)": 10000,
    "Russland – S-Band (3.5 GHz)": 3500,
    "Russland – L-Band (1.4 GHz)": 1400,
}

# Konvertiert sphärisch zu kartesisch
def sph2cart(az, el):
    az = np.radians(az)
    el = np.radians(el)
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return np.stack((x, y, z), axis=-1)

# Raytracing mit Mehrweg-Reflexionen
def trace_ray(mesh, origin, direction, max_depth, reflectivity, freq_ghz):
    try:
        rays = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    except Exception:
        rays = mesh.ray

    energy = 1.0
    rcs = 0.0
    loss_per_reflection = frequency_dependent_loss(freq_ghz)

    for depth in range(max_depth):
        locs, idx_ray, idx_tri = rays.intersects_location([origin], [direction], multiple_hits=False)
        if len(locs) == 0:
            break
        hit = locs[0]
        tri = idx_tri[0]
        normal = mesh.face_normals[tri]
        reflect_dir = direction - 2 * np.dot(direction, normal) * normal
        reflect_dir /= np.linalg.norm(reflect_dir)

        # Check if reflected ray returns to source (simple model)
        if np.dot(reflect_dir, -direction) > 0.98:
            contrib = energy * mesh.area_faces[tri] * (np.dot(normal, -direction))**2
            rcs += contrib

        energy *= reflectivity * loss_per_reflection
        origin = hit + 1e-4 * reflect_dir
        direction = reflect_dir

    return rcs

def frequency_dependent_loss(freq_ghz):
    return max(0.8 - 0.02 * (freq_ghz - 1), 0.2)

# Simulation 
def simulate_rcs(mesh, material, max_reflections, freq_ghz, az_steps=72, el_steps=36):
    az = np.linspace(0, 360, az_steps)
    el = np.linspace(-90, 90, el_steps)
    az_grid, el_grid = np.meshgrid(az, el)
    dirs = sph2cart(az_grid, el_grid)

    rcs_db = np.zeros_like(az_grid)
    for i in range(dirs.shape[0]):
        for j in range(dirs.shape[1]):
            origin = mesh.bounding_sphere.center + 100 * (-dirs[i, j])
            rcs_lin = trace_ray(mesh, origin, dirs[i, j], max_reflections, material["reflectivity"], freq_ghz)
            rcs_db[i, j] = 10 * np.log10(rcs_lin + 1e-6)

    return az, el, rcs_db

# 2D-Polarplot
def plot_2d(az, rcs_db):
    plt.figure()
    plt.polar(np.radians(az), rcs_db, label='Elevation 0°')
    plt.title("2D Radar RCS (dB)")
    plt.legend()
    plt.show()

# 3D-Kugelplot
def plot_3d(az, el, rcs_db):
    az_rad = np.radians(az)
    el_rad = np.radians(el)
    az_grid, el_grid = np.meshgrid(az_rad, el_rad)

    rcs_norm = (rcs_db - rcs_db.min()) / (rcs_db.max() - rcs_db.min())
    r = np.clip(rcs_db, 0.01, None)
    x = r * np.cos(el_grid) * np.cos(az_grid)
    y = r * np.cos(el_grid) * np.sin(az_grid)
    z = r * np.sin(el_grid)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(x, y, z, facecolors=plt.cm.jet(rcs_norm), rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.set_title("3D Radar RCS")
    plt.colorbar(surface, label="RCS (dB)")
    plt.show()

# CSV Export
def export_to_csv(az, el, rcs_db, filename):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Azimuth (°)', 'Elevation (°)', 'RCS (dB)'])
        for i in range(len(el)):
            for j in range(len(az)):
                writer.writerow([az[j], el[i], rcs_db[i, j]])

# GUI
class RadarGUI:
    def __init__(self, master):
        self.master = master
        master.title("Radar-RCS Simulation")

        tk.Label(master, text="STL-Modell:").grid(row=0, column=0)
        self.file_label = tk.Label(master, text="Keine Datei gewählt")
        self.file_label.grid(row=0, column=1)
        tk.Button(master, text="Datei wählen", command=self.load_file).grid(row=0, column=2)

        tk.Label(master, text="Material:").grid(row=1, column=0)
        self.material_var = tk.StringVar(master)
        self.material_var.set("Aluminium")
        ttk.OptionMenu(master, self.material_var, "Aluminium", *material_db.keys()).grid(row=1, column=1)

        tk.Label(master, text="Radarprofil:").grid(row=2, column=0)
        self.preset_var = tk.StringVar(master)
        self.preset_var.set("NATO – X-Band (10 GHz)")
        ttk.OptionMenu(master, self.preset_var, "NATO – X-Band (10 GHz)", *radar_presets.keys(), command=self.apply_preset).grid(row=2, column=1)

        tk.Label(master, text="Max. Reflexionen:").grid(row=3, column=0)
        self.reflections_scale = tk.Scale(master, from_=1, to=10, orient=tk.HORIZONTAL)
        self.reflections_scale.set(3)
        self.reflections_scale.grid(row=3, column=1)

        tk.Label(master, text="Radarfrequenz (MHz):").grid(row=4, column=0)
        self.freq_scale = tk.Scale(master, from_=100, to=10000, orient=tk.HORIZONTAL, resolution=100)
        self.freq_scale.set(10000)
        self.freq_scale.grid(row=4, column=1)

        tk.Button(master, text="Simulation starten", command=self.run_simulation).grid(row=5, column=0, columnspan=3)

        self.mesh = None

    def apply_preset(self, selection):
        if selection in radar_presets:
            self.freq_scale.set(radar_presets[selection])

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("3D-Dateien", "*.stl *.obj *.ply *.glb *.gltf")])
        if path:
            self.mesh = trimesh.load_mesh(path, force='mesh')
            self.file_label.config(text=os.path.basename(path))

    def run_simulation(self):
        if self.mesh is None:
            print("Kein Modell geladen")
            return

        material = material_db[self.material_var.get()]
        max_ref = self.reflections_scale.get()
        freq_mhz = self.freq_scale.get()
        freq_ghz = freq_mhz / 1000.0

        az, el, rcs_db = simulate_rcs(self.mesh, material, max_ref, freq_ghz)

        # 2D Plot bei Elevation = 0
        mid = el.shape[0] // 2
        plot_2d(az, rcs_db[mid, :])

        # 3D Plot
        plot_3d(az, el, rcs_db)

        # CSV Export
        export_to_csv(az, el, rcs_db, "rcs_export.csv")
        print("RCS-Daten exportiert nach rcs_export.csv")

if __name__ == "__main__":
    root = tk.Tk()
    gui = RadarGUI(root)
    root.mainloop()
