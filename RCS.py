# Radar RCS Simulator – Vollständiger Code (GUI + Visualisierung + Export)

import numpy as np
import trimesh
import traceback
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import csv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Imports ---

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
    "Graphit":     {"sigma": 1e3,    "eps_r": 12.0, "tan_delta": 0.1, "reflectivity": 0.5},
"RAM – Iron Ball Paint":      {"sigma": 1e-4, "eps_r": 10.0, "tan_delta": 0.15, "reflectivity": 0.25},
"RAM – Carbon Nanotube":      {"sigma": 1e2,  "eps_r": 12.0, "tan_delta": 0.1,  "reflectivity": 0.1},
"RAM – Conductive Polymer":   {"sigma": 1e0,  "eps_r": 5.0,  "tan_delta": 0.3,  "reflectivity": 0.2},
"RAM – Magnetic Ferrite":     {"sigma": 1e-2, "eps_r": 13.0, "tan_delta": 0.5,  "reflectivity": 0.15},
"RAM – Foam Layer":           {"sigma": 1e-6, "eps_r": 1.5,  "tan_delta": 0.05, "reflectivity": 0.3},
"RAM – Dallenbach Layer":     {"sigma": 1e-5, "eps_r": 7.0,  "tan_delta": 0.2,  "reflectivity": 0.05},
"RAM – Jaumann Layer":        {"sigma": 1e-4, "eps_r": 9.0,  "tan_delta": 0.4,  "reflectivity": 0.05},
 "MagRAM":                {"sigma": 1e-4, "eps_r": 15.0, "tan_delta": 0.25, "reflectivity": 0.05},
    "Metamaterial RAM":      {"sigma": 1e-7, "eps_r": 25.0, "tan_delta": 0.4,  "reflectivity": 0.01},
    "Carbon Nanotube Foam":  {"sigma": 1e1,  "eps_r": 7.5,  "tan_delta": 0.03, "reflectivity": 0.15},
    "Spray-on Polymer RAM":  {"sigma": 5e-3, "eps_r": 9.0,  "tan_delta": 0.12, "reflectivity": 0.2}

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
    "S-400 91N6E Big Bird (6 GHz)": 6000,
    "Su-57 N036 Belka (9 GHz)": 9000,
    "MiG-31 Zaslon (10.2 GHz)": 10200,
    "S-300 Flap Lid (9.2 GHz)": 9200,
    "S-400 Grave Stone (10 GHz)": 10000,
    "S-400 Big Bird (6 GHz)": 6000,
    "Konteyner OTH Radar (0.02 GHz)": 20,
    "Nebo SV (0.15 GHz)": 150,
    "Nebo-M Multiband (1.5 GHz)": 1500,
    "96L6E Cheese Board (3 GHz)": 3000,
    "Kasta 2E1 (1.2 GHz)": 1200
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


# No longer needed, noise is added vectorized in simulate_rcs

# --- Raytracing ---
def trace_ray(mesh, origin, direction, max_depth, reflectivity, freq_ghz):
    """
    Traces a single radar ray on the mesh with recursive reflections.
    
    Parameters:
        mesh: trimesh.Trimesh – The mesh to trace on.
        origin: np.ndarray – Starting point of the ray (3,).
        direction: np.ndarray – Normalized direction of the ray (3,).
        max_depth: int – Maximum number of reflections.
        reflectivity: float – Material reflectivity coefficient.
        freq_ghz: float – Frequency in GHz (affects energy loss).
    
    Returns:
        rcs: float – Accumulated radar cross section contribution.
    """
    rays = mesh.ray  # Always use built-in intersector (safe)
    energy = 1.0
    rcs = 0.0
    loss_per_reflection = frequency_dependent_loss(freq_ghz)

    for _ in range(max_depth):
        try:
            locs, _, tri_idx = rays.intersects_location(
                np.array([origin]), np.array([direction]), multiple_hits=False)
        except Exception as e:
            # Fall back silently if the intersector fails
            print(f"[Warn] Raytracing error: {e}")
            break

        if len(locs) == 0:
            break  # no hit

        hit = locs[0]
        normal = mesh.face_normals[tri_idx[0]]

        # Calculate reflection
        reflect_dir = direction - 2 * np.dot(direction, normal) * normal
        reflect_dir /= np.linalg.norm(reflect_dir)

        # Only add contribution if reflection is strong
        if np.dot(reflect_dir, -direction) > 0.95:
            contrib = energy * mesh.area_faces[tri_idx[0]] * (np.dot(normal, -direction))**2
            rcs += contrib

        # Update for next bounce
        energy *= reflectivity * loss_per_reflection
        origin = hit + 1e-4 * reflect_dir  # offset to avoid self-hit
        direction = reflect_dir

    return rcs

# --- Simulation ---
def simulate_rcs(mesh, material, max_reflections, freq_ghz, az_steps=360, el_steps=181):
    import traceback
    if mesh is None or not hasattr(mesh, 'bounding_sphere'):
        raise ValueError("Kein gültiges 3D-Mesh geladen.")
    
    az = np.linspace(0, 360, az_steps)
    el = np.linspace(-90, 90, el_steps)
    az_grid, el_grid = np.meshgrid(az, el)
    dirs = sph2cart(az_grid, el_grid)
    origins = mesh.bounding_sphere.center + 100 * (-dirs.reshape(-1, 3))
    directions = dirs.reshape(-1, 3)
    reflectivity = material["reflectivity"]

    def ray_func(args):
        origin, direction = args
        try:
            return trace_ray(mesh, origin, direction, max_reflections, reflectivity, freq_ghz)
        except Exception as e:
            print("[simulate_rcs] Fehler beim Raytrace:", e)
            traceback.print_exc()
            return 0.0

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        rcs_lin = list(executor.map(ray_func, zip(origins, directions)))

    rcs_lin = np.clip(np.array(rcs_lin).reshape(dirs.shape[:2]), 1e-10, None)
    rcs_db = 10 * np.log10(rcs_lin)
    rcs_db += np.random.normal(0, 2, rcs_db.shape)
    return az, el, rcs_db

# robustere Version von plot_freq_sweep
def robust_freq_sweep(mesh, material, max_reflections, master_callback=None):
    import matplotlib.pyplot as plt
    import traceback
    from tkinter import messagebox

    if mesh is None:
        print("[Frequenz-Sweep] Kein Mesh geladen.")
        if master_callback:
            master_callback(lambda: messagebox.showerror("Fehler", "Kein 3D-Modell geladen."))
        return

    freqs = np.linspace(1, 35, 30)
    values = []

    for f in freqs:
        try:
            _, _, rcs = simulate_rcs(mesh, material, max_reflections, f)
            mid_el = rcs.shape[0] // 2
            avg_rcs = np.mean(rcs[mid_el])
            values.append(avg_rcs)
        except Exception as e:
            print(f"[freq_sweep] Fehler bei {f} GHz:", e)
            traceback.print_exc()
            values.append(-100)

    def plot_result():
        plt.plot(freqs, values)
        plt.xlabel("Frequenz (GHz)")
        plt.ylabel("RCS (dBsm)")
        plt.title("Frequenz-Sweep")
        plt.grid(True)
        plt.show()

    if master_callback:
        master_callback(plot_result)
    else:
        plot_result()

class RadarGUI:
    def __init__(self, master):
        self.master = master
        master.title("Radar RCS Simulation")

        self.main_frame = ttk.Frame(master, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        

   
        self.master = master
        master.title("Radar RCS Simulation")
        master.geometry("800x600")
        master.minsize(700, 500)
        master.configure(bg="#222831")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background="#222831", foreground="#eeeeee", font=("Segoe UI", 11))
        style.configure('TButton', font=("Segoe UI", 11), padding=6)
        style.configure('TCheckbutton', background="#222831", foreground="#eeeeee", font=("Segoe UI", 11))
        style.configure('TMenubutton', font=("Segoe UI", 11))

        self.main_frame = ttk.Frame(master, padding=20, style='TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(tuple(range(13)), weight=1)
        # All widgets must be created before using them in toggle_live_mode
        # --- Controls ---
        ttk.Label(self.main_frame, text="3D-Modell (STL, OBJ, GLB):").grid(row=0, column=0, sticky="e", pady=4)
        self.file_label = ttk.Label(self.main_frame, text="Keine Datei", style='TLabel')
        self.file_label.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(self.main_frame, text="Laden", command=self.load_file).grid(row=0, column=2, padx=5, pady=4)

        ttk.Label(self.main_frame, text="Material:").grid(row=1, column=0, sticky="e", pady=4)
        self.material_var = tk.StringVar(master)
        self.material_var.set("Aluminium")
        material_menu = ttk.OptionMenu(self.main_frame, self.material_var, "Aluminium", *material_db.keys())
        material_menu.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(self.main_frame, text="Radarprofil:").grid(row=2, column=0, sticky="e", pady=4)
        self.preset_var = tk.StringVar(master)
        self.preset_var.set("NATO – X-Band (10 GHz)")
        preset_menu = ttk.OptionMenu(self.main_frame, self.preset_var, "NATO – X-Band (10 GHz)", *radar_presets.keys(), command=self.apply_preset)
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

        self.show_model = tk.BooleanVar(value=True)
        self.show_rays = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.main_frame, text="Modell anzeigen", variable=self.show_model, style='TCheckbutton').grid(row=9, column=0, sticky="w", pady=4)
        ttk.Checkbutton(self.main_frame, text="Rayvisualisierung", variable=self.show_rays, style='TCheckbutton').grid(row=9, column=1, sticky="w", pady=4)

        ttk.Button(self.main_frame, text="Simulation starten", command=self.run_simulation).grid(row=10, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Button(self.main_frame, text="Frequenz-Sweep", command=lambda: robust_freq_sweep(self.mesh,material_db[self.material_var.get()],self.refl_slider.get(),lambda func: self.master.after(0, func))).grid(row=11, column=0, sticky="ew", pady=4)
        ttk.Button(self.main_frame, text="Heatmap exportieren", command=self.export_heatmap).grid(row=11, column=1, sticky="ew", pady=4)
        
        self.mesh = None
        self.last_rcs = None
        self.last_az = None
        self.last_el = None

        self.live_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.main_frame, text="Live Simulation (experimentell)", variable=self.live_mode, style='TCheckbutton', command=self.toggle_live_mode).grid(row=12, column=0, columnspan=2, sticky="w", pady=4)
    def toggle_live_mode(self):
        if self.live_mode.get():
            self.freq_scale.bind('<ButtonRelease-1>', self.live_update)
            self.refl_slider.bind('<ButtonRelease-1>', self.live_update)
            self.el_slider.bind('<ButtonRelease-1>', self.live_update)
            self.yaw.bind('<ButtonRelease-1>', self.live_update)
            self.pitch.bind('<ButtonRelease-1>', self.live_update)
            self.roll.bind('<ButtonRelease-1>', self.live_update)
            self.material_var.trace_add('write', lambda *args: self.live_update())
            self.preset_var.trace_add('write', lambda *args: self.live_update())
        else:
            self.freq_scale.unbind('<ButtonRelease-1>')
            self.refl_slider.unbind('<ButtonRelease-1>')
            self.el_slider.unbind('<ButtonRelease-1>')
            self.yaw.unbind('<ButtonRelease-1>')
            self.pitch.unbind('<ButtonRelease-1>')
            self.roll.unbind('<ButtonRelease-1>')
    def live_update(self, event=None):
        import threading
        if self.mesh is None or not self.live_mode.get():
            return
        def worker():
            try:
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
                self.master.after(0, self.plot_2d)
                self.master.after(0, lambda: self.plot_3d(rotated))
                # Export to Documents/RCS_Live_Exports
                import os
                export_dir = os.path.expanduser('~/Documents/RCS_Live_Exports')
                os.makedirs(export_dir, exist_ok=True)
                self.export_csv(export_dir, live=True)
                self.export_heatmap(export_dir, live=True)
            except Exception as e:
                pass
        threading.Thread(target=worker, daemon=True).start()

        ttk.Label(self.main_frame, text="3D-Modell (STL, OBJ, GLB):").grid(row=0, column=0, sticky="e", pady=4)
        self.file_label = ttk.Label(self.main_frame, text="Keine Datei", style='TLabel')
        self.file_label.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(self.main_frame, text="Laden", command=self.load_file).grid(row=0, column=2, padx=5, pady=4)

        ttk.Label(self.main_frame, text="Material:").grid(row=1, column=0, sticky="e", pady=4)
        self.material_var = tk.StringVar(self.master)
        self.material_var.set("Aluminium")
        material_menu = ttk.OptionMenu(self.main_frame, self.material_var, "Aluminium", *material_db.keys())
        material_menu.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(self.main_frame, text="Radarprofil:").grid(row=2, column=0, sticky="e", pady=4)
        self.preset_var = tk.StringVar(self.master)
        self.preset_var.set("NATO – X-Band (10 GHz)")
        preset_menu = ttk.OptionMenu(self.main_frame, self.preset_var, "NATO – X-Band (10 GHz)", *radar_presets.keys(), command=self.apply_preset)
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

        self.show_model = tk.BooleanVar(value=True)
        self.show_rays = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.main_frame, text="Modell anzeigen", variable=self.show_model, style='TCheckbutton').grid(row=9, column=0, sticky="w", pady=4)
        ttk.Checkbutton(self.main_frame, text="Rayvisualisierung", variable=self.show_rays, style='TCheckbutton').grid(row=9, column=1, sticky="w", pady=4)

        ttk.Button(self.main_frame, text="Simulation starten", command=self.run_simulation).grid(row=10, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Button(self.main_frame, text="Frequenz-Sweep", command=self.plot_freq_sweep).grid(row=11, column=0, sticky="ew", pady=4)
        ttk.Button(self.main_frame, text="Heatmap exportieren", command=self.export_heatmap).grid(row=11, column=1, sticky="ew", pady=4)

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
            try:
                mesh = trimesh.load_mesh(path, force='mesh')

                if not isinstance(mesh, trimesh.Trimesh):
                    if isinstance(mesh, trimesh.Scene):
                        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
                    else:
                        raise ValueError("Unbekannter Mesh-Typ geladen.")

                mesh.remove_unreferenced_vertices()
                mesh.update_faces(mesh.nondegenerate_faces())  # Statt remove_degenerate_faces (veraltet)

                from trimesh.ray.ray_triangle import RayMeshIntersector
                try:
                    mesh.ray = RayMeshIntersector(mesh)
                except Exception as e:
                    messagebox.showerror("Raytracer Fehler", f"Raytracer konnte nicht initialisiert werden:\n{e}")
                    self.mesh = None
                    return

                self.mesh = mesh
                self.file_label.config(text=os.path.basename(path))

            except Exception as e:
                messagebox.showerror("Fehler beim Laden", f"Das Modell konnte nicht geladen werden:\n{e}")
                self.mesh = None
                self.file_label.config(text="Keine Datei")




    def run_simulation(self):
        if self.mesh is None:
            messagebox.showwarning("Kein Modell", "Kein Modell geladen.")
            return
        try:
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
        except Exception as e:
            messagebox.showerror("Fehler bei Simulation", f"Fehler während der Simulation:\n{e}")

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
        r_raw = 10 ** (self.last_rcs / 10.0)
        # Scale RCS bubble to match model size
        model_radius = np.linalg.norm(mesh.bounding_box.extents) / 2
        r = r_raw / np.max(r_raw) * model_radius * 1.2  # 1.2 = extra margin
        x = r * np.cos(el_grid) * np.cos(az_grid)
        y = r * np.cos(el_grid) * np.sin(az_grid)
        z = r * np.sin(el_grid)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis((r_raw - r_raw.min()) / (r_raw.max() - r_raw.min())), rstride=1, cstride=1, alpha=0.5)
        mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        mappable.set_array(r_raw)
        fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="RCS (linear)")
        if self.show_model.get():
            ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces], facecolor='gray', edgecolor='k', alpha=0.2))
        ax.set_box_aspect([1, 1, 1])
        ax.set_title("3D RCS Plot")
        plt.show()

    def export_csv(self, export_dir=None, live=False):
        import os
        if export_dir is None and not live:
            from tkinter import filedialog
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Speichern unter...")
            if not file_path:
                return
        elif export_dir is not None:
            file_path = os.path.join(export_dir, "rcs_export.csv")
        else:
            file_path = "rcs_export.csv"
        try:
            with open(file_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Azimuth (°)', 'Elevation (°)', 'RCS (dBsm)'])
                for i in range(len(self.last_el)):
                    for j in range(len(self.last_az)):
                        writer.writerow([self.last_az[j], self.last_el[i], self.last_rcs[i, j]])
            if not live:
                messagebox.showinfo("Export erfolgreich", f"RCS-Daten wurden als {os.path.basename(file_path)} gespeichert.")
        except Exception as e:
            if not live:
                messagebox.showerror("Fehler beim Export", f"Fehler beim Exportieren der CSV:\n{e}")

    def export_heatmap(self, export_dir=None, live=False):
        import threading
        import os
        def worker():
            if self.last_rcs is None:
                return
            try:
                if export_dir is None and not live:
                    from tkinter import filedialog
                    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")], title="Speichern unter...")
                    if not file_path:
                        return
                elif export_dir is not None:
                    file_path = os.path.join(export_dir, "rcs_heatmap.png")
                else:
                    file_path = "rcs_heatmap.png"
                plt.imshow(self.last_rcs, extent=[0, 360, -90, 90], aspect='auto', cmap='hot', origin='lower')
                plt.colorbar(label="RCS (dBsm)")
                plt.title("RCS Heatmap")
                plt.xlabel("Azimuth")
                plt.ylabel("Elevation")
                plt.savefig(file_path)
                plt.close()
                if not live:
                    self.master.after(0, lambda: messagebox.showinfo("Export erfolgreich", f"Heatmap wurde als {os.path.basename(file_path)} gespeichert."))
            except Exception as e:
                if not live:
                    self.master.after(0, lambda e=e: messagebox.showerror("Fehler beim Export", f"Fehler beim Exportieren der Heatmap:\n{e}"))
        threading.Thread(target=worker, daemon=True).start()


if __name__ == '__main__':
    root = tk.Tk()
    gui = RadarGUI(root)
    root.mainloop()
