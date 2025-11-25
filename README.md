# Radar RCS Simulator

A Tkinter-based tool for estimating radar cross section (RCS) distributions of 3D models using a simple multi-bounce ray tracing approach. Results can be visualized in 2D/3D and exported as CSV or PNG heatmaps.

## Installation
```
pip install -r requirements.txt
```

## Usage
Run the graphical interface via either command:
```
python -m rcs.main
# or
python RCS.py
```

Load an STL/OBJ/GLB/GLTF model, select a material and radar preset, then start the simulation. You can also perform a frequency sweep and export results for further analysis. To speed up runs and get a radar-focused perspective, you can now:

- Limit the simulated field of view with a radar azimuth/elevation look direction and beam width.
- Tune azimuth/elevation resolution to trade fidelity for speed.
- Adjust the radar range (relative to the object size) to emphasize realistic decay over distance.

## Automated releases
Each push to the `main` branch triggers a GitHub Actions workflow that builds a Windows `.exe` with PyInstaller and publishes it as a new release artifact.
