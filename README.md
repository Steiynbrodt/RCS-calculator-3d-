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

Load an STL/OBJ/GLB/GLTF model, select a material and radar preset, then start the simulation. You can also perform a frequency sweep and export results for further analysis. The PyQt5 front-end now includes:

- Predefined NATO and Russian SAM radar presets (S-300/400, Patriot, NASAMS, MEADS, etc.) that automatically configure band and frequency settings.
- An aircraft speed field (m/s) that is preserved with simulation results, templates, and project files for Doppler-focused analysis.
- A frequency selector on the 3D plot tab so you can inspect individual tones from a sweep instead of only the first sample.
- A dedicated heatmap tab that renders the azimuth/elevation RCS grid per frequency and exports alongside the polar and 3D plots.
- A toggle between multi-bounce ray tracing and a physical-optics-style facet summation, so you can pick the solver that best matches your scenario.
- Engine/propeller placement editors with RPM and yaw settings that inject simplified powerplant returns into the RCS solution and are saved inside project files.

### PyPOFacets-inspired facet summation
PyPOFacets computes monostatic RCS by summing illuminated mesh facets using a physical-optics flat-plate approximation. A similar mode is now available from code by setting `SimulationSettings.method="facet_po"`, which accumulates `4πA²/λ² cos²θ` per visible facet and scales by material reflectivity. This approach avoids multi-bounce ray tracing noise and provides a faster baseline for complex meshes when you want results closer to a physical-optics summation.

### NCTR preview
Use the **NCTR-Vorschau** button to generate a micro-Doppler signature preview. The resulting spectrogram can be exported as CSV and reused as a simple NCTR template for aircraft identification workflows.
