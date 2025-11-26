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

### NCTR preview
Use the **NCTR-Vorschau** button to generate a micro-Doppler signature preview. The resulting spectrogram can be exported as CSV and reused as a simple NCTR template for aircraft identification workflows.
