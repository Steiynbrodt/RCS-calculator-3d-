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
- Additional radar presets (SA-2/SA-3/SA-6, Aegis SPY-1, Giraffe AMB) plus more realistic material presets including naval, composite, and ground clutter substrates.
- Surface-roughness and speckle sliders plus an optional random seed to add grain when plots look overly smooth.
- Rotor/NCTR helpers (blade count and RPM) and compressor blade inputs that estimate micro-Doppler lines and carry them into templates for recognition workflows.
- An engine/prop placement table that marks nacelles/props on the 3D preview, saves them with projects/templates, and includes an auto-pair helper along the fuselage X-axis.

### NCTR preview
Use the **NCTR-Vorschau** button to generate a micro-Doppler signature preview. The resulting spectrogram can be exported as CSV and reused as a simple NCTR template for aircraft identification workflows.

For more realistic NCTR signatures, combine high-fidelity meshes (include antennae/rotors/weapon pylons/compressor stages) with accurate material choices, populate blade-count/RPM for rotating parts (including the inlet compressor), and add a small surface roughness sigma plus speckle grain to break up perfectly smooth specular returns. The viewer normalizes RCS to the maximum value, so low roughness settings will appear smooth and round by design.
