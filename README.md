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
- A radar-style heatmap tab that plots azimuth/elevation RCS in article-like panels with adjustable color maps, clip limits, and
  either single-tone, median-of-sweep, or all-tone grid layouts for quick comparisons.
- Additional radar presets (SA-2/SA-3/SA-6, Aegis SPY-1, Giraffe AMB) plus more realistic material presets including naval, composite, and ground clutter substrates.
- Surface-roughness and speckle sliders plus an optional random seed to add grain when plots look overly smooth.
- Rotor/NCTR helpers (blade count and RPM) and compressor blade inputs that estimate micro-Doppler lines and carry them into templates for recognition workflows.
- An engine/prop placement table that marks nacelles/props on the 3D preview, saves them with projects/templates, and includes an auto-pair helper along the fuselage X-axis.
- A bundled F-16A article-style reference dataset (9.6 GHz) that you can load from **Reference data → Load F-16A article-style dataset** to compare your own meshes against the blog example without re-running a simulation.

### Loading reference datasets
- Pick **Reference data → Load F-16A article-style dataset** to populate the plots with the canned 9.6 GHz sweep and Doppler annotations from the blog post.
- Switch between the **3D plot** and **Radar heatmap** tabs to see the same dataset rendered as either polar slices or radar-panel grids; both views stay synchronized with the frequency selector.
- Start your own simulation afterward to overwrite the loaded dataset with fresh results while retaining the same visualization settings.

### NCTR preview
Use the **NCTR-Vorschau** button to generate a micro-Doppler signature preview. The resulting spectrogram can be exported as CSV and reused as a simple NCTR template for aircraft identification workflows.

### Radar heatmap view
Switch to the **Radar heatmap** tab to see the same data rendered as article-style azimuth/elevation panels. You can pick a
radar-like color map, tighten dB clip bounds to accentuate hot lobes, and choose between the selected tone, the median of a
sweep, or an all-tone grid to mirror the multi-band radar figure from the F-16A example article.

For more realistic NCTR signatures, combine high-fidelity meshes (include antennae/rotors/weapon pylons/compressor stages) with accurate material choices, populate blade-count/RPM for rotating parts (including the inlet compressor), and add a small surface roughness sigma plus speckle grain to break up perfectly smooth specular returns. The viewer normalizes RCS to the maximum value, so low roughness settings will appear smooth and round by design.
