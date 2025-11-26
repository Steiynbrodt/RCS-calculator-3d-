# Radar RCS Studio

A PyQt5 application for estimating radar cross section (RCS) distributions of
3D models. Load an STL/OBJ/GLTF mesh, select radar parameters, run sweeps, and
visualize polar or 3D RCS plots. Results can be exported as CSV/PNG, saved into
project files, or captured as NCTR-style signature templates for later matching.

## Features
- Band presets for L/S/X with editable frequencies or sweeps
- Azimuth/elevation sweeps and polarization selection (H/V)
- Material database stored in JSON with add/edit/delete UI
- Background simulation worker to keep the UI responsive
- 2D polar overlays, 3D RCS visualization, and mesh preview
- CSV/PNG export plus JSON project save/load
- Signature template library with matching scores

## Installation
```bash
pip install -r requirements.txt
```

If you see an error about missing Qt binding modules (PyQt5/PySide2), install
PyQt5 manually:

```bash
pip install PyQt5
```

## Running the GUI
```bash
python -m rcs.main
# or
python RCS.py
```

## Basic workflow
1. **Load a mesh** via *File → Open STL*.
2. **Choose radar settings** on the left: band, frequency (single or sweep),
   angles, polarization, material, and maximum reflections.
3. Click **Run simulation** to start a threaded RCS sweep. Progress is shown in
   the status bar.
4. Inspect the **3D model, 2D polar plot, and 3D RCS** tabs. Use plot controls
   to change elevation slices or color limits.
5. **Export** data with *File → Export CSV* or *File → Export Plots*.
6. **Save projects** with *File → Save Project* to preserve all current
   settings and reload them later.
7. **Templates:** after a simulation, use *Templates → Create from result* to
   store a signature. Run *Templates → Run template matching* to compare the
   current result against the library and view scores in the Templates tab.

