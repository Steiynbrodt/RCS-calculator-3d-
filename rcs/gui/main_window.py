"""PyQt5 main window for the RCS calculator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PyQt5 import QtCore, QtWidgets

from ..materials import MaterialDatabase
from ..project_io import ProjectState, load_project, save_project
from ..rcs_engine import BAND_DEFAULTS, FrequencySweep, Material, RCSEngine, SimulationResult, SimulationSettings
from ..templates import TemplateLibrary


class SimulationWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        engine: RCSEngine,
        mesh: trimesh.Trimesh,
        material: Material,
        settings: SimulationSettings,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.mesh = mesh
        self.material = material
        self.settings = settings

    def run(self) -> None:  # noqa: D401
        try:
            self.engine.reset()
            result = self.engine.compute(self.mesh, self.material, self.settings, progress=self.progress.emit)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(5, 4))
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111)

    def clear(self) -> None:
        self.figure.clf()
        self.ax = self.figure.add_subplot(111)
        self.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Radar RCS Studio")
        self.resize(1200, 800)

        self.material_db = MaterialDatabase()
        self.template_lib = TemplateLibrary()
        self.engine = RCSEngine()

        self.mesh: Optional[trimesh.Trimesh] = None
        self.mesh_path: Optional[Path] = None
        self.result: Optional[SimulationResult] = None

        self._build_ui()
        self._connect_actions()
        self._update_band_defaults()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        self.controls = self._build_controls()
        layout.addWidget(self.controls, 0)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        # Model preview tab
        self.model_canvas = PlotCanvas()
        model_tab = QtWidgets.QWidget()
        model_layout = QtWidgets.QVBoxLayout(model_tab)
        model_layout.addWidget(self.model_canvas)
        self.tabs.addTab(model_tab, "3D Model")

        # 2D polar plot tab
        self.polar_canvas = PlotCanvas()
        polar_tab = QtWidgets.QWidget()
        polar_layout = QtWidgets.QVBoxLayout(polar_tab)
        controls = QtWidgets.QHBoxLayout()
        self.elevation_spin = QtWidgets.QDoubleSpinBox()
        self.elevation_spin.setRange(-90, 90)
        self.elevation_spin.setValue(0.0)
        controls.addWidget(QtWidgets.QLabel("Elevation slice (deg):"))
        controls.addWidget(self.elevation_spin)
        self.scale_mode = QtWidgets.QComboBox()
        self.scale_mode.addItems(["dBsm", "Linear"])
        controls.addWidget(QtWidgets.QLabel("Scale:"))
        controls.addWidget(self.scale_mode)
        controls.addStretch(1)
        polar_layout.addLayout(controls)
        polar_layout.addWidget(self.polar_canvas)
        self.tabs.addTab(polar_tab, "2D Polar")

        # 3D RCS plot
        self.rcs3d_canvas = PlotCanvas()
        rcs_tab = QtWidgets.QWidget()
        rcs_layout = QtWidgets.QVBoxLayout(rcs_tab)
        rcs_controls = QtWidgets.QHBoxLayout()
        self.clip_min = QtWidgets.QDoubleSpinBox()
        self.clip_min.setRange(-100, 100)
        self.clip_min.setValue(-40)
        self.clip_max = QtWidgets.QDoubleSpinBox()
        self.clip_max.setRange(-20, 100)
        self.clip_max.setValue(20)
        rcs_controls.addWidget(QtWidgets.QLabel("Min dB:"))
        rcs_controls.addWidget(self.clip_min)
        rcs_controls.addWidget(QtWidgets.QLabel("Max dB:"))
        rcs_controls.addWidget(self.clip_max)
        rcs_controls.addStretch(1)
        rcs_layout.addLayout(rcs_controls)
        rcs_layout.addWidget(self.rcs3d_canvas)
        self.tabs.addTab(rcs_tab, "3D RCS")

        # Templates tab
        self.templates_table = QtWidgets.QTableWidget(0, 4)
        self.templates_table.setHorizontalHeaderLabels(["Name", "Class", "Band", "Score"])
        template_tab = QtWidgets.QWidget()
        template_layout = QtWidgets.QVBoxLayout(template_tab)
        btns = QtWidgets.QHBoxLayout()
        self.refresh_templates_btn = QtWidgets.QPushButton("Refresh")
        self.create_template_btn = QtWidgets.QPushButton("Create from result")
        self.match_template_btn = QtWidgets.QPushButton("Run template matching")
        btns.addWidget(self.refresh_templates_btn)
        btns.addWidget(self.create_template_btn)
        btns.addWidget(self.match_template_btn)
        btns.addStretch(1)
        template_layout.addLayout(btns)
        template_layout.addWidget(self.templates_table)
        self.tabs.addTab(template_tab, "Templates")

        # Status bar elements
        self.progress = QtWidgets.QProgressBar()
        self.status_label = QtWidgets.QLabel("Ready")
        status = QtWidgets.QStatusBar()
        status.addWidget(self.progress, 1)
        status.addWidget(self.status_label, 1)
        self.setStatusBar(status)

    def _build_controls(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        # File chooser
        file_layout = QtWidgets.QHBoxLayout()
        self.file_label = QtWidgets.QLabel("No file loaded")
        self.file_btn = QtWidgets.QPushButton("Open STL")
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_btn)
        form.addRow("Model:", file_layout)

        # Band
        self.band_combo = QtWidgets.QComboBox()
        self.band_combo.addItems(["L", "S", "X"])
        form.addRow("Band:", self.band_combo)

        # Frequency controls
        freq_layout = QtWidgets.QGridLayout()
        self.freq_mode = QtWidgets.QComboBox()
        self.freq_mode.addItems(["Single", "Sweep"])
        freq_layout.addWidget(QtWidgets.QLabel("Mode"), 0, 0)
        freq_layout.addWidget(self.freq_mode, 0, 1)
        self.single_freq = QtWidgets.QDoubleSpinBox()
        self.single_freq.setRange(0.5, 40.0)
        self.single_freq.setSuffix(" GHz")
        self.single_freq.setValue(10.0)
        freq_layout.addWidget(self.single_freq, 1, 0, 1, 2)
        self.sweep_start = QtWidgets.QDoubleSpinBox()
        self.sweep_start.setRange(0.5, 40.0)
        self.sweep_stop = QtWidgets.QDoubleSpinBox()
        self.sweep_stop.setRange(0.5, 40.0)
        self.sweep_steps = QtWidgets.QSpinBox()
        self.sweep_steps.setRange(2, 200)
        self.sweep_steps.setValue(5)
        freq_layout.addWidget(QtWidgets.QLabel("Start (GHz)"), 2, 0)
        freq_layout.addWidget(self.sweep_start, 2, 1)
        freq_layout.addWidget(QtWidgets.QLabel("Stop (GHz)"), 3, 0)
        freq_layout.addWidget(self.sweep_stop, 3, 1)
        freq_layout.addWidget(QtWidgets.QLabel("Steps"), 4, 0)
        freq_layout.addWidget(self.sweep_steps, 4, 1)
        form.addRow("Frequency:", freq_layout)

        # Angles
        self.az_start = QtWidgets.QDoubleSpinBox()
        self.az_start.setRange(0, 360)
        self.az_stop = QtWidgets.QDoubleSpinBox()
        self.az_stop.setRange(0, 360)
        self.az_stop.setValue(360)
        self.az_step = QtWidgets.QDoubleSpinBox()
        self.az_step.setRange(0.5, 45)
        self.az_step.setSingleStep(0.5)
        self.az_step.setValue(5)
        self.el_start = QtWidgets.QDoubleSpinBox()
        self.el_start.setRange(-90, 90)
        self.el_start.setValue(-90)
        self.el_stop = QtWidgets.QDoubleSpinBox()
        self.el_stop.setRange(-90, 90)
        self.el_stop.setValue(90)
        self.el_step = QtWidgets.QDoubleSpinBox()
        self.el_step.setRange(0.5, 30)
        self.el_step.setSingleStep(0.5)
        self.el_step.setValue(5)
        angle_grid = QtWidgets.QGridLayout()
        angle_grid.addWidget(QtWidgets.QLabel("Az start"), 0, 0)
        angle_grid.addWidget(self.az_start, 0, 1)
        angle_grid.addWidget(QtWidgets.QLabel("Az stop"), 1, 0)
        angle_grid.addWidget(self.az_stop, 1, 1)
        angle_grid.addWidget(QtWidgets.QLabel("Az step"), 2, 0)
        angle_grid.addWidget(self.az_step, 2, 1)
        angle_grid.addWidget(QtWidgets.QLabel("El start"), 3, 0)
        angle_grid.addWidget(self.el_start, 3, 1)
        angle_grid.addWidget(QtWidgets.QLabel("El stop"), 4, 0)
        angle_grid.addWidget(self.el_stop, 4, 1)
        angle_grid.addWidget(QtWidgets.QLabel("El step"), 5, 0)
        angle_grid.addWidget(self.el_step, 5, 1)
        form.addRow("Angles:", angle_grid)

        # Polarization / reflections
        self.pol_combo = QtWidgets.QComboBox()
        self.pol_combo.addItems(["H", "V"])
        form.addRow("Polarization:", self.pol_combo)
        self.reflections_spin = QtWidgets.QSpinBox()
        self.reflections_spin.setRange(1, 10)
        self.reflections_spin.setValue(3)
        form.addRow("Max reflections:", self.reflections_spin)

        # Material selection
        self.material_combo = QtWidgets.QComboBox()
        self.material_combo.addItems(self.material_db.names())
        form.addRow("Material:", self.material_combo)

        # Buttons
        self.run_btn = QtWidgets.QPushButton("Run simulation")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        form.addRow(self.run_btn, self.stop_btn)

        return widget

    def _connect_actions(self) -> None:
        self.file_btn.clicked.connect(self._open_mesh)
        self.band_combo.currentTextChanged.connect(self._update_band_defaults)
        self.run_btn.clicked.connect(self._run_simulation)
        self.stop_btn.clicked.connect(self._stop_simulation)
        self.create_template_btn.clicked.connect(self._create_template)
        self.match_template_btn.clicked.connect(self._match_templates)
        self.refresh_templates_btn.clicked.connect(self._refresh_templates)
        self.elevation_spin.valueChanged.connect(self._update_polar_plot)
        self.scale_mode.currentTextChanged.connect(self._update_polar_plot)
        self.clip_min.valueChanged.connect(self._update_rcs_plot)
        self.clip_max.valueChanged.connect(self._update_rcs_plot)

        # Menu bar
        file_menu = self.menuBar().addMenu("File")
        open_action = file_menu.addAction("Open STL")
        save_project_action = file_menu.addAction("Save Project")
        load_project_action = file_menu.addAction("Load Project")
        export_csv_action = file_menu.addAction("Export CSV")
        export_plot_action = file_menu.addAction("Export Plots")
        exit_action = file_menu.addAction("Exit")

        materials_menu = self.menuBar().addMenu("Materials")
        edit_materials_action = materials_menu.addAction("Edit materials")

        templates_menu = self.menuBar().addMenu("Templates / NCTR")
        templates_menu.addAction(self.create_template_btn.text(), self._create_template)
        templates_menu.addAction(self.match_template_btn.text(), self._match_templates)

        open_action.triggered.connect(self._open_mesh)
        save_project_action.triggered.connect(self._save_project)
        load_project_action.triggered.connect(self._load_project)
        export_csv_action.triggered.connect(self._export_csv)
        export_plot_action.triggered.connect(self._export_plots)
        exit_action.triggered.connect(self.close)
        edit_materials_action.triggered.connect(self._edit_materials)

    # ------------------------------------------------------------------
    def _update_band_defaults(self) -> None:
        band = self.band_combo.currentText()
        start, stop = BAND_DEFAULTS.get(band, BAND_DEFAULTS["S"])
        self.single_freq.setValue((start + stop) / 2 / 1e9)
        self.sweep_start.setValue(start / 1e9)
        self.sweep_stop.setValue(stop / 1e9)

    def _open_mesh(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open STL", str(Path.home()), "Mesh (*.stl *.obj *.glb *.gltf)")
        if not path:
            return
        mesh = trimesh.load_mesh(path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )
        mesh.remove_unreferenced_vertices()
        mesh.update_faces(mesh.nondegenerate_faces())
        if len(mesh.faces) > 50000:
            mesh = mesh.simplify_quadratic_decimation(50000)
        self.mesh = mesh
        self.mesh_path = Path(path)
        self.file_label.setText(Path(path).name)
        self._draw_mesh_preview()

    def _settings_from_ui(self) -> SimulationSettings:
        sweep = None
        if self.freq_mode.currentText() == "Sweep":
            sweep = FrequencySweep(
                start_hz=self.sweep_start.value() * 1e9,
                stop_hz=self.sweep_stop.value() * 1e9,
                steps=self.sweep_steps.value(),
            )
        freq = None if sweep else self.single_freq.value() * 1e9
        return SimulationSettings(
            band=self.band_combo.currentText(),
            polarization=self.pol_combo.currentText(),
            max_reflections=self.reflections_spin.value(),
            frequency_hz=freq,
            sweep=sweep,
            azimuth_start=self.az_start.value(),
            azimuth_stop=self.az_stop.value(),
            azimuth_step=self.az_step.value(),
            elevation_start=self.el_start.value(),
            elevation_stop=self.el_stop.value(),
            elevation_step=self.el_step.value(),
        )

    def _run_simulation(self) -> None:
        if self.mesh is None:
            QtWidgets.QMessageBox.warning(self, "No mesh", "Please load a 3D model first.")
            return
        settings = self._settings_from_ui()
        material = self.material_db.get(self.material_combo.currentText())
        self.status_label.setText("Running simulation...")
        self.progress.setValue(0)
        self.worker = SimulationWorker(self.engine, self.mesh, material, settings)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_simulation_finished)
        self.worker.failed.connect(self._on_simulation_failed)
        self.worker.start()

    def _stop_simulation(self) -> None:
        self.engine.request_stop()
        self.status_label.setText("Stopping...")

    def _on_simulation_failed(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Simulation failed", message)
        self.status_label.setText("Failed")

    def _on_simulation_finished(self, result: SimulationResult) -> None:
        self.result = result
        self.status_label.setText("Simulation complete")
        self.progress.setValue(100)
        self._update_polar_plot()
        self._update_rcs_plot()
        self._draw_mesh_preview()

    # ------------------------------------------------------------------
    def _draw_mesh_preview(self) -> None:
        self.model_canvas.clear()
        ax = self.model_canvas.figure.add_subplot(111, projection="3d")
        if self.mesh is None:
            self.model_canvas.draw_idle()
            return
        mesh = self.mesh
        ax.add_collection3d(
            Poly3DCollection(mesh.vertices[mesh.faces], facecolor="#6baed6", edgecolor="k", alpha=0.4)
        )
        bounds = mesh.bounds
        max_range = (bounds[1] - bounds[0]).max()
        mid = (bounds[1] + bounds[0]) / 2
        ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
        ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
        ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)
        ax.set_title("Model preview")
        self.model_canvas.draw_idle()

    def _update_polar_plot(self) -> None:
        if self.result is None:
            return
        self.polar_canvas.clear()
        ax = self.polar_canvas.figure.add_subplot(111, polar=True)
        az, rcs = self.result.slice_for_elevation(self.elevation_spin.value())
        freq_labels = []
        for idx, freq in enumerate(self.result.frequencies_hz):
            values = rcs[idx]
            if self.scale_mode.currentText() == "Linear":
                values = 10 ** (values / 10)
            ax.plot(np.radians(az), values, label=f"{freq/1e9:.2f} GHz")
            freq_labels.append(freq)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.legend()
        ax.set_title(f"Polar plot ({self.result.band}-band, {self.result.polarization})")
        self.polar_canvas.draw_idle()

    def _update_rcs_plot(self) -> None:
        if self.result is None:
            return
        self.rcs3d_canvas.clear()
        ax = self.rcs3d_canvas.figure.add_subplot(111, projection="3d")
        freq_idx = 0
        rcs = self.result.rcs_dbsm[freq_idx]
        rcs = np.clip(rcs, self.clip_min.value(), self.clip_max.value())
        az_rad = np.radians(self.result.azimuth_deg)
        el_rad = np.radians(self.result.elevation_deg)
        az_grid, el_grid = np.meshgrid(az_rad, el_rad)
        r_lin = 10 ** (rcs / 10)
        r_norm = r_lin / (np.nanmax(r_lin) + 1e-9)
        radius = np.cbrt(r_norm)
        x = radius * np.cos(el_grid) * np.cos(az_grid)
        y = radius * np.cos(el_grid) * np.sin(az_grid)
        z = radius * np.sin(el_grid)
        surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(r_norm), rstride=1, cstride=1, alpha=0.9)
        mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        mappable.set_array(rcs)
        self.rcs3d_canvas.figure.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="RCS (dBsm)")
        ax.set_title(f"3D RCS at {self.result.frequencies_hz[freq_idx]/1e9:.2f} GHz")
        ax.set_box_aspect((1, 1, 1))
        self.rcs3d_canvas.draw_idle()

    # ------------------------------------------------------------------
    def _refresh_templates(self) -> None:
        self.templates_table.setRowCount(0)
        for path in self.template_lib.list_templates():
            template = self.template_lib.load_template(path)
            row = self.templates_table.rowCount()
            self.templates_table.insertRow(row)
            self.templates_table.setItem(row, 0, QtWidgets.QTableWidgetItem(template.name))
            self.templates_table.setItem(row, 1, QtWidgets.QTableWidgetItem(template.target_class))
            self.templates_table.setItem(row, 2, QtWidgets.QTableWidgetItem(template.band))
            self.templates_table.setItem(row, 3, QtWidgets.QTableWidgetItem("-"))

    def _create_template(self) -> None:
        if self.result is None:
            QtWidgets.QMessageBox.information(self, "No result", "Run a simulation first.")
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Template name", "Name")
        if not ok or not name:
            return
        cls, ok = QtWidgets.QInputDialog.getText(self, "Target class", "Class")
        if not ok or not cls:
            return
        template = self.template_lib.create_from_result(self.result, name=name, target_class=cls)
        path = self.template_lib.save_template(template)
        QtWidgets.QMessageBox.information(self, "Template saved", f"Saved to {path}")
        self._refresh_templates()

    def _match_templates(self) -> None:
        if self.result is None:
            QtWidgets.QMessageBox.warning(self, "No result", "Run a simulation first.")
            return
        matches = self.template_lib.match(self.result)
        self.templates_table.setRowCount(0)
        for template, score in matches:
            row = self.templates_table.rowCount()
            self.templates_table.insertRow(row)
            self.templates_table.setItem(row, 0, QtWidgets.QTableWidgetItem(template.name))
            self.templates_table.setItem(row, 1, QtWidgets.QTableWidgetItem(template.target_class))
            self.templates_table.setItem(row, 2, QtWidgets.QTableWidgetItem(template.band))
            self.templates_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{score:.2f}"))

    # ------------------------------------------------------------------
    def _save_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", str(Path.home()), "Project (*.json)")
        if not path:
            return
        state = ProjectState(mesh_path=str(self.mesh_path) if self.mesh_path else None, settings=self._settings_from_ui(), material_name=self.material_combo.currentText())
        save_project(path, state)

    def _load_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load project", str(Path.home()), "Project (*.json)")
        if not path:
            return
        state = load_project(path)
        self.band_combo.setCurrentText(state.settings.band)
        self.pol_combo.setCurrentText(state.settings.polarization)
        self.reflections_spin.setValue(state.settings.max_reflections)
        if state.settings.sweep:
            self.freq_mode.setCurrentText("Sweep")
            self.sweep_start.setValue(state.settings.sweep.start_hz / 1e9)
            self.sweep_stop.setValue(state.settings.sweep.stop_hz / 1e9)
            self.sweep_steps.setValue(state.settings.sweep.steps)
        else:
            self.freq_mode.setCurrentText("Single")
            if state.settings.frequency_hz:
                self.single_freq.setValue(state.settings.frequency_hz / 1e9)
        self.az_start.setValue(state.settings.azimuth_start)
        self.az_stop.setValue(state.settings.azimuth_stop)
        self.az_step.setValue(state.settings.azimuth_step)
        self.el_start.setValue(state.settings.elevation_start)
        self.el_stop.setValue(state.settings.elevation_stop)
        self.el_step.setValue(state.settings.elevation_step)
        if state.material_name in self.material_db.materials:
            self.material_combo.setCurrentText(state.material_name)
        if state.mesh_path and Path(state.mesh_path).exists():
            self.mesh_path = Path(state.mesh_path)
            self._open_mesh_from_path(self.mesh_path)

    def _open_mesh_from_path(self, path: Path) -> None:
        mesh = trimesh.load_mesh(path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )
        mesh.remove_unreferenced_vertices()
        mesh.update_faces(mesh.nondegenerate_faces())
        self.mesh = mesh
        self.file_label.setText(path.name)
        self._draw_mesh_preview()

    def _export_csv(self) -> None:
        if self.result is None:
            QtWidgets.QMessageBox.warning(self, "No result", "Run a simulation first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", str(Path.home()), "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("band,frequency_hz,azimuth_deg,elevation_deg,polarization,rcs_linear,rcs_db\n")
            for fi, freq in enumerate(self.result.frequencies_hz):
                for ei, el in enumerate(self.result.elevation_deg):
                    for ai, az in enumerate(self.result.azimuth_deg):
                        rcs_db = self.result.rcs_dbsm[fi, ei, ai]
                        rcs_lin = 10 ** (rcs_db / 10)
                        fh.write(
                            f"{self.result.band},{freq},{az},{el},{self.result.polarization},{rcs_lin},{rcs_db}\n"
                        )

    def _export_plots(self) -> None:
        if self.result is None:
            QtWidgets.QMessageBox.warning(self, "No result", "Run a simulation first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export plots", str(Path.home()), "PNG (*.png)")
        if not path:
            return
        self.polar_canvas.figure.savefig(path)
        rcs_path = Path(path).with_name(Path(path).stem + "_3d.png")
        self.rcs3d_canvas.figure.savefig(rcs_path)

    def _edit_materials(self) -> None:
        dialog = MaterialsDialog(self.material_db, parent=self)
        if dialog.exec_():
            self.material_combo.clear()
            self.material_combo.addItems(self.material_db.names())


def run_app() -> None:
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    win._refresh_templates()
    app.exec_()


class MaterialsDialog(QtWidgets.QDialog):
    def __init__(self, db: MaterialDatabase, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Materials")
        layout = QtWidgets.QVBoxLayout(self)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Name", "ε'", "ε''", "σ", "Reflectivity"])
        layout.addWidget(self.table)
        self._reload()

        btns = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add")
        edit_btn = QtWidgets.QPushButton("Edit")
        delete_btn = QtWidgets.QPushButton("Delete")
        btns.addWidget(add_btn)
        btns.addWidget(edit_btn)
        btns.addWidget(delete_btn)
        btns.addStretch(1)
        layout.addLayout(btns)

        add_btn.clicked.connect(self._add)
        edit_btn.clicked.connect(self._edit)
        delete_btn.clicked.connect(self._delete)

    def _reload(self) -> None:
        self.table.setRowCount(0)
        for mat in self.db.as_list():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(mat.name))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{mat.epsilon_real:.2f}"))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{mat.epsilon_imag:.2f}"))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{mat.conductivity:.2e}"))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{mat.reflectivity:.2f}"))

    def _add(self) -> None:
        material = self._prompt_material()
        if material:
            self.db.add_material(material)
            self._reload()

    def _edit(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        name = self.table.item(row, 0).text()
        material = self._prompt_material(name)
        if material:
            self.db.update_material(name, **material.__dict__)
            self._reload()

    def _delete(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        name = self.table.item(row, 0).text()
        self.db.delete_material(name)
        self._reload()

    def _prompt_material(self, existing: str | None = None) -> Material | None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Material")
        form = QtWidgets.QFormLayout(dialog)
        name_edit = QtWidgets.QLineEdit(existing or "")
        eps_r = QtWidgets.QDoubleSpinBox()
        eps_r.setRange(0, 100)
        eps_i = QtWidgets.QDoubleSpinBox()
        eps_i.setRange(0, 100)
        sigma = QtWidgets.QDoubleSpinBox()
        sigma.setRange(0, 1e8)
        sigma.setDecimals(3)
        sigma.setSingleStep(1e4)
        refl = QtWidgets.QDoubleSpinBox()
        refl.setRange(0, 1)
        refl.setSingleStep(0.05)
        if existing:
            mat = self.db.get(existing)
            eps_r.setValue(mat.epsilon_real)
            eps_i.setValue(mat.epsilon_imag)
            sigma.setValue(mat.conductivity)
            refl.setValue(mat.reflectivity)
        form.addRow("Name", name_edit)
        form.addRow("ε'", eps_r)
        form.addRow("ε''", eps_i)
        form.addRow("σ", sigma)
        form.addRow("Reflectivity", refl)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        form.addRow(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            return Material(
                name=name_edit.text(),
                epsilon_real=eps_r.value(),
                epsilon_imag=eps_i.value(),
                conductivity=sigma.value(),
                reflectivity=refl.value(),
            )
        return None


__all__ = ["run_app", "MainWindow"]
