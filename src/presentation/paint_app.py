from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QImage, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSpinBox,
    QToolBar,
    QToolButton,
    QWidget,
)

from src.application.services.boundary_validator import BoundaryValidator
from src.domain.entities.mesh import Mesh
from src.presentation.canvas import Canvas
from src.presentation.tools import Tool, TriangulationMode
from src.presentation.triangulation_adapter import TriangulationAdapter, TriangulationSettings

TRIANGULATION_INFO_DIALOG_TEXT = (
    "Зверніть увагу на особливості роботи з включеннями та розрізами перед початком тріангуляції:\n\n"
    "1. Включення (замкнені області всередині основної області):\n"
    "• Під час тріангуляції основної області включення може не враховуватись.\n"
    "• Якщо включення враховується, тоді сітка будується окремо для основної області та для "
    "включення. По їхній спільній межі трикутники не заходять один в один. Вузли на цій межі "
    "матимуть окремі номери для кожної з областей.\n\n"
    "2. Розріз (лінія всередині області):\n"
    "• Під час тріангуляції такий розріз обходиться з обох сторін. Тобто він працює як межа, "
    "у якої є дві окремі сторони."
)


class TriangulationWorker(QObject):
    finished = Signal(object, float)
    failed = Signal(str)

    def __init__(
        self,
        adapter: TriangulationAdapter,
        image_data: QImage,
        contours: list[list[tuple[int, int]]],
        mode: TriangulationMode,
        custom_settings: TriangulationSettings | None,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._image_data = image_data
        self._contours = contours
        self._mode = mode
        self._custom_settings = custom_settings

    def run(self) -> None:
        try:
            if self._contours:
                mesh, coefficient = self._adapter.run_from_contours(
                    self._contours,
                    mode=self._mode,
                    custom_settings=self._custom_settings,
                )
            else:
                mesh, coefficient = self._adapter.run(
                    self._image_data,
                    mode=self._mode,
                    custom_settings=self._custom_settings,
                )
        except Exception as error:  # noqa: BLE001
            self.failed.emit(str(error))
            return
        self.finished.emit(mesh, coefficient)


class PaintApp(QMainWindow):
    _CANVAS_SIZE_PRESETS: tuple[tuple[str, tuple[int, int]], ...] = (
        ("VGA 640 x 480", (640, 480)),
        ("SVGA 800 x 600", (800, 600)),
        ("XGA 1024 x 768", (1024, 768)),
        ("HD 1280 x 720", (1280, 720)),
        ("Full HD 1920 x 1080", (1920, 1080)),
    )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AFM Mask Editor")
        self._canvas = Canvas()
        self._triangulation_adapter = TriangulationAdapter()
        self._boundary_validator = BoundaryValidator()
        self._actions: dict[str, QAction] = {}
        self._tool_actions: dict[Tool, QAction] = {}
        self._tools_toolbar: QToolBar | None = None
        self._actions_toolbar: QToolBar | None = None
        self._settings_dock: QDockWidget | None = None
        self._triangulation_thread: QThread | None = None
        self._triangulation_worker: TriangulationWorker | None = None
        self._triangulation_busy = False
        self._triangulation_mode = TriangulationMode.BALANCED
        self._tool_status_label = QLabel("Tool: Pen")
        self._state_status_label = QLabel("Idle")
        self._mode_status_label = QLabel("Triangulation: Balanced")
        self._status_progress = QProgressBar()
        self._canvas_resize_in_progress = False

        self._mode_combo: QComboBox
        self._theme_combo: QComboBox
        self._size_preset_combo: QComboBox
        self._canvas_width_spin: QSpinBox
        self._canvas_height_spin: QSpinBox
        self._canvas_size_feedback: QLabel
        self._custom_checkbox: QCheckBox
        self._h_spin: QDoubleSpinBox
        self._smooth_spin: QSpinBox
        self._epsilon_spin: QDoubleSpinBox
        self._iter_factor_spin: QSpinBox
        self._quality_spin: QDoubleSpinBox
        self._meters_per_pixel_spin: QDoubleSpinBox

        self._build_central_workspace()
        self._build_toolbar()
        self._build_settings_panel()
        self._configure_status_bar()
        self._apply_theme(self._theme_combo.currentData())
        self._apply_preset_controls(self._triangulation_mode)
        self._update_custom_controls_enabled()
        self.statusBar().showMessage("Ready")
        self.resize(1200, 780)

    def run_triangulation(self, image_data: QImage) -> tuple[Mesh, float]:
        mode, custom_settings = self._resolve_mode_and_settings()
        contours = self._contours_for_triangulation(image_data, prefer_strokes=True)
        if contours:
            return self._triangulation_adapter.run_from_contours(contours, mode=mode, custom_settings=custom_settings)
        return self._triangulation_adapter.run(image_data, mode=mode, custom_settings=custom_settings)

    def display_triangulation_result(self, mesh: Mesh, coefficient: float) -> None:
        self._canvas.set_mesh_overlay(mesh)
        self.statusBar().showMessage(f"Triangulation done. coefficient={coefficient:.4f}")
        self._state_status_label.setText("Idle")

    def _build_toolbar(self) -> None:
        tools_toolbar = QToolBar("Drawing Tools", self)
        tools_toolbar.setMovable(False)
        tools_toolbar.setObjectName("DrawingToolsBar")
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, tools_toolbar)
        self._tools_toolbar = tools_toolbar

        actions_toolbar = QToolBar("Actions", self)
        actions_toolbar.setMovable(False)
        actions_toolbar.setObjectName("ActionsBar")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, actions_toolbar)
        self._actions_toolbar = actions_toolbar

        action_group = QActionGroup(self)
        action_group.setExclusive(True)

        tool_handlers: dict[str, tuple[str, Callable[[], None]]] = {
            "Pen": ("Draw with black pen", lambda: self._set_tool(Tool.PEN)),
            "Eraser": ("Erase to white", lambda: self._set_tool(Tool.ERASER)),
            "Fill": ("Flood fill with black", lambda: self._set_tool(Tool.FILL)),
            "Point": ("Draw filled point", lambda: self._set_tool(Tool.POINT)),
            "Segment": ("Draw filled segment", lambda: self._set_tool(Tool.SEGMENT)),
            "Rectangle": ("Draw filled rectangle", lambda: self._set_tool(Tool.RECTANGLE)),
            "Circle": ("Draw filled circle", lambda: self._set_tool(Tool.CIRCLE)),
        }
        tool_shortcuts: dict[str, str] = {
            "Pen": "P",
            "Eraser": "E",
            "Fill": "F",
            "Point": "1",
            "Segment": "2",
            "Rectangle": "3",
            "Circle": "4",
        }
        command_handlers: dict[str, tuple[str, Callable[[], None]]] = {
            "Undo": ("Undo last action", self._on_undo),
            "Redo": ("Redo last action", self._on_redo),
            "Save": ("Save image to PNG/JPG", self._on_save),
            "Load": ("Load image from PNG/JPG", self._on_load),
            "Info": ("Show triangulation info", self.show_triangulation_info_dialog),
            "Triangulation": ("Run triangulation", self._on_triangulation),
            "Clear": ("Clear canvas", self._canvas.clear),
        }

        for title, (tooltip, handler) in tool_handlers.items():
            action = QAction(title, self)
            action.setToolTip(tooltip)
            action.setCheckable(True)
            action.setShortcut(tool_shortcuts[title])
            action.triggered.connect(handler)
            action_group.addAction(action)
            tools_toolbar.addAction(action)
            self._actions[title] = action
            self._tool_actions[Tool(title.lower())] = action

        self._tool_actions[Tool.PEN].setChecked(True)
        self._set_tool(Tool.PEN)

        for title, (tooltip, handler) in command_handlers.items():
            action = QAction(title, self)
            action.setToolTip(tooltip)
            if title == "Undo":
                action.setShortcut(QKeySequence.StandardKey.Undo)
            elif title == "Redo":
                action.setShortcut(QKeySequence.StandardKey.Redo)
            action.triggered.connect(handler)
            actions_toolbar.addAction(action)
            self._actions[title] = action

    def _build_settings_panel(self) -> None:
        dock = QDockWidget("Properties", self)
        dock.setObjectName("PropertiesDock")
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self._settings_dock = dock

        panel = QWidget()
        panel.setObjectName("PropertiesPanel")
        layout = QFormLayout(panel)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._theme_combo = QComboBox()
        self._theme_combo.addItem("Dark", "dark")
        self._theme_combo.addItem("Light", "light")
        initial_theme = os.getenv("AFM_THEME", "dark").strip().lower()
        self._theme_combo.setCurrentIndex(1 if initial_theme == "light" else 0)
        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        layout.addRow("Theme:", self._theme_combo)

        size_group = QGroupBox("Canvas Size")
        size_form = QFormLayout(size_group)

        self._size_preset_combo = QComboBox()
        for label, size in self._CANVAS_SIZE_PRESETS:
            if (
                Canvas.MIN_WIDTH <= size[0] <= Canvas.MAX_WIDTH
                and Canvas.MIN_HEIGHT <= size[1] <= Canvas.MAX_HEIGHT
            ):
                self._size_preset_combo.addItem(label, size)
        self._size_preset_combo.addItem("Custom", "custom")
        self._size_preset_combo.currentIndexChanged.connect(self._on_canvas_preset_changed)
        size_form.addRow("Preset:", self._size_preset_combo)

        self._canvas_width_spin = QSpinBox()
        self._canvas_width_spin.setRange(Canvas.MIN_WIDTH, Canvas.MAX_WIDTH)
        self._canvas_width_spin.setSingleStep(16)
        self._canvas_width_spin.setSuffix(" px")
        self._canvas_width_spin.setKeyboardTracking(False)
        self._canvas_width_spin.valueChanged.connect(self._on_custom_canvas_size_changed)
        size_form.addRow("Width:", self._canvas_width_spin)

        self._canvas_height_spin = QSpinBox()
        self._canvas_height_spin.setRange(Canvas.MIN_HEIGHT, Canvas.MAX_HEIGHT)
        self._canvas_height_spin.setSingleStep(16)
        self._canvas_height_spin.setSuffix(" px")
        self._canvas_height_spin.setKeyboardTracking(False)
        self._canvas_height_spin.valueChanged.connect(self._on_custom_canvas_size_changed)
        size_form.addRow("Height:", self._canvas_height_spin)

        limits_label = QLabel(
            f"Allowed range: {Canvas.MIN_WIDTH}-{Canvas.MAX_WIDTH}px width, "
            f"{Canvas.MIN_HEIGHT}-{Canvas.MAX_HEIGHT}px height."
        )
        limits_label.setWordWrap(True)
        size_form.addRow("", limits_label)

        self._canvas_size_feedback = QLabel("")
        self._canvas_size_feedback.setWordWrap(True)
        self._canvas_size_feedback.hide()
        size_form.addRow("", self._canvas_size_feedback)

        self._sync_canvas_size_controls(*self._current_canvas_dimensions())
        layout.addRow(size_group)

        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Fast", TriangulationMode.FAST.value)
        self._mode_combo.addItem("Balanced", TriangulationMode.BALANCED.value)
        self._mode_combo.addItem("Accurate", TriangulationMode.ACCURATE.value)
        self._mode_combo.addItem("Custom", TriangulationMode.CUSTOM.value)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_combo_changed)
        layout.addRow("Preset:", self._mode_combo)

        self._custom_checkbox = QCheckBox("Use custom values")
        self._custom_checkbox.stateChanged.connect(self._on_custom_toggle_changed)
        layout.addRow("", self._custom_checkbox)

        manual_group = QGroupBox("Triangulation Settings")
        manual_form = QFormLayout(manual_group)

        self._h_spin = QDoubleSpinBox()
        self._h_spin.setRange(
            TriangulationAdapter.MIN_TARGET_EDGE_LENGTH,
            TriangulationAdapter.MAX_TARGET_EDGE_LENGTH,
        )
        self._h_spin.setSingleStep(1.0)
        self._h_spin.setDecimals(1)
        self._h_spin.setToolTip("Target edge length (h). Smaller = finer mesh, slower.")
        manual_form.addRow("Target edge length (h):", self._h_spin)

        self._smooth_spin = QSpinBox()
        self._smooth_spin.setRange(
            TriangulationAdapter.MIN_SMOOTHING_ITERATIONS,
            TriangulationAdapter.MAX_SMOOTHING_ITERATIONS,
        )
        self._smooth_spin.setSingleStep(1)
        self._smooth_spin.setToolTip("Smoothing iterations.")
        manual_form.addRow("Smoothing iterations:", self._smooth_spin)

        self._epsilon_spin = QDoubleSpinBox()
        self._epsilon_spin.setRange(
            TriangulationAdapter.MIN_CONTOUR_EPSILON,
            TriangulationAdapter.MAX_CONTOUR_EPSILON,
        )
        self._epsilon_spin.setSingleStep(0.1)
        self._epsilon_spin.setDecimals(2)
        self._epsilon_spin.setToolTip("Contour simplify epsilon.")
        manual_form.addRow("Contour simplify epsilon:", self._epsilon_spin)

        self._iter_factor_spin = QSpinBox()
        self._iter_factor_spin.setRange(
            TriangulationAdapter.MIN_MAX_ITERATIONS_FACTOR,
            TriangulationAdapter.MAX_MAX_ITERATIONS_FACTOR,
        )
        self._iter_factor_spin.setSingleStep(10)
        self._iter_factor_spin.setToolTip("Max iterations factor for AFM.")
        manual_form.addRow("Max iterations factor:", self._iter_factor_spin)

        self._quality_spin = QDoubleSpinBox()
        self._quality_spin.setRange(
            TriangulationAdapter.MIN_TRIANGLE_QUALITY,
            TriangulationAdapter.MAX_TRIANGLE_QUALITY,
        )
        self._quality_spin.setSingleStep(0.01)
        self._quality_spin.setDecimals(3)
        self._quality_spin.setToolTip("Minimum accepted triangle quality.")
        manual_form.addRow("Min triangle quality:", self._quality_spin)

        self._meters_per_pixel_spin = QDoubleSpinBox()
        self._meters_per_pixel_spin.setRange(
            TriangulationAdapter.MIN_METERS_PER_PIXEL,
            TriangulationAdapter.MAX_METERS_PER_PIXEL,
        )
        self._meters_per_pixel_spin.setSingleStep(0.001)
        self._meters_per_pixel_spin.setDecimals(6)
        self._meters_per_pixel_spin.setValue(1.0)
        self._meters_per_pixel_spin.setToolTip(
            "Physical scale in meters per pixel. Physical area = pixel area * (m/px)^2."
        )
        manual_form.addRow("Scale (m/px):", self._meters_per_pixel_spin)

        layout.addRow(manual_group)
        self._mode_combo.setCurrentIndex(1)
        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_central_workspace(self) -> None:
        workspace = QWidget()
        workspace.setObjectName("Workspace")
        self.setCentralWidget(workspace)
        self._canvas.setObjectName("Canvas")
        self._canvas.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._canvas.setAutoFillBackground(True)

        from PySide6.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(0)
        layout.addWidget(self._canvas, 0, Qt.AlignmentFlag.AlignCenter)

    def _configure_status_bar(self) -> None:
        self._status_progress.setRange(0, 0)
        self._status_progress.setFixedWidth(120)
        self._status_progress.setFixedHeight(8)
        self._status_progress.setTextVisible(False)
        self._status_progress.hide()

        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().addPermanentWidget(self._tool_status_label)
        self.statusBar().addPermanentWidget(self._mode_status_label)
        self.statusBar().addPermanentWidget(self._state_status_label)
        self.statusBar().addPermanentWidget(self._status_progress)

    def _apply_theme(self, theme_name: str) -> None:
        if theme_name == "light":
            self.setStyleSheet(self._light_theme_stylesheet())
        else:
            self.setStyleSheet(self._dark_theme_stylesheet())
        triangulation_action = self._actions.get("Triangulation")
        if self._actions_toolbar is not None and triangulation_action is not None:
            triangulation_button = self._actions_toolbar.widgetForAction(triangulation_action)
            if isinstance(triangulation_button, QToolButton):
                triangulation_button.setStyleSheet(
                    """
                    QToolButton {
                        background: #0d3a88;
                        border: 1px solid #3b82f6;
                        color: #eff6ff;
                        border-radius: 8px;
                        padding: 6px 10px;
                    }
                    QToolButton:hover {
                        background: #1d4ed8;
                    }
                    QToolButton:pressed {
                        background: #1e3a8a;
                    }
                    """
                )

    def _on_theme_changed(self) -> None:
        self._apply_theme(str(self._theme_combo.currentData()))

    def _current_canvas_dimensions(self) -> tuple[int, int]:
        size = self._canvas.canvas_size()
        return size.width(), size.height()

    def _sync_canvas_size_controls(self, width: int, height: int) -> None:
        self._canvas_resize_in_progress = True
        try:
            self._canvas_width_spin.setValue(width)
            self._canvas_height_spin.setValue(height)
            preset_index = self._matching_canvas_preset_index(width, height)
            self._size_preset_combo.setCurrentIndex(preset_index)
        finally:
            self._canvas_resize_in_progress = False

    def _matching_canvas_preset_index(self, width: int, height: int) -> int:
        for index in range(self._size_preset_combo.count()):
            data = self._size_preset_combo.itemData(index)
            if isinstance(data, tuple) and data == (width, height):
                return index
        return self._size_preset_combo.count() - 1

    def _set_canvas_feedback(self, message: str, *, error: bool) -> None:
        self._canvas_size_feedback.setText(message)
        self._canvas_size_feedback.setStyleSheet("color: #dc2626;" if error else "color: #16a34a;")
        self._canvas_size_feedback.setVisible(bool(message))

    def _resize_window_for_canvas_change(
        self,
        old_width: int,
        old_height: int,
        new_width: int,
        new_height: int,
    ) -> None:
        if self.isMaximized() or self.isFullScreen():
            return
        width_delta = new_width - old_width
        height_delta = new_height - old_height
        target_width = max(1, self.width() + width_delta)
        target_height = max(1, self.height() + height_delta)
        self.resize(target_width, target_height)

    def _apply_canvas_resize(self, width: int, height: int, *, show_error_dialog: bool) -> bool:
        if self._canvas_resize_in_progress:
            return False
        self._canvas_resize_in_progress = True
        try:
            old_width, old_height = self._current_canvas_dimensions()
            success, message = self._canvas.resize_canvas(width, height)
            if not success:
                current_width, current_height = self._current_canvas_dimensions()
                self._sync_canvas_size_controls(current_width, current_height)
                error_message = message or "Failed to resize canvas."
                self._set_canvas_feedback(error_message, error=True)
                if show_error_dialog:
                    QMessageBox.warning(self, "Canvas Resize Blocked", error_message)
                return False
            self._sync_canvas_size_controls(width, height)
            self._resize_window_for_canvas_change(old_width, old_height, width, height)
            self._set_canvas_feedback(f"Canvas resized to {width} x {height}px.", error=False)
            self.statusBar().showMessage(f"Canvas resized to {width} x {height}px")
            return True
        finally:
            self._canvas_resize_in_progress = False

    def _on_canvas_preset_changed(self) -> None:
        if self._canvas_resize_in_progress:
            return
        size = self._size_preset_combo.currentData()
        if size == "custom" or not isinstance(size, tuple):
            return
        width, height = size
        self._apply_canvas_resize(width, height, show_error_dialog=True)

    def _on_custom_canvas_size_changed(self) -> None:
        if self._canvas_resize_in_progress:
            return
        width = int(self._canvas_width_spin.value())
        height = int(self._canvas_height_spin.value())
        custom_index = self._size_preset_combo.count() - 1
        self._canvas_resize_in_progress = True
        try:
            self._size_preset_combo.setCurrentIndex(custom_index)
        finally:
            self._canvas_resize_in_progress = False
        self._apply_canvas_resize(width, height, show_error_dialog=True)

    def _spinbox_controls_stylesheet(self, *, dark: bool) -> str:
        icons_dir = (Path(__file__).resolve().parent / "assets" / "icons").as_posix()
        up_icon = f"{icons_dir}/spin_up.svg"
        down_icon = f"{icons_dir}/spin_down.svg"
        hover_bg = "rgba(59, 130, 246, 0.22)" if dark else "rgba(59, 130, 246, 0.14)"
        pressed_bg = "rgba(59, 130, 246, 0.34)" if dark else "rgba(59, 130, 246, 0.24)"
        divider = "rgba(96, 165, 250, 0.28)" if dark else "rgba(59, 130, 246, 0.20)"
        text_color = "#60a5fa" if dark else "#0f172a"
        return f"""
            QSpinBox, QDoubleSpinBox {{
                color: {text_color};
                padding-right: 34px;
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                subcontrol-origin: padding;
                background: transparent;
                border: none;
                width: 24px;
                margin-right: 2px;
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button {{
                subcontrol-position: top right;
                height: 13px;
                margin-top: 2px;
            }}
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                subcontrol-position: bottom right;
                height: 13px;
                margin-bottom: 2px;
                border-top: 1px solid {divider};
            }}
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
                background: {hover_bg};
                border-radius: 6px;
            }}
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
                background: {pressed_bg};
            }}
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
                image: url({up_icon});
                width: 10px;
                height: 10px;
            }}
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                image: url({down_icon});
                width: 10px;
                height: 10px;
            }}
        """

    def _dark_theme_stylesheet(self) -> str:
        spinbox_controls = self._spinbox_controls_stylesheet(dark=True)
        return """
            QMainWindow {
                background: #17191d;
                color: #e5e7eb;
                font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
                font-size: 12px;
            }
            QWidget#Workspace {
                background: #111318;
            }
            QWidget#Canvas {
                background: #ffffff;
                border: 1px solid #3f4754;
                border-radius: 8px;
            }
            QToolBar#DrawingToolsBar {
                spacing: 6px;
                padding: 8px 6px;
                background: #1d2129;
                border-right: 1px solid #2d3340;
            }
            QToolBar#ActionsBar {
                spacing: 8px;
                padding: 8px 10px;
                background: #1a1e25;
                border-bottom: 1px solid #2d3340;
            }
            QToolBar QToolButton {
                color: #e5e7eb;
                background: #262b35;
                border: 1px solid #303747;
                border-radius: 8px;
                padding: 6px 10px;
                qproperty-cursor: PointingHandCursor;
            }
            QToolBar QToolButton:hover {
                background: #30384a;
                border-color: #3b4558;
            }
            QToolBar QToolButton:checked {
                background: #1e40af;
                border-color: #3b82f6;
                color: #eff6ff;
            }
            QToolBar QToolButton:pressed {
                background: #374151;
            }
            QDockWidget#PropertiesDock {
                background: #181c23;
                color: #e5e7eb;
            }
            QDockWidget#PropertiesDock::title {
                background: #1f2430;
                padding: 8px 10px;
                border-bottom: 1px solid #2d3340;
                font-weight: 600;
            }
            QWidget#PropertiesPanel {
                background: #181c23;
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #2d3340;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px 10px 10px 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #cfd5df;
            }
            QLabel {
                color: #cfd5df;
            }
            QMessageBox {
                background: #1a1e25;
            }
            QMessageBox QLabel {
                color: #e5e7eb;
            }
            QMessageBox QPushButton {
                color: #e5e7eb;
                background: #262b35;
                border: 1px solid #3b4558;
                border-radius: 6px;
                padding: 4px 10px;
                min-width: 70px;
            }
            QMessageBox QPushButton:hover {
                background: #30384a;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background: #11151d;
                color: #e5e7eb;
                border: 1px solid #364153;
                border-radius: 6px;
                min-height: 30px;
                padding: 3px 8px;
                qproperty-cursor: PointingHandCursor;
            }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #4b5c77;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #3b82f6;
            }
            __SPINBOX_CONTROLS__
            QComboBox QAbstractItemView {
                background: #11151d;
                color: #e5e7eb;
                border: 1px solid #364153;
                selection-background-color: #1e40af;
                selection-color: #eff6ff;
                outline: none;
            }
            QCheckBox {
                spacing: 8px;
                color: #e5e7eb;
                qproperty-cursor: PointingHandCursor;
            }
            QCheckBox:disabled {
                color: #94a3b8;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #4b5563;
                border-radius: 4px;
                background: #11151d;
            }
            QCheckBox::indicator:checked {
                background: #3b82f6;
                border-color: #3b82f6;
            }
            QStatusBar {
                background: #131720;
                border-top: 1px solid #2d3340;
                color: #9aa4b2;
            }
            QStatusBar::item {
                border: none;
            }
            QProgressBar {
                background: #262b35;
                border: 1px solid #3b4558;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: #3b82f6;
                border-radius: 4px;
            }
            """.replace("__SPINBOX_CONTROLS__", spinbox_controls)

    def _light_theme_stylesheet(self) -> str:
        spinbox_controls = self._spinbox_controls_stylesheet(dark=False)
        return """
            QMainWindow {
                background: #f3f5f8;
                color: #111827;
                font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
                font-size: 12px;
            }
            QWidget#Workspace {
                background: #eef2f7;
            }
            QWidget#Canvas {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
            }
            QToolBar#DrawingToolsBar {
                spacing: 6px;
                padding: 8px 6px;
                background: #e8edf4;
                border-right: 1px solid #d3dbe6;
            }
            QToolBar#ActionsBar {
                spacing: 8px;
                padding: 8px 10px;
                background: #f8fafc;
                border-bottom: 1px solid #d3dbe6;
            }
            QToolBar QToolButton {
                color: #0f172a;
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 6px 10px;
                qproperty-cursor: PointingHandCursor;
            }
            QToolBar QToolButton:hover {
                background: #f1f5f9;
                border-color: #94a3b8;
            }
            QToolBar QToolButton:checked {
                background: #dbeafe;
                border-color: #3b82f6;
                color: #1e3a8a;
            }
            QToolBar QToolButton:pressed {
                background: #e2e8f0;
            }
            QDockWidget#PropertiesDock {
                background: #f8fafc;
                color: #0f172a;
            }
            QDockWidget#PropertiesDock::title {
                background: #eef2f7;
                padding: 8px 10px;
                border-bottom: 1px solid #d3dbe6;
                font-weight: 600;
            }
            QWidget#PropertiesPanel {
                background: #f8fafc;
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #d3dbe6;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px 10px 10px 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #334155;
            }
            QLabel {
                color: #334155;
            }
            QMessageBox {
                background: #f8fafc;
            }
            QMessageBox QLabel {
                color: #0f172a;
            }
            QMessageBox QPushButton {
                color: #0f172a;
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 4px 10px;
                min-width: 70px;
            }
            QMessageBox QPushButton:hover {
                background: #f1f5f9;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background: #ffffff;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                min-height: 30px;
                padding: 3px 8px;
                qproperty-cursor: PointingHandCursor;
            }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #94a3b8;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #3b82f6;
            }
            __SPINBOX_CONTROLS__
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                selection-background-color: #dbeafe;
                selection-color: #1e3a8a;
                outline: none;
            }
            QCheckBox {
                spacing: 8px;
                color: #0f172a;
                qproperty-cursor: PointingHandCursor;
            }
            QCheckBox:disabled {
                color: #64748b;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #94a3b8;
                border-radius: 4px;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                background: #3b82f6;
                border-color: #3b82f6;
            }
            QStatusBar {
                background: #f8fafc;
                border-top: 1px solid #d3dbe6;
                color: #475569;
            }
            QStatusBar::item {
                border: none;
            }
            QProgressBar {
                background: #e2e8f0;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: #3b82f6;
                border-radius: 4px;
            }
            """.replace("__SPINBOX_CONTROLS__", spinbox_controls)

    def _set_tool(self, tool: Tool) -> None:
        if tool in (Tool.PEN, Tool.SEGMENT, Tool.RECTANGLE, Tool.CIRCLE, Tool.POINT) and self._canvas.geometry_is_dirty():
            self._refresh_geometry_from_canvas_image(self._canvas.image_data(), prefer_strokes=True)
        self._canvas.set_tool(tool)
        self._tool_status_label.setText(f"Tool: {tool.value.capitalize()}")
        action = self._tool_actions.get(tool)
        if action is not None and not action.isChecked():
            action.setChecked(True)

    def _set_triangulation_mode(self, mode: TriangulationMode) -> None:
        self._triangulation_mode = mode
        self._mode_status_label.setText(f"Triangulation: {mode.value.capitalize()}")

    def _on_mode_combo_changed(self) -> None:
        mode = TriangulationMode(self._mode_combo.currentData())
        self._set_triangulation_mode(mode)
        self._apply_preset_controls(mode)
        self._update_custom_controls_enabled()

    def _on_custom_toggle_changed(self) -> None:
        self._update_custom_controls_enabled()
        if self._custom_checkbox.isChecked():
            self._set_triangulation_mode(TriangulationMode.CUSTOM)
        else:
            self._set_triangulation_mode(TriangulationMode(self._mode_combo.currentData()))

    def _apply_preset_controls(self, mode: TriangulationMode) -> None:
        if mode == TriangulationMode.CUSTOM:
            return
        settings = self._triangulation_adapter.preset_settings(mode)
        self._h_spin.setValue(settings.target_edge_length)
        self._smooth_spin.setValue(settings.smoothing_iterations)
        self._epsilon_spin.setValue(settings.contour_epsilon)
        self._iter_factor_spin.setValue(settings.max_iterations_factor)
        self._quality_spin.setValue(settings.min_triangle_quality)

    def _update_custom_controls_enabled(self) -> None:
        is_custom_preset = TriangulationMode(self._mode_combo.currentData()) == TriangulationMode.CUSTOM
        enable_custom = is_custom_preset or self._custom_checkbox.isChecked()
        if is_custom_preset:
            self._custom_checkbox.setChecked(True)
            self._custom_checkbox.setEnabled(False)
            self._set_triangulation_mode(TriangulationMode.CUSTOM)
        else:
            self._custom_checkbox.setEnabled(True)
            self._set_triangulation_mode(
                TriangulationMode.CUSTOM if enable_custom else TriangulationMode(self._mode_combo.currentData())
            )
        for control in (
            self._h_spin,
            self._smooth_spin,
            self._epsilon_spin,
            self._iter_factor_spin,
            self._quality_spin,
        ):
            control.setEnabled(enable_custom)
        self._meters_per_pixel_spin.setEnabled(True)

    def _resolve_mode_and_settings(self) -> tuple[TriangulationMode, TriangulationSettings | None]:
        mode = self._triangulation_mode
        settings = TriangulationSettings(
            target_edge_length=float(self._h_spin.value()),
            smoothing_iterations=int(self._smooth_spin.value()),
            contour_epsilon=float(self._epsilon_spin.value()),
            max_iterations_factor=int(self._iter_factor_spin.value()),
            min_triangle_quality=float(self._quality_spin.value()),
            meters_per_pixel=float(self._meters_per_pixel_spin.value()),
        )
        if mode != TriangulationMode.CUSTOM:
            return mode, settings
        return mode, settings

    def _build_triangulation_info_dialog(self) -> QMessageBox:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Triangulation Info")
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setTextFormat(Qt.TextFormat.PlainText)
        dialog.setText(TRIANGULATION_INFO_DIALOG_TEXT)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        ok_button = dialog.button(QMessageBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText("Продовжити")
        return dialog

    def show_triangulation_info_dialog(self) -> None:
        self._build_triangulation_info_dialog().exec()

    def _on_save(self) -> None:
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Canvas",
            str(Path.cwd() / "mask.png"),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)",
        )
        if not path:
            return
        image = self._canvas.image_data()
        image_format = "PNG" if "PNG" in selected_filter else "JPEG"
        if not image.save(path, image_format):
            QMessageBox.critical(self, "Save Error", "Failed to save image.")
            return
        self.statusBar().showMessage(f"Saved: {path}")

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg)",
        )
        if not path:
            return
        image = QImage(path)
        if image.isNull():
            QMessageBox.critical(self, "Load Error", "Failed to load image.")
            return
        binary = Canvas.binarize_image(image, threshold=127)
        self._canvas.set_image(binary)
        self._refresh_geometry_from_canvas_image(binary, prefer_strokes=True)
        self.statusBar().showMessage(f"Loaded: {path}")

    def _on_undo(self) -> None:
        if self._triangulation_busy:
            return
        if not self._canvas.undo():
            self.statusBar().showMessage("Nothing to undo")

    def _on_redo(self) -> None:
        if self._triangulation_busy:
            return
        if not self._canvas.redo():
            self.statusBar().showMessage("Nothing to redo")

    def _on_triangulation(self) -> None:
        if self._triangulation_busy:
            return
        image_data = self._canvas.image_data()
        self._canvas.set_invalid_points([])
        contours = self._contours_for_triangulation(
            image_data,
            prefer_strokes=True,
        )
        if len(contours) <= 1:
            branch_points = self._triangulation_adapter.detect_stroke_branch_points_from_image(image_data)
            if branch_points:
                self._canvas.set_invalid_points(branch_points[:12])
                QMessageBox.warning(
                    self,
                    "Invalid Geometry",
                    "Detected a branch or self-crossing in the drawn stroke. "
                    "Triangulation requires a simple closed boundary without intersections.",
                )
                return
        if contours:
            prepared_contours = self._prepare_contours_for_triangulation(contours)
            if not prepared_contours:
                return
            contours = prepared_contours
        else:
            self._canvas.set_invalid_segments([])

        mode, custom_settings = self._resolve_mode_and_settings()
        if mode == TriangulationMode.CUSTOM and custom_settings is not None:
            try:
                self._triangulation_adapter.validate_custom_settings(custom_settings)
            except ValueError as error:
                QMessageBox.warning(self, "Invalid Parameters", str(error))
                return
        self._set_triangulation_busy(True)
        self.statusBar().showMessage("Triangulation in progress...")
        self._state_status_label.setText("Triangulating...")

        self._triangulation_thread = QThread(self)
        self._triangulation_worker = TriangulationWorker(
            self._triangulation_adapter,
            image_data,
            contours,
            mode,
            custom_settings,
        )
        self._triangulation_worker.moveToThread(self._triangulation_thread)

        self._triangulation_thread.started.connect(self._triangulation_worker.run)
        self._triangulation_worker.finished.connect(self._on_triangulation_finished)
        self._triangulation_worker.failed.connect(self._on_triangulation_failed)
        self._triangulation_worker.finished.connect(self._cleanup_triangulation_thread)
        self._triangulation_worker.failed.connect(self._cleanup_triangulation_thread)

        self._triangulation_thread.start()

    def _refresh_geometry_from_canvas_image(
        self,
        image_data: QImage,
        *,
        prefer_strokes: bool = False,
    ) -> list[list[tuple[int, int]]]:
        try:
            if prefer_strokes:
                contours = self._triangulation_adapter.extract_stroke_contours_from_image(image_data)
            else:
                contours = self._triangulation_adapter.extract_contours_from_image(image_data)
        except ValueError:
            self._canvas.set_geometry_contours([], preprocess=False, redraw_image=False, emit_change=False)
            return []
        self._canvas.set_geometry_contours(contours, preprocess=False, redraw_image=False, emit_change=False)
        return self._canvas.geometry_contours()

    def _contours_for_triangulation(
        self,
        image_data: QImage,
        *,
        prefer_strokes: bool,
    ) -> list[list[tuple[int, int]]]:
        contours = self._canvas.geometry_contours()
        cached_closed_contours, cached_open_contours = self._split_closed_and_open_contours(contours)
        if cached_closed_contours:
            return [*cached_closed_contours, *cached_open_contours]
        if prefer_strokes:
            try:
                boundary_contours = self._triangulation_adapter.extract_contours_from_image(image_data)
            except ValueError:
                boundary_contours = []
            closed_boundary_contours, open_boundary_contours = self._split_closed_and_open_contours(boundary_contours)
            if closed_boundary_contours:
                merged_open_contours = self._merge_unique_contours([*cached_open_contours, *open_boundary_contours])
                self._canvas.set_geometry_contours(
                    [*closed_boundary_contours, *merged_open_contours],
                    preprocess=False,
                    redraw_image=False,
                    emit_change=False,
                )
                return self._canvas.geometry_contours()
        return self._refresh_geometry_from_canvas_image(image_data, prefer_strokes=prefer_strokes)

    def _merge_unique_contours(
        self,
        contours: list[list[tuple[float, float]]],
    ) -> list[list[tuple[float, float]]]:
        seen: set[tuple[tuple[float, float], ...]] = set()
        merged: list[list[tuple[float, float]]] = []
        for contour in contours:
            key = tuple((float(x), float(y)) for x, y in contour)
            if key in seen:
                continue
            seen.add(key)
            merged.append(contour)
        return merged

    def _split_closed_and_open_contours(
        self,
        contours: list[list[tuple[float, float]]],
    ) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
        closed_contours: list[list[tuple[float, float]]] = []
        open_contours: list[list[tuple[float, float]]] = []
        for contour in contours:
            normalized = self._boundary_validator.normalize_valid_closed_contours([contour])
            if normalized:
                closed_contours.append(normalized[0])
            else:
                open_contours.append(contour)
        return closed_contours, open_contours

    def _prepare_contours_for_triangulation(
        self,
        contours: list[list[tuple[float, float]]],
    ) -> list[list[tuple[float, float]]]:
        closed_contours, open_contours = self._split_closed_and_open_contours(contours)
        if not closed_contours:
            if open_contours:
                QMessageBox.information(
                    self,
                    "No Closed Region",
                    "Triangulation needs at least one closed outer boundary. "
                    "The open lines will stay available, but the region cannot be triangulated yet.",
                )
            return []

        outer_contour = self._boundary_validator.select_outer_contour(closed_contours)
        if outer_contour is None:
            validation = self._boundary_validator.validate(closed_contours)
        else:
            validation = self._boundary_validator.validate([outer_contour])
        if not validation.is_valid:
            self._canvas.set_invalid_segments(self._latest_open_segment(validation))
            if validation.open_contours:
                QMessageBox.information(
                    self,
                    "Open Contour Detected",
                    "Region boundary is not closed. "
                    "Please close the highlighted gap manually before triangulation.",
                )
                return []
            if validation.self_intersections:
                QMessageBox.warning(
                    self,
                    "Invalid Geometry",
                    "Contour contains self-intersections. Fix geometry before triangulation.",
                )
                return []

        closed_contours = self._boundary_validator.normalize_valid_closed_contours(closed_contours)
        return [*closed_contours, *open_contours]

    def _latest_open_segment(
        self,
        validation,  # BoundaryValidationResult
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        open_issues = [issue for issue in validation.open_contours if issue.segment is not None]
        if not open_issues:
            return []
        latest = max(open_issues, key=lambda issue: issue.contour_index)
        return [latest.segment] if latest.segment is not None else []

    def _on_triangulation_finished(self, mesh: object, coefficient: float) -> None:
        self._set_triangulation_busy(False)
        self.display_triangulation_result(mesh, coefficient)

    def _on_triangulation_failed(self, message: str) -> None:
        self._set_triangulation_busy(False)
        QMessageBox.warning(self, "Triangulation Error", message)
        self.statusBar().showMessage("Triangulation failed")
        self._state_status_label.setText("Idle")

    def _cleanup_triangulation_thread(self) -> None:
        if self._triangulation_thread is not None:
            self._triangulation_thread.quit()
            self._triangulation_thread.wait()
        if self._triangulation_worker is not None:
            self._triangulation_worker.deleteLater()
        if self._triangulation_thread is not None:
            self._triangulation_thread.deleteLater()
        self._triangulation_thread = None
        self._triangulation_worker = None

    def _set_triangulation_busy(self, busy: bool) -> None:
        self._triangulation_busy = busy
        triangulation_action = self._actions.get("Triangulation")
        if triangulation_action is not None:
            triangulation_action.setEnabled(not busy)
        self._mode_combo.setEnabled(not busy)
        self._custom_checkbox.setEnabled(not busy and TriangulationMode(self._mode_combo.currentData()) != TriangulationMode.CUSTOM)
        for control in (
            self._h_spin,
            self._smooth_spin,
            self._epsilon_spin,
            self._iter_factor_spin,
            self._quality_spin,
        ):
            control.setEnabled(control.isEnabled() and not busy)
        self._meters_per_pixel_spin.setEnabled(not busy)
        self._canvas.setEnabled(not busy)
        self._status_progress.setVisible(busy)
        self.setCursor(Qt.CursorShape.WaitCursor if busy else Qt.CursorShape.ArrowCursor)
        if not busy:
            self._update_custom_controls_enabled()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._triangulation_busy:
            QMessageBox.information(self, "Please wait", "Triangulation is still running.")
            event.ignore()
            return
        super().closeEvent(event)
