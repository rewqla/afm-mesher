from __future__ import annotations

from collections import deque

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from src.domain.entities.mesh import Mesh
from src.presentation.tools import Tool


class Canvas(QWidget):
    image_changed = Signal()

    def __init__(self, width: int = 900, height: int = 650, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = QImage(width, height, QImage.Format.Format_RGB32)
        self._image.fill(Qt.GlobalColor.white)
        self._tool: Tool = Tool.PEN
        self._drawing = False
        self._last_point = QPoint()
        self._shape_start: QPoint | None = None
        self._shape_end: QPoint | None = None
        self._mesh_overlay: Mesh | None = None
        self._undo_stack: list[QImage] = []
        self._redo_stack: list[QImage] = []
        self._max_history = 30
        self.setMinimumSize(width, height)
        self.setMouseTracking(True)

    def sizeHint(self) -> QSize:
        return QSize(self._image.width(), self._image.height())

    def set_tool(self, tool: Tool) -> None:
        self._tool = tool
        if tool == Tool.FILL:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif tool == Tool.ERASER:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def clear(self) -> None:
        self._push_undo_state()
        self._image.fill(Qt.GlobalColor.white)
        self._mesh_overlay = None
        self.update()
        self.image_changed.emit()

    def set_image(self, image: QImage, *, record_history: bool = True) -> None:
        if record_history:
            self._push_undo_state()
        converted = image.convertToFormat(QImage.Format.Format_RGB32)
        self._image = converted
        self._mesh_overlay = None
        self.setMinimumSize(converted.width(), converted.height())
        self.resize(converted.size())
        self.update()
        self.image_changed.emit()

    def image_data(self) -> QImage:
        return self._image.copy()

    def set_mesh_overlay(self, mesh: Mesh | None) -> None:
        self._mesh_overlay = mesh
        self.update()

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._image.copy())
        self._image = self._undo_stack.pop()
        self._mesh_overlay = None
        self.update()
        self.image_changed.emit()
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._image.copy())
        self._image = self._redo_stack.pop()
        self._mesh_overlay = None
        self.update()
        self.image_changed.emit()
        return True

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return

        point = event.position().toPoint()
        if not self._in_bounds(point):
            return

        self._mesh_overlay = None
        if self._tool in (Tool.PEN, Tool.ERASER):
            self._push_undo_state()
            self._drawing = True
            self._last_point = point
        elif self._tool == Tool.FILL:
            self._push_undo_state()
            self._flood_fill(point)
            self.image_changed.emit()
        elif self._tool == Tool.POINT:
            self._push_undo_state()
            self._draw_point(point)
            self.image_changed.emit()
        else:
            self._push_undo_state()
            self._shape_start = point
            self._shape_end = point
            self._drawing = True
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        point = event.position().toPoint()
        if not self._in_bounds(point):
            return
        if not self._drawing:
            return

        if self._tool in (Tool.PEN, Tool.ERASER):
            painter = QPainter(self._image)
            painter.setPen(self._stroke_pen())
            painter.drawLine(self._last_point, point)
            painter.end()
            self._last_point = point
            self.update()
            self.image_changed.emit()
            return

        if self._tool in (Tool.SEGMENT, Tool.RECTANGLE, Tool.CIRCLE):
            self._shape_end = point
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self._drawing:
            return

        if self._tool in (Tool.SEGMENT, Tool.RECTANGLE, Tool.CIRCLE):
            self._shape_end = event.position().toPoint()
            self._commit_shape()
            self.image_changed.emit()

        self._drawing = False
        self._shape_start = None
        self._shape_end = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.drawImage(0, 0, self._image)

        if self._drawing and self._tool in (Tool.SEGMENT, Tool.RECTANGLE, Tool.CIRCLE):
            self._draw_shape_preview(painter)
        if self._mesh_overlay is not None:
            self._draw_mesh_overlay(painter, self._mesh_overlay)

    def _stroke_pen(self) -> QPen:
        color = Qt.GlobalColor.black if self._tool == Tool.PEN else Qt.GlobalColor.white
        width = 4 if self._tool == Tool.PEN else 12
        return QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)

    def _draw_point(self, point: QPoint) -> None:
        painter = QPainter(self._image)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(Qt.GlobalColor.black)
        painter.drawEllipse(point, 4, 4)
        painter.end()

    def _commit_shape(self) -> None:
        if self._shape_start is None or self._shape_end is None:
            return

        painter = QPainter(self._image)
        pen = QPen(Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.GlobalColor.black)

        if self._tool == Tool.SEGMENT:
            painter.drawLine(self._shape_start, self._shape_end)
        elif self._tool == Tool.RECTANGLE:
            rect = QRect(self._shape_start, self._shape_end).normalized()
            painter.drawRect(rect)
        elif self._tool == Tool.CIRCLE:
            rect = QRect(self._shape_start, self._shape_end).normalized()
            painter.drawEllipse(rect)
        painter.end()

    def _draw_shape_preview(self, painter: QPainter) -> None:
        if self._shape_start is None or self._shape_end is None:
            return
        preview_pen = QPen(Qt.GlobalColor.darkGray, 1, Qt.PenStyle.DashLine)
        painter.setPen(preview_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._tool == Tool.SEGMENT:
            painter.drawLine(self._shape_start, self._shape_end)
        elif self._tool == Tool.RECTANGLE:
            painter.drawRect(QRect(self._shape_start, self._shape_end).normalized())
        elif self._tool == Tool.CIRCLE:
            painter.drawEllipse(QRect(self._shape_start, self._shape_end).normalized())

    def _flood_fill(self, seed: QPoint) -> None:
        target = self._image.pixelColor(seed)
        fill_color = QColor(Qt.GlobalColor.black)
        if target == fill_color:
            return

        width = self._image.width()
        height = self._image.height()
        queue: deque[QPoint] = deque([seed])
        visited: set[tuple[int, int]] = set()

        while queue:
            point = queue.popleft()
            x, y = point.x(), point.y()
            if not (0 <= x < width and 0 <= y < height):
                continue
            key = (x, y)
            if key in visited:
                continue
            visited.add(key)

            if self._image.pixelColor(x, y) != target:
                continue
            self._image.setPixelColor(x, y, fill_color)
            queue.append(QPoint(x + 1, y))
            queue.append(QPoint(x - 1, y))
            queue.append(QPoint(x, y + 1))
            queue.append(QPoint(x, y - 1))

    def _draw_mesh_overlay(self, painter: QPainter, mesh: Mesh) -> None:
        pen = QPen(Qt.GlobalColor.red, 1, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for triangle in mesh.triangles:
            a = QPoint(int(triangle.a.x), int(triangle.a.y))
            b = QPoint(int(triangle.b.x), int(triangle.b.y))
            c = QPoint(int(triangle.c.x), int(triangle.c.y))
            painter.drawLine(a, b)
            painter.drawLine(b, c)
            painter.drawLine(c, a)

    def _in_bounds(self, point: QPoint) -> bool:
        return 0 <= point.x() < self._image.width() and 0 <= point.y() < self._image.height()

    def _push_undo_state(self) -> None:
        self._undo_stack.append(self._image.copy())
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    @staticmethod
    def binarize_image(image: QImage, threshold: int = 127) -> QImage:
        grayscale = image.convertToFormat(QImage.Format.Format_Grayscale8)
        output = QImage(grayscale.size(), QImage.Format.Format_RGB32)
        white = QColor(Qt.GlobalColor.white)
        black = QColor(Qt.GlobalColor.black)
        for y in range(grayscale.height()):
            for x in range(grayscale.width()):
                value = QColor(grayscale.pixel(x, y)).value()
                output.setPixelColor(x, y, black if value <= threshold else white)
        return output

    def to_pixmap(self) -> QPixmap:
        return QPixmap.fromImage(self._image)
