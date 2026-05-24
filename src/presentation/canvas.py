from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap, QPolygon
from PySide6.QtWidgets import QWidget

from src.domain.entities.point import Point
from src.domain.geometry.contour_pipeline import preprocess_contour, smart_append_contour
from src.domain.entities.mesh import Mesh
from src.presentation.tools import Tool


class Canvas(QWidget):
    image_changed = Signal()
    MIN_WIDTH = 256
    MIN_HEIGHT = 256
    MAX_WIDTH = 4096
    MAX_HEIGHT = 4096

    def __init__(self, width: int = 900, height: int = 650, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._preprocess_epsilon = 0.75
        self._closure_tolerance = 8.0
        self._continuation_tolerance_sq = self._closure_tolerance * self._closure_tolerance
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
        self._geometry_contours: list[list[tuple[float, float]]] = []
        self._filled_contour_keys: set[tuple[tuple[float, float], ...]] = set()
        self._invalid_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._invalid_points: list[tuple[float, float]] = []
        self._current_stroke: list[tuple[float, float]] = []
        self._active_base_contour: list[tuple[float, float]] | None = None
        self._active_base_contour_index: int | None = None
        self._active_attach_side = "end"
        self._geometry_dirty = False
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
        self._geometry_contours.clear()
        self._filled_contour_keys.clear()
        self._invalid_segments.clear()
        self._invalid_points.clear()
        self._geometry_dirty = False
        self._mesh_overlay = None
        self.update()
        self.image_changed.emit()

    def set_image(self, image: QImage, *, record_history: bool = True) -> None:
        if record_history:
            self._push_undo_state()
        converted = image.convertToFormat(QImage.Format.Format_RGB32)
        self._image = converted
        self._geometry_contours.clear()
        self._filled_contour_keys.clear()
        self._invalid_segments.clear()
        self._invalid_points.clear()
        self._geometry_dirty = False
        self._mesh_overlay = None
        self.setMinimumSize(converted.width(), converted.height())
        self.resize(converted.size())
        self.update()
        self.image_changed.emit()

    def image_data(self) -> QImage:
        return self._image.copy()

    def canvas_size(self) -> QSize:
        return self._image.size()

    def geometry_contours(self) -> list[list[tuple[float, float]]]:
        return [[(x, y) for x, y in contour] for contour in self._geometry_contours]

    def geometry_is_dirty(self) -> bool:
        return self._geometry_dirty

    def resize_canvas(self, width: int, height: int, *, record_history: bool = True) -> tuple[bool, str | None]:
        if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
            return False, (
                f"Canvas size must be at least {self.MIN_WIDTH} x {self.MIN_HEIGHT}px."
            )
        if width > self.MAX_WIDTH or height > self.MAX_HEIGHT:
            return False, (
                f"Canvas size must not exceed {self.MAX_WIDTH} x {self.MAX_HEIGHT}px."
            )

        content_bounds = self._content_bounds()
        if content_bounds is not None:
            min_x, min_y, max_x, max_y = content_bounds
            if max_x >= width or max_y >= height:
                return False, (
                    f"Cannot apply {width} x {height}px. Existing content occupies "
                    f"x={min_x}..{max_x}, y={min_y}..{max_y} and would fall outside the canvas."
                )

        if width == self._image.width() and height == self._image.height():
            return True, None

        if record_history:
            self._push_undo_state()

        resized = QImage(width, height, QImage.Format.Format_RGB32)
        resized.fill(Qt.GlobalColor.white)
        painter = QPainter(resized)
        painter.drawImage(0, 0, self._image)
        painter.end()

        self._image = resized
        self._mesh_overlay = None
        self.setMinimumSize(width, height)
        self.resize(width, height)
        self.updateGeometry()
        self.update()
        self.image_changed.emit()
        return True, None

    def set_geometry_contours(
        self,
        contours: list[list[tuple[float, float]]],
        *,
        preprocess: bool = True,
        redraw_image: bool = True,
        emit_change: bool = True,
    ) -> None:
        self._geometry_contours = []
        self._filled_contour_keys.clear()
        for contour in contours:
            self._store_contour(contour, preprocess=preprocess)
        self._geometry_dirty = False
        if redraw_image:
            self._redraw_geometry_layer()
        self.update()
        if emit_change:
            self.image_changed.emit()

    def set_invalid_segments(self, segments: list[tuple[tuple[float, float], tuple[float, float]]]) -> None:
        normalized = [((a[0], a[1]), (b[0], b[1])) for a, b in segments]
        self._invalid_segments = normalized[-1:]  # Always keep only latest visual hint.
        self.update()

    def set_invalid_points(self, points: list[tuple[float, float]]) -> None:
        self._invalid_points = [(float(x), float(y)) for x, y in points]
        self.update()

    def set_mesh_overlay(self, mesh: Mesh | None) -> None:
        self._mesh_overlay = mesh
        self.update()

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._image.copy())
        self._image = self._undo_stack.pop()
        self._geometry_contours.clear()
        self._filled_contour_keys.clear()
        self._invalid_segments.clear()
        self._invalid_points.clear()
        self._geometry_dirty = True
        self._mesh_overlay = None
        self.update()
        self.image_changed.emit()
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._image.copy())
        self._image = self._redo_stack.pop()
        self._geometry_contours.clear()
        self._filled_contour_keys.clear()
        self._invalid_segments.clear()
        self._invalid_points.clear()
        self._geometry_dirty = True
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
        self._invalid_segments.clear()
        self._invalid_points.clear()
        self._active_base_contour = None
        self._active_base_contour_index = None
        self._active_attach_side = "end"
        if self._tool in (Tool.PEN, Tool.ERASER):
            self._push_undo_state()
            self._drawing = True
            self._last_point = point
            if self._tool == Tool.PEN:
                (
                    self._active_base_contour,
                    self._active_base_contour_index,
                    self._active_attach_side,
                    stroke_start,
                ) = self._take_continuation_target(point.x(), point.y())
                self._current_stroke = [stroke_start]
                if self._current_stroke:
                    sx, sy = self._current_stroke[-1]
                    self._last_point = QPoint(int(round(sx)), int(round(sy)))
            else:
                self._geometry_contours.clear()
                self._filled_contour_keys.clear()
                self._geometry_dirty = True
        elif self._tool == Tool.FILL:
            self._push_undo_state()
            self._geometry_contours.clear()
            self._filled_contour_keys.clear()
            self._geometry_dirty = True
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
            if self._tool == Tool.PEN:
                self._current_stroke.append((float(point.x()), float(point.y())))
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
        elif self._tool == Tool.PEN:
            self._maybe_commit_pen_contour()

        self._drawing = False
        self._shape_start = None
        self._shape_end = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.drawImage(0, 0, self._image)

        if self._drawing and self._tool in (Tool.SEGMENT, Tool.RECTANGLE, Tool.CIRCLE):
            self._draw_shape_preview(painter)
        if self._invalid_segments:
            self._draw_invalid_segments(painter)
        if self._invalid_points:
            self._draw_invalid_points(painter)
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
        self._append_circle_contour(point, 4)
        self._geometry_dirty = False

    def _commit_shape(self) -> None:
        if self._shape_start is None or self._shape_end is None:
            return

        painter = QPainter(self._image)
        pen = QPen(Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.GlobalColor.black)

        if self._tool == Tool.SEGMENT:
            painter.drawLine(self._shape_start, self._shape_end)
            self._store_contour(
                [
                    (float(self._shape_start.x()), float(self._shape_start.y())),
                    (float(self._shape_end.x()), float(self._shape_end.y())),
                ],
                preprocess=False,
            )
            self._geometry_dirty = False
        elif self._tool == Tool.RECTANGLE:
            rect = QRect(self._shape_start, self._shape_end).normalized()
            painter.drawRect(rect)
            contour = [
                (rect.left(), rect.top()),
                (rect.right(), rect.top()),
                (rect.right(), rect.bottom()),
                (rect.left(), rect.bottom()),
                (rect.left(), rect.top()),
            ]
            self._geometry_contours.append(contour)
            self._filled_contour_keys.add(self._contour_key(contour))
            self._geometry_dirty = False
        elif self._tool == Tool.CIRCLE:
            rect = QRect(self._shape_start, self._shape_end).normalized()
            painter.drawEllipse(rect)
            self._append_ellipse_contour(rect)
            self._geometry_dirty = False
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
        target = self._image.pixel(seed.x(), seed.y())
        fill_color = QColor(Qt.GlobalColor.black).rgb()
        if target == fill_color:
            return

        width = self._image.width()
        height = self._image.height()
        stack: list[tuple[int, int]] = [(seed.x(), seed.y())]

        while stack:
            x, y = stack.pop()
            if not (0 <= x < width and 0 <= y < height):
                continue
            if self._image.pixel(x, y) != target:
                continue

            left = x
            while left >= 0 and self._image.pixel(left, y) == target:
                left -= 1
            left += 1

            right = x
            while right < width and self._image.pixel(right, y) == target:
                right += 1
            right -= 1

            for fill_x in range(left, right + 1):
                self._image.setPixel(fill_x, y, fill_color)

            for neighbor_y in (y - 1, y + 1):
                if not (0 <= neighbor_y < height):
                    continue
                run_start: int | None = None
                for scan_x in range(left, right + 1):
                    if self._image.pixel(scan_x, neighbor_y) == target:
                        if run_start is None:
                            run_start = scan_x
                    elif run_start is not None:
                        stack.append(((run_start + scan_x - 1) // 2, neighbor_y))
                        run_start = None
                if run_start is not None:
                    stack.append(((run_start + right) // 2, neighbor_y))

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

    def _draw_invalid_segments(self, painter: QPainter) -> None:
        pen = QPen(Qt.GlobalColor.yellow, 3, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for (x1, y1), (x2, y2) in self._invalid_segments:
            painter.drawLine(QPoint(int(round(x1)), int(round(y1))), QPoint(int(round(x2)), int(round(y2))))

    def _draw_invalid_points(self, painter: QPainter) -> None:
        pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for x, y in self._invalid_points:
            center = QPoint(int(round(x)), int(round(y)))
            painter.drawEllipse(center, 7, 7)

    def _in_bounds(self, point: QPoint) -> bool:
        return 0 <= point.x() < self._image.width() and 0 <= point.y() < self._image.height()

    def _push_undo_state(self) -> None:
        self._undo_stack.append(self._image.copy())
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _maybe_commit_pen_contour(self) -> None:
        min_points = 2 if self._active_base_contour is not None else 4
        if len(self._current_stroke) < min_points:
            self._current_stroke = []
            self._active_base_contour = None
            self._active_base_contour_index = None
            self._active_attach_side = "end"
            return
        contour_points = self._to_points(self._current_stroke)
        contour_points = preprocess_contour(
            contour_points,
            epsilon=self._preprocess_epsilon,
            closure_tolerance=self._closure_tolerance,
        )
        if self._active_base_contour is not None:
            base_points = self._to_points(self._active_base_contour)
            contour_points = smart_append_contour(
                base_points,
                contour_points,
                self._closure_tolerance,
                attach_to=self._active_attach_side,
            )
            contour_points = preprocess_contour(
                contour_points,
                epsilon=self._preprocess_epsilon,
                closure_tolerance=self._closure_tolerance,
            )
        if self._active_base_contour_index is not None:
            removed = self._geometry_contours.pop(self._active_base_contour_index)
            self._filled_contour_keys.discard(self._contour_key(removed))
        self._store_contour([(point.x, point.y) for point in contour_points])
        self._redraw_geometry_layer()
        self._geometry_dirty = False
        self._current_stroke = []
        self._active_base_contour = None
        self._active_base_contour_index = None
        self._active_attach_side = "end"

    def _append_circle_contour(self, center: QPoint, radius: int, segments: int = 20) -> None:
        from math import cos, pi, sin

        contour: list[tuple[int, int]] = []
        for i in range(segments):
            angle = 2.0 * pi * i / segments
            x = int(round(center.x() + radius * cos(angle)))
            y = int(round(center.y() + radius * sin(angle)))
            contour.append((x, y))
        contour.append(contour[0])
        self._store_contour(contour)
        self._filled_contour_keys.add(self._contour_key(self._geometry_contours[-1]))
        self._geometry_dirty = False

    def _append_ellipse_contour(self, rect: QRect, segments: int = 40) -> None:
        from math import cos, pi, sin

        cx = rect.center().x()
        cy = rect.center().y()
        rx = max(1, rect.width() / 2.0)
        ry = max(1, rect.height() / 2.0)
        contour: list[tuple[int, int]] = []
        for i in range(segments):
            angle = 2.0 * pi * i / segments
            x = int(round(cx + rx * cos(angle)))
            y = int(round(cy + ry * sin(angle)))
            contour.append((x, y))
        contour.append(contour[0])
        self._store_contour(contour)
        self._filled_contour_keys.add(self._contour_key(self._geometry_contours[-1]))
        self._geometry_dirty = False

    def _redraw_geometry_layer(self) -> None:
        self._image.fill(Qt.GlobalColor.white)
        painter = QPainter(self._image)
        pen = QPen(Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        for contour in self._geometry_contours:
            if len(contour) < 2:
                continue
            if self._contour_is_closed(contour) and self._contour_key(contour) in self._filled_contour_keys:
                painter.setBrush(Qt.GlobalColor.black)
                polygon_points = [
                    QPoint(int(round(x)), int(round(y)))
                    for x, y in contour
                ]
                painter.drawPolygon(QPolygon(polygon_points))
                continue
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(len(contour) - 1):
                x1, y1 = contour[i]
                x2, y2 = contour[i + 1]
                painter.drawLine(
                    QPoint(int(round(x1)), int(round(y1))),
                    QPoint(int(round(x2)), int(round(y2))),
                )
        painter.end()

    def _take_continuation_target(
        self,
        x: float,
        y: float,
    ) -> tuple[list[tuple[float, float]] | None, int | None, str, tuple[float, float]]:
        if not self._geometry_contours:
            return None, None, "end", (float(x), float(y))

        best_idx = -1
        attach_side = "end"
        best_dist = self._continuation_tolerance_sq + 1.0
        for idx, contour in enumerate(self._geometry_contours):
            if len(contour) < 2 or self._contour_is_closed(contour):
                continue
            start = contour[0]
            end = contour[-1]
            start_dist = self._distance_sq(start, (x, y))
            end_dist = self._distance_sq(end, (x, y))
            if start_dist <= self._continuation_tolerance_sq and start_dist < best_dist:
                best_idx = idx
                attach_side = "start"
                best_dist = start_dist
            if end_dist <= self._continuation_tolerance_sq and end_dist < best_dist:
                best_idx = idx
                attach_side = "end"
                best_dist = end_dist

        if best_idx < 0:
            return None, None, "end", (float(x), float(y))

        target = self._geometry_contours[best_idx]
        anchor = target[0] if attach_side == "start" else target[-1]
        return list(target), best_idx, attach_side, anchor

    def _store_contour(self, contour: list[tuple[float, float]], *, preprocess: bool = True) -> None:
        if len(contour) < 2:
            return
        if preprocess:
            processed = preprocess_contour(
                self._to_points(contour),
                epsilon=self._preprocess_epsilon,
                closure_tolerance=self._closure_tolerance,
            )
        else:
            processed = self._to_points(contour)
        if len(processed) < 2:
            return
        self._geometry_contours.append([(point.x, point.y) for point in processed])

    def _contour_key(self, contour: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
        return tuple((round(float(x), 3), round(float(y), 3)) for x, y in contour)

    def _contour_is_closed(self, contour: list[tuple[float, float]]) -> bool:
        if len(contour) < 4:
            return False
        return self._distance_sq(contour[0], contour[-1]) <= 1e-6

    def _to_points(self, contour: list[tuple[float, float]]) -> list[Point]:
        return [Point(float(x), float(y)) for x, y in contour]

    def _distance_sq(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        return dx * dx + dy * dy

    def _content_bounds(self) -> tuple[int, int, int, int] | None:
        raster_bounds = self._raster_content_bounds()
        contour_bounds = self._contour_content_bounds()
        if raster_bounds is None:
            return contour_bounds
        if contour_bounds is None:
            return raster_bounds
        return (
            min(raster_bounds[0], contour_bounds[0]),
            min(raster_bounds[1], contour_bounds[1]),
            max(raster_bounds[2], contour_bounds[2]),
            max(raster_bounds[3], contour_bounds[3]),
        )

    def _raster_content_bounds(self) -> tuple[int, int, int, int] | None:
        white = QColor(Qt.GlobalColor.white).rgb()
        min_x = self._image.width()
        min_y = self._image.height()
        max_x = -1
        max_y = -1
        for y in range(self._image.height()):
            for x in range(self._image.width()):
                if self._image.pixel(x, y) == white:
                    continue
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        if max_x < 0 or max_y < 0:
            return None
        return (min_x, min_y, max_x, max_y)

    def _contour_content_bounds(self) -> tuple[int, int, int, int] | None:
        if not self._geometry_contours:
            return None
        xs = [point[0] for contour in self._geometry_contours for point in contour]
        ys = [point[1] for contour in self._geometry_contours for point in contour]
        if not xs or not ys:
            return None
        return (
            int(round(min(xs))),
            int(round(min(ys))),
            int(round(max(xs))),
            int(round(max(ys))),
        )

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
