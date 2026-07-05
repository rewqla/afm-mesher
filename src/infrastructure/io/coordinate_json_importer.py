from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ImportedCoordinates:
    outer_boundary: list[tuple[float, float]]
    inclusions: list[tuple[list[tuple[float, float]], str]]
    cuts: list[list[tuple[float, float]]]


class CoordinateJsonImporter:
    """
    Завантажує та валідує JSON-файл з координатами контурів.
    Повертає ImportedCoordinates або кидає ValueError з описом помилки.
    """

    def load(self, path: str) -> ImportedCoordinates:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        return self._parse(data)

    def loads(self, text: str) -> ImportedCoordinates:
        data = json.loads(text)
        return self._parse(data)

    def _parse(self, data: dict[str, Any]) -> ImportedCoordinates:
        if "outer_boundary" not in data:
            raise ValueError("JSON не містить поля 'outer_boundary'")
        boundary = self._parse_vertices(data["outer_boundary"], "outer_boundary")
        if len(boundary) < 3:
            raise ValueError("'outer_boundary' має містити щонайменше 3 вершини")

        inclusions: list[tuple[list[tuple[float, float]], str]] = []
        for i, inc in enumerate(data.get("inclusions", [])):
            if not isinstance(inc, dict):
                raise ValueError(f"Включення [{i}] має бути об'єктом")
            if "vertices" not in inc:
                raise ValueError(f"Включення [{i}] не містить поля 'vertices'")
            verts = self._parse_vertices(inc["vertices"], f"inclusions[{i}].vertices")
            if len(verts) < 3:
                raise ValueError(f"Включення [{i}] має щонайменше 3 вершини")
            inc_type = inc.get("type", "ignore")
            if inc_type not in ("ignore", "merge"):
                raise ValueError(
                    f"Включення [{i}]: невідомий тип '{inc_type}'. "
                    "Допустимі значення: 'ignore', 'merge'"
                )
            inclusions.append((verts, inc_type))

        cuts: list[list[tuple[float, float]]] = []
        for i, cut in enumerate(data.get("cuts", [])):
            verts = self._parse_vertices(cut, f"cuts[{i}]")
            if len(verts) < 2:
                raise ValueError(f"Розріз [{i}] має щонайменше 2 вершини")
            cuts.append(verts)

        return ImportedCoordinates(
            outer_boundary=boundary,
            inclusions=inclusions,
            cuts=cuts,
        )

    def _parse_vertices(self, raw: Any, field_name: str) -> list[tuple[float, float]]:
        if not isinstance(raw, list):
            raise ValueError(f"'{field_name}' має бути списком")
        result: list[tuple[float, float]] = []
        for i, pt in enumerate(raw):
            if (
                not isinstance(pt, (list, tuple))
                or len(pt) != 2
                or not all(isinstance(c, (int, float)) for c in pt)
            ):
                raise ValueError(
                    f"'{field_name}[{i}]' має бути парою чисел [x, y], отримано: {pt!r}"
                )
            result.append((float(pt[0]), float(pt[1])))
        return result

    def to_canvas_contours(self, imported: ImportedCoordinates) -> list[list[tuple[float, float]]]:
        """
        Конвертує ImportedCoordinates у формат canvas:
        list[list[tuple[float, float]]]
        - closed contours: outer_boundary + inclusions (з повтором першої точки)
        - open contours: cuts
        """
        contours: list[list[tuple[float, float]]] = []

        boundary = list(imported.outer_boundary)
        if boundary and boundary[0] != boundary[-1]:
            boundary.append(boundary[0])
        contours.append(boundary)

        for verts, _ in imported.inclusions:
            pts = list(verts)
            if pts and pts[0] != pts[-1]:
                pts.append(pts[0])
            contours.append(pts)

        for cut in imported.cuts:
            contours.append(list(cut))

        return contours
