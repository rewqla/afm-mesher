from dataclasses import dataclass, field
from math import isfinite

from src.domain.entities.indexed_mesh import IndexedMesh
from src.domain.entities.linear_triangle import LinearTriangle
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle


@dataclass(slots=True)
class MeshSourceContours:
    """Original coordinate-import contours for round-trip export.

    This provenance is available only when the mesh is built from imported
    coordinate JSON. Hand-drawn canvas geometry does not preserve inclusion
    types separately from generic closed/open contours.
    """

    outer_boundary: list[tuple[float, float]]
    inclusions: list[tuple[list[tuple[float, float]], str]]
    cuts: list[list[tuple[float, float]]]


@dataclass(frozen=True, slots=True)
class TriangulationTimings:
    """Wall-clock seconds (time.perf_counter) spent by one triangulation run, per stage.

    Stages may not add up to `total_s`: the rest is glue code between them."""

    total_s: float
    contours_s: float = 0.0
    afm_s: float = 0.0
    postprocess_s: float = 0.0
    numbering_s: float = 0.0


@dataclass(slots=True)
class Mesh:
    triangles: list[Triangle]
    meters_per_pixel: float = 1.0
    node_order: tuple[Point, ...] | None = None
    indexed_mesh_data: IndexedMesh | None = None
    source_contours: MeshSourceContours | None = None
    # Cut segments exactly as the mesher constrained them (each is an edge of the mesh).
    # Display/export must use these instead of the raw drawn stroke, which the adapter simplifies.
    cut_edges: tuple[tuple[Point, Point], ...] = ()
    # Boundary of merge inclusions as meshed (each segment is an edge shared by both subdomains).
    interface_edges: tuple[tuple[Point, Point], ...] = ()
    timings: TriangulationTimings | None = None
    # User-facing notes about geometry that could not be taken into account (e.g. a dropped cut).
    warnings: tuple[str, ...] = ()
    _linear_triangles_cache: dict[tuple[int, float], tuple[LinearTriangle, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.triangles:
            raise ValueError("Mesh triangles must not be empty.")
        if not _is_positive_finite_number(self.meters_per_pixel):
            raise ValueError("Mesh meters_per_pixel must be a positive finite number.")
        if self.indexed_mesh_data is not None and not isinstance(self.indexed_mesh_data, IndexedMesh):
            raise ValueError("Mesh indexed_mesh_data must be an IndexedMesh instance when provided.")

    def linear_triangles(self, key_precision: int = 10, meters_per_pixel: float | None = None) -> list[LinearTriangle]:
        """Build cached LinearTriangle objects keyed by precision and scale."""
        if key_precision < 0:
            raise ValueError("key_precision must be >= 0")
        resolved_meters_per_pixel = self.meters_per_pixel if meters_per_pixel is None else meters_per_pixel
        if not _is_positive_finite_number(resolved_meters_per_pixel):
            raise ValueError("meters_per_pixel must be a positive finite number.")
        cache_key = (key_precision, float(resolved_meters_per_pixel))
        cached = self._linear_triangles_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        indexed_mesh = self.indexed_mesh(key_precision=key_precision)

        linear: list[LinearTriangle] = []
        for triangle_index, triangle in enumerate(indexed_mesh.triangles, start=1):
            node_coordinates = tuple(
                Point(*indexed_mesh.nodes[node_id - indexed_mesh.index_base])
                for node_id in triangle
            )
            linear.append(
                LinearTriangle(
                    triangle_number=triangle_index,
                    node_coordinates=node_coordinates,
                    node_numbers=triangle,
                    meters_per_pixel=float(resolved_meters_per_pixel),
                )
            )
        # Lazy import to avoid circular dependency: mesh_postprocessing depends on LinearTriangle,
        # while Mesh imports LinearTriangle at module load time.
        from src.application.services.mesh_postprocessing import renumber_triangles

        renumbered = tuple(renumber_triangles(linear))
        self._linear_triangles_cache[cache_key] = renumbered
        return list(renumbered)

    def indexed_mesh(
        self,
        boundary_points: list[Point] | None = None,
        interface_points: list[Point] | None = None,
        key_precision: int = 10,
    ) -> IndexedMesh:
        if key_precision < 0:
            raise ValueError("key_precision must be >= 0")
        if self.indexed_mesh_data is not None:
            if boundary_points is None and interface_points is None:
                return self.indexed_mesh_data
            # Keep the mesher's numbering (cut twins share coordinates and must not be glued back);
            # only re-derive the node attributes from the given points.
            return self._reclassified_indexed_mesh(
                self.indexed_mesh_data, boundary_points, interface_points, key_precision
            )

        key_to_node_number: dict[tuple[float, float], int] = {}
        node_coordinates_by_id: dict[int, tuple[float, float]] = {}
        triangles: list[tuple[int, int, int]] = []

        def point_key(point: Point) -> tuple[float, float]:
            return (round(point.x, key_precision), round(point.y, key_precision))

        def register_point(point: Point) -> int:
            key = point_key(point)
            node_number = key_to_node_number.get(key)
            if node_number is None:
                node_number = len(key_to_node_number) + 1
                key_to_node_number[key] = node_number
                node_coordinates_by_id[node_number] = (point.x, point.y)
            return node_number

        if self.node_order is not None:
            for point in self.node_order:
                register_point(point)

        for triangle in self.triangles:
            triangles.append(
                (
                    register_point(triangle.a),
                    register_point(triangle.b),
                    register_point(triangle.c),
                )
            )

        nodes = [node_coordinates_by_id[node_id] for node_id in range(1, len(node_coordinates_by_id) + 1)]
        boundary_nodes: set[int] = set()
        interface_nodes: set[int] = set()
        if boundary_points is not None:
            for point in boundary_points:
                node_id = key_to_node_number.get(point_key(point))
                if node_id is not None:
                    boundary_nodes.add(node_id)
        if interface_points is not None:
            for point in interface_points:
                node_id = key_to_node_number.get(point_key(point))
                if node_id is not None:
                    interface_nodes.add(node_id)

        bandwidth = max((max(triangle) - min(triangle) for triangle in triangles), default=0)
        indexed_mesh = IndexedMesh(
            nodes=nodes,
            triangles=triangles,
            boundary_nodes=boundary_nodes,
            interface_nodes=interface_nodes,
            bandwidth=bandwidth,
            index_base=1,
            meters_per_pixel=self.meters_per_pixel,
        )
        if boundary_points is None and interface_points is None:
            self.indexed_mesh_data = indexed_mesh
        return indexed_mesh

    def _reclassified_indexed_mesh(
        self,
        source: IndexedMesh,
        boundary_points: list[Point] | None,
        interface_points: list[Point] | None,
        key_precision: int,
    ) -> IndexedMesh:
        ids_by_key: dict[tuple[float, float], list[int]] = {}
        for offset, (x, y) in enumerate(source.nodes):
            ids_by_key.setdefault((round(x, key_precision), round(y, key_precision)), []).append(
                source.index_base + offset
            )

        def ids_of(points: list[Point] | None) -> set[int]:
            found: set[int] = set()
            for point in points or ():
                found.update(ids_by_key.get((round(point.x, key_precision), round(point.y, key_precision)), ()))
            return found

        return IndexedMesh(
            nodes=list(source.nodes),
            triangles=list(source.triangles),
            boundary_nodes=ids_of(boundary_points),
            interface_nodes=ids_of(interface_points),
            bandwidth=source.bandwidth,
            index_base=source.index_base,
            meters_per_pixel=source.meters_per_pixel,
            twin_groups=list(source.twin_groups),
            triangle_subdomains=list(source.triangle_subdomains),
        )


def _is_positive_finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return isfinite(float(value)) and float(value) > 0.0
