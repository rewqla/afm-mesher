from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from src.domain.entities.node_type import NodeType


@dataclass(slots=True)
class IndexedMesh:
    nodes: list[tuple[float, float]]
    triangles: list[tuple[int, int, int]]
    boundary_nodes: set[int]
    bandwidth: int
    interface_nodes: set[int] = field(default_factory=set)
    index_base: int = 1
    meters_per_pixel: float = 1.0

    def __post_init__(self) -> None:
        if self.index_base not in (0, 1):
            raise ValueError("IndexedMesh index_base must be 0 or 1.")
        if self.bandwidth < 0:
            raise ValueError("IndexedMesh bandwidth must be >= 0.")
        if not _is_positive_finite_number(self.meters_per_pixel):
            raise ValueError("IndexedMesh meters_per_pixel must be a positive finite number.")

        node_ids = set(range(self.index_base, self.index_base + len(self.nodes)))
        for triangle in self.triangles:
            if len(triangle) != 3:
                raise ValueError("IndexedMesh triangles must contain exactly 3 node indices.")
            if len(set(triangle)) != 3:
                raise ValueError("IndexedMesh triangle node indices must be unique inside one triangle.")
            if any(node_id not in node_ids for node_id in triangle):
                raise ValueError("IndexedMesh triangle node indices are out of bounds.")

        if any(node_id not in node_ids for node_id in self.boundary_nodes):
            raise ValueError("IndexedMesh boundary node indices are out of bounds.")
        if any(node_id not in node_ids for node_id in self.interface_nodes):
            raise ValueError("IndexedMesh interface node indices are out of bounds.")

    def node_type(self, node_id: int) -> NodeType:
        if node_id in self.interface_nodes:
            # INTERFACE wins over BOUNDARY on collisions because it is the more
            # specific classification for internal constrained/interface nodes.
            return NodeType.INTERFACE
        if node_id in self.boundary_nodes:
            return NodeType.BOUNDARY
        return NodeType.INTERIOR


def _is_positive_finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return isfinite(float(value)) and float(value) > 0.0
