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
    # Nodes sharing coordinates on a cut or a merge-inclusion boundary: one number per side, ascending.
    twin_groups: list[tuple[int, ...]] = field(default_factory=list)
    # Subdomain of each triangle (aligned with `triangles`): 0 main domain, k the k-th merge inclusion.
    # Empty means every triangle belongs to the main domain.
    triangle_subdomains: list[int] = field(default_factory=list)

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
        seen_twins: set[int] = set()
        for group in self.twin_groups:
            if len(group) < 2 or len(set(group)) != len(group):
                raise ValueError("IndexedMesh twin group must contain at least 2 distinct node indices.")
            if any(node_id not in node_ids for node_id in group):
                raise ValueError("IndexedMesh twin node indices are out of bounds.")
            if seen_twins.intersection(group):
                raise ValueError("IndexedMesh twin groups must not overlap.")
            seen_twins.update(group)
        if self.triangle_subdomains and len(self.triangle_subdomains) != len(self.triangles):
            raise ValueError("IndexedMesh triangle_subdomains must match triangles.")
        if any(not isinstance(k, int) or k < 0 for k in self.triangle_subdomains):
            raise ValueError("IndexedMesh triangle subdomains must be non-negative integers.")

    def twin_map(self) -> dict[int, tuple[int, ...]]:
        """Node number -> numbers of its twins (same coordinates on a cut or merge-inclusion boundary)."""
        return {
            node_id: tuple(other for other in group if other != node_id)
            for group in self.twin_groups
            for node_id in group
        }

    def triangle_subdomain(self, triangle_index: int) -> int:
        """Subdomain of the triangle at 0-based position `triangle_index` in `triangles`."""
        return self.triangle_subdomains[triangle_index] if self.triangle_subdomains else 0

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
