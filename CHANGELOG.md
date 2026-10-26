## [1.0.1] - 2026-10-26

### Added

- MIT license

### Updated

- README

## [1.0.0] - 2026-10-24

### Added

- Advancing front triangulation of planar domains with inclusions and zero-thickness cuts, using a two-sided representation of cuts in the front
- Ear clipping fallback for the remaining front when no candidate is found
- Domain input by drawing, from a raster image, or from a JSON coordinate file
- Contour preprocessing: duplicate removal, tolerance-based closure, Ramer–Douglas–Peucker simplification and self-intersection check
- Merging of overlapping inclusions and contour role detection by nesting depth parity
- Laplacian smoothing of interior nodes
- Continuous node and triangle numbering with node classification (interior, boundary, interface) and no duplicate numbers on inclusion boundaries
- Multi-start reverse Cuthill–McKee renumbering to reduce the global matrix bandwidth
- Mesh quality metrics (min/mean/max and histogram)
- Mesh export to `OBJ`, `PLY` and `VTK`
- Graphical interface for drawing and editing the domain, setting triangulation parameters (Fast, Balanced, Precise presets) and saving/loading data
