# AFM Mesher

Triangulation of planar domains with inclusions and zero-thickness cuts by the advancing front method, with a single continuous numbering of nodes and triangles.

The program takes a domain (an outer boundary, inclusions and cuts), builds a conforming triangular mesh, and exports it. It is the software part of the paper *Triangulation of planar domains with inclusions and cuts by the advancing front method with continuous node numbering* (Mosiichuk Yu., Martyniuk P.).

## What it does

- **Domain input.** Draw the domain in the graphical editor, load a raster image (contours are extracted from it), or import a JSON file with coordinates.
- **Inclusions.** An inclusion is either a *hole* (excluded from the mesh) or a *subdomain* (meshed together with the main region, with the common boundary discretized identically on both sides).
- **Cuts.** A cut is an open polyline of zero thickness. Each cut segment enters the advancing front twice, in opposite directions, so the mesh grows on both sides and both banks are made of the same chain of mesh edges.
- **Meshing.** Advancing front with Steiner points; ear clipping as a fallback for the remaining front; edge flips, removal of valence-3 nodes and Laplacian smoothing afterwards.
- **Numbering.** One continuous numbering of nodes and of triangles. Nodes on cuts and subdomain interfaces are split into twins (one number per side), and every node gets a type: interior, boundary or interface.
- **Bandwidth reduction.** Multi-start reverse Cuthill–McKee renumbering of the nodes to reduce the bandwidth of the global matrix.
- **Export.** JSON (nodes with types and twin groups, triangles with subdomain numbers, mesh statistics) and `OBJ`, `PLY`, `VTK`.

## Installation

Python 3 is required (developed and run with Python 3.14).

```bash
pip install -r requirements.txt
```

Dependencies: PySide6, Pillow, matplotlib, shapely, numpy (pinned in `requirements.txt`).

## Usage

Graphical editor:

```bash
python src/main.py --mode ui
```

Batch run on the reference images in `data/images` (writes debug images and `OBJ`/`PLY`/`VTK` files to `data/output`):

```bash
python src/main.py --mode batch
```

In the editor, choose a preset (`Fast`, `Balanced`, `Accurate`) or `Custom` (target edge length `h`, smoothing iterations, contour tolerance, iteration factor, minimum triangle quality).

## Coordinate file format

```json
{
  "outer_boundary": [[100, 100], [500, 100], [500, 500], [100, 500], [100, 100]],
  "inclusions": [
    {
      "vertices": [[220, 220], [380, 220], [380, 380], [220, 380], [220, 220]],
      "type": "ignore"
    }
  ],
  "cuts": [
    [[130, 130], [190, 300]]
  ]
}
```

- `outer_boundary` is required: a list of `[x, y]` pairs.
- `inclusions` is optional. Each inclusion has `vertices` and a `type`: `"ignore"` (hole) or `"merge"` (subdomain meshed together with the main region).
- `cuts` is optional: a list of polylines, each with at least two points.
- Coordinates can be integers or floats. Coordinates from JSON are used as given, without simplification.

Examples are in `diagnostics/cases/` and `tests/fixtures/`.

## Project structure

```text
src/
  application/    mesher, region topology, boundary validation, postprocessing
  domain/         points, triangles, mesh, node types, geometry utilities
  infrastructure/ image preprocessing, coordinate import, export, visualization
  presentation/   graphical editor and the triangulation adapter
scripts/          test image generation, benchmarks
diagnostics/      diagnostic scripts and example domains
tests/            automated tests
```

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The suite has 357 tests in 37 modules. Line coverage is 87.4% for the core (without the user interface modules) and 84.3% for the whole system.

## License

MIT, see [LICENSE](LICENSE).

## Citation

If you use this software, please cite the paper above. The reference and the archived release DOI will be added here after publication.
