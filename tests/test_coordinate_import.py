from __future__ import annotations

import json
import unittest

from src.infrastructure.io.coordinate_json_importer import CoordinateJsonImporter


class CoordinateJsonImporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.importer = CoordinateJsonImporter()

    def test_loads_valid_json_with_outer_boundary_only(self) -> None:
        imported = self.importer.loads(
            json.dumps(
                {
                    "outer_boundary": [
                        [0.0, 0.0],
                        [10.0, 0.0],
                        [10.0, 10.0],
                        [0.0, 10.0],
                        [0.0, 0.0],
                    ]
                }
            )
        )

        self.assertEqual(
            imported.outer_boundary,
            [
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
                (0.0, 0.0),
            ],
        )
        self.assertEqual(imported.inclusions, [])
        self.assertEqual(imported.cuts, [])

    def test_loads_json_with_ignore_inclusion(self) -> None:
        imported = self.importer.loads(
            json.dumps(
                {
                    "outer_boundary": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
                    "inclusions": [
                        {
                            "vertices": [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]],
                            "type": "ignore",
                        }
                    ],
                }
            )
        )

        self.assertEqual(len(imported.inclusions), 1)
        self.assertEqual(imported.inclusions[0][1], "ignore")

    def test_loads_json_with_merge_inclusion(self) -> None:
        imported = self.importer.loads(
            json.dumps(
                {
                    "outer_boundary": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
                    "inclusions": [
                        {
                            "vertices": [[5, 5], [7, 5], [7, 7], [5, 7], [5, 5]],
                            "type": "merge",
                        }
                    ],
                }
            )
        )

        self.assertEqual(len(imported.inclusions), 1)
        self.assertEqual(imported.inclusions[0][1], "merge")

    def test_loads_json_with_cuts(self) -> None:
        imported = self.importer.loads(
            json.dumps(
                {
                    "outer_boundary": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
                    "cuts": [
                        [[1, 1], [9, 9]],
                        [[1, 9], [9, 1]],
                    ],
                }
            )
        )

        self.assertEqual(imported.inclusions, [])
        self.assertEqual(len(imported.cuts), 2)
        self.assertEqual(imported.cuts[0], [(1.0, 1.0), (9.0, 9.0)])

    def test_loads_json_with_all_sections(self) -> None:
        imported = self.importer.loads(
            json.dumps(
                {
                    "outer_boundary": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
                    "inclusions": [
                        {
                            "vertices": [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]],
                            "type": "ignore",
                        },
                        {
                            "vertices": [[6, 6], [8, 6], [8, 8], [6, 8], [6, 6]],
                            "type": "merge",
                        },
                    ],
                    "cuts": [
                        [[1, 5], [9, 5]],
                    ],
                }
            )
        )

        self.assertEqual(len(imported.outer_boundary), 5)
        self.assertEqual(len(imported.inclusions), 2)
        self.assertEqual(len(imported.cuts), 1)

    def test_accepts_integer_coordinates(self) -> None:
        imported = self.importer.loads(
            json.dumps(
                {
                    "outer_boundary": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
                }
            )
        )

        self.assertEqual(imported.outer_boundary[0], (0.0, 0.0))
        self.assertEqual(imported.outer_boundary[1], (10.0, 0.0))

    def test_to_canvas_contours_closes_open_boundary(self) -> None:
        imported = self.importer.loads(
            json.dumps(
                {
                    "outer_boundary": [[0, 0], [10, 0], [10, 10], [0, 10]],
                }
            )
        )

        contours = self.importer.to_canvas_contours(imported)

        self.assertEqual(contours[0][0], contours[0][-1])
        self.assertEqual(
            contours[0],
            [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)],
        )

    def test_missing_outer_boundary_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "outer_boundary"):
            self.importer.loads(json.dumps({"cuts": [[[0, 0], [1, 1]]]}))

    def test_outer_boundary_with_two_vertices_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "outer_boundary"):
            self.importer.loads(json.dumps({"outer_boundary": [[0, 0], [1, 1]]}))

    def test_inclusion_without_vertices_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "vertices"):
            self.importer.loads(
                json.dumps(
                    {
                        "outer_boundary": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
                        "inclusions": [{"type": "ignore"}],
                    }
                )
            )

    def test_inclusion_with_unknown_type_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "невідомий тип"):
            self.importer.loads(
                json.dumps(
                    {
                        "outer_boundary": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
                        "inclusions": [
                            {
                                "vertices": [[2, 2], [4, 2], [4, 4], [2, 4]],
                                "type": "unknown",
                            }
                        ],
                    }
                )
            )

    def test_cut_with_one_vertex_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "щонайменше 2 вершини"):
            self.importer.loads(
                json.dumps(
                    {
                        "outer_boundary": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
                        "cuts": [[[1, 1]]],
                    }
                )
            )

    def test_invalid_json_syntax_raises_json_decode_error(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            self.importer.loads('{"outer_boundary": [[0, 0], [1, 1]]')

    def test_point_with_single_element_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[x, y\]"):
            self.importer.loads(json.dumps({"outer_boundary": [[0], [1, 1], [2, 2]]}))

    def test_point_with_three_elements_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[x, y\]"):
            self.importer.loads(json.dumps({"outer_boundary": [[0, 0, 0], [1, 1], [2, 2]]}))

    def test_point_with_non_numeric_values_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[x, y\]"):
            self.importer.loads(json.dumps({"outer_boundary": [["a", "b"], [1, 1], [2, 2]]}))

    def test_to_canvas_contours_output_order_and_closure(self) -> None:
        imported = self.importer.loads(
            json.dumps(
                {
                    "outer_boundary": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
                    "inclusions": [
                        {
                            "vertices": [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]],
                            "type": "ignore",
                        },
                        {
                            "vertices": [[6, 6], [8, 6], [8, 8], [6, 8], [6, 6]],
                            "type": "merge",
                        },
                    ],
                    "cuts": [
                        [[1, 5], [9, 5]],
                    ],
                }
            )
        )

        contours = self.importer.to_canvas_contours(imported)

        self.assertEqual(contours[0], imported.outer_boundary)
        self.assertEqual(contours[1][0], contours[1][-1])
        self.assertEqual(contours[2][0], contours[2][-1])
        self.assertEqual(contours[3], [(1.0, 5.0), (9.0, 5.0)])
        self.assertNotEqual(contours[3][0], contours[3][-1])

    def test_to_canvas_contours_places_inclusions_after_boundary(self) -> None:
        imported = self.importer.loads(
            json.dumps(
                {
                    "outer_boundary": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
                    "inclusions": [
                        {
                            "vertices": [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]],
                            "type": "ignore",
                        }
                    ],
                }
            )
        )

        contours = self.importer.to_canvas_contours(imported)

        self.assertEqual(contours[0], imported.outer_boundary)
        self.assertEqual(contours[1][0], contours[1][-1])

    def test_to_canvas_contours_places_cuts_last(self) -> None:
        imported = self.importer.loads(
            json.dumps(
                {
                    "outer_boundary": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
                    "inclusions": [
                        {
                            "vertices": [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]],
                            "type": "ignore",
                        }
                    ],
                    "cuts": [
                        [[1, 5], [9, 5]],
                    ],
                }
            )
        )

        contours = self.importer.to_canvas_contours(imported)

        self.assertEqual(contours[-1], [(1.0, 5.0), (9.0, 5.0)])
        self.assertNotEqual(contours[-1][0], contours[-1][-1])


if __name__ == "__main__":
    unittest.main()
