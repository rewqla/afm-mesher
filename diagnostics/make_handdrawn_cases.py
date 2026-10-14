"""Генерує "намальовані від руки" варіанти рис. 2.2: щільні цілочисельні полілінії (як із canvas).

    .venv\\Scripts\\python diagnostics\\make_handdrawn_cases.py
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

CASES = Path(__file__).resolve().parent / "cases"


def densify(points, step: float, rng: random.Random, jitter: float, closed: bool):
    pts = [tuple(p) for p in points]
    if closed and pts[0] == pts[-1]:
        pts = pts[:-1]
    pairs = list(zip(pts, pts[1:] + pts[:1])) if closed else list(zip(pts[:-1], pts[1:]))
    out = []
    for a, b in pairs:
        n = max(1, math.ceil(math.dist(a, b) / step))
        for k in range(n):
            t = k / n
            x = a[0] + (b[0] - a[0]) * t + rng.uniform(-jitter, jitter)
            y = a[1] + (b[1] - a[1]) * t + rng.uniform(-jitter, jitter)
            p = [round(x), round(y)]
            if not out or p != out[-1]:
                out.append(p)
    if not closed:
        out.append([round(pts[-1][0]), round(pts[-1][1])])
    else:
        out.append(out[0])
    return out


def main() -> None:
    base = json.loads((CASES / "fig22_like.json").read_text(encoding="utf-8"))
    rng = random.Random(22)
    outer = densify(base["outer_boundary"], 3.0, rng, 0.8, closed=True)
    hole = densify(base["inclusions"][0]["vertices"], 3.0, rng, 0.8, closed=True)
    cut = densify(base["cuts"][0], 3.0, rng, 0.8, closed=False)
    case = {"outer_boundary": outer, "inclusions": [{"vertices": hole, "type": "ignore"}], "cuts": [cut]}
    (CASES / "fig22_handdrawn.json").write_text(json.dumps(case), encoding="utf-8")

    # той самий, але розріз починається на зовнішній межі (верхній край, x=470 -> y=15)
    cut_b = densify([[470, 15], *base["cuts"][0]], 3.0, random.Random(23), 0.8, closed=False)
    case_b = {"outer_boundary": outer, "inclusions": [{"vertices": hole, "type": "ignore"}], "cuts": [cut_b]}
    (CASES / "fig22_handdrawn_cut_from_boundary.json").write_text(json.dumps(case_b), encoding="utf-8")
    print("ok:", len(outer), len(hole), len(cut), len(cut_b))


if __name__ == "__main__":
    main()
