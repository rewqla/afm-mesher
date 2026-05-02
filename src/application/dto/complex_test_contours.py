from src.domain.entities.point import Point


def star_contour() -> list[Point]:
    """
    10-point concave star, alternating radii R=200 and r=80.
    Centered at (250, 250), coordinates in [0, 500].
    """
    return [
        Point(250.0, 50.0),
        Point(297.0, 185.0),
        Point(440.0, 188.0),
        Point(326.0, 275.0),
        Point(368.0, 412.0),
        Point(250.0, 330.0),
        Point(132.0, 412.0),
        Point(174.0, 275.0),
        Point(60.0, 188.0),
        Point(203.0, 185.0),
    ]


def u_shape_contour() -> list[Point]:
    """
    Thin-wall U-shape (concave polygon), coordinates in [0, 500].
    """
    return [
        Point(100.0, 80.0),
        Point(180.0, 80.0),
        Point(180.0, 340.0),
        Point(320.0, 340.0),
        Point(320.0, 80.0),
        Point(400.0, 80.0),
        Point(400.0, 420.0),
        Point(100.0, 420.0),
    ]


def hourglass_contour() -> list[Point]:
    """
    Hourglass / narrow-neck shape:
    two wide blocks connected by a thin channel (~40 px width).
    """
    return [
        Point(60.0, 120.0),
        Point(220.0, 120.0),
        Point(220.0, 230.0),
        Point(280.0, 230.0),
        Point(280.0, 120.0),
        Point(440.0, 120.0),
        Point(440.0, 380.0),
        Point(280.0, 380.0),
        Point(280.0, 270.0),
        Point(220.0, 270.0),
        Point(220.0, 380.0),
        Point(60.0, 380.0),
    ]
