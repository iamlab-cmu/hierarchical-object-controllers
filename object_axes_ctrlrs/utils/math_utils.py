import numpy as np


def points_on_same_side(p1, p2, a, b):
    """Check if points a and b lie on the same or opposite side of the line.

    Line goes from p1 to p2.

    p1: (x, y) Tuple.
    p2: (x, y) Tuple.
    a: (x, y) Tuple.
    b: (x, y) Tuple.

    Return: Tuple. (points on opposite_side, atleast one point is on line.)
    """
    x1, y1 = p1
    x2, y2 = p2
    ax, ay = a
    bx, by = b

    diff = ((y1 - y2) * (ax - x1) + (x2 - x1) * (ay - y1)) * \
            ((y1 - y2) * (bx - x1) + (x2 - x1) * (by - y1))
    # Equality would mean that one of the points lies on the line.
    same_side = not (diff < 0 or abs(diff) < 1e-6)
    return same_side


def angle_between_axis(v1, v2):
    """Find angle between vectrors
    
    v1: Tuple (x, y)
    v2: Tuple (x, y)
    """
    v1_norm = np.linalg.norm(v1)
    if v1_norm <= 1e-8:
        return 0.0
    v1 = v1 / v1_norm

    v2_norm = np.linalg.norm(v2)
    if v2_norm <= 1e-8:
        return 0.0
    v2 = v2 / v2_norm

    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def check_underflow(value, precision=1e-6, new_value=1e-6):
    if abs(value) < precision:
        if value > 0:
            return new_value
        else:
            return -new_value
    else:
        return value
