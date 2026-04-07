"""Route segmentation: decimation, RDP/VW simplification, gradient segments."""

import bisect
import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RouteSegment:
    """A simplified route segment with constant gradient.

    Attributes
    ----------
    start_distance_m : float
        Start of segment in meters from ride start.
    end_distance_m : float
        End of segment in meters from ride start.
    gradient : float
        Elevation change over distance as a fraction (not percent).
        Positive = uphill, negative = downhill.
    """

    start_distance_m: float
    end_distance_m: float
    gradient: float


def decimate(df: pd.DataFrame, spacing_m: float = 20.0) -> list[tuple[float, float]]:
    """Reduce an elevation profile to one point per spacing_m meters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with distance_m and elevation_m columns.
    spacing_m : float, optional
        Minimum distance between retained points, by default 20.0.

    Returns
    -------
    list of (float, float)
        List of (distance_m, elevation_m) tuples. Always includes last point.
    """
    points = list(zip(df["distance_m"], df["elevation_m"]))
    result = [points[0]]
    for d, e in points[1:]:
        if d - result[-1][0] >= spacing_m:
            result.append((d, e))
    if result[-1] != points[-1]:
        result.append(points[-1])
    return result


def perpendicular_distance(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    """Perpendicular distance from point p to line segment a-b.

    Parameters
    ----------
    p : tuple of float
        Point (x, y) to measure from.
    a : tuple of float
        Start of line segment (x, y).
    b : tuple of float
        End of line segment (x, y).

    Returns
    -------
    float
        Perpendicular distance in the same units as the input coordinates.
    """
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return np.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def ramer_douglas_peucker(
    points: list[tuple[float, float]],
    epsilon: float,
) -> list[tuple[float, float]]:
    """Simplify a polyline using the Ramer-Douglas-Peucker algorithm.

    Parameters
    ----------
    points : list of (float, float)
        Input polyline as (x, y) pairs. Both axes in meters.
    epsilon : float
        Maximum perpendicular deviation to retain a point (meters).

    Returns
    -------
    list of (float, float)
        Simplified polyline.
    """
    if len(points) < 3:
        return points
    dmax = 0.0
    idx = 0
    for i in range(1, len(points) - 1):
        d = perpendicular_distance(points[i], points[0], points[-1])
        if d > dmax:
            dmax, idx = d, i
    if dmax >= epsilon:
        left = ramer_douglas_peucker(points[: idx + 1], epsilon)
        right = ramer_douglas_peucker(points[idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def _triangle_area(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """Unsigned area of triangle (a, b, c) via the shoelace formula."""
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def visvalingam_whyatt(
    points: list[tuple[float, float]],
    min_area: float,
) -> list[tuple[float, float]]:
    """Simplify a polyline using the Visvalingam–Whyatt algorithm.

    Iteratively removes the interior point whose triangle (with its
    two neighbors) has the smallest effective area, until all remaining
    triangles exceed `min_area`.

    Parameters
    ----------
    points : list of (float, float)
        Input polyline as (x, y) pairs.
    min_area : float
        Minimum effective triangle area to retain a point.

    Returns
    -------
    list of (float, float)
        Simplified polyline. First and last points are always kept.
    """
    n = len(points)
    if n < 3:
        return list(points)

    # Doubly-linked list via prev/next arrays
    prev_idx = list(range(-1, n - 1))
    next_idx = list(range(1, n + 1))
    next_idx[-1] = -1  # sentinel

    removed = [False] * n
    counter = 0  # tie-breaker for heap stability
    heap: list[tuple[float, float, int]] = []

    for i in range(1, n - 1):
        area = _triangle_area(points[prev_idx[i]], points[i], points[next_idx[i]])
        heapq.heappush(heap, (area, counter, i))
        counter += 1

    while heap:
        area, _, idx = heapq.heappop(heap)
        if removed[idx]:
            continue
        if area >= min_area:
            break

        removed[idx] = True
        p, nx = prev_idx[idx], next_idx[idx]
        next_idx[p] = nx
        prev_idx[nx] = p

        # Recompute neighbors — effective area never shrinks (VW property)
        for neighbor in (p, nx):
            if prev_idx[neighbor] == -1 or next_idx[neighbor] == -1:
                continue
            new_area = _triangle_area(
                points[prev_idx[neighbor]], points[neighbor], points[next_idx[neighbor]]
            )
            new_area = max(new_area, area)
            heapq.heappush(heap, (new_area, counter, neighbor))
            counter += 1

    return [p for p, r in zip(points, removed) if not r]


def merge_short_segments(
    points: list[tuple[float, float]],
    min_length_m: float,
) -> list[tuple[float, float]]:
    """Merge consecutive segments shorter than min_length_m into neighbors.

    Parameters
    ----------
    points : list of (float, float)
        Simplified polyline as (distance_m, elevation_m) pairs.
    min_length_m : float
        Minimum segment length in meters.

    Returns
    -------
    list of (float, float)
        Points with short intermediate segments removed. First and last
        points are always retained.
    """
    result = [points[0]]
    for p in points[1:-1]:
        if p[0] - result[-1][0] >= min_length_m:
            result.append(p)
    result.append(points[-1])
    return result


def build_segments(points: list[tuple[float, float]]) -> list[RouteSegment]:
    """Build a list of RouteSegments from a simplified point list.

    Parameters
    ----------
    points : list of (float, float)
        Polyline as (distance_m, elevation_m) pairs.

    Returns
    -------
    list of RouteSegment
        One segment per consecutive pair of points.
    """
    segments = []
    for i in range(len(points) - 1):
        d0, e0 = points[i]
        d1, e1 = points[i + 1]
        delta_d = d1 - d0
        gradient = (e1 - e0) / delta_d if delta_d > 0 else 0.0
        segments.append(RouteSegment(d0, d1, gradient))
    return segments


def decimate_to_gradient_segments(
    df: pd.DataFrame,
    min_area: float = 2.0,
    min_length_m: float = 200.0,
) -> tuple[list[tuple[float, float]], list[RouteSegment]]:
    """Decimate an elevation profile into gradient segments via VW simplification.

    Parameters
    ----------
    df : pd.DataFrame
        Ride DataFrame with `distance_m` and `elevation_m` columns.
    min_area : float
        Visvalingam-Whyatt minimum triangle area.
    min_length_m : float
        Minimum segment length in meters after merging.

    Returns
    -------
    points : list of (float, float)
        Simplified (distance_m, elevation_m) polyline.
    segments : list of RouteSegment
        One segment per consecutive pair of points.
    """
    points = list(zip(df["distance_m"], df["elevation_m"]))
    points = visvalingam_whyatt(points, min_area)
    points = merge_short_segments(points, min_length_m)
    return points, build_segments(points)


def gradient_at_distance(distance_m: float, segments: list[RouteSegment]) -> float:
    """Return the gradient at a given distance along the route.

    Parameters
    ----------
    distance_m : float
        Distance from ride start in meters.
    segments : list of RouteSegment
        Sorted list of route segments.

    Returns
    -------
    float
        Gradient as a fraction at the given distance.
    """
    ends = [s.end_distance_m for s in segments]
    idx = bisect.bisect_left(ends, distance_m)
    idx = min(idx, len(segments) - 1)
    return segments[idx].gradient
