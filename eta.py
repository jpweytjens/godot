from pathlib import Path
import bisect
from dataclasses import dataclass

import gpxpy
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# GPX parsing
# ---------------------------------------------------------------------------


def read_gpx(path: Path) -> pd.DataFrame:
    """Parse a GPX file into a raw DataFrame.

    Parameters
    ----------
    path : Path
        Path to the GPX file.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp_ms, lat, lon, elevation_m, speed_ms.
        Additional extension fields (hr, cad, watts, atemp) included if present.
        No distance or derived speed columns — apply pipe functions for those.
    """
    with open(path) as f:
        gpx = gpxpy.parse(f)
    rows = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                ext = {}
                if pt.extensions:
                    for child in pt.extensions:
                        for field in child:
                            tag = field.tag.split("}")[-1]  # strip namespace
                            try:
                                ext[tag] = float(field.text)
                            except (TypeError, ValueError):
                                pass
                rows.append(
                    {
                        "timestamp_ms": int(pt.time.timestamp() * 1000),
                        "lat": pt.latitude,
                        "lon": pt.longitude,
                        "elevation_m": pt.elevation or 0.0,
                        "speed_ms": ext.get("speed", np.nan),
                        **{
                            k: ext[k]
                            for k in ("hr", "cad", "watts", "atemp")
                            if k in ext
                        },
                    }
                )
    return pd.DataFrame(rows).sort_values("timestamp_ms").reset_index(drop=True)


def add_haversine_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative distance using the Haversine formula.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with lat and lon columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with distance_m column added. First row is 0.0.
    """
    R = 6_371_000.0
    lat = np.radians(df["lat"].values)
    lon = np.radians(df["lon"].values)
    dphi = np.diff(lat, prepend=lat[0])
    dlam = np.diff(lon, prepend=lon[0])
    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(np.roll(lat, 1)) * np.cos(lat) * np.sin(dlam / 2) ** 2
    )
    a[0] = 0.0
    step = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return df.assign(distance_m=np.cumsum(step))


def add_integrated_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative distance by integrating recorded GPS speed over time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamp_ms and speed_ms columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with distance_m column added. First row is 0.0.
    """
    dt_s = df["timestamp_ms"].diff().fillna(0) / 1000.0
    step = (df["speed_ms"].fillna(0) * dt_s).clip(lower=0)
    return df.assign(distance_m=step.cumsum())


def add_smooth_speed(
    df: pd.DataFrame,
    window: str = "5s",
    clip_lower: float = 0.0,
) -> pd.DataFrame:
    """Add smoothed speed in km/h via a rolling mean over speed_ms.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamp_ms and speed_ms columns.
    window : str, optional
        Rolling window size as a pandas time offset string, by default '5s'.
    clip_lower : float, optional
        Minimum value to clip the smoothed speed to, by default 0.0.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with speed_kmh column added.
    """
    idx = pd.to_datetime(df["timestamp_ms"], unit="ms")
    speed_kmh = (
        pd.Series(df["speed_ms"].values * 3.6, index=idx)
        .rolling(window)
        .mean()
        .fillna(0.0)
        .clip(lower=clip_lower)
        .values
    )
    return df.assign(speed_kmh=speed_kmh)


# ---------------------------------------------------------------------------
# Route segmentation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Naive estimators
# ---------------------------------------------------------------------------

ROLLING_WINDOW_S = 300.0


class AvgSpeedEstimator:
    """Estimates remaining time using cumulative average moving speed."""

    def reset(self) -> None:
        self._total_distance_m = 0.0
        self._total_time_s = 0.0
        self._prev_distance_m: float | None = None
        self._prev_timestamp_ms: int | None = None

    def update(
        self, timestamp_ms: int, distance_m: float, speed_kmh: float, elevation_m: float
    ) -> None:
        if self._prev_timestamp_ms is not None and speed_kmh >= 1.0:
            dt_s = (timestamp_ms - self._prev_timestamp_ms) / 1000.0
            dd_m = distance_m - self._prev_distance_m
            if dt_s > 0 and dd_m >= 0:
                self._total_distance_m += dd_m
                self._total_time_s += dt_s
        self._prev_distance_m = distance_m
        self._prev_timestamp_ms = timestamp_ms

    def predict(
        self, current_distance_m: float, total_distance_m: float, now_ms: int
    ) -> float:
        if self._total_time_s == 0 or self._total_distance_m == 0:
            return np.nan
        avg_speed_ms = self._total_distance_m / self._total_time_s
        remaining_m = max(total_distance_m - current_distance_m, 0.0)
        return remaining_m / avg_speed_ms


class RollingAvgSpeedEstimator:
    """Estimates remaining time using rolling average speed over a time window."""

    def __init__(self, window_s: float = ROLLING_WINDOW_S):
        self._window_s = window_s
        self._observations: list[tuple[int, float]] = []

    def reset(self) -> None:
        self._observations = []

    def update(
        self, timestamp_ms: int, distance_m: float, speed_kmh: float, elevation_m: float
    ) -> None:
        if speed_kmh >= 1.0:
            self._observations.append((timestamp_ms, speed_kmh))
        cutoff_ms = timestamp_ms - int(self._window_s * 1000)
        self._observations = [(t, v) for t, v in self._observations if t >= cutoff_ms]

    def predict(
        self, current_distance_m: float, total_distance_m: float, now_ms: int
    ) -> float:
        if not self._observations:
            return np.nan
        rolling_avg_ms = np.mean([v for _, v in self._observations]) / 3.6
        remaining_m = max(total_distance_m - current_distance_m, 0.0)
        return remaining_m / rolling_avg_ms
