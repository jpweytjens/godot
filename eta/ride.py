"""Ride loading, preparation, and classification."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from eta.gpx import (
    add_haversine_distance,
    add_integrated_distance,
    add_smooth_speed,
    fill_pauses,
    read_gpx,
)
from eta.plot import pause_intervals, prep_time_axis

_DISTANCE_PIPES = {
    "haversine": add_haversine_distance,
    "integrated": add_integrated_distance,
}


@dataclass
class Ride:
    """A fully prepared ride ready for backtesting and plotting.
    Parameters
    ----------
    name : str
        GPX file stem, e.g. `"criquielion"`.
    label : str
        Human-readable name (underscores replaced with spaces).
    df : pd.DataFrame
        Prepared DataFrame with columns: time, lat, lon, elevation_m,
        speed_ms, distance_m, paused, speed_kmh, delta_distance,
        delta_time, elapsed_min.
    route_type : str
        One of `"uphill"`, `"downhill"`, `"flat"`, `"hilly"`,
        `"mountain"`.
    contains_pauses : bool
        Whether the ride contains pauses >= 60 s.
    pauses : pd.DataFrame
        Pause intervals with `start_min` and `end_min` columns.
    distance_method : str
        Distance pipeline used: `"haversine"` or `"integrated"`.
    speed_smoothed : bool
        Whether speed was smoothed with a rolling mean.
    distance : float
        Total route distance in metres.
    total_time : float
        Total elapsed time in seconds.
    ride_time : float
        Time spent moving in seconds.
    paused_time : float
        Time spent stopped in seconds.
    """

    name: str
    label: str
    df: pd.DataFrame
    route_type: str
    contains_pauses: bool
    pauses: pd.DataFrame
    distance_method: str
    speed_smoothed: bool
    distance: float
    total_time: float
    ride_time: float
    paused_time: float

    def __str__(self) -> str:
        dist_km = self.distance / 1000
        ride_min = self.ride_time / 60
        return (
            f"{self.label} — {dist_km:.1f} km, {ride_min:.0f} min ({self.route_type})"
        )

    def __repr__(self) -> str:
        return (
            f"Ride(name={self.name!r}, distance={self.distance:.0f}, "
            f"route_type={self.route_type!r}, "
            f"distance_method={self.distance_method!r}, "
            f"points={len(self.df)})"
        )


def classify_route(
    df: pd.DataFrame,
    dominance_ratio: float = 1.25,
    flat_m: float = 500.0,
    mountain_m: float = 2000.0,
) -> str:
    """Classify a ride by its elevation profile.

    Parameters
    ----------
    df : pd.DataFrame
        Ride DataFrame with an `elevation_m` column.
    dominance_ratio : float, optional
        If one direction exceeds the other by this factor, the route is
        classified as uphill or downhill. Default 1.25 (25% more).
    flat_m : float, optional
        Maximum cumulative ascent (m) for a balanced route to be called flat.
    mountain_m : float, optional
        Minimum cumulative ascent (m) for a balanced route to be called mountain.

    Returns
    -------
    str
        One of: `"uphill"`, `"downhill"`, `"flat"`, `"hilly"`,
        `"mountain"`.
    """
    diff = df["elevation_m"].diff().fillna(0)
    ascent_m = diff[diff > 0].sum()
    descent_m = -diff[diff < 0].sum()

    if ascent_m > descent_m * dominance_ratio:
        return "uphill"
    if descent_m > ascent_m * dominance_ratio:
        return "downhill"
    if ascent_m < flat_m:
        return "flat"
    if ascent_m < mountain_m:
        return "hilly"
    return "mountain"


def has_pauses(
    df: pd.DataFrame,
    min_pause_s: float = 60.0,
) -> bool:
    """Return True if the ride contains at least one pause long enough to matter.

    Requires a `paused` boolean column produced by `fill_pauses`.

    Parameters
    ----------
    df : pd.DataFrame
        Ride DataFrame with `paused` and `time` columns.
    min_pause_s : float, optional
        Minimum duration (seconds) for a paused run to count. Default 60.

    Returns
    -------
    bool
    """
    is_paused = df["paused"]
    run_id = (is_paused != is_paused.shift()).cumsum()
    pause_durations = (
        df[is_paused]
        .groupby(run_id[is_paused])["time"]
        .agg(lambda t: (t.iloc[-1] - t.iloc[0]).total_seconds())
    )
    return bool(len(pause_durations) > 0 and pause_durations.max() >= min_pause_s)


def load_ride(
    gpx_path: Path,
    distance_method: str = "haversine",
    smooth_speed: bool = True,
    smooth_window: str = "5s",
) -> Ride:
    """Load a GPX file and return a fully prepared `Ride`.

    Pipeline: read → distance → fill_pauses → smooth speed → add deltas
    → add elapsed_min → compute scalars → classify → build pause intervals.

    Parameters
    ----------
    gpx_path : Path
        Path to the GPX file.
    distance_method : str, optional
        `"haversine"` (default) or `"integrated"`.
    smooth_speed : bool, optional
        Whether to apply the rolling speed smoother. Default True.
    smooth_window : str, optional
        Rolling window size as a pandas time offset string. Default `"5s"`.

    Returns
    -------
    Ride
    """
    if distance_method not in _DISTANCE_PIPES:
        raise ValueError(f"distance_method must be one of {list(_DISTANCE_PIPES)}")

    df = read_gpx(gpx_path).pipe(_DISTANCE_PIPES[distance_method]).pipe(fill_pauses)

    if smooth_speed:
        df = add_smooth_speed(df, window=smooth_window)
    else:
        df = df.assign(speed_kmh=df["speed_ms"] * 3.6)

    # Precompute per-row deltas
    df = df.assign(
        delta_distance=df["distance_m"].diff().clip(lower=0).fillna(0),
        delta_time=df["time"].diff().dt.total_seconds().fillna(0),
    )

    # Add elapsed_min (no warmup clipping)
    df = prep_time_axis(df)

    # Scalar summaries
    distance = df["distance_m"].iloc[-1]
    total_time = df["delta_time"].sum()
    paused_mask = df["paused"]
    ride_time = df.loc[~paused_mask, "delta_time"].sum()
    paused_time = df.loc[paused_mask, "delta_time"].sum()

    name = gpx_path.stem
    return Ride(
        name=name,
        label=name.replace("_", " "),
        df=df,
        route_type=classify_route(df),
        contains_pauses=has_pauses(df),
        pauses=pause_intervals(df),
        distance_method=distance_method,
        speed_smoothed=smooth_speed,
        distance=distance,
        total_time=total_time,
        ride_time=ride_time,
        paused_time=paused_time,
    )
