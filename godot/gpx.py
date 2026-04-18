"""GPX file parsing and distance/speed pipe functions."""

from functools import lru_cache
from pathlib import Path

import gpxpy
import numpy as np
import pandas as pd
from godot.convert import ms_to_kmh


@lru_cache(maxsize=None)
def read_gpx(path: Path) -> pd.DataFrame:
    """Parse a GPX file into a raw DataFrame.

    Parameters
    ----------
    path : Path
        Path to the GPX file.

    Returns
    -------
    pd.DataFrame
        Columns: time, lat, lon, elevation_m, speed_ms.
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
                        "time": pd.Timestamp(pt.time.timestamp(), unit="s"),
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
    if not rows:
        raise ValueError("GPX file contains no trackpoints")
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


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
        DataFrame with time and speed_ms columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with distance_m column added. First row is 0.0.
    """
    dt_s = df["time"].diff().dt.total_seconds().fillna(0)
    step = (df["speed_ms"].fillna(0) * dt_s).clip(lower=0)
    return df.assign(distance_m=step.cumsum())


def fill_pauses(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to 1-second frequency, filling gaps left by auto-pause.

    Inserted rows get ``speed_ms = 0`` and a ``paused = True`` flag.
    Rows where ``distance_m`` hasn't changed are also flagged as paused,
    catching recorded-but-stationary points (e.g. waiting at a light).
    Lat, lon, elevation, and distance are forward-filled from the last
    recorded point.

    Parameters
    ----------
    df : pd.DataFrame
        Ride DataFrame with a time column (1-second recording interval).

    Returns
    -------
    pd.DataFrame
        DataFrame at 1-second frequency with a ``paused`` boolean column.
    """
    out = df.set_index("time").asfreq("1s")
    resampled = out["lat"].isna()
    out[["lat", "lon", "elevation_m"]] = out[["lat", "lon", "elevation_m"]].ffill()
    if "distance_m" in out.columns:
        out["distance_m"] = out["distance_m"].ffill()
    out["speed_ms"] = out["speed_ms"].fillna(0.0)
    # Recorded but not moving — treat as paused to avoid zero-speed estimates
    stationary = out["distance_m"].diff().fillna(0) == 0
    out["paused"] = resampled | stationary
    return out.reset_index()


def add_smooth_speed(
    df: pd.DataFrame,
    window: str = "5s",
    clip_lower: float = 0.0,
) -> pd.DataFrame:
    """Add smoothed speed in km/h via a rolling mean over speed_ms.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time and speed_ms columns.
    window : str, optional
        Rolling window size as a pandas time offset string, by default '5s'.
    clip_lower : float, optional
        Minimum value to clip the smoothed speed to, by default 0.0.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with speed_kmh column added.
    """
    speed_kmh = (
        pd.Series(ms_to_kmh(df["speed_ms"].values), index=df["time"])
        .rolling(window)
        .mean()
        .fillna(0.0)
        .clip(lower=clip_lower)
        .values
    )
    return df.assign(speed_kmh=speed_kmh)


def pause_run_id(paused: pd.Series) -> pd.Series:
    """Assign a unique integer ID to each contiguous pause/riding run.

    Consecutive rows with the same `paused` value share an ID.
    Each transition (paused to riding or vice versa) increments the ID.

    Parameters
    ----------
    paused : pd.Series
        Boolean series indicating paused state at each row.

    Returns
    -------
    pd.Series
        Integer series with a unique ID per contiguous block.
    """
    return (paused != paused.shift()).cumsum()
