"""GPX file parsing and distance/speed pipe functions."""

from pathlib import Path

import gpxpy
import numpy as np
import pandas as pd


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
