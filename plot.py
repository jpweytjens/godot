"""Plotting utilities for ETA backtest results.

Color palettes from Paul Tol's color schemes:
https://personal.sron.nl/~pault/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# ---------------------------------------------------------------------------
# Paul Tol color palettes
# ---------------------------------------------------------------------------

# Qualitative — up to 7 colors, good contrast and color-blind safe
TOL_BRIGHT = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
]

# Alternative qualitative — warmer tones, good for 6 categories
TOL_VIBRANT = [
    "#EE7733",
    "#0077BB",
    "#33BBEE",
    "#EE3377",
    "#CC3311",
    "#009988",
    "#BBBBBB",
]

# Larger qualitative set — up to 10 colors
TOL_MUTED = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#DDDDDD",
]

# Default palette for line plots
COLORS = TOL_BRIGHT


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clip(df: pd.DataFrame, warmup_pct: float | None) -> pd.DataFrame:
    """Filter out the first warmup_pct fraction of a backtest result or ride DataFrame."""
    if warmup_pct is None:
        return df
    cutoff = df["distance_m"].iloc[-1] * warmup_pct
    return df[df["distance_m"] >= cutoff]


def _draw_pause_bands(
    ax: Axes,
    ride_clipped: pd.DataFrame,
    t0: pd.Timestamp,
    pause_kmh: float,
    min_pause_s: float,
) -> None:
    """Shade paused sections as full-height TOL-blue bands in time space."""
    is_slow = ride_clipped["speed_kmh"] < pause_kmh
    run_id = (is_slow != is_slow.shift()).cumsum()
    for _, group in ride_clipped[is_slow].groupby(run_id[is_slow]):
        duration_s = (group["time"].iloc[-1] - group["time"].iloc[0]).total_seconds()
        if duration_s >= min_pause_s:
            ax.axvspan(
                (group["time"].iloc[0] - t0).total_seconds() / 60,
                (group["time"].iloc[-1] - t0).total_seconds() / 60,
                alpha=0.2,
                color=TOL_BRIGHT[0],
                linewidth=0,
            )


def _align_twin_at(
    ax_primary: Axes,
    ax_twin: Axes,
    twin_ref: float,
    primary_ref: float,
    twin_hi: float,
) -> None:
    """Set twin axis ylim so twin_ref maps to the same chart position as primary_ref.

    Parameters
    ----------
    ax_primary : Axes
        Primary axes whose ylim determines the reference position.
    ax_twin : Axes
        Twin axes whose ylim will be adjusted.
    twin_ref : float
        The twin-axis value that should land on primary_ref's chart position.
    primary_ref : float
        The primary-axis value whose chart position we are targeting.
    twin_hi : float
        The desired upper limit of the twin axis.
    """
    lo, hi = ax_primary.get_ylim()
    if hi == lo:
        return
    frac = np.clip((primary_ref - lo) / (hi - lo), 0.05, 0.95)
    twin_lo = (twin_ref - frac * twin_hi) / (1.0 - frac)
    ax_twin.set_ylim(twin_lo, twin_hi)


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------


def add_elevation_profile(
    df: pd.DataFrame,
    ax: Axes,
    alpha: float = 0.15,
    color: str = "#BBBBBB",
    height_fraction: float = 0.35,
) -> Axes:
    """Overlay the elevation profile as a semi-transparent fill on an existing axes.

    Uses a twin y-axis so the elevation scale does not interfere with the
    primary axis. The fill is pushed to the lower portion of the chart via
    the twin axis y-limits.

    Parameters
    ----------
    df : pd.DataFrame
        Ride DataFrame with distance_m and elevation_m columns.
    ax : Axes
        Primary axes to add the overlay to.
    alpha : float, optional
        Opacity of the fill, by default 0.15.
    color : str, optional
        Fill color, by default Tol light grey.
    height_fraction : float, optional
        Fraction of the chart height the elevation profile occupies, by default 0.35.

    Returns
    -------
    Axes
        The twin axes (can be ignored if no further customisation needed).
    """
    if len(df) < 3:
        return ax.twinx()

    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    dist_km = df["distance_m"] / 1000
    elev = df["elevation_m"]

    ax2.fill_between(dist_km, elev, elev.min(), alpha=alpha, color=color, linewidth=0)
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # Anchor the floor of the fill (elev.min()) to primary y=0
    _align_twin_at(ax, ax2, elev.min(), 0.0, elev.max() / height_fraction)

    return ax2


def add_gradient_profile(
    df: pd.DataFrame,
    ax: Axes,
    smooth_m: float = 200.0,
    alpha: float = 0.2,
    clamp_pct: float = 15.0,
) -> Axes:
    """Overlay uphill/downhill gradient bands as background fill on an existing axes.

    Gradient is computed from the elevation and distance columns, then
    smoothed with a rolling window to reduce GPS noise. Uphill sections
    are filled with Paul Tol red, downhill with Paul Tol blue, both anchored
    so the zero-gradient line aligns with y=0 on the primary axis.

    Parameters
    ----------
    df : pd.DataFrame
        Ride DataFrame with distance_m and elevation_m columns.
    ax : Axes
        Primary axes to add the overlay to.
    smooth_m : float, optional
        Rolling smoothing window in meters, by default 200.0.
    alpha : float, optional
        Opacity of the fills, by default 0.5.
    clamp_pct : float, optional
        Maximum gradient percentage shown (±), by default 15.0.

    Returns
    -------
    Axes
        The twin axes.
    """
    if len(df) < 3:
        return ax.twinx()

    dist_km = df["distance_m"] / 1000

    # Gradient in percent, smoothed over ~smooth_m meters
    dd = df["distance_m"].diff().replace(0, np.nan)
    raw_gradient_pct = (df["elevation_m"].diff() / dd * 100).fillna(0)

    spacing_m = df["distance_m"].diff().median()
    window = max(1, int(smooth_m / spacing_m))
    gradient_pct = raw_gradient_pct.rolling(window, center=True, min_periods=1).mean()

    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax2.fill_between(
        dist_km,
        gradient_pct,
        where=gradient_pct > 0,
        alpha=alpha,
        color=TOL_VIBRANT[3],  # Paul Tol red for uphill
        linewidth=0,
        interpolate=True,
    )
    ax2.fill_between(
        dist_km,
        gradient_pct,
        where=gradient_pct < 0,
        alpha=alpha,
        color=TOL_BRIGHT[0],  # Paul Tol blue for downhill
        linewidth=0,
        interpolate=True,
    )
    ax2.axhline(0, color="#CCCCCC", linewidth=0.4, zorder=0)
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # Anchor gradient y=0 to primary axis y=0
    _align_twin_at(ax, ax2, 0.0, 0.0, clamp_pct)

    return ax2


# ---------------------------------------------------------------------------
# Backtest result plots
# ---------------------------------------------------------------------------


def plot_backtest(
    result: pd.DataFrame,
    title: str,
    ax: Axes | None = None,
    ride_df: pd.DataFrame | None = None,
    warmup_pct: float | None = None,
) -> None:
    """Plot predicted vs actual remaining time over distance.

    Parameters
    ----------
    result : pd.DataFrame
        Output of backtest().
    title : str
        Plot title.
    ax : Axes, optional
        Axes to draw on. Creates a new figure if None.
    ride_df : pd.DataFrame, optional
        Original ride DataFrame. If provided, adds elevation overlay.
    warmup_pct : float, optional
        If set, hides data before this distance (km) to skip cold-start noise.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    r = _clip(result, warmup_pct)
    dist_km = r["distance_m"] / 1000
    ax.plot(
        dist_km, r["ata_remaining_s"] / 60, label="Actual", color="black", linewidth=1.5
    )
    ax.plot(
        dist_km,
        r["eta_remaining_s"] / 60,
        label="Predicted",
        color=COLORS[0],
        alpha=0.85,
    )
    if ride_df is not None:
        add_elevation_profile(_clip(ride_df, warmup_pct), ax)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Remaining time (min)")
    ax.set_title(title)
    ax.legend()


def plot_delta(
    result: pd.DataFrame,
    title: str,
    ax: Axes | None = None,
    ride_df: pd.DataFrame | None = None,
    warmup_pct: float | None = None,
    overlay: str | None = "gradient",
    x_axis: str = "distance",
) -> None:
    """Plot ETA error (delta = ETA - ATA) over distance or elapsed time.

    Parameters
    ----------
    result : pd.DataFrame
        Output of backtest().
    title : str
        Plot title.
    ax : Axes, optional
        Axes to draw on. Creates a new figure if None.
    ride_df : pd.DataFrame, optional
        Original ride DataFrame. Required for any overlay.
    warmup_pct : float, optional
        If set, hides data before this fraction of total distance.
    overlay : str or None, optional
        Background overlay: ``"gradient"`` (default), ``"elevation"``, or
        ``None``. Ignored when ``x_axis="time"`` (overlays are position-based).
    x_axis : str, optional
        ``"distance"`` (default) or ``"time"`` (elapsed minutes).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    r = _clip(result, warmup_pct)
    if x_axis == "time":
        t0 = r["time"].iloc[0]
        x = (r["time"] - t0).dt.total_seconds() / 60
        xlabel = "Elapsed time (min)"
        effective_overlay = None  # overlays are position-based
    else:
        x = r["distance_m"] / 1000
        xlabel = "Distance (km)"
        effective_overlay = overlay

    if ride_df is not None and x_axis == "time":
        ride_clipped = _clip(ride_df, warmup_pct)
        _draw_pause_bands(ax, ride_clipped, r["time"].iloc[0], 1.0, 60.0)

    ax.plot(x, r["delta_s"] / 60, color=TOL_VIBRANT[5], linewidth=1.2)
    ax.axhline(5, linestyle="--", color="#BBBBBB", linewidth=1, label="+5 min")
    ax.axhline(-5, linestyle="--", color="#BBBBBB", linewidth=1, label="\u22125 min")
    ax.axhline(0, linestyle="-", color="black", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ETA \u2212 ATA (min)")
    ax.set_title(title)
    ax.legend()

    if ride_df is not None and effective_overlay is not None:
        ride_clipped = _clip(ride_df, warmup_pct)
        if effective_overlay == "gradient":
            add_gradient_profile(ride_clipped, ax)
        elif effective_overlay == "elevation":
            add_elevation_profile(ride_clipped, ax)


def plot_speed(
    result: pd.DataFrame,
    title: str,
    ax: Axes | None = None,
    ride_df: pd.DataFrame | None = None,
    warmup_pct: float | None = None,
    pause_kmh: float = 1.0,
    min_pause_s: float = 60.0,
    x_axis: str = "time",
) -> None:
    """Plot actual speed and estimator average speed over elapsed time or distance.

    Pause bands (time-space only) highlight sections where speed stays below
    ``pause_kmh`` for at least ``min_pause_s`` seconds.

    Parameters
    ----------
    result : pd.DataFrame
        Output of backtest(), must include time and speed_ms columns.
    title : str
        Plot title.
    ax : Axes, optional
        Axes to draw on. Creates a new figure if None.
    ride_df : pd.DataFrame, optional
        Original ride DataFrame with speed_kmh and time columns.
    warmup_pct : float, optional
        If set, hides data before this fraction of total distance.
    pause_kmh : float, optional
        Speed threshold below which a point counts as stopped. Default 1.0.
    min_pause_s : float, optional
        Minimum pause duration (s) to draw a band. Default 60.
    x_axis : str, optional
        ``"time"`` (default, elapsed minutes) or ``"distance"`` (km).
        Pause bands are only drawn in time space.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    r = _clip(result, warmup_pct)

    if x_axis == "time":
        t0 = r["time"].iloc[0]
        x_est = (r["time"] - t0).dt.total_seconds() / 60
        xlabel = "Elapsed time (min)"
    else:
        x_est = r["distance_m"] / 1000
        xlabel = "Distance (km)"

    if ride_df is not None:
        ride_clipped = _clip(ride_df, warmup_pct)

        if x_axis == "time":
            t0_ride = ride_clipped["time"].iloc[0]
            x_ride = (ride_clipped["time"] - t0_ride).dt.total_seconds() / 60
            _draw_pause_bands(ax, ride_clipped, t0_ride, pause_kmh, min_pause_s)
        else:
            x_ride = ride_clipped["distance_m"] / 1000

        ax.plot(
            x_ride,
            ride_clipped["speed_kmh"],
            color="black",
            linewidth=0.6,
            alpha=0.4,
            label="Actual",
        )

    ax.plot(
        x_est,
        r["speed_ms"] * 3.6,
        color=TOL_VIBRANT[5],
        linewidth=1.2,
        label="Avg speed",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(title)
    ax.legend()


def plot_comparison(
    results: dict[str, pd.DataFrame],
    title: str,
    ax: Axes | None = None,
    ride_df: pd.DataFrame | None = None,
    warmup_pct: float | None = None,
) -> None:
    """Plot remaining time for multiple estimators on the same axes.

    The actual remaining time (ATA) is shown as a black reference line.
    Each estimator gets a distinct Paul Tol color.

    Parameters
    ----------
    results : dict of str -> pd.DataFrame
        Mapping of estimator name to backtest() output DataFrame.
    title : str
        Plot title.
    ax : Axes, optional
        Axes to draw on. Creates a new figure if None.
    ride_df : pd.DataFrame, optional
        Original ride DataFrame. If provided, adds elevation overlay.
    warmup_pct : float, optional
        If set, hides data before this distance (km) to skip cold-start noise.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))

    if ride_df is not None:
        add_elevation_profile(_clip(ride_df, warmup_pct), ax)

    # ATA reference — use any result for the actual line
    first = _clip(next(iter(results.values())), warmup_pct)
    dist_km = first["distance_m"] / 1000
    ax.plot(
        dist_km,
        first["ata_remaining_s"] / 60,
        label="Actual",
        color="black",
        linewidth=2.0,
        zorder=10,
    )

    # Use TOL_MUTED for many estimators, fall back to cycling if needed
    palette = TOL_MUTED if len(results) > len(COLORS) else COLORS
    for i, (name, result) in enumerate(results.items()):
        r = _clip(result, warmup_pct)
        ax.plot(
            r["distance_m"] / 1000,
            r["eta_remaining_s"] / 60,
            label=name,
            color=palette[i % len(palette)],
            alpha=0.85,
            linewidth=1.2,
        )

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Remaining time (min)")
    ax.set_title(title)
    ax.legend(loc="upper right")
