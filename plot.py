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


def _clip(df: pd.DataFrame, warmup_km: float | None) -> pd.DataFrame:
    """Filter out rows before warmup_km from a backtest result or ride DataFrame."""
    if warmup_km is None:
        return df
    return df[df["distance_m"] >= warmup_km * 1000]


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
    ax2 = ax.twinx()
    dist_km = df["distance_m"] / 1000
    elev = df["elevation_m"]

    ax2.fill_between(dist_km, elev, alpha=alpha, color=color, linewidth=0)
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # Push profile to lower portion: set top of twin axis well above max elevation
    ax2.set_ylim(elev.min(), elev.max() / height_fraction)

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
    are filled with an orange-red, downhill with blue.

    Parameters
    ----------
    df : pd.DataFrame
        Ride DataFrame with distance_m and elevation_m columns.
    ax : Axes
        Primary axes to add the overlay to.
    smooth_m : float, optional
        Rolling smoothing window in meters, by default 200.0.
    alpha : float, optional
        Opacity of the fills, by default 0.2.
    clamp_pct : float, optional
        Maximum gradient percentage shown (±), by default 15.0.

    Returns
    -------
    Axes
        The twin axes.
    """
    dist_km = df["distance_m"] / 1000

    # Gradient in percent, smoothed over ~smooth_m meters
    dd = df["distance_m"].diff().replace(0, np.nan)
    raw_gradient_pct = (df["elevation_m"].diff() / dd * 100).fillna(0)

    spacing_m = df["distance_m"].diff().median()
    window = max(1, int(smooth_m / spacing_m))
    gradient_pct = raw_gradient_pct.rolling(window, center=True, min_periods=1).mean()

    ax2 = ax.twinx()
    ax2.fill_between(
        dist_km,
        gradient_pct,
        where=gradient_pct > 0,
        alpha=alpha,
        color=TOL_VIBRANT[0],  # orange-red for uphill
        linewidth=0,
        interpolate=True,
    )
    ax2.fill_between(
        dist_km,
        gradient_pct,
        where=gradient_pct < 0,
        alpha=alpha,
        color=TOL_BRIGHT[0],  # blue for downhill
        linewidth=0,
        interpolate=True,
    )
    ax2.axhline(0, color="#CCCCCC", linewidth=0.4, zorder=0)
    ax2.set_ylim(-clamp_pct, clamp_pct)
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    return ax2


# ---------------------------------------------------------------------------
# Backtest result plots
# ---------------------------------------------------------------------------


def plot_backtest(
    result: pd.DataFrame,
    title: str,
    ax: Axes | None = None,
    ride_df: pd.DataFrame | None = None,
    warmup_km: float | None = None,
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
    warmup_km : float, optional
        If set, hides data before this distance (km) to skip cold-start noise.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    r = _clip(result, warmup_km)
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
        add_elevation_profile(_clip(ride_df, warmup_km), ax)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Remaining time (min)")
    ax.set_title(title)
    ax.legend()


def plot_delta(
    result: pd.DataFrame,
    title: str,
    ax: Axes | None = None,
    ride_df: pd.DataFrame | None = None,
    warmup_km: float | None = None,
) -> None:
    """Plot ETA error (delta = ETA - ATA) over distance.

    Parameters
    ----------
    result : pd.DataFrame
        Output of backtest().
    title : str
        Plot title.
    ax : Axes, optional
        Axes to draw on. Creates a new figure if None.
    ride_df : pd.DataFrame, optional
        Original ride DataFrame. If provided, adds gradient overlay.
    warmup_km : float, optional
        If set, hides data before this distance (km) to skip cold-start noise.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    if ride_df is not None:
        add_gradient_profile(_clip(ride_df, warmup_km), ax)
    r = _clip(result, warmup_km)
    dist_km = r["distance_m"] / 1000
    ax.plot(dist_km, r["delta_s"] / 60, color=COLORS[1], linewidth=1.2)
    ax.axhline(5, linestyle="--", color="#BBBBBB", linewidth=1, label="+5 min")
    ax.axhline(-5, linestyle="--", color="#BBBBBB", linewidth=1, label="\u22125 min")
    ax.axhline(0, linestyle="-", color="black", linewidth=0.5)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("ETA \u2212 ATA (min)")
    ax.set_title(title)
    ax.legend()


def plot_comparison(
    results: dict[str, pd.DataFrame],
    title: str,
    ax: Axes | None = None,
    ride_df: pd.DataFrame | None = None,
    warmup_km: float | None = None,
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
    warmup_km : float, optional
        If set, hides data before this distance (km) to skip cold-start noise.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))

    if ride_df is not None:
        add_elevation_profile(_clip(ride_df, warmup_km), ax)

    # ATA reference — use any result for the actual line
    first = _clip(next(iter(results.values())), warmup_km)
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
        r = _clip(result, warmup_km)
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
