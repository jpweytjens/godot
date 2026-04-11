"""Composable Altair plotting layers for ETA backtest results."""

import altair as alt
import pandas as pd

from godot.gradient import gradient_bin
from godot.gpx import pause_run_id
from godot.palettes import GRADIENT_COLORS
from godot.segmentation import decimate_to_gradient_segments
from godot.theme import COLORS, TOL_BRIGHT, TOL_MUTED, TOL_VIBRANT

# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------


def prep_time_axis(df: pd.DataFrame, warmup_pct: float | None = None) -> pd.DataFrame:
    """Clip warmup and add elapsed_min column.

    Parameters
    ----------
    df : pd.DataFrame
        Ride or result DataFrame with time and distance_m columns.
    warmup_pct : float, optional
        Fraction of total distance to clip from the start.

    Returns
    -------
    pd.DataFrame
        Copy with elapsed_min column added. Warmup rows removed.
    """
    if warmup_pct is not None:
        cutoff = df["distance_m"].iloc[-1] * warmup_pct
        df = df[df["distance_m"] >= cutoff]
    t0 = df["time"].iloc[0]
    return df.assign(elapsed_min=(df["time"] - t0).dt.total_seconds() / 60)


def _pause_intervals(df: pd.DataFrame, min_pause_s: float = 60.0) -> pd.DataFrame:
    """Build pause interval bounds from a DataFrame with `paused` and `elapsed_min`."""
    if "paused" not in df.columns:
        return pd.DataFrame(columns=["start_min", "end_min"])
    is_paused = df["paused"]
    run_id = pause_run_id(is_paused)
    intervals = []
    for _, grp in df[is_paused].groupby(run_id[is_paused]):
        dur_s = (grp["time"].iloc[-1] - grp["time"].iloc[0]).total_seconds()
        if dur_s >= min_pause_s:
            intervals.append(
                {
                    "start_min": grp["elapsed_min"].iloc[0],
                    "end_min": grp["elapsed_min"].iloc[-1],
                }
            )
    return pd.DataFrame(intervals, columns=["start_min", "end_min"])


# ---------------------------------------------------------------------------
# Layer functions — each returns an alt.Chart
# ---------------------------------------------------------------------------

X_ELAPSED = alt.X("elapsed_min:Q").title("Elapsed time (min)")


def pause_bands(df: pd.DataFrame, min_pause_s: float = 10.0) -> alt.Chart:
    """Blue semi-transparent bands for paused sections.

    Parameters
    ----------
    df : pd.DataFrame
        Ride DataFrame with `paused`, `time`, and `elapsed_min` columns.
    min_pause_s : float, optional
        Minimum pause duration in seconds to show. Default 10.
    """
    intervals = _pause_intervals(df, min_pause_s)
    return (
        alt.Chart(intervals)
        .mark_rect(opacity=0.15, color=TOL_BRIGHT[0])
        .encode(x="start_min:Q", x2="end_min:Q")
    )


def speed_actual(ride_df: pd.DataFrame) -> alt.Chart:
    """Thin grey line of actual recorded speed."""
    return (
        alt.Chart(ride_df)
        .mark_line(strokeWidth=0.6, opacity=0.4, color="black")
        .encode(x=X_ELAPSED, y=alt.Y("speed_kmh:Q").title("Speed (km/h)"))
    )


def speed_estimated(
    result_df: pd.DataFrame,
    color: str = TOL_VIBRANT[5],
    opacity: float = 1.0,
    stroke_width: float = 1.5,
) -> alt.Chart:
    """Estimated average speed line from the estimator."""
    df = result_df.assign(speed_kmh=result_df["speed_ms"] * 3.6)
    return (
        alt.Chart(df)
        .mark_line(
            strokeWidth=stroke_width,
            color=color,
            opacity=opacity,
            invalid="break-paths-filter-domains",
        )
        .encode(x=X_ELAPSED, y=alt.Y("speed_kmh:Q").title("Speed (km/h)"))
    )


def eta_error(
    result_df: pd.DataFrame,
    color: str = TOL_VIBRANT[5],
    opacity: float = 1.0,
    stroke_width: float = 1.2,
) -> alt.Chart:
    """ETA error (delta) line in minutes."""
    df = result_df.assign(delta_min=result_df["delta_s"] / 60)
    return (
        alt.Chart(df)
        .mark_line(
            strokeWidth=stroke_width,
            color=color,
            opacity=opacity,
            invalid="break-paths-filter-domains",
        )
        .encode(x=X_ELAPSED, y=alt.Y("delta_min:Q").title("ETA \u2212 ATA (min)"))
    )


def eta_error_pct(
    result_df: pd.DataFrame,
    color: str = TOL_VIBRANT[5],
    opacity: float = 1.0,
    stroke_width: float = 1.2,
) -> alt.Chart:
    """ETA error as percentage of actual remaining time."""
    ata = result_df["ata_remaining_s"]
    pct = (result_df["delta_s"] / ata).where(ata > 0) * 100
    df = result_df.assign(delta_pct=pct)
    return (
        alt.Chart(df)
        .mark_line(
            strokeWidth=stroke_width,
            color=color,
            opacity=opacity,
            invalid="break-paths-filter-domains",
        )
        .encode(x=X_ELAPSED, y=alt.Y("delta_pct:Q").title("ETA error (%)"))
    )


def error_pct_refs() -> alt.Chart:
    """Horizontal reference lines at 0%, +10%, -10%."""
    zero = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="black", strokeWidth=0.5)
        .encode(y="y:Q")
    )
    bounds = (
        alt.Chart(pd.DataFrame({"y": [10, -10]}))
        .mark_rule(color="#BBBBBB", strokeWidth=1, strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    return zero + bounds


def error_refs() -> alt.Chart:
    """Horizontal reference lines at 0, +5, -5 minutes."""
    zero = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="black", strokeWidth=0.5)
        .encode(y="y:Q")
    )
    bounds = (
        alt.Chart(pd.DataFrame({"y": [5, -5]}))
        .mark_rule(color="#BBBBBB", strokeWidth=1, strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    return zero + bounds


def eta_countdown(result_df: pd.DataFrame) -> alt.Chart:
    """Estimated and actual time remaining, in minutes."""
    combined = pd.concat(
        [
            result_df[["elapsed_min"]].assign(
                remaining_min=result_df["ata_remaining_s"] / 60, series="Actual"
            ),
            result_df[["elapsed_min"]].assign(
                remaining_min=result_df["eta_remaining_s"] / 60, series="Estimated"
            ),
        ],
        ignore_index=True,
    )
    return (
        alt.Chart(combined)
        .mark_line(strokeWidth=1.2, invalid="break-paths-filter-domains")
        .encode(
            x=X_ELAPSED,
            y=alt.Y("remaining_min:Q").title("Time remaining (min)"),
            color=alt.Color("series:N")
            .scale(domain=["Actual", "Estimated"], range=["#888", TOL_VIBRANT[5]])
            .legend(alt.Legend(title=None, orient="top-right")),
        )
    )


LEGEND_BOTTOM = alt.Legend(
    title=None,
    orient="bottom",
    direction="horizontal",
    strokeColor="#ccc",
    padding=6,
    symbolStrokeWidth=3,
    symbolSize=200,
)


def avg_speed_overview(
    ride_df: pd.DataFrame,
    ref_total_df: pd.DataFrame | None = None,
    ref_moving_df: pd.DataFrame | None = None,
) -> alt.Chart:
    """Estimated expanding avg speeds converging toward actual avg (horizontal).

    Shows:
    - Actual speed (60s mean) — faint background, the quantity being averaged
    - Actual total avg — horizontal dashed line (final ride value)
    - Actual moving avg — horizontal dashed line (final ride value)
    - Est. total avg — solid expanding line converging toward actual total
    - Est. moving avg — solid expanding line converging toward actual moving

    Parameters
    ----------
    ride_df : pd.DataFrame
        Prepped ride DataFrame.
    ref_total_df : pd.DataFrame, optional
        Backtest result from total-time avg speed estimator.
    ref_moving_df : pd.DataFrame, optional
        Backtest result from moving-time avg speed estimator.
    """
    smoothed = ride_df["speed_kmh"].rolling(60, min_periods=1, center=True).mean()

    # Actual final averages — constant horizontal lines via full-length series
    dd = ride_df["delta_distance"]
    dt = ride_df["delta_time"]
    moving = ~ride_df["paused"]
    total_dist = dd.sum()
    actual_total_kmh = total_dist / dt.sum() * 3.6
    actual_moving_kmh = total_dist / (dt * moving).sum() * 3.6

    elapsed = ride_df[["elapsed_min"]]
    frames = [
        elapsed.assign(speed_kmh=smoothed, series="Actual (60s mean)"),
        elapsed.assign(speed_kmh=actual_total_kmh, series="Actual total avg"),
        elapsed.assign(speed_kmh=actual_moving_kmh, series="Actual moving avg"),
    ]
    series = ["Actual (60s mean)", "Actual total avg", "Actual moving avg"]
    colors = ["#888", TOL_BRIGHT[3], TOL_BRIGHT[4]]
    widths = [0.8, 1.2, 1.2]
    dashes = [[1, 0], [6, 3], [6, 3]]

    if ref_total_df is not None:
        frames.append(
            ref_total_df[["elapsed_min"]].assign(
                speed_kmh=ref_total_df["speed_ms"] * 3.6, series="Est. total avg"
            )
        )
        series.append("Est. total avg")
        colors.append(TOL_BRIGHT[3])
        widths.append(1.5)
        dashes.append([1, 0])

    if ref_moving_df is not None:
        frames.append(
            ref_moving_df[["elapsed_min"]].assign(
                speed_kmh=ref_moving_df["speed_ms"] * 3.6, series="Est. moving avg"
            )
        )
        series.append("Est. moving avg")
        colors.append(TOL_BRIGHT[4])
        widths.append(1.5)
        dashes.append([1, 0])

    combined = pd.concat(frames, ignore_index=True)
    return (
        alt.Chart(combined)
        .mark_line(strokeWidth=1, invalid="break-paths-filter-domains")
        .encode(
            x=X_ELAPSED,
            y=alt.Y("speed_kmh:Q")
            .title("Speed (km/h)")
            .scale(domain=[0, 40], clamp=True),
            color=alt.Color("series:N")
            .scale(domain=series, range=colors)
            .legend(LEGEND_BOTTOM),
            strokeWidth=alt.StrokeWidth("series:N")
            .scale(domain=series, range=widths)
            .legend(None),
            strokeDash=alt.StrokeDash("series:N")
            .scale(domain=series, range=dashes)
            .legend(None),
        )
    )


def actual_speed(
    ride_df: pd.DataFrame,
    result_df: pd.DataFrame | None = None,
) -> alt.Chart:
    """Actual speed with optional estimator effective speed overlay.

    Parameters
    ----------
    ride_df : pd.DataFrame
        Prepped ride DataFrame with `elapsed_min` and `speed_kmh`.
    result_df : pd.DataFrame, optional
        Estimator backtest result with `speed_ms` and `elapsed_min`.
    """
    smoothed = ride_df[["elapsed_min"]].assign(
        speed_kmh=ride_df["speed_kmh"].rolling(60, min_periods=1, center=True).mean(),
        series="Actual (60s mean)",
    )
    frames = [
        ride_df[["elapsed_min", "speed_kmh"]].assign(series="Actual"),
        smoothed,
    ]
    series = ["Actual", "Actual (60s mean)"]
    colors = ["#888", "#333"]
    opacities = [0.2, 0.8]
    widths = [0.5, 1.2]

    if result_df is not None:
        speed_col = (
            "current_speed_ms" if "current_speed_ms" in result_df else "speed_ms"
        )
        frames.append(
            result_df[["elapsed_min"]].assign(
                speed_kmh=result_df[speed_col] * 3.6, series="Estimated"
            )
        )
        series.append("Estimated")
        colors.append(TOL_VIBRANT[5])
        opacities.append(1.0)
        widths.append(1.5)

    combined = pd.concat(frames, ignore_index=True)
    return (
        alt.Chart(combined)
        .mark_line(strokeWidth=1, invalid="break-paths-filter-domains")
        .encode(
            x=X_ELAPSED,
            y=alt.Y("speed_kmh:Q")
            .title("Speed (km/h)")
            .scale(domain=[0, 50], clamp=True),
            color=alt.Color("series:N")
            .scale(domain=series, range=colors)
            .legend(LEGEND_BOTTOM),
            opacity=alt.Opacity("series:N")
            .scale(domain=series, range=opacities)
            .legend(None),
            strokeWidth=alt.StrokeWidth("series:N")
            .scale(domain=series, range=widths)
            .legend(None),
        )
    )


def speed_comparison(
    ride_df: pd.DataFrame,
    result_df: pd.DataFrame,
    ref_df: pd.DataFrame | None = None,
) -> alt.Chart:
    """Effective speed (remaining_dist / TTG) vs reference estimators.

    Parameters
    ----------
    ride_df : pd.DataFrame
        Prepped ride DataFrame with `elapsed_min` and `speed_kmh`.
    result_df : pd.DataFrame
        Estimator backtest result with `speed_ms` and `elapsed_min`.
    ref_df : pd.DataFrame, optional
        Reference estimator backtest result with `speed_ms` and `elapsed_min`.
        When provided, shown as a third "Reference" series. When omitted,
        only actual and estimated lines are drawn.
    """
    est_df = result_df.assign(speed_kmh=result_df["speed_ms"] * 3.6)
    smoothed_actual = ride_df[["elapsed_min"]].assign(
        speed_kmh=ride_df["speed_kmh"].rolling(60, min_periods=1, center=True).mean(),
        series="Actual (60s mean)",
    )
    frames = [
        ride_df[["elapsed_min", "speed_kmh"]].assign(series="Actual"),
        smoothed_actual,
        est_df[["elapsed_min", "speed_kmh"]].assign(series="Estimated"),
    ]
    series = ["Actual", "Actual (60s mean)", "Estimated"]
    colors = ["#888", "#555", TOL_VIBRANT[5]]
    opacities = [0.2, 0.7, 1.0]
    widths = [0.5, 1.2, 1.5]

    if ref_df is not None:
        ref_speed = ref_df.assign(speed_kmh=ref_df["speed_ms"] * 3.6)
        frames.append(
            ref_speed[["elapsed_min", "speed_kmh"]].assign(series="Reference")
        )
        series.append("Reference")
        colors.append(TOL_BRIGHT[3])
        opacities.append(1.0)
        widths.append(1.5)

    combined = pd.concat(frames, ignore_index=True)
    lines = (
        alt.Chart(combined)
        .mark_line(strokeWidth=1, invalid="break-paths-filter-domains")
        .encode(
            x=X_ELAPSED,
            y=alt.Y("speed_kmh:Q")
            .title("Effective speed (km/h)")
            .scale(domain=[0, 40], clamp=True),
            color=alt.Color("series:N")
            .scale(domain=series, range=colors)
            .legend(LEGEND_BOTTOM),
            opacity=alt.Opacity("series:N")
            .scale(domain=series, range=opacities)
            .legend(None),
            strokeWidth=alt.StrokeWidth("series:N")
            .scale(domain=series, range=widths)
            .legend(None),
        )
    )
    # Horizontal line at the ride's final total average speed
    total_s = (ride_df["time"].iloc[-1] - ride_df["time"].iloc[0]).total_seconds()
    total_avg_kmh = (
        (ride_df["distance_m"].iloc[-1] / total_s) * 3.6 if total_s > 0 else 0.0
    )
    ref_line = (
        alt.Chart(pd.DataFrame({"y": [total_avg_kmh]}))
        .mark_rule(color=TOL_BRIGHT[1], strokeWidth=1.2, strokeDash=[6, 3])
        .encode(y="y:Q")
    )
    return lines + ref_line


def comparison_errors(
    results: dict[str, pd.DataFrame], warmup_pct: float | None = None
) -> alt.Chart:
    """Overlay ETA error lines for multiple estimators with distinct colors."""
    frames = []
    for name, result in results.items():
        prepped = prep_time_axis(result, warmup_pct)
        frames.append(prepped.assign(estimator=name, delta_min=prepped["delta_s"] / 60))
    combined = pd.concat(frames, ignore_index=True)
    palette = TOL_MUTED if len(results) > len(COLORS) else COLORS
    return (
        alt.Chart(combined)
        .mark_line(strokeWidth=1.2, invalid="break-paths-filter-domains")
        .encode(
            x=X_ELAPSED,
            y=alt.Y("delta_min:Q").title("ETA \u2212 ATA (min)"),
            color=alt.Color("estimator:N")
            .scale(range=palette[: len(results)])
            .legend(alt.Legend(title=None, orient="top-right")),
        )
    )


def elevation_profile(
    ride_name: str,
    df: pd.DataFrame,
    min_area: float = 5000.0,
    min_length_m: float = 200.0,
) -> alt.Chart:
    """Two-row chart: original elevation profile + gradient-colored segments.

    Parameters
    ----------
    ride_name : str
        Display name for the chart title.
    df : pd.DataFrame
        Ride DataFrame with `distance_m` and `elevation_m` columns.
    min_area : float
        Visvalingam-Whyatt minimum triangle area.
    min_length_m : float
        Minimum segment length in meters after merging.
    """
    points, segments = decimate_to_gradient_segments(df, min_area, min_length_m)

    # Build long-form plotting DataFrame from segments
    rows = []
    for i, seg in enumerate(segments):
        b = gradient_bin(seg.gradient)
        label = f"{b:+d}%"
        for j in (i, i + 1):
            rows.append(
                {
                    "distance_km": points[j][0] / 1000,
                    "elevation_m": points[j][1],
                    "gradient_bin": label,
                    "segment_id": i,
                }
            )
    segments_df = pd.DataFrame(rows)

    dist_km = df["distance_m"] / 1000
    original = (
        alt.Chart(
            pd.DataFrame({"distance_km": dist_km, "elevation_m": df["elevation_m"]})
        )
        .mark_line(strokeWidth=0.8, color="#555")
        .encode(
            x=alt.X("distance_km:Q").title("Distance (km)"),
            y=alt.Y("elevation_m:Q").title("Elevation (m)"),
        )
        .properties(
            width=900,
            height=200,
            title=f"Original elevation profile ({len(df):,} points)",
        )
    )

    n_points = len(segments) + 1
    bin_labels = sorted(GRADIENT_COLORS.keys())
    domain = [f"{b:+d}%" for b in bin_labels]
    range_ = [GRADIENT_COLORS[b] for b in bin_labels]

    segmented = (
        alt.Chart(segments_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("distance_km:Q").title("Distance (km)"),
            y=alt.Y("elevation_m:Q").title("Elevation (m)"),
            color=alt.Color(
                "gradient_bin:N", scale=alt.Scale(domain=domain, range=range_)
            )
            .title("Gradient")
            .legend(orient="right"),
            detail="segment_id:N",
        )
        .properties(
            width=900,
            height=200,
            title=f"VW-simplified — gradient bins (3%) ({n_points:,} points)",
        )
    )

    return (
        (original & segmented)
        .resolve_scale(y="independent")
        .properties(title=alt.Title(ride_name))
        .configure_legend(disable=False)
    )
