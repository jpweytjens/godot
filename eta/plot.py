"""Composable Altair plotting layers for ETA backtest results."""

import altair as alt
import pandas as pd

from eta.gpx import pause_run_id
from eta.theme import COLORS, TOL_BRIGHT, TOL_MUTED, TOL_VIBRANT

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


def pause_intervals(ride_df: pd.DataFrame, min_pause_s: float = 60.0) -> pd.DataFrame:
    """Convert paused column into a DataFrame of pause intervals.

    Parameters
    ----------
    ride_df : pd.DataFrame
        Ride DataFrame with paused, time, and elapsed_min columns.
        Must have been processed by `prep_time_axis` first.
    min_pause_s : float, optional
        Minimum pause duration in seconds to include. Default 60.

    Returns
    -------
    pd.DataFrame
        Columns: start_min, end_min. One row per pause interval.
    """
    if "paused" not in ride_df.columns:
        return pd.DataFrame(columns=["start_min", "end_min"])
    is_paused = ride_df["paused"]
    run_id = pause_run_id(is_paused)
    intervals = []
    for _, grp in ride_df[is_paused].groupby(run_id[is_paused]):
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


def pause_bands(pauses_df: pd.DataFrame) -> alt.Chart:
    """Blue semi-transparent bands for paused sections."""
    return (
        alt.Chart(pauses_df)
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


def speed_estimated(result_df: pd.DataFrame) -> alt.Chart:
    """Estimated average speed line from the estimator."""
    df = result_df.assign(speed_kmh=result_df["speed_ms"] * 3.6)
    return (
        alt.Chart(df)
        .mark_line(
            strokeWidth=1.5, color=TOL_VIBRANT[5], invalid="break-paths-filter-domains"
        )
        .encode(x=X_ELAPSED, y=alt.Y("speed_kmh:Q").title("Speed (km/h)"))
    )


def eta_error(result_df: pd.DataFrame) -> alt.Chart:
    """ETA error (delta) line in minutes."""
    df = result_df.assign(delta_min=result_df["delta_s"] / 60)
    return (
        alt.Chart(df)
        .mark_line(
            strokeWidth=1.2, color=TOL_VIBRANT[5], invalid="break-paths-filter-domains"
        )
        .encode(x=X_ELAPSED, y=alt.Y("delta_min:Q").title("ETA \u2212 ATA (min)"))
    )


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


def speed_comparison(ride_df: pd.DataFrame, result_df: pd.DataFrame) -> alt.Chart:
    """Actual vs estimated speed with legend."""
    est_df = result_df.assign(speed_kmh=result_df["speed_ms"] * 3.6)
    combined = pd.concat(
        [
            ride_df[["elapsed_min", "speed_kmh"]].assign(series="Actual"),
            est_df[["elapsed_min", "speed_kmh"]].assign(series="Estimated"),
        ],
        ignore_index=True,
    )
    return (
        alt.Chart(combined)
        .mark_line(strokeWidth=1, invalid="break-paths-filter-domains")
        .encode(
            x=X_ELAPSED,
            y=alt.Y("speed_kmh:Q").title("Speed (km/h)"),
            color=alt.Color("series:N")
            .scale(domain=["Actual", "Estimated"], range=["#888", TOL_VIBRANT[5]])
            .legend(alt.Legend(title=None, orient="top-right")),
            opacity=alt.Opacity("series:N")
            .scale(domain=["Actual", "Estimated"], range=[0.4, 1.0])
            .legend(None),
            strokeWidth=alt.StrokeWidth("series:N")
            .scale(domain=["Actual", "Estimated"], range=[0.6, 1.5])
            .legend(None),
        )
    )


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
