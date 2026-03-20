"""Run ETA estimators against GPX files and report accuracy metrics."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from benchmark import backtest
from estimators import AvgSpeedEstimator, RollingAvgSpeedEstimator
from gpx import add_haversine_distance, add_smooth_speed, read_gpx
from plot import plot_comparison, plot_delta

ESTIMATORS = {
    "AvgSpeed": AvgSpeedEstimator(),
    "Rolling 1min": RollingAvgSpeedEstimator(window_s=60),
    "Rolling 5min": RollingAvgSpeedEstimator(window_s=300),
    "Rolling 10min": RollingAvgSpeedEstimator(window_s=600),
    "Rolling 30min": RollingAvgSpeedEstimator(window_s=1800),
}


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
        Ride DataFrame with an elevation_m column.
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
        One of: "uphill", "downhill", "flat", "hilly", "mountain".
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


def run(gpx_path: Path) -> dict:
    """Run all estimators against a single GPX file, save plots, return metrics.

    Parameters
    ----------
    gpx_path : Path
        Path to the GPX file to backtest.

    Returns
    -------
    dict
        Row dict with keys: ride, route_type, and per-estimator MAE/RMSE columns.
    """
    df = read_gpx(gpx_path).pipe(add_haversine_distance).pipe(add_smooth_speed)
    ride_name = gpx_path.stem
    route_type = classify_route(df)

    results = {name: backtest(df, est) for name, est in ESTIMATORS.items()}

    n_est = len(ESTIMATORS)
    height_ratios = [3] * n_est + [5]
    fig, axes = plt.subplots(
        n_est + 1,
        1,
        figsize=(14, sum(height_ratios)),
        gridspec_kw={"height_ratios": height_ratios},
    )
    for i, (name, result) in enumerate(results.items()):
        plot_delta(
            result, f"{name} \u2014 {ride_name}", ax=axes[i], ride_df=df, warmup_km=5.0
        )
    plot_comparison(
        results,
        f"All estimators \u2014 {ride_name}",
        ax=axes[-1],
        ride_df=df,
        warmup_km=5.0,
    )
    plt.tight_layout()
    out = Path(f"backtest_{ride_name}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    row: dict = {"ride": ride_name, "route_type": route_type}
    for name, result in results.items():
        trimmed = result[result["distance_m"] > 5000]["delta_s"].dropna()
        col = name.lower().replace(" ", "_")
        row[f"{col}_mae"] = trimmed.abs().mean() / 60
        row[f"{col}_rmse"] = (trimmed**2).mean() ** 0.5 / 60
    return row


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or list(Path("data").glob("*.gpx"))
    if not paths:
        print(
            "No GPX files found. Place .gpx files in data/ or pass paths as arguments."
        )
        sys.exit(1)

    rows = [run(p) for p in paths]
    results_df = pd.DataFrame(rows)

    metric_cols = [
        c for c in results_df.columns if c.endswith("_mae") or c.endswith("_rmse")
    ]

    print("\n--- Per-ride metrics ---")
    print(results_df.to_string(index=False, float_format="{:.2f}".format))

    print("\n--- Global averages ---")
    print(results_df[metric_cols].mean().to_string(float_format="{:.2f}".format))

    print("\n--- Averages by route type ---")
    print(
        results_df.groupby("route_type")[metric_cols]
        .mean()
        .to_string(float_format="{:.2f}".format)
    )
