"""Run ETA estimators against GPX files and report accuracy metrics."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from benchmark import backtest
from eta import (
    AvgSpeedEstimator,
    RollingAvgSpeedEstimator,
    add_haversine_distance,
    add_smooth_speed,
    read_gpx,
)
from plot import plot_backtest, plot_delta


def run(gpx_path: Path) -> None:
    """Run all estimators against a single GPX file and save plots.

    Parameters
    ----------
    gpx_path : Path
        Path to the GPX file to backtest.
    """
    df = read_gpx(gpx_path).pipe(add_haversine_distance).pipe(add_smooth_speed)
    ride_name = gpx_path.stem

    estimators = {
        "AvgSpeed": AvgSpeedEstimator(),
        "RollingAvg (5min)": RollingAvgSpeedEstimator(),
    }

    fig, axes = plt.subplots(len(estimators), 2, figsize=(16, 4 * len(estimators)))
    for i, (name, est) in enumerate(estimators.items()):
        result = backtest(df, est)
        plot_backtest(result, f"{name} \u2014 {ride_name}", ax=axes[i, 0], ride_df=df)
        plot_delta(result, f"Delta \u2014 {name}", ax=axes[i, 1], ride_df=df)

    plt.tight_layout()
    out = Path(f"backtest_{ride_name}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    print(f"\n{'Estimator':<25} {'MAE (min)':>10} {'RMSE (min)':>10}")
    print("-" * 47)
    for name, est in estimators.items():
        result = backtest(df, est)
        trimmed = result[result["distance_m"] > 5000]["delta_s"].dropna()
        mae = trimmed.abs().mean() / 60
        rmse = (trimmed**2).mean() ** 0.5 / 60
        print(f"{name:<25} {mae:>10.2f} {rmse:>10.2f}")


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or list(Path("data").glob("*.gpx"))
    if not paths:
        print(
            "No GPX files found. Place .gpx files in data/ or pass paths as arguments."
        )
        sys.exit(1)
    for p in paths:
        print(f"\n=== {p.stem} ===")
        run(p)
