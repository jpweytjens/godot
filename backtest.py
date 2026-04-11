"""Run ETA estimators against GPX files and report accuracy metrics."""

from pathlib import Path

import altair as alt
import pandas as pd
from tqdm.contrib.concurrent import process_map

from godot.benchmark import backtest, compute_metrics
from godot.estimators import (
    AdaptiveLerpSpeedEstimator,
    AdaptivePhysicsEstimator,
    AvgSpeedEstimator,
    BinnedAdaptivePhysicsEstimator,
    NoisyOracleAdaptiveLerpEstimator,
    OracleAdaptiveLerpEstimator,
    GradientPriorEstimator,
    PhysicsGradientPriorEstimator,
)
from godot.pause import NoPause, WallClockPause
from godot.plot import (
    actual_speed,
    avg_speed_overview,
    comparison_errors,
    error_pct_refs,
    error_refs,
    eta_error,
    eta_error_pct,
    pause_bands,
    prep_time_axis,
)
from godot.ride import load_ride
from godot.theme import TOL_BRIGHT

REF_TOTAL = "Average speed (total)"
REF_MOVING = "Average speed (moving)"
REF_TOTAL_COLOR = TOL_BRIGHT[3]  # yellow
REF_MOVING_COLOR = TOL_BRIGHT[4]  # cyan
REF_OPACITY = 0.7

# Load empirical gradient ratios
_ratio_df = pd.read_parquet(Path("data/gradient_ratios.parquet"))
GRADIENT_RATIOS: dict[int, float] = _ratio_df["mean_ratio"].to_dict()
GLOBAL_PRIOR_KMH = 28.8  # tunable flat-ground speed

ESTIMATORS = {
    "Average speed (moving)": (AvgSpeedEstimator(moving_only=True), WallClockPause()),
    "Average speed (total)": (AvgSpeedEstimator(moving_only=False), NoPause()),
    # "Rolling 5 min (moving)": (
    #     RollingAvgSpeedEstimator(window_s=300, moving_only=True),
    #     WallClockPause(),
    # ),
    # "Rolling 10 min (moving)": (
    #     RollingAvgSpeedEstimator(window_s=600, moving_only=True),
    #     WallClockPause(),
    # ),
    # "Rolling 30 min (moving)": (
    #     RollingAvgSpeedEstimator(window_s=1800, moving_only=True),
    #     WallClockPause(),
    # ),
    # "Rolling 60 min (moving)": (
    #     RollingAvgSpeedEstimator(window_s=3600, moving_only=True),
    #     WallClockPause(),
    # ),
    # "Rolling median 30 min (moving)": (
    #     RollingMedianSpeedEstimator(window_s=1800, moving_only=True),
    #     WallClockPause(),
    # ),
    # "EWMA 60 min (moving)": (
    #     EWMASpeedEstimator(span_s=3600, moving_only=True),
    #     WallClockPause(),
    # ),
    # "EWMA 10 min (moving)": (
    #     EWMASpeedEstimator(span_s=600, moving_only=True),
    #     WallClockPause(),
    # ),
    # "DEWMA 10+60 min (moving)": (
    #     DEWMASpeedEstimator(
    #         slow_span_s=3600,
    #         fast_span_s=600,
    #         slow_weight=0.7,
    #         fast_weight=0.3,
    #         moving_only=True,
    #     ),
    #     WallClockPause(),
    # ),
    # "EWMA 10 min + prior (moving)": (
    #     PriorEWMASpeedEstimator(
    #         span_s=600,
    #         prior_ms=28.8 / 3.6,
    #         moving_only=True,
    #     ),
    #     WallClockPause(),
    # ),
    # "Lerp (moving)": (
    #     LerpSpeedEstimator(
    #         prior_ms=28.8 / 3.6,
    #         fast_span_s=600,
    #         ramp_s=600,
    #         fast_weight=0.15,
    #         moving_only=True,
    #     ),
    #     WallClockPause(),
    # ),
    "Adaptive lerp": (
        AdaptiveLerpSpeedEstimator(
            prior_ms=28.8 / 3.6,
            tau=60 * 60,
            k=0.5,
            fast_span_s=3600,
            fast_weight=0.25,
        ),
        NoPause(),
    ),
    "Adaptive lerp (oracle prior)": (
        OracleAdaptiveLerpEstimator(tau=300, k=2.0, fast_span_s=3600, fast_weight=0.15),
        NoPause(),
    ),
    "Adaptive lerp (noisy prior)": (
        NoisyOracleAdaptiveLerpEstimator(
            tau=300, k=2.0, fast_span_s=3600, fast_weight=0.15, cv=0.10, seed=42
        ),
        NoPause(),
    ),
    "Static gradient prior": (
        GradientPriorEstimator(v_flat_kmh=GLOBAL_PRIOR_KMH, ratios=GRADIENT_RATIOS),
        NoPause(),
    ),
    "Physics gradient prior": (
        PhysicsGradientPriorEstimator(
            mass_kg=80,
            v_flat_kmh=GLOBAL_PRIOR_KMH,
        ),
        WallClockPause(),
    ),
    "Adaptive physics (flat cal)": (
        AdaptivePhysicsEstimator(
            mass_kg=80,
            v_flat_kmh=GLOBAL_PRIOR_KMH,
            cal_max_grad=0.02,
        ),
        WallClockPause(),
    ),
    "Adaptive physics (all grad)": (
        AdaptivePhysicsEstimator(
            mass_kg=80,
            v_flat_kmh=GLOBAL_PRIOR_KMH,
            cal_max_grad=1.0,
        ),
        WallClockPause(),
    ),
    "Binned adaptive physics": (
        BinnedAdaptivePhysicsEstimator(
            mass_kg=80,
            v_flat_kmh=GLOBAL_PRIOR_KMH,
        ),
        WallClockPause(),
    ),
}


def run(
    gpx_path: Path,
    distance_method: str = "haversine",
    smooth_speed: bool = True,
    smooth_window: str = "5s",
) -> dict:
    """Run all estimators against a single GPX file, save plots, return metrics.

    Parameters
    ----------
    gpx_path : Path
        Path to the GPX file to backtest.
    distance_method : str, optional
        Distance pipeline to apply: ``"haversine"`` (default) or ``"integrated"``.
    smooth_speed : bool, optional
        Whether to apply the rolling speed smoother. Default True.
    smooth_window : str, optional
        Rolling window size as a pandas time offset string. Default `"5s"`.

    Returns
    -------
    dict
        Row dict with keys: ride, distance_method, speed_smoothed, route_type,
        and per-estimator MAE/RMSE columns.
    """
    ride = load_ride(gpx_path, distance_method, smooth_speed, smooth_window)
    results = {
        name: backtest(ride, est, pause) for name, (est, pause) in ESTIMATORS.items()
    }

    out_dir = Path("output") / "backtests" / ride.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare ride data for plotting (with warmup clipped)
    ride_prepped = prep_time_axis(ride.df, warmup_pct=0.02)

    # Reference results: total (yellow) and moving (cyan)
    ref_total = results.get(REF_TOTAL)
    ref_total_prepped = (
        prep_time_axis(ref_total, warmup_pct=0.02) if ref_total is not None else None
    )
    ref_moving = results.get(REF_MOVING)
    ref_moving_prepped = (
        prep_time_axis(ref_moving, warmup_pct=0.02) if ref_moving is not None else None
    )

    # Per-estimator: 1x3 (ETA error | MPE % | speed), time domain
    for name, result in results.items():
        result_prepped = prep_time_axis(result, warmup_pct=0.02)
        is_ref = name in (REF_TOTAL, REF_MOVING)

        # Reference layers (behind the estimator line)
        ref_error: list[alt.Chart] = []
        ref_pct: list[alt.Chart] = []
        if not is_ref:
            for ref_prep, color in [
                (ref_total_prepped, REF_TOTAL_COLOR),
                (ref_moving_prepped, REF_MOVING_COLOR),
            ]:
                if ref_prep is not None:
                    ref_error.append(
                        eta_error(
                            ref_prep, color=color, opacity=REF_OPACITY, stroke_width=1.0
                        )
                    )
                    ref_pct.append(
                        eta_error_pct(
                            ref_prep, color=color, opacity=REF_OPACITY, stroke_width=1.0
                        )
                    )

        error_chart = alt.layer(
            pause_bands(ride_prepped),
            *ref_error,
            eta_error(result_prepped),
            error_refs(),
        ).properties(width=800, height=200)

        error_pct_chart = alt.layer(
            pause_bands(ride_prepped),
            *ref_pct,
            eta_error_pct(result_prepped),
            error_pct_refs(),
        ).properties(width=800, height=200)

        # Avg speed overview: estimated + actual cumulative averages
        speed_chart = alt.layer(
            pause_bands(ride_prepped),
            avg_speed_overview(
                ride_prepped,
                ref_total_df=ref_total_prepped,
                ref_moving_df=ref_moving_prepped,
            ),
        ).properties(width=800, height=200)

        actual_speed_chart = alt.layer(
            pause_bands(ride_prepped),
            actual_speed(ride_prepped, result_prepped),
        ).properties(width=800, height=200)

        chart = (
            error_chart & error_pct_chart & speed_chart & actual_speed_chart
        ).properties(title=alt.Title(f"{name} \u2014 {ride.label}"))
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        chart.save(str(out_dir / f"{safe_name}.png"), scale_factor=2)

    # Comparison: all estimators' ETA error on one chart
    comp_chart = alt.layer(
        pause_bands(ride_prepped),
        comparison_errors(results, warmup_pct=0.02),
        error_refs(),
    ).properties(
        title=alt.Title(f"All estimators \u2014 {ride.label}"),
        width=900,
        height=350,
    )
    comp_chart.save(str(out_dir / "comparison.png"), scale_factor=2)

    row: dict = {
        "ride": ride.name,
        "distance_method": distance_method,
        "speed_smoothed": smooth_speed,
        "route_type": ride.route_type,
        "contains_pauses": ride.contains_pauses,
    }
    warmup_m = ride.distance * 0.02
    for name, result in results.items():
        _, pause = ESTIMATORS[name]
        moving = isinstance(pause, WallClockPause)
        metrics = compute_metrics(result, warmup_m, moving_only=moving)
        col = name.lower().replace(" ", "_")
        row[f"{col}_mae"] = metrics["mae_min"]
        row[f"{col}_rmse"] = metrics["rmse_min"]
        row[f"{col}_mpe"] = metrics["mpe_pct"]
        row[f"{col}_mape"] = metrics["mape_pct"]
    return row


if __name__ == "__main__":
    import argparse
    from itertools import product

    parser = argparse.ArgumentParser(description="ETA estimator backtest")
    parser.add_argument(
        "paths", nargs="*", type=Path, help="GPX files (default: data/*.gpx)"
    )
    parser.add_argument(
        "--distance",
        nargs="+",
        choices=["haversine", "integrated"],
        default=["integrated"],
        metavar="METHOD",
        help="Distance method(s) to use (default: integrated)",
    )
    parser.add_argument(
        "--smoothing",
        nargs="+",
        choices=["on", "off"],
        default=["off"],
        metavar="ON|OFF",
        help="Speed smoothing option(s) to include (default: off)",
    )
    parser.add_argument(
        "--smooth-window",
        default="5s",
        metavar="WINDOW",
        help="Rolling window size for speed smoothing (default: 5s)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["mae", "rmse", "mpe", "mape"],
        default=["mae", "mpe", "mape"],
        metavar="METRIC",
        help="Metrics to display (default: mae mpe mape)",
    )
    args = parser.parse_args()

    paths = [p.resolve() for p in (args.paths or list(Path("data").glob("*.gpx")))]
    if not paths:
        parser.error(
            "No GPX files found. Place .gpx files in data/ or pass paths as arguments."
        )

    smooth_options = [s == "on" for s in args.smoothing]
    combos = list(product(paths, args.distance, smooth_options))
    rows = process_map(
        run,
        [c[0] for c in combos],
        [c[1] for c in combos],
        [c[2] for c in combos],
        [args.smooth_window] * len(combos),
        desc="Backtesting",
        unit="run",
    )
    results_df = (
        pd.DataFrame(rows)
        .sort_values(["route_type", "ride", "distance_method", "speed_smoothed"])
        .reset_index(drop=True)
    )

    from godot.report import write_html_report

    write_html_report(results_df, args.metrics, Path("output") / "results.html")
