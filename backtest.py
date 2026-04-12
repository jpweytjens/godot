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
    AdaptiveGradientPriorEstimator,
    BinnedAdaptiveEstimator,
    OracleTrustedBinnedEstimator,
    TrustedBinnedAdaptiveEstimator,
    WeightedGainVFlat,
    NoisyOracleAdaptiveLerpEstimator,
    OracleAdaptiveLerpEstimator,
    GradientPriorEstimator,
    PhysicsGradientPriorEstimator,
)
from godot.pause import NoPause, WallClockPause
from godot.plot import (
    _pause_intervals,
    avg_speed_overview,
    comparison_errors,
    downsample_for_plot,
    error_pct_refs,
    error_refs,
    eta_error,
    eta_error_moving,
    eta_error_moving_pct,
    eta_error_pct,
    pause_bands,
    prep_time_axis,
    speed_raw,
    speed_smoothed_comparison,
)
from godot.ride import load_ride
from godot.theme import TOL_BRIGHT

REF_TOTAL = "0_T_avg_speed"
REF_MOVING = "0_M_avg_speed"
REF_TOTAL_COLOR = TOL_BRIGHT[3]  # yellow
REF_MOVING_COLOR = TOL_BRIGHT[4]  # cyan
REF_OPACITY = 0.7

# Load empirical gradient ratios
_ratio_df = pd.read_parquet(Path("data/gradient_ratios.parquet"))
GRADIENT_RATIOS: dict[int, float] = _ratio_df["mean_ratio"].to_dict()
GLOBAL_PRIOR_KMH = 28.8  # tunable flat-ground speed
TOTAL_SYSTEM_MASS = 85 + 10  # rider + bike, for physics-based estimators


def _time_basis(name: str) -> str:
    """Extract time basis ('T' or 'M') from estimator name."""
    parts = name.split("_")
    return parts[1].upper() if len(parts) >= 2 else "T"


# Estimator naming convention: {level}_{T|M}_{prior}_{name}
#   level: 0-4 (complexity / correction layers)
#   T|M:   Total time (NoPause) or Moving time (WallClockPause)
#   prior: global (fixed), oracle, or estimator name (e.g. wgain)
#   name:  descriptive correction layer
ESTIMATORS = {
    # --- Level 0: no gradient awareness ---
    "0_T_avg_speed": (
        AvgSpeedEstimator(moving_only=False),
        NoPause(),
    ),
    "0_M_avg_speed": (
        AvgSpeedEstimator(moving_only=True),
        WallClockPause(),
    ),
    "0_T_adaptive_lerp": (
        AdaptiveLerpSpeedEstimator(
            prior_ms=28.8 / 3.6,
            tau=60 * 60,
            k=0.5,
            fast_span_s=3600,
            fast_weight=0.25,
        ),
        NoPause(),
    ),
    # --- Level 1: gradient prior only (no online correction) ---
    "1_T_global_empirical_prior": (
        GradientPriorEstimator(v_flat_kmh=GLOBAL_PRIOR_KMH, ratios=GRADIENT_RATIOS),
        NoPause(),
    ),
    "1_M_global_physics_prior": (
        PhysicsGradientPriorEstimator(
            mass_kg=TOTAL_SYSTEM_MASS,
            v_flat_kmh=GLOBAL_PRIOR_KMH,
        ),
        WallClockPause(),
    ),
    # --- Level 2: + slow EWMA or adaptive v_flat ---
    "2_T_wgain_empirical_adaptive_vflat": (
        AdaptiveGradientPriorEstimator(
            v_flat_kmh=GLOBAL_PRIOR_KMH,
            ratios=GRADIENT_RATIOS,
            vflat_estimator=WeightedGainVFlat(),
        ),
        NoPause(),
    ),
    "2_M_global_physics_slow_cal": (
        AdaptivePhysicsEstimator(
            mass_kg=TOTAL_SYSTEM_MASS,
            v_flat_kmh=GLOBAL_PRIOR_KMH,
            cal_max_grad=0.02,
        ),
        WallClockPause(),
    ),
    # --- Level 3: + per-bin fast EWMA ---
    "3_T_global_empirical_binned": (
        BinnedAdaptiveEstimator(
            prior=GradientPriorEstimator(
                v_flat_kmh=GLOBAL_PRIOR_KMH,
                ratios=GRADIENT_RATIOS,
            ),
        ),
        NoPause(),
    ),
    "3_M_global_physics_binned": (
        BinnedAdaptiveEstimator(
            prior=PhysicsGradientPriorEstimator(
                mass_kg=TOTAL_SYSTEM_MASS,
                v_flat_kmh=GLOBAL_PRIOR_KMH,
            ),
        ),
        WallClockPause(),
    ),
    # --- Level 4: + trust ramp + clamping ---
    "4_T_global_empirical_trusted": (
        TrustedBinnedAdaptiveEstimator(
            prior=GradientPriorEstimator(
                v_flat_kmh=GLOBAL_PRIOR_KMH,
                ratios=GRADIENT_RATIOS,
            ),
        ),
        NoPause(),
    ),
    "4_M_global_physics_trusted": (
        TrustedBinnedAdaptiveEstimator(
            prior=PhysicsGradientPriorEstimator(
                mass_kg=TOTAL_SYSTEM_MASS,
                v_flat_kmh=GLOBAL_PRIOR_KMH,
            ),
        ),
        WallClockPause(),
    ),
    "4_T_oracle_empirical_trusted": (
        OracleTrustedBinnedEstimator(
            prior=GradientPriorEstimator(
                v_flat_kmh=GLOBAL_PRIOR_KMH,
                ratios=GRADIENT_RATIOS,
            ),
        ),
        NoPause(),
    ),
    "4_M_oracle_physics_trusted": (
        OracleTrustedBinnedEstimator(
            prior=PhysicsGradientPriorEstimator(
                mass_kg=TOTAL_SYSTEM_MASS,
                v_flat_kmh=GLOBAL_PRIOR_KMH,
            ),
        ),
        WallClockPause(),
    ),
}


def run(
    gpx_path: Path,
    distance_method: str = "haversine",
    smooth_speed: bool = True,
    smooth_window: str = "5s",
    save_plots: bool = True,
    chart_width: int = 1200,
    chart_height: int = 200,
    comparison_height: int = 400,
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
    save_plots : bool, optional
        Whether to generate and save Altair charts. Default True.
    chart_width : int, optional
        Width in pixels for individual charts. Default 1200.
    chart_height : int, optional
        Height in pixels for individual charts. Default 200.
    comparison_height : int, optional
        Height in pixels for the comparison chart. Default 400.

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

    if save_plots:
        out_dir = Path("output") / "backtests" / ride.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Prepare ride data for plotting (with warmup clipped, then downsampled)
        ride_prepped = downsample_for_plot(prep_time_axis(ride.df, warmup_pct=0.02))
        pause_df = _pause_intervals(ride_prepped)

        # Reference results: total (yellow) and moving (cyan)
        ref_total = results.get(REF_TOTAL)
        ref_total_prepped = (
            downsample_for_plot(prep_time_axis(ref_total, warmup_pct=0.02))
            if ref_total is not None
            else None
        )
        ref_moving = results.get(REF_MOVING)
        ref_moving_prepped = (
            downsample_for_plot(prep_time_axis(ref_moving, warmup_pct=0.02))
            if ref_moving is not None
            else None
        )

        # Per-estimator: 5-panel stack
        for name, result in results.items():
            result_prepped = downsample_for_plot(
                prep_time_axis(result, warmup_pct=0.02)
            )
            is_ref = name in (REF_TOTAL, REF_MOVING)

            # Select error functions based on time basis
            is_moving = _time_basis(name) == "M"
            err_fn = eta_error_moving if is_moving else eta_error
            err_pct_fn = eta_error_moving_pct if is_moving else eta_error_pct

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
                            err_fn(
                                ref_prep,
                                color=color,
                                opacity=REF_OPACITY,
                                stroke_width=1.0,
                            )
                        )
                        ref_pct.append(
                            err_pct_fn(
                                ref_prep,
                                color=color,
                                opacity=REF_OPACITY,
                                stroke_width=1.0,
                            )
                        )

            error_chart = alt.layer(
                pause_bands(ride_prepped, pause_df=pause_df),
                *ref_error,
                err_fn(result_prepped),
                error_refs(),
            ).properties(width=chart_width, height=chart_height)

            error_pct_chart = alt.layer(
                pause_bands(ride_prepped, pause_df=pause_df),
                *ref_pct,
                err_pct_fn(result_prepped),
                error_pct_refs(),
            ).properties(width=chart_width, height=chart_height)

            # Avg speed overview: estimated + actual cumulative averages
            speed_chart = alt.layer(
                pause_bands(ride_prepped, pause_df=pause_df),
                avg_speed_overview(
                    ride_prepped,
                    ref_total_df=ref_total_prepped,
                    ref_moving_df=ref_moving_prepped,
                ),
            ).properties(width=chart_width, height=chart_height)

            # Raw speed: unsmoothed actual vs predicted
            raw_speed_chart = alt.layer(
                pause_bands(ride_prepped, pause_df=pause_df),
                speed_raw(ride_prepped, result_prepped),
            ).properties(width=chart_width, height=chart_height)

            # Smoothed speed: both 60s-smoothed
            smooth_speed_chart = alt.layer(
                pause_bands(ride_prepped, pause_df=pause_df),
                speed_smoothed_comparison(ride_prepped, result_prepped),
            ).properties(width=chart_width, height=chart_height)

            chart = (
                error_chart
                & error_pct_chart
                & speed_chart
                & raw_speed_chart
                & smooth_speed_chart
            ).properties(title=alt.Title(f"{name} \u2014 {ride.label}"))
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            chart.save(str(out_dir / f"{safe_name}.png"), scale_factor=2)

        # Comparison: all estimators' ETA error on one chart
        comp_chart = alt.layer(
            pause_bands(ride_prepped, pause_df=pause_df),
            comparison_errors(results, warmup_pct=0.02),
            error_refs(),
        ).properties(
            title=alt.Title(f"All estimators \u2014 {ride.label}"),
            width=chart_width,
            height=comparison_height,
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
        col = name.lower().replace(" ", "_")
        basis = _time_basis(name)
        if basis == "T":
            # Wall-clock metrics (vs total remaining time)
            wc = compute_metrics(result, warmup_m, moving_only=False)
            row[f"{col}_mae"] = wc["mae_min"]
            row[f"{col}_rmse"] = wc["rmse_min"]
            row[f"{col}_mpe"] = wc["mpe_pct"]
            row[f"{col}_mape"] = wc["mape_pct"]
        else:
            # Moving-time metrics (vs remaining moving time)
            mv = compute_metrics(result, warmup_m, moving_only=True)
            row[f"{col}_mov_mae"] = mv["mae_min"]
            row[f"{col}_mov_rmse"] = mv["rmse_min"]
            row[f"{col}_mov_mpe"] = mv["mpe_pct"]
            row[f"{col}_mov_mape"] = mv["mape_pct"]
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
        "--no-plots",
        action="store_true",
        help="Skip chart generation for faster runs",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=[
            "mae",
            "rmse",
            "mpe",
            "mape",
            "mov_mae",
            "mov_rmse",
            "mov_mpe",
            "mov_mape",
        ],
        default=None,
        metavar="METRIC",
        help="Metrics to display (default depends on --no-plots)",
    )
    args = parser.parse_args()

    if args.metrics is None:
        args.metrics = (
            ["rmse", "mpe", "mape", "mov_rmse", "mov_mpe", "mov_mape"]
            if args.no_plots
            else [
                "mae",
                "rmse",
                "mpe",
                "mape",
                "mov_mae",
                "mov_rmse",
                "mov_mpe",
                "mov_mape",
            ]
        )

    paths = [p.resolve() for p in (args.paths or list(Path("data").glob("*.gpx")))]
    if not paths:
        parser.error(
            "No GPX files found. Place .gpx files in data/ or pass paths as arguments."
        )

    smooth_options = [s == "on" for s in args.smoothing]
    combos = list(product(paths, args.distance, smooth_options))
    save_plots = not args.no_plots
    rows = process_map(
        run,
        [c[0] for c in combos],
        [c[1] for c in combos],
        [c[2] for c in combos],
        [args.smooth_window] * len(combos),
        [save_plots] * len(combos),
        chunksize=1,
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
