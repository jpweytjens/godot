"""Run ETA estimators against GPX files and report accuracy metrics."""

from pathlib import Path

import altair as alt
import pandas as pd
from tqdm.contrib.concurrent import process_map

from godot.benchmark import backtest, compute_metrics
from godot.estimators import (
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
    RealisticPhysicsEstimator,
    realistic_physics_ratios,
    EwmaLockVFlat,
    MedianLockVFlat,
    FlatSpeedVFlat,
    PriorFreeVFlat,
    PriorFreeEwmaVFlat,
    CalibratingPhysicsEstimator,
    PIPhysicsEstimator,
)
from godot.pause import WallClockPause
from godot.plot import (
    X_ELAPSED,
    _end_labels,
    _pause_intervals,
    avg_speed_overview,
    comparison_errors,
    downsample_for_plot,
    error_pct_refs,
    error_refs,
    pause_bands,
    prep_time_axis,
    speed_residual_raw,
    speed_residual_smoothed,
)
from godot.theme import TOL_VIBRANT
from godot.ride import load_ride
from godot.theme import TOL_BRIGHT

REF_MOVING = "0_M_avg_speed"
REF_MOVING_COLOR = TOL_BRIGHT[4]  # cyan
REF_OPACITY = 0.7

# Load empirical gradient ratios
_ratio_df = pd.read_parquet(Path("data/gradient_ratios.parquet"))
GRADIENT_RATIOS: dict[int, float] = _ratio_df["mean_ratio"].to_dict()
GLOBAL_PRIOR_KMH = 28.8  # tunable flat-ground speed
TOTAL_SYSTEM_MASS = 85 + 10  # rider + bike, for physics-based estimators
REALISTIC_RATIOS: dict[int, float] = realistic_physics_ratios(
    mass_kg=TOTAL_SYSTEM_MASS,
    v_flat_ms=GLOBAL_PRIOR_KMH / 3.6,
)


# Estimator naming convention: {level}_M_{prior}_{name}
#   level: 0-5 (complexity / correction layers)
#   M:     Moving time — all estimators use WallClockPause
#   prior: global (fixed), oracle, or estimator name (e.g. wgain)
#   name:  descriptive correction layer
ESTIMATORS = {
    # --- Level 0: no gradient awareness ---
    "0_M_avg_speed": (
        AvgSpeedEstimator(moving_only=True),
        WallClockPause(),
    ),
    # --- Level 1: gradient prior only (no online correction) ---
    "1_M_global_empirical_prior": (
        GradientPriorEstimator(v_flat_kmh=GLOBAL_PRIOR_KMH, ratios=GRADIENT_RATIOS),
        WallClockPause(),
    ),
    "1_M_global_physics_prior": (
        PhysicsGradientPriorEstimator(
            mass_kg=TOTAL_SYSTEM_MASS,
            v_flat_kmh=GLOBAL_PRIOR_KMH,
        ),
        WallClockPause(),
    ),
    # --- Level 2: + slow EWMA or adaptive v_flat ---
    "2_M_wgain_empirical_adaptive_vflat": (
        AdaptiveGradientPriorEstimator(
            v_flat_kmh=GLOBAL_PRIOR_KMH,
            ratios=GRADIENT_RATIOS,
            vflat_estimator=WeightedGainVFlat(),
        ),
        WallClockPause(),
    ),
    # "2_M_global_physics_slow_cal": (
    #     AdaptivePhysicsEstimator(
    #         mass_kg=TOTAL_SYSTEM_MASS,
    #         v_flat_kmh=GLOBAL_PRIOR_KMH,
    #         cal_max_grad=0.02,
    #     ),
    #     WallClockPause(),
    # ),
    # --- Level 3: + per-bin fast EWMA ---
    "3_M_global_empirical_binned": (
        BinnedAdaptiveEstimator(
            prior=GradientPriorEstimator(
                v_flat_kmh=GLOBAL_PRIOR_KMH,
                ratios=GRADIENT_RATIOS,
            ),
        ),
        WallClockPause(),
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
    "4_M_global_empirical_trusted": (
        TrustedBinnedAdaptiveEstimator(
            prior=GradientPriorEstimator(
                v_flat_kmh=GLOBAL_PRIOR_KMH,
                ratios=GRADIENT_RATIOS,
            ),
        ),
        WallClockPause(),
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
    # "4_M_oracle_empirical_trusted": (
    #     OracleTrustedBinnedEstimator(
    #         prior=GradientPriorEstimator(
    #             v_flat_kmh=GLOBAL_PRIOR_KMH,
    #             ratios=GRADIENT_RATIOS,
    #         ),
    #     ),
    #     WallClockPause(),
    # ),
    # "4_M_oracle_physics_trusted": (
    #     OracleTrustedBinnedEstimator(
    #         prior=PhysicsGradientPriorEstimator(
    #             mass_kg=TOTAL_SYSTEM_MASS,
    #             v_flat_kmh=GLOBAL_PRIOR_KMH,
    #         ),
    #     ),
    #     WallClockPause(),
    # ),
    # --- Level 5: realistic physics prior ---
    "1_M_global_realistic_prior": (
        RealisticPhysicsEstimator(
            mass_kg=TOTAL_SYSTEM_MASS,
            v_flat_kmh=GLOBAL_PRIOR_KMH,
        ),
        WallClockPause(),
    ),
    "5_M_global_realistic_trusted": (
        TrustedBinnedAdaptiveEstimator(
            prior=RealisticPhysicsEstimator(
                mass_kg=TOTAL_SYSTEM_MASS,
                v_flat_kmh=GLOBAL_PRIOR_KMH,
            ),
        ),
        WallClockPause(),
    ),
    # --- Calibrating physics: self-contained realistic + EWMA v_flat ---
    "5_M_calibrating_physics": (
        CalibratingPhysicsEstimator(
            mass_kg=TOTAL_SYSTEM_MASS,
            v_flat_kmh=GLOBAL_PRIOR_KMH,
        ),
        WallClockPause(),
    ),
    "5_M_pi_physics": (
        PIPhysicsEstimator(
            mass_kg=TOTAL_SYSTEM_MASS,
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

        # Reference result: moving avg speed (cyan)
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
            is_ref = name == REF_MOVING

            # All estimators use moving-time basis
            delta_col = "delta_moving_s"
            ata_col = "ata_moving_s"
            err_title = "ETA \u2212 moving ATA (min)"
            pct_title = "Moving ETA error (%)"

            # Build combined error DataFrames with series column
            def _build_error_frames(delta_c, ata_c):
                frames_min = []
                frames_pct = []
                series = []
                colors = []
                opacities = []
                widths = []
                if not is_ref:
                    for ref_prep, ref_color, ref_name in [
                        (ref_moving_prepped, REF_MOVING_COLOR, "Avg moving"),
                    ]:
                        if ref_prep is not None:
                            frames_min.append(
                                ref_prep[["elapsed_min"]].assign(
                                    value=ref_prep[delta_c] / 60,
                                    series=ref_name,
                                )
                            )
                            ata = ref_prep[ata_c]
                            pct = (ref_prep[delta_c] / ata).where(ata > 0) * 100
                            frames_pct.append(
                                ref_prep[["elapsed_min"]].assign(
                                    value=pct,
                                    series=ref_name,
                                )
                            )
                            series.append(ref_name)
                            colors.append(ref_color)
                            opacities.append(REF_OPACITY)
                            widths.append(1.0)
                # Main estimator
                frames_min.append(
                    result_prepped[["elapsed_min"]].assign(
                        value=result_prepped[delta_c] / 60,
                        series="Estimator",
                    )
                )
                ata = result_prepped[ata_c]
                pct = (result_prepped[delta_c] / ata).where(ata > 0) * 100
                frames_pct.append(
                    result_prepped[["elapsed_min"]].assign(
                        value=pct,
                        series="Estimator",
                    )
                )
                series.append("Estimator")
                colors.append(TOL_VIBRANT[5])
                opacities.append(1.0)
                widths.append(1.5)
                return frames_min, frames_pct, series, colors, opacities, widths

            (
                err_frames,
                pct_frames,
                err_series,
                err_colors,
                err_opacities,
                err_widths,
            ) = _build_error_frames(delta_col, ata_col)

            def _encoded_chart(frames, y_title, series, colors, opacities, widths):
                combined = pd.concat(frames, ignore_index=True)
                lines = (
                    alt.Chart(combined)
                    .mark_line(strokeWidth=1, invalid="break-paths-filter-domains")
                    .encode(
                        x=X_ELAPSED,
                        y=alt.Y("value:Q").title(y_title),
                        color=alt.Color("series:N")
                        .scale(domain=series, range=colors)
                        .legend(None),
                        opacity=alt.Opacity("series:N")
                        .scale(domain=series, range=opacities)
                        .legend(None),
                        strokeWidth=alt.StrokeWidth("series:N")
                        .scale(domain=series, range=widths)
                        .legend(None),
                    )
                )
                labels = _end_labels(combined, "value", series, colors)
                return lines + labels

            error_chart = alt.layer(
                pause_bands(ride_prepped, pause_df=pause_df),
                _encoded_chart(
                    err_frames,
                    err_title,
                    err_series,
                    err_colors,
                    err_opacities,
                    err_widths,
                ),
                error_refs(),
            ).properties(width=chart_width, height=chart_height)

            error_pct_chart = alt.layer(
                pause_bands(ride_prepped, pause_df=pause_df),
                _encoded_chart(
                    pct_frames,
                    pct_title,
                    err_series,
                    err_colors,
                    err_opacities,
                    err_widths,
                ),
                error_pct_refs(),
            ).properties(width=chart_width, height=chart_height)

            # Avg speed overview: estimated + actual cumulative averages
            speed_chart = alt.layer(
                pause_bands(ride_prepped, pause_df=pause_df),
                avg_speed_overview(
                    ride_prepped,
                    ref_moving_df=ref_moving_prepped,
                ),
            ).properties(width=chart_width, height=chart_height)

            # Speed residual: raw (predicted - actual)
            raw_residual_chart = alt.layer(
                pause_bands(ride_prepped, pause_df=pause_df),
                speed_residual_raw(ride_prepped, result_prepped),
            ).properties(width=chart_width, height=chart_height)

            # Speed residual: 60s smoothed
            smooth_residual_chart = alt.layer(
                pause_bands(ride_prepped, pause_df=pause_df),
                speed_residual_smoothed(ride_prepped, result_prepped),
            ).properties(width=chart_width, height=chart_height)

            chart = (
                (
                    error_chart
                    & error_pct_chart
                    & speed_chart
                    & raw_residual_chart
                    & smooth_residual_chart
                )
                .resolve_scale(
                    color="independent",
                    opacity="independent",
                    strokeWidth="independent",
                    strokeDash="independent",
                )
                .properties(title=alt.Title(f"{name} \u2014 {ride.label}"))
            )
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
        mv = compute_metrics(result, warmup_m, moving_only=True)
        row[f"{col}_mae"] = mv["mae_min"]
        row[f"{col}_rmse"] = mv["rmse_min"]
        row[f"{col}_mpe"] = mv["mpe_pct"]
        row[f"{col}_mape"] = mv["mape_pct"]
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
        choices=["mae", "rmse", "mpe", "mape"],
        default=None,
        metavar="METRIC",
        help="Metrics to display (default: all)",
    )
    args = parser.parse_args()

    if args.metrics is None:
        args.metrics = ["mae", "rmse", "mpe", "mape"]

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
