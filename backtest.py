"""Run ETA estimators against GPX files and report accuracy metrics."""

from pathlib import Path

import altair as alt
import pandas as pd
from tqdm.contrib.concurrent import process_map

from godot.benchmark import backtest, compute_metrics, compute_ride_divergence
from godot.config import RideConfig
from godot.estimators import (
    AvgSpeedEstimator,
    AdaptiveGradientPriorEstimator,
    BinnedAdaptiveEstimator,
    TrustedBinnedAdaptiveEstimator,
    WeightedGainVFlat,
    GradientPriorEstimator,
    PhysicsGradientPriorEstimator,
    RealisticPhysicsEstimator,
    CalibratingPhysicsEstimator,
    PIPhysicsEstimator,
    CalibratedFtpPhysicsEstimator,
    CalibratedPhysicsEstimator,
    CalibratedSplitPhysicsEstimator,
    CalibratedVerySplitPhysicsEstimator,
    DynamicSplitPhysicsEstimator,
    DynamicVerySplitPhysicsEstimator,
    PauseResetSplitIntegralPhysicsEstimator,
    PauseResetVerySplitIntegralPhysicsEstimator,
    WarmupDynamicSplitPhysicsEstimator,
    WarmupDynamicVerySplitPhysicsEstimator,
    IntegralPhysicsEstimator,
    PriorFreeFtpPhysicsEstimator,
    PriorFreePhysicsEstimator,
    RelevantDynamicSplitPhysicsEstimator,
    RelevantIntegralDynamicSplitPhysicsEstimator,
    RelevantSplitIntegralPhysicsEstimator,
    SplitIntegralPhysicsEstimator,
    VeryRealisticPhysicsEstimator,
    VerySplitIntegralPhysicsEstimator,
    QuadIntegralPhysicsEstimator,
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
    plot_resample_freq,
    prep_time_axis,
    speed_residual_raw,
    speed_residual_smoothed,
)
from godot.theme import TOL_VIBRANT
from godot.ride import load_ride
from godot.theme import TOL_BRIGHT

REF_MOVING = "L0_none_moving_average"
REF_MOVING_COLOR = TOL_BRIGHT[4]  # cyan
REF_OPACITY = 0.7

CFG = RideConfig()


# Estimator naming convention: L{level}_{vflat_source}_{name}
# Each estimator exposes its `key` as a property — see godot.estimators.BaseEstimator.
# `vflat_source`: fixed_vflat | online_<strategy> | oracle_vflat | none
_ESTIMATOR_INSTANCES = [
    AvgSpeedEstimator(moving_only=True),
    GradientPriorEstimator(CFG),
    PhysicsGradientPriorEstimator(CFG),
    RealisticPhysicsEstimator(CFG),
    VeryRealisticPhysicsEstimator(CFG),
    AdaptiveGradientPriorEstimator(CFG, vflat_estimator=WeightedGainVFlat()),
    PriorFreePhysicsEstimator(CFG),
    PriorFreeFtpPhysicsEstimator(CFG),
    CalibratedPhysicsEstimator(CFG),
    CalibratedFtpPhysicsEstimator(CFG),
    CalibratedSplitPhysicsEstimator(CFG),
    CalibratedVerySplitPhysicsEstimator(CFG),
    DynamicSplitPhysicsEstimator(CFG),
    DynamicVerySplitPhysicsEstimator(CFG),
    PauseResetSplitIntegralPhysicsEstimator(CFG),
    PauseResetVerySplitIntegralPhysicsEstimator(CFG),
    WarmupDynamicSplitPhysicsEstimator(CFG),
    WarmupDynamicVerySplitPhysicsEstimator(CFG),
    BinnedAdaptiveEstimator(CFG, prior=GradientPriorEstimator(CFG)),
    BinnedAdaptiveEstimator(CFG, prior=PhysicsGradientPriorEstimator(CFG)),
    TrustedBinnedAdaptiveEstimator(CFG, prior=GradientPriorEstimator(CFG)),
    TrustedBinnedAdaptiveEstimator(CFG, prior=PhysicsGradientPriorEstimator(CFG)),
    TrustedBinnedAdaptiveEstimator(CFG, prior=RealisticPhysicsEstimator(CFG)),
    CalibratingPhysicsEstimator(CFG),
    PIPhysicsEstimator(CFG),
    IntegralPhysicsEstimator(CFG),
    SplitIntegralPhysicsEstimator(CFG),
    VerySplitIntegralPhysicsEstimator(CFG),
    QuadIntegralPhysicsEstimator(CFG),
    RelevantSplitIntegralPhysicsEstimator(CFG),
    RelevantIntegralDynamicSplitPhysicsEstimator(CFG),
    RelevantDynamicSplitPhysicsEstimator(CFG),
]
ESTIMATORS = {est.key: (est, WallClockPause()) for est in _ESTIMATOR_INSTANCES}


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

        # Prepare ride data for plotting (with warmup clipped, then downsampled).
        # Coarsen the resample on long rides so ultras don't drown the renderer.
        freq = plot_resample_freq(ride.total_time)
        ride_prepped = downsample_for_plot(
            prep_time_axis(ride.df, warmup_pct=0.02), freq=freq
        )
        pause_df = _pause_intervals(ride_prepped)
        bands_layer = pause_bands(ride_prepped, pause_df=pause_df)

        # Reference result: moving avg speed (cyan)
        ref_moving = results.get(REF_MOVING)
        ref_moving_prepped = (
            downsample_for_plot(prep_time_axis(ref_moving, warmup_pct=0.02), freq=freq)
            if ref_moving is not None
            else None
        )

        # Per-estimator: 5-panel stack
        for name, result in results.items():
            result_prepped = downsample_for_plot(
                prep_time_axis(result, warmup_pct=0.02), freq=freq
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
                bands_layer,
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
                bands_layer,
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
                bands_layer,
                avg_speed_overview(
                    ride_prepped,
                    ref_moving_df=ref_moving_prepped,
                ),
            ).properties(width=chart_width, height=chart_height)

            # Speed residual: raw (predicted - actual)
            raw_residual_chart = alt.layer(
                bands_layer,
                speed_residual_raw(ride_prepped, result_prepped),
            ).properties(width=chart_width, height=chart_height)

            # Speed residual: 60s smoothed
            smooth_residual_chart = alt.layer(
                bands_layer,
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
            chart.save(str(out_dir / f"{safe_name}.svg"))

        # Comparison: all estimators' ETA error on one chart
        comp_chart = alt.layer(
            bands_layer,
            comparison_errors(results, warmup_pct=0.02, freq=freq),
            error_refs(),
        ).properties(
            title=alt.Title(f"All estimators \u2014 {ride.label}"),
            width=chart_width,
            height=comparison_height,
        )
        comp_chart.save(str(out_dir / "comparison.svg"))

    row: dict = {
        "ride": ride.name,
        "distance_method": distance_method,
        "speed_smoothed": smooth_speed,
        "route_type": ride.route_type,
        "contains_pauses": ride.contains_pauses,
    }
    div = compute_ride_divergence(ride, CFG.realistic_ratios)
    row["divergence_max_pct"] = div["divergence_max_pct"]
    row["divergence_mean_pct"] = div["divergence_mean_pct"]
    row["divergence_time_above_pct_s"] = div["divergence_time_above_pct_s"]
    warmup_m = ride.distance * 0.02
    for name, result in results.items():
        col = name.lower().replace(" ", "_")
        mv = compute_metrics(result, warmup_m, moving_only=True)
        row[f"{col}_mae"] = mv["mae_min"]
        row[f"{col}_rmse"] = mv["rmse_min"]
        row[f"{col}_mpe"] = mv["mpe_pct"]
        row[f"{col}_mape"] = mv["mape_pct"]
        row[f"{col}_settle"] = mv["settle_min"]
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
        choices=["mae", "rmse", "mpe", "mape", "settle"],
        default=None,
        metavar="METRIC",
        help="Metrics to display (default: all)",
    )
    args = parser.parse_args()

    if args.metrics is None:
        args.metrics = ["mae", "rmse", "mpe", "mape", "settle"]

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
