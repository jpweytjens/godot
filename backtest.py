"""Run ETA estimators against GPX files and report accuracy metrics."""

from pathlib import Path

import altair as alt
import pandas as pd
from tqdm.contrib.concurrent import process_map

from eta.benchmark import backtest, compute_metrics
from eta.estimators import (
    AdaptiveLerpSpeedEstimator,
    AvgSpeedEstimator,
    DEWMASpeedEstimator,
    EWMASpeedEstimator,
    LerpSpeedEstimator,
    PriorEWMASpeedEstimator,
    RollingAvgSpeedEstimator,
    RollingMedianSpeedEstimator,
)
from eta.pause import NoPause, SubtractElapsed
from eta.plot import (
    comparison_errors,
    error_pct_refs,
    error_refs,
    eta_error,
    eta_error_pct,
    pause_bands,
    prep_time_axis,
    speed_comparison,
)
from eta.ride import load_ride

_N_INFO_COLS = 5  # ride, distance_method, speed_smoothed, route_type, contains_pauses

_TABLE_STYLES = [
    # Booktabs-style outer rules + clean font
    {
        "selector": "",
        "props": (
            "border-collapse: collapse;"
            "border-top: 2px solid #222;"
            "border-bottom: 2px solid #222;"
            "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;"
            "font-size: 12.5px;"
        ),
    },
    # Mid-rule below header
    {"selector": "thead", "props": "border-bottom: 1px solid #888;"},
    # No cell borders, comfortable padding
    {"selector": "th, td", "props": "padding: 5px 12px; border: none;"},
    # Numbers right-aligned, headers right-aligned
    {"selector": "td", "props": "text-align: right;"},
    {
        "selector": "th",
        "props": "text-align: right; font-weight: 600; background-color: #fff;",
    },
    # Info columns left-aligned
    {
        "selector": f"td:nth-child(-n+{_N_INFO_COLS}), th:nth-child(-n+{_N_INFO_COLS})",
        "props": "text-align: left;",
    },
    # Alternating row shading
    {"selector": "tbody tr:nth-child(even)", "props": "background-color: #f0f4f8;"},
    {"selector": "tbody tr:hover", "props": "background-color: #dce8f5;"},
    # Caption
    {
        "selector": "caption",
        "props": "caption-side: top; font-size: 14px; font-weight: bold; padding-bottom: 8px; text-align: left;",
    },
]

ESTIMATORS = {
    # "Average speed (moving)": (AvgSpeedEstimator(moving_only=True), SubtractElapsed()),
    "Average speed (total)": (AvgSpeedEstimator(moving_only=False), NoPause()),
    # "Rolling 5 min (moving)": (
    #     RollingAvgSpeedEstimator(window_s=300, moving_only=True),
    #     SubtractElapsed(),
    # ),
    # "Rolling 10 min (moving)": (
    #     RollingAvgSpeedEstimator(window_s=600, moving_only=True),
    #     SubtractElapsed(),
    # ),
    # "Rolling 30 min (moving)": (
    #     RollingAvgSpeedEstimator(window_s=1800, moving_only=True),
    #     SubtractElapsed(),
    # ),
    # "Rolling 60 min (moving)": (
    #     RollingAvgSpeedEstimator(window_s=3600, moving_only=True),
    #     SubtractElapsed(),
    # ),
    # "Rolling median 30 min (moving)": (
    #     RollingMedianSpeedEstimator(window_s=1800, moving_only=True),
    #     SubtractElapsed(),
    # ),
    # "EWMA 60 min (moving)": (
    #     EWMASpeedEstimator(span_s=3600, moving_only=True),
    #     SubtractElapsed(),
    # ),
    # "EWMA 10 min (moving)": (
    #     EWMASpeedEstimator(span_s=600, moving_only=True),
    #     SubtractElapsed(),
    # ),
    # "DEWMA 10+60 min (moving)": (
    #     DEWMASpeedEstimator(
    #         slow_span_s=3600,
    #         fast_span_s=600,
    #         slow_weight=0.7,
    #         fast_weight=0.3,
    #         moving_only=True,
    #     ),
    #     SubtractElapsed(),
    # ),
    # "EWMA 10 min + prior (moving)": (
    #     PriorEWMASpeedEstimator(
    #         span_s=600,
    #         prior_ms=28.8 / 3.6,
    #         moving_only=True,
    #     ),
    #     SubtractElapsed(),
    # ),
    # "Lerp (moving)": (
    #     LerpSpeedEstimator(
    #         prior_ms=28.8 / 3.6,
    #         fast_span_s=600,
    #         ramp_s=600,
    #         fast_weight=0.15,
    #         moving_only=True,
    #     ),
    #     SubtractElapsed(),
    # ),
    "Adaptive lerp": (
        AdaptiveLerpSpeedEstimator(
            prior_ms=28.8 / 3.6,
            tau=300,
            k=2.0,
            fast_span_s=3600,
            fast_weight=0.15,
        ),
        NoPause(),
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

    # Per-estimator: 1x3 (ETA error | MPE % | speed), time domain
    for name, result in results.items():
        result_prepped = prep_time_axis(result, warmup_pct=0.02)

        error_chart = alt.layer(
            pause_bands(ride_prepped),
            eta_error(result_prepped),
            error_refs(),
        ).properties(width=800, height=200)

        error_pct_chart = alt.layer(
            pause_bands(ride_prepped),
            eta_error_pct(result_prepped),
            error_pct_refs(),
        ).properties(width=800, height=200)

        speed_chart = alt.layer(
            pause_bands(ride_prepped),
            speed_comparison(ride_prepped, result_prepped),
        ).properties(width=800, height=200)

        chart = (error_chart & error_pct_chart & speed_chart).properties(
            title=alt.Title(f"{name} \u2014 {ride.label}")
        )
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
        metrics = compute_metrics(result, warmup_m)
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

    # --- Metric metadata ---
    _METRIC_META = {
        "mae": {"suffix": "_mae", "label": "MAE", "fmt": "{:.2f}"},
        "rmse": {"suffix": "_rmse", "label": "RMSE", "fmt": "{:.2f}"},
        "mpe": {"suffix": "_mpe", "label": "MPE%", "fmt": "{:.1f}"},
        "mape": {"suffix": "_mape", "label": "MAPE%", "fmt": "{:.1f}"},
    }
    selected_metrics = args.metrics

    col_to_name = {name.lower().replace(" ", "_"): name for name in ESTIMATORS}
    info_cols = [
        "ride",
        "distance_method",
        "speed_smoothed",
        "route_type",
        "contains_pauses",
    ]

    # Collect columns per metric group, in display order
    metric_groups: dict[str, list[str]] = {}
    for m in selected_metrics:
        suffix = _METRIC_META[m]["suffix"]
        metric_groups[m] = [c for c in results_df.columns if c.endswith(suffix)]
    metric_cols = [c for m in selected_metrics for c in metric_groups[m]]

    results_df = results_df[info_cols + metric_cols]

    print("\n--- Per-ride metrics ---")
    print(results_df.to_string(index=False, float_format="{:.2f}".format))

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    # --- Rename columns for display ---
    rename_map: dict[str, str] = {
        "ride": "Ride",
        "distance_method": "Distance",
        "speed_smoothed": "Smoothed",
        "route_type": "Type",
        "contains_pauses": "Pauses",
    }
    for m in selected_metrics:
        meta = _METRIC_META[m]
        for c in metric_groups[m]:
            est_key = c[: -len(meta["suffix"])]
            rename_map[c] = f"{col_to_name[est_key]} {meta['label']}"

    display_df = results_df.rename(columns=rename_map)
    display_groups: dict[str, list[str]] = {
        m: [rename_map[c] for c in metric_groups[m]] for m in selected_metrics
    }
    display_metric = [c for m in selected_metrics for c in display_groups[m]]

    # Format map: minutes get 2 decimals, percentages get 1
    fmt_map = {
        c: _METRIC_META[m]["fmt"] for m in selected_metrics for c in display_groups[m]
    }

    # Separator styles between metric groups
    sep_styles: list[dict] = []
    pos = _N_INFO_COLS + 1  # nth-child is 1-indexed
    for i, m in enumerate(selected_metrics):
        if i > 0:
            sep_styles.append(
                {
                    "selector": f"td:nth-child({pos}), th:nth-child({pos})",
                    "props": "border-left: 1px solid #ccc;",
                }
            )
        pos += len(display_groups[m])

    # --- Global averages (sorted by MAE if available) ---
    global_data: dict[str, pd.Series] = {}
    for m in selected_metrics:
        meta = _METRIC_META[m]
        cols = metric_groups[m]
        global_data[meta["label"]] = (
            results_df[cols]
            .mean()
            .rename(lambda c, s=meta["suffix"]: col_to_name[c[: -len(s)]])
        )
    global_avg = pd.DataFrame(global_data)
    sort_col = "MAE" if "mae" in selected_metrics else global_avg.columns[0]
    global_avg = global_avg.sort_values(sort_col)

    # --- By-route-type and by-pipeline aggregations ---
    agg_rename = {c: rename_map[c] for c in metric_cols}
    by_type_df = (
        results_df.groupby("route_type")[metric_cols].mean().rename(columns=agg_rename)
    )
    by_pipeline_df = (
        results_df.groupby(["distance_method", "speed_smoothed"])[metric_cols]
        .mean()
        .rename(columns=agg_rename)
    )

    print("\n--- Global averages ---")
    print(global_avg.to_string(float_format="{:.2f}".format))

    print("\n--- Averages by route type ---")
    print(by_type_df.to_string(float_format="{:.2f}".format))

    print("\n--- Averages by pipeline ---")
    print(by_pipeline_df.to_string(float_format="{:.2f}".format))

    # --- Highlighting helpers ---
    _BOLD = "font-weight: bold"

    def _highlight_min_all(s: pd.Series) -> list[str]:
        """Bold all values that equal the row/column minimum."""
        return [_BOLD if v == s.min() else "" for v in s]

    def _highlight_min_abs_all(s: pd.Series) -> list[str]:
        """Bold all values closest to zero (for signed metrics like MPE)."""
        return [_BOLD if abs(v) == s.abs().min() else "" for v in s]

    # --- HTML: per-ride table ---
    styler = display_df.style
    for m in selected_metrics:
        cols = pd.Index(display_groups[m])
        if m == "mpe":
            styler = styler.apply(_highlight_min_abs_all, axis=1, subset=cols)
        else:
            styler = styler.apply(_highlight_min_all, axis=1, subset=cols)
    per_ride_html = (
        styler.format(fmt_map)
        .set_table_styles(_TABLE_STYLES + sep_styles)
        .set_caption("Per-ride metrics")
        .hide(axis="index")
        .to_html()
    )

    # --- HTML: global averages (highlight best per column) ---
    global_fmt = {
        _METRIC_META[m]["label"]: _METRIC_META[m]["fmt"] for m in selected_metrics
    }
    global_styler = global_avg.style
    for m in selected_metrics:
        label = _METRIC_META[m]["label"]
        if m == "mpe":
            global_styler = global_styler.apply(
                _highlight_min_abs_all, axis=0, subset=[label]
            )
        else:
            global_styler = global_styler.apply(
                _highlight_min_all, axis=0, subset=[label]
            )
    global_html = (
        global_styler.format(global_fmt)
        .set_table_styles(_TABLE_STYLES)
        .set_caption("Global averages")
        .to_html()
    )

    by_type_html = (
        by_type_df.style.format(fmt_map)
        .set_table_styles(_TABLE_STYLES + sep_styles)
        .set_caption("Averages by route type")
        .to_html()
    )
    by_pipeline_html = (
        by_pipeline_df.style.format(fmt_map)
        .set_table_styles(_TABLE_STYLES + sep_styles)
        .set_caption("Averages by pipeline (distance method × speed smoothing)")
        .to_html()
    )

    html_path = out_dir / "results.html"
    html_path.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:2em;color:#222;}"
        "h2{margin-top:2.5em;font-size:1.05em;color:#555;border-bottom:1px solid #ddd;padding-bottom:4px;}"
        "</style></head><body>"
        f"<h2>Per-ride metrics</h2>{per_ride_html}"
        f"<h2>Global averages</h2>{global_html}"
        f"<h2>Averages by route type</h2>{by_type_html}"
        f"<h2>Averages by pipeline</h2>{by_pipeline_html}"
        "</body></html>"
    )
    print(f"Saved {html_path}")
