"""Run ETA estimators against GPX files and report accuracy metrics."""

from pathlib import Path

import altair as alt
import pandas as pd
from tqdm.contrib.concurrent import process_map

from eta.benchmark import backtest, compute_metrics
from eta.estimators import AvgSpeedEstimator, RollingAvgSpeedEstimator
from eta.pause import NoPause, SubtractElapsed
from eta.plot import (
    comparison_errors,
    error_refs,
    eta_error,
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
    "Average speed (moving)": (AvgSpeedEstimator(moving_only=True), SubtractElapsed()),
    "Average speed (total)": (AvgSpeedEstimator(moving_only=False), NoPause()),
    # "Rolling 1 min (moving)": (RollingAvgSpeedEstimator(window_s=60, moving_only=True), SubtractElapsed()),
    # "Rolling 1 min (total)": (RollingAvgSpeedEstimator(window_s=60, moving_only=False), NoPause()),
    "Rolling 5 min (moving)": (
        RollingAvgSpeedEstimator(window_s=300, moving_only=True),
        SubtractElapsed(),
    ),
    "Rolling 5 min (total)": (
        RollingAvgSpeedEstimator(window_s=300, moving_only=False),
        NoPause(),
    ),
    # "Rolling 10 min (moving)": (RollingAvgSpeedEstimator(window_s=600, moving_only=True), SubtractElapsed()),
    # "Rolling 10 min (total)": (RollingAvgSpeedEstimator(window_s=600, moving_only=False), NoPause()),
    "Rolling 30 min (moving)": (
        RollingAvgSpeedEstimator(window_s=1800, moving_only=True),
        SubtractElapsed(),
    ),
    "Rolling 30 min (total)": (
        RollingAvgSpeedEstimator(window_s=1800, moving_only=False),
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

    # Per-estimator: 1x2 (ETA error | speed), time domain
    for name, result in results.items():
        result_prepped = prep_time_axis(result, warmup_pct=0.02)

        error_chart = alt.layer(
            pause_bands(ride.pauses),
            eta_error(result_prepped),
            error_refs(),
        ).properties(width=800, height=200)

        speed_chart = alt.layer(
            pause_bands(ride.pauses),
            speed_comparison(ride_prepped, result_prepped),
        ).properties(width=800, height=200)

        chart = (error_chart & speed_chart).properties(
            title=alt.Title(f"{name} \u2014 {ride.label}")
        )
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        chart.save(str(out_dir / f"{safe_name}.png"), scale_factor=2)

    # Comparison: all estimators' ETA error on one chart
    comp_chart = alt.layer(
        pause_bands(ride.pauses),
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

    col_to_name = {name.lower().replace(" ", "_"): name for name in ESTIMATORS}
    info_cols = [
        "ride",
        "distance_method",
        "speed_smoothed",
        "route_type",
        "contains_pauses",
    ]
    mae_cols = [c for c in results_df.columns if c.endswith("_mae")]
    rmse_cols = [c for c in results_df.columns if c.endswith("_rmse")]
    metric_cols = mae_cols + rmse_cols

    # Reorder: info | all MAE | all RMSE
    results_df = results_df[info_cols + metric_cols]

    print("\n--- Per-ride metrics ---")
    print(results_df.to_string(index=False, float_format="{:.2f}".format))

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    rename_map = {
        "ride": "Ride",
        "distance_method": "Distance",
        "speed_smoothed": "Smoothed",
        "route_type": "Type",
        "contains_pauses": "Pauses",
        **{c: f"{col_to_name[c[:-4]]} MAE" for c in mae_cols},
        **{c: f"{col_to_name[c[:-5]]} RMSE" for c in rmse_cols},
    }
    display_df = results_df.rename(columns=rename_map)
    display_mae = [rename_map[c] for c in mae_cols]
    display_rmse = [rename_map[c] for c in rmse_cols]
    display_metric = display_mae + display_rmse

    # Left border on first RMSE column to visually separate the two groups
    first_rmse_pos = _N_INFO_COLS + len(display_mae) + 1  # nth-child is 1-indexed
    sep_style = [
        {
            "selector": f"td:nth-child({first_rmse_pos}), th:nth-child({first_rmse_pos})",
            "props": "border-left: 1px solid #ccc;",
        }
    ]

    global_avg = pd.DataFrame(
        {
            "MAE": results_df[mae_cols].mean().rename(lambda c: col_to_name[c[:-4]]),
            "RMSE": results_df[rmse_cols].mean().rename(lambda c: col_to_name[c[:-5]]),
        }
    )
    by_type_df = (
        results_df.groupby("route_type")[metric_cols]
        .mean()
        .rename(
            columns={
                **{c: f"{col_to_name[c[:-4]]} MAE" for c in mae_cols},
                **{c: f"{col_to_name[c[:-5]]} RMSE" for c in rmse_cols},
            }
        )
    )

    print("\n--- Global averages ---")
    print(global_avg.to_string(float_format="{:.2f}".format))

    print("\n--- Averages by route type ---")
    print(by_type_df.to_string(float_format="{:.2f}".format))

    by_pipeline_df = (
        results_df.groupby(["distance_method", "speed_smoothed"])[metric_cols]
        .mean()
        .rename(
            columns={
                **{c: f"{col_to_name[c[:-4]]} MAE" for c in mae_cols},
                **{c: f"{col_to_name[c[:-5]]} RMSE" for c in rmse_cols},
            }
        )
    )

    print("\n--- Averages by pipeline ---")
    print(by_pipeline_df.to_string(float_format="{:.2f}".format))

    # --- HTML output ---
    per_ride_html = (
        display_df.style.highlight_min(
            axis=1, subset=pd.Index(display_mae), props="font-weight: bold"
        )
        .highlight_min(axis=1, subset=pd.Index(display_rmse), props="font-weight: bold")
        .format({c: "{:.2f}" for c in display_metric})
        .set_table_styles(_TABLE_STYLES + sep_style)
        .set_caption("Per-ride metrics")
        .hide(axis="index")
        .to_html()
    )
    global_html = (
        global_avg.style.format("{:.2f}")
        .set_table_styles(_TABLE_STYLES)
        .set_caption("Global averages")
        .to_html()
    )
    by_type_html = (
        by_type_df.style.format("{:.2f}")
        .set_table_styles(_TABLE_STYLES + sep_style)
        .set_caption("Averages by route type")
        .to_html()
    )
    by_pipeline_html = (
        by_pipeline_df.style.format("{:.2f}")
        .set_table_styles(_TABLE_STYLES + sep_style)
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
