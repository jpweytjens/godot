"""Run ETA estimators against GPX files and report accuracy metrics."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.contrib.concurrent import process_map

from benchmark import backtest
from estimators import AvgSpeedEstimator, RollingAvgSpeedEstimator
from gpx import (
    add_haversine_distance,
    add_integrated_distance,
    add_smooth_speed,
    read_gpx,
)
from plot import plot_comparison, plot_delta

_N_INFO_COLS = 4  # ride, distance_method, speed_smoothed, route_type

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
    "AvgSpeed": AvgSpeedEstimator(),
    # "Rolling 1min": RollingAvgSpeedEstimator(window_s=60),
    "Rolling 5min": RollingAvgSpeedEstimator(window_s=300),
    # "Rolling 10min": RollingAvgSpeedEstimator(window_s=600),
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


_DISTANCE_PIPES = {
    "haversine": add_haversine_distance,
    "integrated": add_integrated_distance,
}


def run(
    gpx_path: Path,
    distance_method: str = "haversine",
    smooth_speed: bool = True,
) -> dict:
    """Run all estimators against a single GPX file, save plots, return metrics.

    Parameters
    ----------
    gpx_path : Path
        Path to the GPX file to backtest.
    distance_method : str, optional
        Distance pipeline to apply: ``"haversine"`` (default) or ``"integrated"``.
    smooth_speed : bool, optional
        Whether to apply the 5s rolling speed smoother. Default True.

    Returns
    -------
    dict
        Row dict with keys: ride, distance_method, speed_smoothed, route_type,
        and per-estimator MAE/RMSE columns.
    """
    if distance_method not in _DISTANCE_PIPES:
        raise ValueError(f"distance_method must be one of {list(_DISTANCE_PIPES)}")
    df = read_gpx(gpx_path).pipe(_DISTANCE_PIPES[distance_method])
    if smooth_speed:
        df = add_smooth_speed(df)
    else:
        df = df.assign(speed_kmh=df["speed_ms"] * 3.6)
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
            result,
            f"{name} \u2014 {ride_name}",
            ax=axes[i],
            ride_df=df,
            warmup_pct=0.02,
        )
    plot_comparison(
        results,
        f"All estimators \u2014 {ride_name}",
        ax=axes[-1],
        ride_df=df,
        warmup_pct=0.02,
    )
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    plt.tight_layout()
    out = out_dir / f"backtest_{ride_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()

    row: dict = {
        "ride": ride_name,
        "distance_method": distance_method,
        "speed_smoothed": smooth_speed,
        "route_type": route_type,
    }
    warmup_cutoff = df["distance_m"].iloc[-1] * 0.02
    for name, result in results.items():
        trimmed = result[result["distance_m"] >= warmup_cutoff]["delta_s"].dropna()
        col = name.lower().replace(" ", "_")
        row[f"{col}_mae"] = trimmed.abs().mean() / 60
        row[f"{col}_rmse"] = (trimmed**2).mean() ** 0.5 / 60
    return row


if __name__ == "__main__":
    from itertools import product

    paths = [Path(p) for p in sys.argv[1:]] or list(Path("data").glob("*.gpx"))
    if not paths:
        print(
            "No GPX files found. Place .gpx files in data/ or pass paths as arguments."
        )
        sys.exit(1)

    combos = list(product(paths, ["haversine", "integrated"], [True, False]))
    rows = process_map(
        run,
        [c[0] for c in combos],
        [c[1] for c in combos],
        [c[2] for c in combos],
        desc="Backtesting",
        unit="run",
    )
    results_df = (
        pd.DataFrame(rows)
        .sort_values(["route_type", "ride", "distance_method", "speed_smoothed"])
        .reset_index(drop=True)
    )

    col_to_name = {name.lower().replace(" ", "_"): name for name in ESTIMATORS}
    info_cols = ["ride", "distance_method", "speed_smoothed", "route_type"]
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
