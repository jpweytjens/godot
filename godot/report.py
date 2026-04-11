"""HTML report generation for ETA backtest results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_METRIC_META = {
    "mae": {"suffix": "_mae", "label": "MAE", "fmt": "{:.2f}"},
    "rmse": {"suffix": "_rmse", "label": "RMSE", "fmt": "{:.2f}"},
    "mpe": {"suffix": "_mpe", "label": "MPE%", "fmt": "{:.1f}"},
    "mape": {"suffix": "_mape", "label": "MAPE%", "fmt": "{:.1f}"},
    "mov_mae": {"suffix": "_mov_mae", "label": "Mov MAE", "fmt": "{:.2f}"},
    "mov_mpe": {"suffix": "_mov_mpe", "label": "Mov MPE%", "fmt": "{:.1f}"},
    "mov_mape": {"suffix": "_mov_mape", "label": "Mov MAPE%", "fmt": "{:.1f}"},
}

_INFO_COLS = [
    "ride",
    "distance_method",
    "speed_smoothed",
    "route_type",
    "contains_pauses",
]

_INFO_RENAME = {
    "ride": "Ride",
    "distance_method": "Distance",
    "speed_smoothed": "Smoothed",
    "route_type": "Type",
    "contains_pauses": "Pauses",
}

_STYLES = [
    {
        "selector": "",
        "props": (
            "border-collapse:collapse;"
            "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "font-size:12.5px;"
        ),
    },
    {"selector": "th,td", "props": "padding:5px 12px; text-align:right;"},
    {"selector": "thead", "props": "border-bottom:1px solid #888;"},
    {"selector": "tbody tr:nth-child(even)", "props": "background:#f0f4f8;"},
    {
        "selector": "caption",
        "props": "caption-side:top; font-weight:bold; text-align:left; padding-bottom:6px;",
    },
]

_BOLD = "font-weight: bold"


def _highlight_min(s: pd.Series) -> list[str]:
    return [_BOLD if v == s.min() else "" for v in s]


def _highlight_min_abs(s: pd.Series) -> list[str]:
    return [_BOLD if abs(v) == s.abs().min() else "" for v in s]


def write_html_report(
    results_df: pd.DataFrame,
    selected_metrics: list[str],
    output_path: Path,
) -> None:
    """Write a styled HTML report of backtest metrics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Raw results with info columns and per-estimator metric columns
        (e.g. ``adaptive_lerp_mae``, ``average_speed_(total)_mpe``).
    selected_metrics : list[str]
        Which metrics to include (e.g. ``["mae", "mpe", "mape"]``).
    output_path : Path
        Where to write the HTML file.
    """
    # --- Discover estimator names from columns ---
    all_names = {
        c[: -len(s)]
        for m in _METRIC_META.values()
        for s in [m["suffix"]]
        for c in results_df.columns
        if c.endswith(s)
    }
    col_to_name = {k: k.replace("_", " ").title() for k in all_names}

    # --- Select and order columns ---
    metric_groups: dict[str, list[str]] = {}
    for m in selected_metrics:
        suffix = _METRIC_META[m]["suffix"]
        metric_groups[m] = [c for c in results_df.columns if c.endswith(suffix)]
    metric_cols = [c for m in selected_metrics for c in metric_groups[m]]
    results_df = results_df[_INFO_COLS + metric_cols]

    # --- Rename columns for display ---
    rename_map: dict[str, str] = dict(_INFO_RENAME)
    for m in selected_metrics:
        meta = _METRIC_META[m]
        for c in metric_groups[m]:
            est_key = c[: -len(meta["suffix"])]
            rename_map[c] = f"{col_to_name[est_key]} {meta['label']}"
    display_df = results_df.rename(columns=rename_map)

    # --- Global averages ---
    # Pivot: one row per estimator, one column per metric
    est_keys = sorted(col_to_name.keys())
    global_rows = []
    for ek in est_keys:
        row: dict[str, object] = {"Estimator": col_to_name[ek]}
        for m in selected_metrics:
            meta = _METRIC_META[m]
            col = ek + meta["suffix"]
            if col in results_df.columns:
                row[meta["label"]] = results_df[col].mean()
        global_rows.append(row)
    global_avg = pd.DataFrame(global_rows).set_index("Estimator")
    sort_col = "MAE" if "mae" in selected_metrics else global_avg.columns[0]
    global_avg = global_avg.sort_values(sort_col)

    # --- Aggregations ---
    agg_rename = {c: rename_map[c] for c in metric_cols}
    by_type_df = (
        results_df.groupby("route_type")[metric_cols].mean().rename(columns=agg_rename)
    )
    by_type_pauses_df = (
        results_df.groupby(["route_type", "contains_pauses"])[metric_cols]
        .mean()
        .rename(columns=agg_rename)
    )
    by_pipeline_df = (
        results_df.groupby(["distance_method", "speed_smoothed"])[metric_cols]
        .mean()
        .rename(columns=agg_rename)
    )

    # --- Format maps ---
    display_groups: dict[str, list[str]] = {
        m: [rename_map[c] for c in metric_groups[m]] for m in selected_metrics
    }
    fmt_map = {
        c: _METRIC_META[m]["fmt"] for m in selected_metrics for c in display_groups[m]
    }
    global_fmt = {
        _METRIC_META[m]["label"]: _METRIC_META[m]["fmt"] for m in selected_metrics
    }

    # --- Styled HTML tables ---
    per_ride_styler = display_df.style
    for m in selected_metrics:
        cols = pd.Index(display_groups[m])
        fn = _highlight_min_abs if m == "mpe" else _highlight_min
        per_ride_styler = per_ride_styler.apply(fn, axis=1, subset=cols)
    per_ride_html = (
        per_ride_styler.format(fmt_map)
        .set_table_styles(_STYLES)
        .set_caption("Per-ride metrics")
        .hide(axis="index")
        .to_html()
    )

    global_styler = global_avg.style
    for m in selected_metrics:
        label = _METRIC_META[m]["label"]
        if label not in global_avg.columns:
            continue
        fn = _highlight_min_abs if "mpe" in m else _highlight_min
        global_styler = global_styler.apply(fn, axis=0, subset=[label])
    global_html = (
        global_styler.format(global_fmt)
        .set_table_styles(_STYLES)
        .set_caption("Global averages")
        .to_html()
    )

    by_type_html = (
        by_type_df.style.format(fmt_map)
        .set_table_styles(_STYLES)
        .set_caption("Averages by route type")
        .to_html()
    )
    by_type_pauses_html = (
        by_type_pauses_df.style.format(fmt_map)
        .set_table_styles(_STYLES)
        .set_caption("Averages by route type × pauses")
        .to_html()
    )
    by_pipeline_html = (
        by_pipeline_df.style.format(fmt_map)
        .set_table_styles(_STYLES)
        .set_caption("Averages by pipeline")
        .to_html()
    )

    # --- Write HTML ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:-apple-system,sans-serif;margin:2em;color:#222;}"
        "h2{margin-top:2em;font-size:1.05em;color:#555;border-bottom:1px solid #ddd;padding-bottom:4px;}"
        "</style></head><body>"
        f"<h2>Per-ride metrics</h2>{per_ride_html}"
        f"<h2>Global averages</h2>{global_html}"
        f"<h2>Averages by route type</h2>{by_type_html}"
        f"<h2>Averages by route type × pauses</h2>{by_type_pauses_html}"
        f"<h2>Averages by pipeline</h2>{by_pipeline_html}"
        "</body></html>"
    )
    print(f"Saved {output_path}")
