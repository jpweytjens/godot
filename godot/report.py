"""HTML report generation for ETA backtest results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_METRIC_META = {
    "mae": {"suffix": "_mae", "label": "MAE (min)", "fmt": "{:.2f}"},
    "rmse": {"suffix": "_rmse", "label": "RMSE (min)", "fmt": "{:.2f}"},
    "mpe": {"suffix": "_mpe", "label": "MPE %", "fmt": "{:.1f}"},
    "mape": {"suffix": "_mape", "label": "MAPE %", "fmt": "{:.1f}"},
    "mov_mae": {"suffix": "_mov_mae", "label": "Mov MAE (min)", "fmt": "{:.2f}"},
    "mov_rmse": {"suffix": "_mov_rmse", "label": "Mov RMSE (min)", "fmt": "{:.2f}"},
    "mov_mpe": {"suffix": "_mov_mpe", "label": "Mov MPE %", "fmt": "{:.1f}"},
    "mov_mape": {"suffix": "_mov_mape", "label": "Mov MAPE %", "fmt": "{:.1f}"},
}

# Metrics applicable to each time basis
_WALLCLOCK_METRICS = {"mae", "rmse", "mpe", "mape"}
_MOVING_METRICS = {"mov_mae", "mov_rmse", "mov_mpe", "mov_mape"}

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


def _highlight_max(s: pd.Series) -> list[str]:
    return [_BOLD if v == s.max() else "" for v in s]


def _highlight_min_abs(s: pd.Series) -> list[str]:
    return [_BOLD if abs(v) == s.abs().min() else "" for v in s]


def _discover_estimators(
    results_df: pd.DataFrame, selected_metrics: list[str]
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Discover estimator keys and build column mappings.

    Returns
    -------
    col_to_name : dict[str, str]
        Estimator key (e.g. ``"adaptive_lerp"``) → display name.
    metric_cols : dict[str, dict[str, str]]
        metric key → {estimator_key: column_name}.
    """
    suffixes = sorted(
        [_METRIC_META[m]["suffix"] for m in selected_metrics], key=len, reverse=True
    )
    all_names: set[str] = set()
    for c in results_df.columns:
        for s in suffixes:
            if c.endswith(s):
                all_names.add(c[: -len(s)])
                break
    col_to_name = {k: k.replace("_", " ").title() for k in sorted(all_names)}

    metric_cols: dict[str, dict[str, str]] = {}
    for m in selected_metrics:
        suffix = _METRIC_META[m]["suffix"]
        metric_cols[m] = {
            ek: ek + suffix for ek in col_to_name if ek + suffix in results_df.columns
        }
    return col_to_name, metric_cols


def _pivot_metric(
    results_df: pd.DataFrame,
    metric_key: str,
    col_to_name: dict[str, str],
    metric_cols: dict[str, str],
    info_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build a per-ride table for a single metric.

    Rows = rides, columns = estimators.
    """
    cols = list(info_cols or [])
    rename = {}
    for ek, col in metric_cols.items():
        cols.append(col)
        rename[col] = col_to_name[ek]
    return results_df[cols].rename(columns={**_INFO_RENAME, **rename})


def _style_metric_table(
    df: pd.DataFrame,
    metric_key: str,
    caption: str,
    fmt: str,
    estimator_names: list[str],
    hide_index: bool = True,
) -> str:
    """Apply formatting and highlighting to a metric table."""
    est_cols = pd.Index([c for c in df.columns if c in estimator_names])
    fn = _highlight_min_abs if "mpe" in metric_key else _highlight_min
    styler = df.style.apply(fn, axis=1, subset=est_cols)
    fmt_map = {c: fmt for c in est_cols}
    styler = styler.format(fmt_map).set_table_styles(_STYLES).set_caption(caption)
    if hide_index:
        styler = styler.hide(axis="index")
    return styler.to_html()


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
    col_to_name, metric_cols = _discover_estimators(results_df, selected_metrics)
    est_names = list(col_to_name.values())

    # --- Global summary: split by time basis ---
    def _global_summary(
        metrics: list[str],
        est_filter: set[str],
        caption: str,
    ) -> str:
        filtered_names = [n for ek, n in col_to_name.items() if ek in est_filter]
        if not filtered_names or not metrics:
            return ""
        rows = []
        for m in metrics:
            meta = _METRIC_META[m]
            is_signed = "mpe" in m and "mape" not in m
            mean_row: dict[str, object] = {"Metric": f"{meta['label']} (mean)"}
            median_row: dict[str, object] = {"Metric": f"{meta['label']} (median)"}
            filtered_cols = {
                ek: col for ek, col in metric_cols[m].items() if ek in est_filter
            }
            for ek, col in filtered_cols.items():
                name = col_to_name[ek]
                mean_row[name] = results_df[col].mean()
                median_row[name] = results_df[col].median()
            rows.extend([mean_row, median_row])

            if filtered_cols:
                cols_map = {col: col_to_name[ek] for ek, col in filtered_cols.items()}
                metric_df = results_df[list(cols_map.keys())].rename(columns=cols_map)
                if is_signed:
                    winners = metric_df.abs().idxmin(axis=1)
                else:
                    winners = metric_df.idxmin(axis=1)
                counts = winners.value_counts()
                win_row: dict[str, object] = {"Metric": f"{meta['label']} (wins)"}
                for name in filtered_names:
                    win_row[name] = counts.get(name, 0)
                rows.append(win_row)

        if not rows:
            return ""
        df = pd.DataFrame(rows).set_index("Metric")

        signed_labels = {
            _METRIC_META[m]["label"] for m in metrics if "mpe" in m and "mape" not in m
        }

        def _hl(s: pd.Series) -> list[str]:
            if "(wins)" in s.name:
                return _highlight_max(s)
            label = s.name.rsplit(" (", 1)[0]
            if label in signed_labels:
                return _highlight_min_abs(s)
            return _highlight_min(s)

        return (
            df.style.apply(_hl, axis=1)
            .format("{:.2f}")
            .set_table_styles(_STYLES)
            .set_caption(caption)
            .to_html()
        )

    # Partition estimators by time basis (T vs M) based on their key prefix
    t_estimators = {ek for ek in col_to_name if ek.split("_")[1:2] == ["t"]}
    m_estimators = {ek for ek in col_to_name if ek.split("_")[1:2] == ["m"]}

    wc_metrics = [m for m in selected_metrics if m in _WALLCLOCK_METRICS]
    mv_metrics = [m for m in selected_metrics if m in _MOVING_METRICS]

    global_wc_html = _global_summary(
        wc_metrics, t_estimators, "Wall-clock metrics (T estimators)"
    )
    global_mv_html = _global_summary(
        mv_metrics, m_estimators, "Moving-time metrics (M estimators)"
    )

    # --- Per-ride tables: one table per metric ---
    per_ride_tables: list[str] = []
    for m in selected_metrics:
        meta = _METRIC_META[m]
        df = _pivot_metric(
            results_df, m, col_to_name, metric_cols[m], info_cols=_INFO_COLS
        )
        html = _style_metric_table(
            df, m, caption=meta["label"], fmt=meta["fmt"], estimator_names=est_names
        )
        per_ride_tables.append(html)

    # --- Groupby tables: one table per metric per grouping ---
    def _grouped_tables(groupby_cols: list[str], caption_prefix: str) -> list[str]:
        tables = []
        for m in selected_metrics:
            meta = _METRIC_META[m]
            raw_cols = list(metric_cols[m].values())
            rename = {col: col_to_name[ek] for ek, col in metric_cols[m].items()}
            agg = (
                results_df.groupby(groupby_cols)[raw_cols].mean().rename(columns=rename)
            )
            est_cols = pd.Index([c for c in agg.columns if c in est_names])
            fn = _highlight_min_abs if "mpe" in m else _highlight_min
            html = (
                agg.style.apply(fn, axis=1, subset=est_cols)
                .format({c: meta["fmt"] for c in est_cols})
                .set_table_styles(_STYLES)
                .set_caption(f"{caption_prefix} — {meta['label']}")
                .to_html()
            )
            tables.append(html)
        return tables

    by_type_tables = _grouped_tables(["route_type"], "By route type")
    by_type_pauses_tables = _grouped_tables(
        ["route_type", "contains_pauses"], "By route type × pauses"
    )

    # --- Write HTML ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    if global_wc_html:
        sections.extend(["<h2>Wall-clock metrics (T estimators)</h2>", global_wc_html])
    if global_mv_html:
        sections.extend(["<h2>Moving-time metrics (M estimators)</h2>", global_mv_html])
    for html in per_ride_tables:
        sections.append(f"<h2>Per-ride</h2>{html}")
    for html in by_type_tables:
        sections.append(f"<h2>By route type</h2>{html}")
    for html in by_type_pauses_tables:
        sections.append(f"<h2>By route type × pauses</h2>{html}")

    output_path.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:-apple-system,sans-serif;margin:2em;color:#222;}"
        "h2{margin-top:2em;font-size:1.05em;color:#555;border-bottom:1px solid #ddd;padding-bottom:4px;}"
        "table{margin-bottom:1.5em;}"
        "</style></head><body>" + "\n".join(sections) + "</body></html>"
    )
    print(f"Saved {output_path}")
