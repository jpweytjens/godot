"""Evaluate v_flat estimator convergence across rides."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from godot.estimators import (
    EwmaLockVFlat,
    FlatSpeedVFlat,
    MedianLockVFlat,
    VFlatEstimator,
    WeightedGainVFlat,
    _row_gradients,
)
from godot.ride import load_ride

# Load empirical gradient ratios
_ratio_df = pd.read_parquet(Path("data/gradient_ratios.parquet"))
GRADIENT_RATIOS: dict[int, float] = _ratio_df["mean_ratio"].to_dict()
GLOBAL_PRIOR_KMH = 28.8

ESTIMATORS: dict[str, VFlatEstimator] = {
    "Flat speed (2%)": FlatSpeedVFlat(max_grad=0.02),
    "Flat speed (5%)": FlatSpeedVFlat(max_grad=0.05),
    "Weighted gain (default)": WeightedGainVFlat(),
    "EWMA lock": EwmaLockVFlat(),
    "Median lock": MedianLockVFlat(),
}

# Checkpoints in moving minutes
CHECKPOINTS_MIN = [2, 5, 10, 20, 30]

# Ground truth targets
GROUND_TRUTHS = ["moving_avg", "flat_avg"]


def _ground_truths(ride, gradients):
    """Compute ground-truth v_flat targets for a ride.

    Returns
    -------
    dict with keys:
        'moving_avg': overall moving average speed (km/h)
        'flat_avg': moving average on flat sections |grad| < 2% (km/h)
    """
    df = ride.df
    moving = ~df["paused"]

    total_dist = df.loc[moving, "delta_distance"].sum()
    total_time = df.loc[moving, "delta_time"].sum()
    moving_avg = (total_dist / total_time * 3.6) if total_time > 0 else np.nan

    flat_mask = moving & (gradients.abs() < 0.02)
    flat_dist = df.loc[flat_mask, "delta_distance"].sum()
    flat_time = df.loc[flat_mask, "delta_time"].sum()
    flat_avg = (flat_dist / flat_time * 3.6) if flat_time > 0 else np.nan

    return {"moving_avg": moving_avg, "flat_avg": flat_avg}


def run_one(gpx_path: Path) -> list[dict]:
    """Run all v_flat estimators on one ride, return rows for the results table."""
    ride = load_ride(gpx_path, "haversine", smooth_speed=True, smooth_window="5s")
    df = ride.df
    moving = ~df["paused"]
    gradients, _ = _row_gradients(ride)

    truths = _ground_truths(ride, gradients)
    cum_moving_s = (moving.astype(float) * df["delta_time"]).cumsum()

    rows = []
    for est_name, estimator in ESTIMATORS.items():
        v_flat_series = estimator.estimate(
            ride, GRADIENT_RATIOS, GLOBAL_PRIOR_KMH / 3.6
        )
        v_flat_kmh = v_flat_series * 3.6

        final_vflat = v_flat_kmh.iloc[-1]

        # Stability: std of v_flat in last 50% of ride
        n = len(v_flat_kmh)
        tail = v_flat_kmh.iloc[n // 2 :]
        stability_std = tail.std()

        row = {
            "ride": gpx_path.stem,
            "route_type": ride.route_type,
            "estimator": est_name,
            "moving_avg": round(truths["moving_avg"], 1),
            "flat_avg": round(truths["flat_avg"], 1)
            if not np.isnan(truths["flat_avg"])
            else None,
            "stability_std": round(stability_std, 2),
        }

        # v_flat at each checkpoint
        for cp_min in CHECKPOINTS_MIN:
            mask = cum_moving_s <= cp_min * 60
            if mask.any():
                idx = cum_moving_s[mask].index[-1]
                vf = v_flat_kmh.loc[idx]
            else:
                vf = GLOBAL_PRIOR_KMH
            row[f"vflat_{cp_min}min"] = round(vf, 1)

        row["v_flat_final"] = round(final_vflat, 1)

        # Error vs each ground truth at each checkpoint + final
        for gt_name in GROUND_TRUTHS:
            gt = truths[gt_name]
            for cp_min in CHECKPOINTS_MIN:
                if not np.isnan(gt) and gt > 0:
                    vf = row[f"vflat_{cp_min}min"]
                    row[f"err_{gt_name}_{cp_min}min"] = round((vf - gt) / gt * 100, 1)
                else:
                    row[f"err_{gt_name}_{cp_min}min"] = None

            if not np.isnan(gt) and gt > 0:
                row[f"err_{gt_name}_final"] = round((final_vflat - gt) / gt * 100, 1)
            else:
                row[f"err_{gt_name}_final"] = None

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

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
        "props": (
            "caption-side:top; font-weight:bold; text-align:left; padding-bottom:6px;"
        ),
    },
]

_BOLD = "font-weight: bold"


def _highlight_min(s: pd.Series) -> list[str]:
    numeric = pd.to_numeric(s, errors="coerce")
    return [_BOLD if v == numeric.min() else "" for v in numeric]


def _highlight_min_abs(s: pd.Series) -> list[str]:
    numeric = pd.to_numeric(s, errors="coerce")
    return [_BOLD if abs(v) == numeric.abs().min() else "" for v in numeric]


def _style_table(df, caption, fmt="{:.1f}", highlight="min", hide_index=True):
    """Style a DataFrame as an HTML table."""
    numeric_cols = df.select_dtypes(include="number").columns
    fn = _highlight_min_abs if highlight == "min_abs" else _highlight_min
    styler = df.style
    if len(numeric_cols):
        styler = styler.apply(fn, axis=1, subset=numeric_cols)
        styler = styler.format({c: fmt for c in numeric_cols})
    styler = styler.set_table_styles(_STYLES).set_caption(caption)
    if hide_index:
        styler = styler.hide(axis="index")
    return styler.to_html()


def write_vflat_report(results: pd.DataFrame, output_path: Path) -> None:
    """Write v_flat convergence HTML report."""
    sections: list[str] = []
    estimators = results["estimator"].unique().tolist()

    # --- 0. Global summary: mean error per estimator ---
    for gt_name, gt_label in [("moving_avg", "Moving Avg"), ("flat_avg", "Flat Avg")]:
        err_cols = [f"err_{gt_name}_{m}min" for m in CHECKPOINTS_MIN]
        err_cols.append(f"err_{gt_name}_final")
        cp_labels = [f"{m}min" for m in CHECKPOINTS_MIN] + ["final"]

        summary_rows = []
        for stat_name, agg_fn in [
            ("MAE (mean)", lambda s: s.abs().mean()),
            ("MAE (median)", lambda s: s.abs().median()),
            ("Bias (mean)", lambda s: s.mean()),
        ]:
            row: dict[str, object] = {"Metric": stat_name}
            for est_name in estimators:
                sub = results[results["estimator"] == est_name]
                vals = [agg_fn(sub[c].dropna()) for c in err_cols]
                for label, v in zip(cp_labels, vals):
                    row[f"{est_name} {label}"] = v
            summary_rows.append(row)

        # Win count per checkpoint (lowest absolute error)
        win_row: dict[str, object] = {"Metric": "Wins (MAE)"}
        for col, label in zip(err_cols, cp_labels):
            pivot = results.pivot(index="ride", columns="estimator", values=col)
            abs_pivot = pivot.abs()
            # Drop rides where all estimators are NaN
            abs_pivot = abs_pivot.dropna(how="all")
            winners = abs_pivot.idxmin(axis=1)
            counts = winners.value_counts()
            for est_name in estimators:
                key = f"{est_name} {label}"
                win_row[key] = win_row.get(key, 0) + counts.get(est_name, 0)
        summary_rows.append(win_row)

        summary_df = pd.DataFrame(summary_rows).set_index("Metric")

        def _highlight_summary(s: pd.Series) -> list[str]:
            if "Wins" in s.name:
                numeric = pd.to_numeric(s, errors="coerce")
                return [_BOLD if v == numeric.max() else "" for v in numeric]
            if "Bias" in s.name:
                return _highlight_min_abs(s)
            return _highlight_min(s)

        html = (
            summary_df.style.apply(_highlight_summary, axis=1)
            .format("{:.1f}")
            .set_table_styles(_STYLES)
            .set_caption(f"Global summary vs {gt_label}")
            .to_html()
        )
        sections.append(f"<h2>Global summary vs {gt_label}</h2>")
        sections.append(html)

    # --- 1. Per-ride convergence: v_flat at checkpoints ---
    cp_cols = [f"vflat_{m}min" for m in CHECKPOINTS_MIN] + ["v_flat_final"]
    cp_rename = {f"vflat_{m}min": f"{m}min" for m in CHECKPOINTS_MIN}
    cp_rename["v_flat_final"] = "final"

    for est_name in estimators:
        sub = results[results["estimator"] == est_name].copy()
        display = sub[
            ["ride", "route_type", "moving_avg", "flat_avg"]
            + cp_cols
            + ["stability_std"]
        ].copy()
        display = display.sort_values(["route_type", "ride"]).rename(
            columns={**cp_rename, "stability_std": "std (tail)"}
        )
        sections.append(f"<h2>{est_name} — v_flat at checkpoints (km/h)</h2>")
        sections.append(_style_table(display, caption=est_name, fmt="{:.1f}"))

    # --- 2. Summary: MAE vs ground truths per estimator ---
    for gt_name, gt_label in [("moving_avg", "Moving Avg"), ("flat_avg", "Flat Avg")]:
        err_cols = [f"err_{gt_name}_{m}min" for m in CHECKPOINTS_MIN]
        err_cols.append(f"err_{gt_name}_final")

        # Pivot: estimators as rows, checkpoints as columns
        summary_rows = []
        for est_name in estimators:
            sub = results[results["estimator"] == est_name]
            row = {"Estimator": est_name}
            for col in err_cols:
                vals = sub[col].dropna()
                label = col.replace(f"err_{gt_name}_", "")
                row[f"{label} (MAE)"] = vals.abs().mean() if len(vals) else None
                row[f"{label} (bias)"] = vals.mean() if len(vals) else None
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        # MAE table
        mae_cols = ["Estimator"] + [c for c in summary_df.columns if "MAE" in c]
        mae_df = summary_df[mae_cols].set_index("Estimator")
        sections.append(f"<h2>MAE % vs {gt_label} (by estimator)</h2>")
        sections.append(
            _style_table(mae_df, caption=f"MAE % vs {gt_label}", hide_index=False)
        )

        # Bias table
        bias_cols = ["Estimator"] + [c for c in summary_df.columns if "bias" in c]
        bias_df = summary_df[bias_cols].set_index("Estimator")
        sections.append(f"<h2>Bias % vs {gt_label} (by estimator)</h2>")
        sections.append(
            _style_table(
                bias_df,
                caption=f"Bias % vs {gt_label}",
                highlight="min_abs",
                hide_index=False,
            )
        )

    # --- 3. Summary by route type ---
    for gt_name, gt_label in [("moving_avg", "Moving Avg"), ("flat_avg", "Flat Avg")]:
        err_cols = [f"err_{gt_name}_{m}min" for m in CHECKPOINTS_MIN]
        err_cols.append(f"err_{gt_name}_final")
        col_rename_rt = {c: c.replace(f"err_{gt_name}_", "") for c in err_cols}

        for est_name in estimators:
            sub = results[results["estimator"] == est_name].copy()
            agg = (
                sub.groupby("route_type")[err_cols]
                .agg(lambda x: x.abs().mean())
                .rename(columns=col_rename_rt)
            )
            sections.append(f"<h2>{est_name} — MAE % vs {gt_label} by route type</h2>")
            sections.append(_style_table(agg, caption=est_name, hide_index=False))

    # --- 4. Per-ride error tables (one per ground truth) ---
    for gt_name, gt_label in [("moving_avg", "Moving Avg"), ("flat_avg", "Flat Avg")]:
        err_cols = [f"err_{gt_name}_{m}min" for m in CHECKPOINTS_MIN]
        err_cols.append(f"err_{gt_name}_final")
        col_rename = {c: c.replace(f"err_{gt_name}_", "") for c in err_cols}

        for est_name in estimators:
            sub = results[results["estimator"] == est_name].copy()
            display = (
                sub[["ride", "route_type"] + err_cols]
                .sort_values(["route_type", "ride"])
                .rename(columns=col_rename)
            )
            sections.append(f"<h2>{est_name} — Error % vs {gt_label}</h2>")
            sections.append(
                _style_table(display, caption=est_name, highlight="min_abs")
            )

    # --- Write ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:-apple-system,sans-serif;margin:2em;color:#222;}"
        "h2{margin-top:2em;font-size:1.05em;color:#555;"
        "border-bottom:1px solid #ddd;padding-bottom:4px;}"
        "table{margin-bottom:1.5em;}"
        "</style></head><body>"
        "<h1 style='font-size:1.3em;'>v_flat Convergence Report</h1>"
        + "\n".join(sections)
        + "</body></html>"
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="v_flat estimator convergence backtest"
    )
    parser.add_argument(
        "paths", nargs="*", type=Path, help="GPX files (default: data/gpx/*.gpx)"
    )
    args = parser.parse_args()

    paths = [
        p.resolve() for p in (args.paths or sorted(Path("data/gpx").glob("*.gpx")))
    ]
    if not paths:
        parser.error("No GPX files found.")

    all_rows_nested = process_map(
        run_one, paths, chunksize=1, desc="v_flat backtest", unit="ride"
    )
    all_rows = [r for batch in all_rows_nested for r in batch]
    results_df = pd.DataFrame(all_rows)

    write_vflat_report(results_df, Path("output") / "vflat_results.html")
