"""Evaluate v_flat estimator convergence across rides."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from godot.estimators import (
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
    "Weighted gain (default)": WeightedGainVFlat(),
    "Weighted gain (fast)": WeightedGainVFlat(tau_s=15, lambda_slow=0.005),
    "Weighted gain (slow)": WeightedGainVFlat(tau_s=60, lambda_slow=0.001),
}

# Checkpoints in moving minutes
CHECKPOINTS_MIN = [2, 5, 10, 20, 30]


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

    # Overall moving average
    total_dist = df.loc[moving, "delta_distance"].sum()
    total_time = df.loc[moving, "delta_time"].sum()
    moving_avg = (total_dist / total_time * 3.6) if total_time > 0 else np.nan

    # Flat-only moving average (|gradient| < 2%)
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
            "v_flat_final": round(final_vflat, 1),
            "moving_avg": round(truths["moving_avg"], 1),
            "flat_avg": round(truths["flat_avg"], 1)
            if not np.isnan(truths["flat_avg"])
            else None,
            "stability_std": round(stability_std, 2),
        }

        # Error vs each ground truth at each checkpoint
        for cp_min in CHECKPOINTS_MIN:
            mask = cum_moving_s <= cp_min * 60
            if mask.any():
                idx = cum_moving_s[mask].index[-1]
                vf = v_flat_kmh.loc[idx]
            else:
                vf = GLOBAL_PRIOR_KMH

            row[f"vflat_{cp_min}min"] = round(vf, 1)

            for gt_name in ["moving_avg", "flat_avg"]:
                gt = truths[gt_name]
                if not np.isnan(gt) and gt > 0:
                    row[f"err_{gt_name}_{cp_min}min"] = round((vf - gt) / gt * 100, 1)
                else:
                    row[f"err_{gt_name}_{cp_min}min"] = None

        # Final error vs ground truths
        for gt_name in ["moving_avg", "flat_avg"]:
            gt = truths[gt_name]
            if not np.isnan(gt) and gt > 0:
                row[f"err_{gt_name}_final"] = round((final_vflat - gt) / gt * 100, 1)
            else:
                row[f"err_{gt_name}_final"] = None

        rows.append(row)

    return rows


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
    results = pd.DataFrame(all_rows)

    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 300)
    pd.set_option("display.float_format", "{:.1f}".format)

    # --- Convergence table: v_flat at checkpoints ---
    print("\n=== v_flat at checkpoints (km/h) ===\n")
    cp_cols = ["ride", "route_type", "estimator", "moving_avg", "flat_avg"]
    cp_cols += [f"vflat_{m}min" for m in CHECKPOINTS_MIN]
    cp_cols += ["v_flat_final", "stability_std"]
    print(results[cp_cols].to_string(index=False))

    # --- Summary: mean absolute error vs each ground truth ---
    print("\n=== Mean absolute % error vs ground truths (by estimator) ===\n")
    for gt_name, gt_label in [("moving_avg", "Moving Avg"), ("flat_avg", "Flat Avg")]:
        print(f"--- vs {gt_label} ---")
        err_cols = [f"err_{gt_name}_{m}min" for m in CHECKPOINTS_MIN]
        err_cols.append(f"err_{gt_name}_final")

        summary = results.groupby("estimator")[err_cols].agg(
            ["mean", "median", lambda x: x.abs().mean()]
        )
        # Flatten multi-level columns
        summary.columns = [
            f"{col}|{'mean' if stat == 'mean' else 'median' if stat == 'median' else 'MAE'}"
            for col, stat in summary.columns
        ]
        # Just show MAE columns
        mae_cols = [c for c in summary.columns if "MAE" in c]
        mae_display = summary[mae_cols].copy()
        mae_display.columns = [
            c.split("|")[0].replace(f"err_{gt_name}_", "") for c in mae_cols
        ]
        print(mae_display.to_string())
        print()
