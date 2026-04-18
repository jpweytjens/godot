"""Oracle decomposition table for method_d.pdf.

Computes three oracle baselines across the ride corpus:

1. **Oracle naive** (v_true) — true whole-ride moving average, no physics.
   Measures the within-ride pace variability floor.
2. **Oracle physics** (prior v_flat) — physics model with perfect split
   corrections from row 0, using the configured v_flat prior.
3. **Oracle physics + oracle v_flat** — same, but the ratio table is
   built at the ride's true flat-section average speed.

All metrics are MAE of predicted remaining *moving* time, excluding
the first 2 % of ride distance (warmup).

Run:  uv run python scripts/oracle_table.py data/strava/*.gpx data/strava/*.fit
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from godot.config import RideConfig
from godot.estimators import (
    _row_gradients,
    very_realistic_physics_ratios,
)
from godot.pcs import classify_by_max_climb
from godot.ride import load_ride
from godot.ttg import segment_ttg_from_row


CFG = RideConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _oracle_v_flat_ms(ride) -> float:
    """True flat-section moving average speed (m/s)."""
    df = ride.df
    moving = ~df["paused"]
    gradients, _ = _row_gradients(ride)
    flat = moving & (gradients.abs() < 0.02)
    d = df.loc[flat, "delta_distance"].sum()
    t = df.loc[flat, "delta_time"].sum()
    return d / t if t > 0 else CFG.v_flat_kmh / 3.6


def _v_true_ms(ride) -> float:
    """True whole-ride moving average speed (m/s)."""
    return ride.distance / ride.ride_time


def _build_ratio_table(v_flat_ms: float) -> dict[int, float]:
    """Build the behavioral physics ratio table at a given v_flat."""
    return very_realistic_physics_ratios(
        mass_kg=CFG.rider_mass_kg + CFG.bike_mass_kg,
        v_flat_ms=v_flat_ms,
        cda=CFG.cda,
        crr=CFG.crr,
        rho=CFG.rho,
        headwind_ms=CFG.headwind_kmh / 3.6,
        climb_effort=CFG.climb_effort,
        p_max_multiplier=CFG.p_max_multiplier,
    )


def _seg_ratio_array(segments, ratios: dict[int, float]) -> np.ndarray:
    """Per-segment speed ratio array, clamped to known bins."""
    min_bin, max_bin = min(ratios), max(ratios)
    return np.array(
        [
            ratios.get(max(min_bin, min(max_bin, math.floor(s.gradient * 100))), 1.0)
            for s in segments
        ]
    )


def _remaining_moving_time(ride) -> np.ndarray:
    """Actual remaining moving time (seconds) at each row."""
    df = ride.df
    moving_dt = df["delta_time"].where(~df["paused"], 0.0).values
    cum_moving = np.cumsum(moving_dt)
    return cum_moving[-1] - cum_moving


def _warmup_mask(ride, warmup_frac: float = 0.02) -> np.ndarray:
    """Boolean mask: True for rows past the warmup distance."""
    distances = ride.df["distance_m"].values
    return distances >= warmup_frac * ride.distance


def _mae_minutes(
    predicted_s: np.ndarray, actual_s: np.ndarray, mask: np.ndarray
) -> float:
    """MAE in minutes over masked rows."""
    err = np.abs(predicted_s[mask] - actual_s[mask])
    return float(np.nanmean(err)) / 60.0


# ---------------------------------------------------------------------------
# Oracle predictions
# ---------------------------------------------------------------------------


def oracle_naive(ride) -> np.ndarray:
    """Remaining time = remaining_distance / v_true."""
    v_true = _v_true_ms(ride)
    remaining_dist = ride.distance - ride.df["distance_m"].values
    return remaining_dist / v_true


def oracle_physics(ride, v_flat_ms: float) -> np.ndarray:
    """Physics model with perfect split corrections, given a specific v_flat.

    1. Build ratio table at v_flat_ms.
    2. Compute per-segment speeds = v_flat * ratio(g).
    3. Compute TTG from each row (split into climb/descent).
    4. Run the ride forward to get final Delta_c and Delta_d.
    5. Re-compute TTG using those final corrections from row 0.
    """
    segments = ride.gradient_segments
    df = ride.df
    distances = df["distance_m"].values
    total_dist = ride.distance
    dt = df["delta_time"].values
    dd = df["delta_distance"].values
    speed = df["speed_ms"].values
    paused = df["paused"].values

    ratios = _build_ratio_table(v_flat_ms)
    seg_ratios = _seg_ratio_array(segments, ratios)
    is_climb_seg = np.array([s.gradient >= 0 for s in segments])

    # Per-segment speeds
    climb_speeds = np.where(is_climb_seg, v_flat_ms * seg_ratios, np.inf)
    descent_speeds = np.where(~is_climb_seg, v_flat_ms * seg_ratios, np.inf)

    # --- Compute final Delta_c, Delta_d by running the full ride ---
    gradients, _ = _row_gradients(ride)
    row_ratios = _seg_ratio_array_per_row(ride, ratios)

    cum_pred_climb = 0.0
    cum_actual_climb = 0.0
    cum_pred_descent = 0.0
    cum_actual_descent = 0.0

    for i in range(len(df)):
        if paused[i]:
            continue
        pred_speed = v_flat_ms * row_ratios[i]
        if pred_speed <= 0:
            continue

        pred_dt = dd[i] / pred_speed
        actual_dt = dt[i]
        g = gradients.iloc[i]

        if g >= 0:
            cum_pred_climb += pred_dt
            cum_actual_climb += actual_dt
        else:
            cum_pred_descent += pred_dt
            cum_actual_descent += actual_dt

    delta_c = cum_pred_climb / cum_actual_climb if cum_actual_climb > 0 else 1.0
    delta_d = cum_pred_descent / cum_actual_descent if cum_actual_descent > 0 else 1.0

    # --- Predict with perfect corrections from row 0 ---
    climb_ttg = segment_ttg_from_row(distances, total_dist, segments, climb_speeds)
    descent_ttg = segment_ttg_from_row(distances, total_dist, segments, descent_speeds)

    with np.errstate(divide="ignore", invalid="ignore"):
        ttg = climb_ttg / delta_c + descent_ttg / delta_d

    return ttg


def _seg_ratio_array_per_row(ride, ratios: dict[int, float]) -> np.ndarray:
    """Per-row speed ratio based on the row's gradient."""
    gradients, _ = _row_gradients(ride)
    min_bin, max_bin = min(ratios), max(ratios)
    grad_pct = np.clip(np.floor(gradients.values * 100).astype(int), min_bin, max_bin)
    return np.array([ratios.get(int(g), 1.0) for g in grad_pct])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-ride results into a summary table by difficulty."""
    order = ["pancake", "rolling", "minor_hills", "hills", "mountains"]
    cats = [c for c in order if c in results["difficulty"].values]

    summary_rows = []
    for cat in cats:
        sub = results[results["difficulty"] == cat]
        summary_rows.append(
            {
                "difficulty": cat,
                "rides": len(sub),
                "naive": sub["oracle_naive"].mean(),
                "physprior": sub["oracle_physics_prior_vflat"].mean(),
                "physoracle": sub["oracle_physics_oracle_vflat"].mean(),
            }
        )

    return pd.DataFrame(summary_rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ride_files", nargs="+", type=Path, help="GPX, FIT, or FIT.gz files."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write summary CSV to this path (for LaTeX \\pgfplotstableread).",
    )
    args = parser.parse_args()

    gpx_paths = sorted(args.ride_files)

    rows = []
    for path in gpx_paths:
        try:
            ride = load_ride(path)
        except Exception as e:
            print(f"  Skipping {path.name}: {e}", file=sys.stderr)
            continue
        if ride.distance < 10_000:
            continue

        clf = classify_by_max_climb(ride.gradient_segments)

        ata = _remaining_moving_time(ride)
        mask = _warmup_mask(ride)

        v_flat_prior = CFG.v_flat_kmh / 3.6
        v_flat_oracle = _oracle_v_flat_ms(ride)

        mae_naive = _mae_minutes(oracle_naive(ride), ata, mask)
        mae_physics_prior = _mae_minutes(oracle_physics(ride, v_flat_prior), ata, mask)
        mae_physics_oracle = _mae_minutes(
            oracle_physics(ride, v_flat_oracle), ata, mask
        )

        rows.append(
            {
                "ride": ride.name,
                "difficulty": clf.difficulty,
                "v_true_kmh": _v_true_ms(ride) * 3.6,
                "v_flat_oracle_kmh": v_flat_oracle * 3.6,
                "oracle_naive": mae_naive,
                "oracle_physics_prior_vflat": mae_physics_prior,
                "oracle_physics_oracle_vflat": mae_physics_oracle,
            }
        )

    results = pd.DataFrame(rows)
    summary = _summarise(results)

    # --- Console output ---
    print("\nOracle decomposition: MAE (minutes) by route difficulty")
    print("=" * 75)
    print(
        f"{'Difficulty':<14} {'n':>4}  {'Naive':>8}  {'Phys+prior':>10}  {'Phys+oracle':>11}"
    )
    print(
        f"{'':14} {'':4}  {'(v_true)':>8}  {'(v_flat=cfg)':>10}  {'(v_flat=true)':>11}"
    )
    print("-" * 75)

    for _, row in summary.iterrows():
        print(
            f"{row['difficulty']:<14} {int(row['rides']):>4}  "
            f"{row['naive']:>8.1f}  "
            f"{row['physprior']:>10.1f}  "
            f"{row['physoracle']:>11.1f}"
        )

    print("-" * 75)
    n = len(results)
    print(
        f"{'All':<14} {n:>4}  "
        f"{results['oracle_naive'].mean():>8.1f}  "
        f"{results['oracle_physics_prior_vflat'].mean():>10.1f}  "
        f"{results['oracle_physics_oracle_vflat'].mean():>11.1f}"
    )
    print(f"\nTotal rides: {n} (>10 km filter)")

    # --- CSV output ---
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.csv, index=False, float_format="%.1f")
        print(f"\nSummary CSV written to {args.csv}")


if __name__ == "__main__":
    main()
