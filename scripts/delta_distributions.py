"""Distribution of final Δ_c and Δ_d across the ride corpus.

If these values are tightly clustered per rider, cross-ride persistence
is viable. If they scatter widely, rider behavior is too variable
for a single learned correction to help.

Run:  uv run python scripts/delta_distributions.py data/strava/*.fit.gz data/strava/*.fit data/strava/*.gpx

Outputs:
  - output/delta_distributions.csv   — per-ride final corrections
  - output/delta_climb.html          — Altair histogram of Δ_c
  - output/delta_descent.html        — Altair histogram of Δ_d
  - output/delta_by_difficulty.html   — Δ_c and Δ_d faceted by difficulty
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

from godot.config import RideConfig
from godot.estimators import (
    _row_gradients,
    very_realistic_physics_ratios,
)
from godot.pcs import classify_by_max_climb
from godot.ride import load_ride

CFG = RideConfig()


def _build_ratio_table(v_flat_ms: float) -> dict[int, float]:
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


def _oracle_v_flat_ms(ride) -> float:
    df = ride.df
    moving = ~df["paused"]
    gradients, _ = _row_gradients(ride)
    flat = moving & (gradients.abs() < 0.02)
    d = df.loc[flat, "delta_distance"].sum()
    t = df.loc[flat, "delta_time"].sum()
    return d / t if t > 0 else CFG.v_flat_kmh / 3.6


def compute_final_deltas(ride, v_flat_ms: float) -> dict:
    """Run through a ride and return final Δ_c and Δ_d."""
    df = ride.df
    paused = df["paused"].values
    dt = df["delta_time"].values
    dd = df["delta_distance"].values
    speed = df["speed_ms"].values

    ratios = _build_ratio_table(v_flat_ms)
    gradients, _ = _row_gradients(ride)
    min_bin, max_bin = min(ratios), max(ratios)
    grad_pct = np.clip(np.floor(gradients.values * 100).astype(int), min_bin, max_bin)
    row_ratios = np.array([ratios.get(int(g), 1.0) for g in grad_pct])

    cum_pred_climb = 0.0
    cum_actual_climb = 0.0
    cum_pred_descent = 0.0
    cum_actual_descent = 0.0

    for i in range(len(df)):
        if paused[i] or dd[i] <= 0 or dt[i] <= 0:
            continue
        pred_speed = v_flat_ms * row_ratios[i]
        if pred_speed <= 0:
            continue

        pred_dt = dd[i] / pred_speed
        g = gradients.iloc[i]

        if g >= 0:
            cum_pred_climb += pred_dt
            cum_actual_climb += dt[i]
        else:
            cum_pred_descent += pred_dt
            cum_actual_descent += dt[i]

    delta_c = (
        cum_pred_climb / cum_actual_climb if cum_actual_climb > 0 else float("nan")
    )
    delta_d = (
        cum_pred_descent / cum_actual_descent
        if cum_actual_descent > 0
        else float("nan")
    )

    return {
        "delta_c": delta_c,
        "delta_d": delta_d,
        "climb_time_min": cum_actual_climb / 60,
        "descent_time_min": cum_actual_descent / 60,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ride_files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.ride_files):
        try:
            ride = load_ride(path)
        except Exception as e:
            print(f"  Skipping {path.name}: {e}", file=sys.stderr)
            continue
        if ride.distance < 10_000:
            continue

        clf = classify_by_max_climb(ride.gradient_segments)
        v_flat_oracle = _oracle_v_flat_ms(ride)
        deltas = compute_final_deltas(ride, v_flat_oracle)

        rows.append(
            {
                "ride": ride.name,
                "difficulty": clf.difficulty,
                "distance_km": ride.distance / 1000,
                "v_flat_kmh": v_flat_oracle * 3.6,
                **deltas,
            }
        )

    results = pd.DataFrame(rows)

    # Filter: need meaningful climb/descent time for Δ to be meaningful
    has_climb = results["climb_time_min"] > 5
    has_descent = results["descent_time_min"] > 5

    args.output.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output / "delta_distributions.csv", index=False)

    # --- Console summary ---
    order = ["pancake", "rolling", "minor_hills", "hills", "mountains"]
    print("\nFinal Δ_c and Δ_d by difficulty (rides with >5 min climb/descent time)")
    print("=" * 85)
    print(
        f"{'Difficulty':<14} {'n_c':>4}  {'Δ_c mean':>8} {'Δ_c std':>8} {'Δ_c med':>8}"
        f"  {'n_d':>4}  {'Δ_d mean':>8} {'Δ_d std':>8} {'Δ_d med':>8}"
    )
    print("-" * 85)

    for cat in order:
        sub_c = results[(results["difficulty"] == cat) & has_climb]
        sub_d = results[(results["difficulty"] == cat) & has_descent]
        if len(sub_c) == 0 and len(sub_d) == 0:
            continue
        nc = len(sub_c)
        nd = len(sub_d)
        print(
            f"{cat:<14} {nc:>4}  "
            f"{sub_c['delta_c'].mean():>8.3f} {sub_c['delta_c'].std():>8.3f} {sub_c['delta_c'].median():>8.3f}"
            f"  {nd:>4}  "
            f"{sub_d['delta_d'].mean():>8.3f} {sub_d['delta_d'].std():>8.3f} {sub_d['delta_d'].median():>8.3f}"
            if nc > 0 and nd > 0
            else f"{cat:<14} {nc:>4}  "
            + (
                f"{sub_c['delta_c'].mean():>8.3f} {sub_c['delta_c'].std():>8.3f} {sub_c['delta_c'].median():>8.3f}"
                if nc > 0
                else f"{'—':>8} {'—':>8} {'—':>8}"
            )
            + f"  {nd:>4}  "
            + (
                f"{sub_d['delta_d'].mean():>8.3f} {sub_d['delta_d'].std():>8.3f} {sub_d['delta_d'].median():>8.3f}"
                if nd > 0
                else f"{'—':>8} {'—':>8} {'—':>8}"
            )
        )

    print("-" * 85)
    sc = results[has_climb]
    sd = results[has_descent]
    print(
        f"{'All':<14} {len(sc):>4}  "
        f"{sc['delta_c'].mean():>8.3f} {sc['delta_c'].std():>8.3f} {sc['delta_c'].median():>8.3f}"
        f"  {len(sd):>4}  "
        f"{sd['delta_d'].mean():>8.3f} {sd['delta_d'].std():>8.3f} {sd['delta_d'].median():>8.3f}"
    )
    print(f"\nTotal rides: {len(results)}")

    # --- Altair charts ---
    # Melt for faceted view
    climb_df = results[has_climb][["ride", "difficulty", "delta_c"]].copy()
    climb_df.columns = ["ride", "difficulty", "delta"]
    climb_df["correction"] = "Δ_c (climb)"

    descent_df = results[has_descent][["ride", "difficulty", "delta_d"]].copy()
    descent_df.columns = ["ride", "difficulty", "delta"]
    descent_df["correction"] = "Δ_d (descent)"

    combined = pd.concat([climb_df, descent_df], ignore_index=True)

    # Histogram: Δ_c
    chart_c = (
        alt.Chart(climb_df)
        .mark_bar()
        .encode(
            x=alt.X("delta:Q", bin=alt.Bin(step=0.02), title="Δ_c (climb)"),
            y=alt.Y("count()", title="Rides"),
            color=alt.Color("difficulty:N", sort=order, title="Difficulty"),
        )
        .properties(
            width=600, height=300, title="Distribution of final Δ_c across corpus"
        )
    )
    chart_c.save(str(args.output / "delta_climb.html"))

    # Histogram: Δ_d
    chart_d = (
        alt.Chart(descent_df)
        .mark_bar()
        .encode(
            x=alt.X("delta:Q", bin=alt.Bin(step=0.02), title="Δ_d (descent)"),
            y=alt.Y("count()", title="Rides"),
            color=alt.Color("difficulty:N", sort=order, title="Difficulty"),
        )
        .properties(
            width=600, height=300, title="Distribution of final Δ_d across corpus"
        )
    )
    chart_d.save(str(args.output / "delta_descent.html"))

    # Faceted: both corrections by difficulty
    chart_faceted = (
        alt.Chart(combined)
        .mark_bar()
        .encode(
            x=alt.X("delta:Q", bin=alt.Bin(step=0.02), title="Correction factor"),
            y=alt.Y("count()", title="Rides"),
            color=alt.Color("difficulty:N", sort=order, title="Difficulty"),
        )
        .properties(width=500, height=200)
        .facet(row=alt.Row("correction:N", title=None))
    )
    chart_faceted.save(str(args.output / "delta_by_difficulty.html"))

    print(f"\nCharts saved to {args.output}/")


if __name__ == "__main__":
    main()
