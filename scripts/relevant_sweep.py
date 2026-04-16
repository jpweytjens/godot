"""Sweep `min_relevant_s` for RelevantSplitIntegralPhysicsEstimator.

Runs each threshold against the GPX corpus (or a provided path glob)
and prints per-threshold mean/median of MAE, MAPE, MPE, RMSE, settle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from godot.benchmark import backtest, compute_metrics
from godot.config import RideConfig
from godot.estimators import (
    RelevantCalibratedSplitPhysicsEstimator,
    RelevantSplitIntegralPhysicsEstimator,
    SplitIntegralPhysicsEstimator,
)
from godot.pause import WallClockPause
from godot.pcs import classify_by_max_climb
from godot.ride import load_ride


def score(estimator, rides) -> list[dict]:
    """Return one metric dict per ride, so results can be grouped."""
    out = []
    for ride in rides:
        result = backtest(ride, estimator, pause_strategy=WallClockPause())
        warmup_m = ride.distance * 0.02
        m = compute_metrics(result, warmup_m, moving_only=True)
        out.append(
            {
                "mae": m["mae_min"],
                "mape": m["mape_pct"],
                "mpe": m["mpe_pct"],
                "rmse": m["rmse_min"],
                "settle": m["settle_min"],
            }
        )
    return out


def summary(rows: list[dict]) -> dict:
    if not rows:
        return {k: float("nan") for k in _SUMMARY_KEYS}
    arr = {
        k: np.array([r[k] for r in rows])
        for k in ("mae", "mape", "mpe", "rmse", "settle")
    }
    return {
        "n": len(rows),
        "mae_mean": np.nanmean(arr["mae"]),
        "mae_median": np.nanmedian(arr["mae"]),
        "mape_median": np.nanmedian(arr["mape"]),
        "mpe_median": np.nanmedian(arr["mpe"]),
        "rmse_median": np.nanmedian(arr["rmse"]),
        "settle_mean": np.nanmean(arr["settle"]),
    }


_SUMMARY_KEYS = (
    "n",
    "mae_mean",
    "mae_median",
    "mape_median",
    "mpe_median",
    "rmse_median",
    "settle_mean",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[120.0],
    )
    parser.add_argument(
        "--v-flat",
        nargs="+",
        type=float,
        default=[28.8],
        dest="v_flat_priors",
        metavar="KMH",
        help="v_flat prior(s) to sweep (default: 28.8)",
    )
    args = parser.parse_args()

    paths = args.paths or sorted(Path("data/gpx").glob("*.gpx"))
    print(f"loading {len(paths)} rides...")
    rides = []
    buckets: list[str] = []
    for p in paths:
        try:
            ride = load_ride(p, distance_method="integrated")
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        rides.append(ride)
        buckets.append(classify_by_max_climb(ride.gradient_segments).difficulty)
    print(f"{len(rides)} rides loaded")

    bucket_counts = pd.Series(buckets).value_counts().to_dict()
    print(f"difficulty: {bucket_counts}")
    print()

    all_variants: list[tuple[str, str, object]] = []
    for v_flat_kmh in args.v_flat_priors:
        cfg = RideConfig(v_flat_kmh=v_flat_kmh)
        tag = f"v={v_flat_kmh:.0f}"
        all_variants.append(("Split", tag, SplitIntegralPhysicsEstimator(cfg)))
        for t in args.thresholds:
            all_variants.append(
                (
                    "Relevant",
                    tag,
                    RelevantSplitIntegralPhysicsEstimator(cfg, min_relevant_s=t),
                )
            )
            all_variants.append(
                (
                    "RelCalib",
                    tag,
                    RelevantCalibratedSplitPhysicsEstimator(cfg, min_relevant_s=t),
                )
            )

    rows_all = []
    rows_by_bucket: dict[str, list] = {b: [] for b in bucket_counts}
    for name, tag, est in all_variants:
        per_ride = score(est, rides)
        rows_all.append({"estimator": name, "t_min_s": tag, **summary(per_ride)})
        for b in bucket_counts:
            sub = [r for r, bk in zip(per_ride, buckets) if bk == b]
            rows_by_bucket[b].append(
                {"estimator": name, "t_min_s": tag, **summary(sub)}
            )

    def _fmt(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype.kind == "f":
                df[col] = df[col].round(2)
        return df

    print("=== all rides ===")
    print(_fmt(pd.DataFrame(rows_all)).to_string(index=False))
    for b in sorted(
        bucket_counts, key=lambda x: ["flat", "hills", "mountains"].index(x)
    ):
        print()
        print(f"=== {b} ({bucket_counts[b]} rides) ===")
        print(_fmt(pd.DataFrame(rows_by_bucket[b])).to_string(index=False))


if __name__ == "__main__":
    main()
