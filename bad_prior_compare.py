"""Compare estimators across multiple v_flat priors to test prior sensitivity.

Usage:
    uv run python bad_prior_compare.py 20 22 28.8 40

Each prior is run as a separate corpus pass; results are then printed as
a side-by-side table grouped by estimator.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from functools import partial
from pathlib import Path

import pandas as pd
from tqdm.contrib.concurrent import process_map

from godot.benchmark import backtest, compute_metrics
from godot.config import RideConfig
from godot.estimators import (
    CalibratedFtpPhysicsEstimator,
    CalibratedPhysicsEstimator,
    CalibratedSplitPhysicsEstimator,
    CalibratedVerySplitPhysicsEstimator,
    CubicSplitIntegralPhysicsEstimator,
    DynamicSplitPhysicsEstimator,
    DynamicVerySplitPhysicsEstimator,
    PriorFreeFtpPhysicsEstimator,
    PriorFreePhysicsEstimator,
    SplitIntegralPhysicsEstimator,
    VerySplitIntegralPhysicsEstimator,
    WarmupDynamicSplitPhysicsEstimator,
    WarmupDynamicVerySplitPhysicsEstimator,
)
from godot.pause import WallClockPause
from godot.ride import load_ride

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _build_estimators(cfg: RideConfig):
    return [
        SplitIntegralPhysicsEstimator(cfg),
        VerySplitIntegralPhysicsEstimator(cfg),
        CubicSplitIntegralPhysicsEstimator(cfg),
        PriorFreePhysicsEstimator(cfg),
        PriorFreeFtpPhysicsEstimator(cfg),
        CalibratedPhysicsEstimator(cfg),
        CalibratedFtpPhysicsEstimator(cfg),
        CalibratedSplitPhysicsEstimator(cfg),
        CalibratedVerySplitPhysicsEstimator(cfg),
        DynamicSplitPhysicsEstimator(cfg),
        DynamicVerySplitPhysicsEstimator(cfg),
        WarmupDynamicSplitPhysicsEstimator(cfg),
        WarmupDynamicVerySplitPhysicsEstimator(cfg),
    ]


def run_one(path: Path, v_flat_kmh: float) -> dict:
    cfg = replace(RideConfig(), v_flat_kmh=v_flat_kmh)
    ride = load_ride(path, "integrated", smooth_speed=False)
    warmup_m = ride.distance * 0.02
    pause = WallClockPause()
    row = {"ride": ride.name}
    for est in _build_estimators(cfg):
        result = backtest(ride, est, pause)
        m = compute_metrics(result, warmup_m, moving_only=True)
        row[f"{est.key}_mae"] = m["mae_min"]
    return row


def _summary_for_prior(paths: list[Path], v_flat_kmh: float) -> dict[str, float]:
    rows = process_map(
        partial(run_one, v_flat_kmh=v_flat_kmh),
        paths,
        chunksize=1,
        desc=f"v_flat={v_flat_kmh}",
        unit="ride",
    )
    df = pd.DataFrame(rows)
    mae_cols = sorted(c for c in df.columns if c.endswith("_mae"))
    return {c[:-4]: df[c].dropna().mean() for c in mae_cols}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare estimators across multiple v_flat priors"
    )
    parser.add_argument(
        "priors",
        nargs="+",
        type=float,
        help="v_flat_kmh values to test (e.g. 20 22 28.8 40)",
    )
    args = parser.parse_args()

    paths = [p.resolve() for p in sorted(Path("data/gpx").glob("*.gpx"))] + [
        p.resolve() for p in sorted(Path("data/fit").glob("*.fit"))
    ]
    print(f"Corpus: {len(paths)} rides, sweeping {len(args.priors)} priors\n")

    results: dict[float, dict[str, float]] = {}
    for prior in args.priors:
        results[prior] = _summary_for_prior(paths, prior)

    # Side-by-side table
    estimators = sorted(results[args.priors[0]].keys())
    header_cols = "  ".join(f"{p:>7.1f}" for p in args.priors)
    print()
    print(f"{'estimator':75s}  {header_cols}")
    print("-" * (78 + len(header_cols)))
    for est in sorted(estimators, key=lambda e: results[args.priors[0]][e]):
        cells = "  ".join(f"{results[p][est]:7.2f}" for p in args.priors)
        print(f"{est:75s}  {cells}")
