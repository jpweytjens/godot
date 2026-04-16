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
    AvgSpeedEstimator,
    EmpiricalPowerRelevantSplitEstimator,
    RelevantCalibratedSplitPhysicsEstimator,
    RelevantSplitIntegralPhysicsEstimator,
    SplitIntegralPhysicsEstimator,
)
from godot.fit import read_fit
from godot.gpx import add_haversine_distance
from godot.segmentation import RouteSegment, decimate_to_gradient_segments
from godot.pause import WallClockPause
from godot.pcs import classify_by_max_climb
from godot.ride import load_ride


def _score_one(ride, estimator) -> dict:
    result = backtest(ride, estimator, pause_strategy=WallClockPause())
    warmup_m = ride.distance * 0.02
    m = compute_metrics(result, warmup_m, moving_only=True)
    return {
        "mae": m["mae_min"],
        "mape": m["mape_pct"],
        "mpe": m["mpe_pct"],
        "rmse": m["rmse_min"],
        "settle": m["settle_min"],
    }


def score(estimator, rides) -> list[dict]:
    """Return one metric dict per ride, so results can be grouped."""
    return [_score_one(ride, estimator) for ride in rides]


def _single_ride_p_curve(path: Path, min_samples: int = 30) -> dict[int, float] | None:
    """Compute P(g)/P(0) from a single FIT file."""
    if path.suffix.lower() != ".fit":
        return None
    try:
        df = read_fit(path)
        if "watts" not in df.columns or df["watts"].notna().sum() < 200:
            return None
        df = df.pipe(add_haversine_distance)
        clean = df.dropna(subset=["distance_m", "elevation_m"])
        if len(clean) < 200:
            return None
        _, segs = decimate_to_gradient_segments(clean)
        if not segs:
            return None

        starts = np.array([s.start_distance_m for s in segs])
        grads = np.array([s.gradient for s in segs])
        idx = np.searchsorted(starts, df["distance_m"].to_numpy(), side="right") - 1
        idx = np.clip(idx, 0, len(segs) - 1)
        gradient = grads[idx]

        mask = (df["watts"] > 0) & (df["speed_ms"] > 1.0) & df["watts"].notna()
        flat_mask = mask & (np.abs(gradient) < 0.01)
        if flat_mask.sum() < 50:
            return None
        p_flat = df.loc[flat_mask, "watts"].median()
        if not (50 < p_flat < 500):
            return None

        grad_pct = np.round(gradient[mask.values] * 100).astype(int)
        p_ratio = (df.loc[mask, "watts"] / p_flat).to_numpy()
        result: dict[int, float] = {}
        for g in range(-15, 16):
            in_bin = grad_pct == g
            if in_bin.sum() >= min_samples:
                result[g] = float(np.median(p_ratio[in_bin]))
        return result if len(result) >= 3 else None
    except Exception:
        return None


def score_per_ride_empirical(
    rides: list,
    paths: list[Path],
    cfg: RideConfig,
    min_relevant_s: float,
    fallback_est,
) -> list[dict]:
    """Score using per-ride P(g)/P(0) curves, falling back when unavailable."""
    out = []
    for ride, path in zip(rides, paths):
        p_curve = _single_ride_p_curve(path)
        if p_curve:
            est = EmpiricalPowerRelevantSplitEstimator(
                cfg, p_curve=p_curve, min_relevant_s=min_relevant_s
            )
        else:
            est = fallback_est
        out.append(_score_one(ride, est))
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


def _compute_p_curve(paths: list[Path]) -> dict[int, float] | None:
    """Compute empirical P(g)/P(0) from FIT files with power data."""
    fit_paths = [p for p in paths if p.suffix.lower() == ".fit"]
    if not fit_paths:
        return None

    print(f"computing P(g)/P(0) from {len(fit_paths)} FIT files...")
    all_grad_pct = []
    all_p_ratio = []
    for p in fit_paths:
        try:
            df = read_fit(p)
            if "watts" not in df.columns or df["watts"].notna().sum() < 200:
                continue
            df = df.pipe(add_haversine_distance)
            clean = df.dropna(subset=["distance_m", "elevation_m"])
            if len(clean) < 200:
                continue
            _, segs = decimate_to_gradient_segments(clean)
            if not segs:
                continue

            starts = np.array([s.start_distance_m for s in segs])
            grads = np.array([s.gradient for s in segs])
            idx = np.searchsorted(starts, df["distance_m"].to_numpy(), side="right") - 1
            idx = np.clip(idx, 0, len(segs) - 1)
            gradient = grads[idx]

            mask = (df["watts"] > 0) & (df["speed_ms"] > 1.0) & df["watts"].notna()
            flat_mask = mask & (np.abs(gradient) < 0.01)
            if flat_mask.sum() < 100:
                continue
            p_flat = df.loc[flat_mask, "watts"].median()
            if not (50 < p_flat < 500):
                continue

            sub = df[mask]
            grad_sub = gradient[mask.values]
            all_grad_pct.append(np.round(grad_sub * 100).astype(int))
            all_p_ratio.append((sub["watts"] / p_flat).to_numpy())
        except Exception:
            continue

    if not all_grad_pct:
        return None

    grad_arr = np.concatenate(all_grad_pct)
    ratio_arr = np.concatenate(all_p_ratio)
    result: dict[int, float] = {}
    for g in range(-15, 16):
        in_bin = grad_arr == g
        if in_bin.sum() >= 100:
            result[g] = float(np.median(ratio_arr[in_bin]))
    print(
        f"  P(g)/P(0) curve: {len(result)} bins, range [{min(result)}..{max(result)}]%"
    )
    return result


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

    p_curve = _compute_p_curve(paths)

    all_variants: list[tuple[str, str, object]] = []
    all_variants.append(("MovingAvg", "-", AvgSpeedEstimator(moving_only=True)))
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
            if p_curve:
                all_variants.append(
                    (
                        "EmpPower",
                        tag,
                        EmpiricalPowerRelevantSplitEstimator(
                            cfg, p_curve=p_curve, min_relevant_s=t
                        ),
                    )
                )

    # Per-ride empirical power: each ride uses its own P(g)/P(0) curve
    for v_flat_kmh in args.v_flat_priors:
        cfg = RideConfig(v_flat_kmh=v_flat_kmh)
        tag = f"v={v_flat_kmh:.0f}"
        for t in args.thresholds:
            fallback = RelevantSplitIntegralPhysicsEstimator(cfg, min_relevant_s=t)
            per_ride_results = score_per_ride_empirical(rides, paths, cfg, t, fallback)
            all_variants.append(("PerRidePow", tag, per_ride_results))

    rows_all = []
    rows_by_bucket: dict[str, list] = {b: [] for b in bucket_counts}
    for name, tag, est_or_results in all_variants:
        if isinstance(est_or_results, list):
            per_ride = est_or_results
        else:
            per_ride = score(est_or_results, rides)
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
