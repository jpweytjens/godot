"""Derive empirical speed ratios per gradient bin from FIT ride data."""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm.contrib.concurrent import process_map

from godot.ride import Ride, load_ride
from godot.segmentation import decimate_to_gradient_segments

FINITE_EDGES = list(range(-20, 21, 1))
BIN_EDGES = [-np.inf] + FINITE_EDGES + [np.inf]
BIN_LABELS = [FINITE_EDGES[0] - 1] + FINITE_EDGES  # left-edge int per bin


def _safe_load_ride(path: Path) -> Ride | None:
    try:
        return load_ride(path, distance_method="integrated", smooth_speed=False)
    except Exception:
        logger.error(f"Failed to load {path.name}")
        return None


def _assign_gradient_bins(ride: Ride) -> pd.DataFrame:
    """Assign gradient bins to each trackpoint in a ride."""
    _, segments = decimate_to_gradient_segments(ride.df)

    ends = np.array([s.end_distance_m for s in segments])
    grads = np.array([s.gradient for s in segments])

    df = ride.df.copy()
    idx = np.searchsorted(ends, df["distance_m"].values, side="left")
    idx = idx.clip(max=len(segments) - 1)

    df["gradient_pct"] = grads[idx] * 100
    df["grad_bin"] = pd.cut(
        df["gradient_pct"], bins=BIN_EDGES, labels=BIN_LABELS, right=False
    )

    # Filter: moving only, above noise threshold
    return df[~df["paused"] & (df["speed_kmh"] >= 3.0)]


def ride_speed_ratios(ride: Ride) -> pd.Series | None:
    """Compute speed ratio per gradient bin for a single ride.

    Returns
    -------
    pd.Series or None
        Index = gradient bin (int), values = ratio to the flat (0) bin.
        None if the ride has no data in the base bin.
    """
    df = _assign_gradient_bins(ride)
    mean_speed = df.groupby("grad_bin", observed=True)["speed_kmh"].mean()

    if 0 not in mean_speed.index:
        return None

    base_speed = mean_speed.loc[0]
    return mean_speed / base_speed


def main():
    fit_files = sorted(Path("data/fit").glob("*.fit"))
    logger.info(f"Found {len(fit_files)} FIT files")

    results = process_map(_safe_load_ride, fit_files, desc="Loading rides", unit="file")
    rides = [r for r in results if r is not None]
    failed = len(results) - len(rides)
    if failed:
        logger.warning(f"{failed}/{len(results)} files failed to load")

    ratios = []
    bin_grad_means = []  # mean actual gradient per bin per ride
    for ride in rides:
        df = _assign_gradient_bins(ride)
        mean_speed = df.groupby("grad_bin", observed=True)["speed_kmh"].mean()
        if 0 not in mean_speed.index:
            continue
        base_speed = mean_speed.loc[0]
        ratio = mean_speed / base_speed
        ratio.index = ratio.index.astype(int)
        ratio.name = None
        ratios.append(ratio)
        gm = df.groupby("grad_bin", observed=True)["gradient_pct"].mean()
        gm.index = gm.index.astype(int)
        gm.name = None
        bin_grad_means.append(gm)

    logger.info(f"{len(ratios)}/{len(rides)} rides had data in the base bin")

    ratio_df = pd.DataFrame(ratios)
    summary = pd.DataFrame(
        {
            "mean_ratio": ratio_df.mean(),
            "std": ratio_df.std(),
            "count": ratio_df.count(),
        }
    ).sort_index()

    summary.to_parquet(Path("data/gradient_ratios.parquet"))
    logger.info("Saved to data/gradient_ratios.parquet")

    print("\n=== Gradient Bin Speed Ratios ===\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()
