"""Time-to-go (TTG) integration helpers.

A recurring pattern across gradient-aware estimators is: given a list
of route segments each with a constant speed estimate, compute the
remaining travel time from each row position to the end of the route.
`segment_ttg_from_row` precomputes TTG at segment boundaries in O(S),
then linearly deducts partial in-segment time at every row in O(N).

For estimators with row-varying multiplicative corrections:

- **Uniform scalar** (e.g. `IntegralPhysicsEstimator`): compute the base
  ttg once, then divide by the scalar row correction.
- **Two-group split** (e.g. `SplitIntegralPhysicsEstimator`): call this
  twice with climb-only and descent-only speed arrays (set the other
  group to `inf` so its segments contribute zero time), divide each by
  its group's per-row correction, sum.
- **Per-bin** (e.g. `BinnedAdaptiveEstimator`): correction varies per
  segment and per row, so the forward loop can't be amortized — fall
  back to the row-by-row loop in the estimator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from godot.segmentation import RouteSegment


def segment_ttg_from_row(
    distances: np.ndarray,
    total_dist: float,
    segments: list[RouteSegment],
    seg_speeds_ms: np.ndarray,
) -> np.ndarray:
    """Time-to-go (seconds) from each row distance to the end of route.

    Parameters
    ----------
    distances : np.ndarray
        Row distances along the route in meters, shape (N,).
    total_dist : float
        Total route distance in meters.
    segments : list[RouteSegment]
        Route segments (constant-gradient pieces).
    seg_speeds_ms : np.ndarray
        Effective speed in m/s for each segment, shape (S,). A segment
        with infinite speed contributes zero time (useful for masking
        out a group in split-correction estimators).

    Returns
    -------
    np.ndarray
        TTG in seconds, shape (N,). Zero at the end of the route.
    """
    seg_starts = np.array([s.start_distance_m for s in segments])
    seg_ends = np.array([s.end_distance_m for s in segments])
    seg_lens = seg_ends - seg_starts

    # Safe divide: inf-speed segments contribute zero time.
    with np.errstate(divide="ignore", invalid="ignore"):
        seg_times = np.where(seg_speeds_ms > 0, seg_lens / seg_speeds_ms, 0.0)

    # TTG at each segment start: cumulative suffix sum.
    ttg_at_start = np.concatenate([np.cumsum(seg_times[::-1])[::-1], [0.0]])

    # For each row, locate containing segment and subtract partial time.
    idx = np.searchsorted(seg_starts, distances, side="right") - 1
    idx = idx.clip(0, len(segments) - 1)

    past_in_seg = distances - seg_starts[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        partial = np.where(
            seg_speeds_ms[idx] > 0, past_in_seg / seg_speeds_ms[idx], 0.0
        )
    return ttg_at_start[idx] - partial


def effective_speed_from_ttg(
    distances: np.ndarray, total_dist: float, ttg: np.ndarray
) -> np.ndarray:
    """Back-derive effective speed = remaining / ttg, NaN where invalid."""
    remaining = total_dist - distances
    valid = (ttg > 0) & (remaining > 0)
    safe = np.where(valid, ttg, 1.0)
    return np.where(valid, remaining / safe, np.nan)
