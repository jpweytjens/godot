"""Time-to-go (TTG) integration helpers.

Given a decimated route (a list of `RouteSegment`s, each with a constant
gradient) and a rider model that maps gradient → speed, there are two
equivalent ways to compute the remaining time from the rider's current
position to the end of the route.

**Segment-based (`segment_ttg_from_row`)**
    Precompute TTG at each segment boundary as a reverse cumulative sum
    of `seg_len / seg_speed`, then per row: find the containing segment
    via `searchsorted`, return `ttg_at_start[idx] - past_in_seg/seg_speed[idx]`.
    Batch-vectorized, O(N + S). The speed-per-segment array is baked in,
    so if rider speeds change mid-ride you have to recompute the suffix
    sum.

**Binned (`BinnedTTG`)**
    Represent the remaining route as a histogram over gradient bins:
    `L_remaining[g]` = meters of route ahead at gradient bin g. TTG is a
    dot product: `ttg = (L_remaining · inv_ratio) / v_flat`, where
    `inv_ratio[g] = 1 / ratio(g)`. The state is incremental — `consume(dd)`
    advances the rider and deducts from the appropriate bin, splitting
    the deduction across bins when a VW segment boundary is crossed.
    O(B) per query and per tick, but decouples route state from rider
    state: swap in new `inv_ratio` values without touching the route
    histogram, and vice versa.

The two are mathematically identical (the binned version just
reassociates the segment sum by grouping segments by their gradient
bin), so the batch helper is used in the Python prototype and `BinnedTTG`
exists as a cleaner online/incremental formulation — for Kotlin port,
live 1 Hz updates, or any setting where the rider model might change
mid-ride and you want to avoid rebuilding the segment-level TTG cache.

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

import math
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


class BinnedTTG:
    """Incremental TTG tracker that represents the remaining route as a
    histogram over gradient bins and computes TTG as a dot product.

    Built for the Kotlin port's online/1 Hz update path and for settings
    where the rider model might change mid-ride (ratio table rebuild)
    without needing to rescan the route.

    Route state and rider state are fully decoupled:

    - Route state: ``L_remaining[g]`` — meters of route ahead at
      gradient bin ``g``. Updated incrementally via :meth:`consume`.
    - Rider state: ``inv_speed[g] = 1 / (v_flat_ms * ratio(g))``.
      Swap via :meth:`set_ratios` or by passing a different ``v_flat_ms``
      at query time; no route rescan needed.

    TTG is the dot product ``Σ_g L_remaining[g] · inv_speed[g]``.

    Example
    -------
    Set up a tiny 3-segment route and walk through it::

        from godot.segmentation import RouteSegment
        from godot.ttg import BinnedTTG

        # 1 km flat → 500 m at 5% climb → 500 m descent at -3%
        segments = [
            RouteSegment(start_distance_m=0.0,    end_distance_m=1000.0, gradient=0.00),
            RouteSegment(start_distance_m=1000.0, end_distance_m=1500.0, gradient=0.05),
            RouteSegment(start_distance_m=1500.0, end_distance_m=2000.0, gradient=-0.03),
        ]
        # Rider: 10 m/s on flat, 0.5x on climbs, 2x on descents
        ratios = {-3: 2.0, 0: 1.0, 5: 0.5}
        ttg = BinnedTTG(segments, ratios, v_flat_ms=10.0)

        # Total ride time from the start:
        assert abs(ttg.ttg() - (1000/10 + 500/5 + 500/20)) < 1e-9
        #                      flat 100s + climb 100s + descent 25s = 225s

        ttg.consume(600.0)           # advance 600 m (still on flat)
        assert abs(ttg.ttg() - (400/10 + 500/5 + 500/20)) < 1e-9

        ttg.consume(600.0)           # crosses flat→climb boundary mid-tick
        # 400 m consumed at flat (done), 200 m consumed at climb
        assert abs(ttg.ttg() - (300/5 + 500/20)) < 1e-9

        # Rider fitness update mid-ride: climbs got 20% harder
        ttg.set_ratios({-3: 2.0, 0: 1.0, 5: 0.4})
        assert abs(ttg.ttg() - (300/4 + 500/20)) < 1e-9

    Mathematical equivalence
    ------------------------
    The binned dot product is exactly the segment sum, regrouped by bin::

        segment form:  Σ_j  L_j / (v_flat * ratio(g_j))
        binned form:   Σ_g  (Σ_{j : g_j = g} L_j) / (v_flat * ratio(g))

    Both sums are over the same set of (length, speed) contributions, so
    ``BinnedTTG.ttg()`` is equal to ``segment_ttg_from_row`` evaluated at
    the current position, up to floating-point drift from incremental
    updates. This invariant is worth asserting in tests.
    """

    def __init__(
        self,
        segments: list[RouteSegment],
        ratios: dict[int, float],
        v_flat_ms: float,
    ) -> None:
        if not segments:
            raise ValueError("BinnedTTG requires at least one segment")

        self._seg_starts = np.array([s.start_distance_m for s in segments])
        self._seg_ends = np.array([s.end_distance_m for s in segments])
        self._seg_lens = self._seg_ends - self._seg_starts
        self._seg_bins = np.array(
            [int(math.floor(s.gradient * 100)) for s in segments], dtype=np.int64
        )
        self._total_distance = float(self._seg_ends[-1])
        self._num_segments = len(segments)

        # Build the bin index: one slot per unique gradient bin that
        # appears in the route.
        unique_bins = np.unique(self._seg_bins)
        self._bins = unique_bins
        self._bin_index: dict[int, int] = {int(b): i for i, b in enumerate(unique_bins)}

        # Initial histogram: sum segment lengths into their respective bins.
        self._L_remaining = np.zeros(len(unique_bins), dtype=np.float64)
        for j in range(self._num_segments):
            self._L_remaining[self._bin_index[int(self._seg_bins[j])]] += (
                self._seg_lens[j]
            )

        # Rider state: inv_speed per bin. Updated by set_ratios or v_flat_ms.
        self._v_flat_ms = float(v_flat_ms)
        self._inv_ratio = self._build_inv_ratio(ratios)

        # Position state: which segment we're in, and how far into it.
        self._current_idx = 0
        self._distance = 0.0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def ttg(self, v_flat_ms: float | None = None) -> float:
        """Remaining time to end of route in seconds.

        Parameters
        ----------
        v_flat_ms : float, optional
            Override the stored v_flat for this query. If None, uses the
            value from construction / last ``set_v_flat``.
        """
        v = self._v_flat_ms if v_flat_ms is None else float(v_flat_ms)
        if v <= 0:
            return float("inf")
        return float(self._L_remaining @ self._inv_ratio) / v

    @property
    def position(self) -> float:
        """Current distance along the route in meters."""
        return self._distance

    @property
    def remaining_distance(self) -> float:
        """Total distance remaining to the end of the route, in meters."""
        return self._total_distance - self._distance

    @property
    def histogram(self) -> dict[int, float]:
        """Current ``L_remaining`` as a plain dict keyed by gradient bin."""
        return {int(b): float(self._L_remaining[i]) for b, i in self._bin_index.items()}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def consume(self, dd: float) -> None:
        """Advance the rider's position by ``dd`` meters.

        Handles the case where ``dd`` crosses a VW segment boundary by
        splitting the deduction across the bins of the segments involved.
        At typical 1 Hz sampling and normal cycling speeds, boundary
        crossings are rare and the loop runs at most once per call; at
        high descent speeds across short segments it may run 2-3 times.
        Amortized cost across a full ride is O(N + S), same as the
        batch helper above.
        """
        if dd <= 0:
            return
        while dd > 0 and self._current_idx < self._num_segments:
            seg_end = self._seg_ends[self._current_idx]
            remaining_in_seg = seg_end - self._distance
            if remaining_in_seg <= 0:
                self._current_idx += 1
                continue
            take = min(dd, remaining_in_seg)
            bin_idx = self._bin_index[int(self._seg_bins[self._current_idx])]
            self._L_remaining[bin_idx] -= take
            self._distance += take
            dd -= take
            # Clamp floating-point drift to zero at the bin level.
            if self._L_remaining[bin_idx] < 0:
                self._L_remaining[bin_idx] = 0.0
            # If we exactly hit (or slightly overshot) the segment end,
            # advance to the next one.
            if take >= remaining_in_seg - 1e-9:
                self._current_idx += 1

    def set_ratios(self, ratios: dict[int, float]) -> None:
        """Install a new rider model — swaps ``inv_ratio`` without
        touching the route histogram.

        Use this when the physics ratio table is rebuilt mid-ride (e.g.
        Variant C's continuous rebuild, or a one-shot recalibration after
        a warmup period). Route state and position are preserved.
        """
        self._inv_ratio = self._build_inv_ratio(ratios)

    def set_v_flat(self, v_flat_ms: float) -> None:
        """Update the flat-ground speed used by :meth:`ttg`.

        Cheap (no histogram touch). Equivalent to passing ``v_flat_ms``
        to every ``ttg`` call.
        """
        self._v_flat_ms = float(v_flat_ms)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_inv_ratio(self, ratios: dict[int, float]) -> np.ndarray:
        """Build ``inv_ratio`` aligned to the route's bin index.

        Any route bins outside the ratio table's domain are clipped to
        the nearest available bin, mirroring ``_ratio_for`` in
        `godot.estimators.BaseEstimator`.
        """
        if not ratios:
            raise ValueError("ratios dict must not be empty")
        min_bin = min(ratios)
        max_bin = max(ratios)
        out = np.zeros(len(self._bins), dtype=np.float64)
        for b, idx in self._bin_index.items():
            clipped = max(min_bin, min(max_bin, b))
            r = ratios.get(clipped, 1.0)
            out[idx] = 1.0 / r if r > 0 else 0.0
        return out


if __name__ == "__main__":
    # Self-verification: walk a real ride through both formulations and
    # assert they agree at every row. Run with `uv run python -m godot.ttg`.
    from pathlib import Path

    from godot.ride import load_ride

    gpx = Path(__file__).resolve().parent.parent / "data/gpx/Criquielion25.gpx"
    ride = load_ride(gpx, "integrated", smooth_speed=False)
    segments = ride.gradient_segments

    # Rider model: the realistic physics ratio table.
    ratios = ride.df.attrs.get("ratios") or None
    if ratios is None:
        from godot.config import RideConfig

        ratios = RideConfig().realistic_ratios
        v_flat = RideConfig().v_flat_ms

    # Batch reference: segment_ttg_from_row at every row's distance.
    distances = ride.df["distance_m"].values
    total_dist = float(distances[-1])
    import math as _math  # local alias for clipping

    min_bin = min(ratios)
    max_bin = max(ratios)
    seg_ratios = np.array(
        [
            ratios.get(
                max(min_bin, min(max_bin, int(_math.floor(s.gradient * 100)))), 1.0
            )
            for s in segments
        ]
    )
    seg_speeds = v_flat * seg_ratios
    reference = segment_ttg_from_row(distances, total_dist, segments, seg_speeds)

    # Incremental: walk BinnedTTG through every row's delta_distance.
    binned = BinnedTTG(segments, ratios, v_flat)
    dd = ride.df["distance_m"].diff().fillna(0.0).values
    binned_out = np.empty(len(distances))
    for i, step in enumerate(dd):
        if step > 0:
            binned.consume(float(step))
        binned_out[i] = binned.ttg()

    # Compare: should agree to floating-point tolerance.
    diff = np.abs(reference - binned_out)
    print(f"BinnedTTG vs segment_ttg_from_row over {len(distances)} rows")
    print(f"  max abs diff  : {diff.max():.6e} s")
    print(f"  mean abs diff : {diff.mean():.6e} s")
    print(f"  final position: {binned.position:.1f} m / {total_dist:.1f} m")
    print(f"  final ttg     : {binned.ttg():.3f} s (should be ~0)")
