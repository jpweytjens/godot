"""ProCyclingStats-style route difficulty metrics.

Implements PCS's `profileScore` formula and the 5-icon stage
classification (flat, hills/mountains × flat/uphill finish).

References
----------
https://www.procyclingstats.com/info/profile-score-explained

Per-climb score:

    climb_score = (avg_gradient_pct / 2) ** 2 * length_km

Whole-stage score multiplies each climb by a factor that depends on
how close the climb is to the finish:

    within 10 km → 1.0
    within 25 km → 0.8
    within 50 km → 0.6
    within 75 km → 0.4
    earlier      → 0.2

`profile_score_final` is the same formula restricted to climbs in the
last 25 km of the route. PCS does not publish fixed icon thresholds
(they assign them per-race), so we use conservative defaults derived
from the rider-specialty page: flat < 50, hills 50-150, mountains > 150,
and an uphill finish when `profile_score_final >= 25`.
"""

from __future__ import annotations

from dataclasses import dataclass

from godot.segmentation import RouteSegment


@dataclass(frozen=True)
class Climb:
    """A contiguous uphill stretch of the route."""

    start_distance_m: float
    end_distance_m: float
    elevation_gain_m: float

    @property
    def length_m(self) -> float:
        return self.end_distance_m - self.start_distance_m

    @property
    def avg_gradient_pct(self) -> float:
        return 100.0 * self.elevation_gain_m / self.length_m

    @property
    def climb_score(self) -> float:
        """PCS per-climb profile score, no distance-to-finish factor."""
        return (self.avg_gradient_pct / 2.0) ** 2 * (self.length_m / 1000.0)


def extract_climbs(
    segments: list[RouteSegment],
    min_length_m: float = 200.0,
) -> list[Climb]:
    """Group consecutive positive-gradient segments into climbs.

    A climb is a maximal run of `RouteSegment`s with `gradient > 0`.
    Runs shorter than `min_length_m` are dropped (matching PCS's 200 m
    minimum segment length for steepness computation).

    Parameters
    ----------
    segments : list of RouteSegment
        Simplified route, e.g. from `decimate_to_gradient_segments`.
    min_length_m : float, optional
        Minimum climb length to include. Default 200.

    Returns
    -------
    list of Climb
        Climbs in route order.
    """
    climbs: list[Climb] = []
    start: float | None = None
    end: float = 0.0
    gain: float = 0.0

    def flush() -> None:
        nonlocal start
        if start is not None and end - start >= min_length_m:
            climbs.append(Climb(start, end, gain))
        start = None

    for s in segments:
        if s.gradient > 0:
            if start is None:
                start = s.start_distance_m
                gain = 0.0
            gain += (s.end_distance_m - s.start_distance_m) * s.gradient
            end = s.end_distance_m
        else:
            flush()
    flush()
    return climbs


_DISTANCE_FACTOR_BINS: list[tuple[float, float]] = [
    (10_000.0, 1.0),
    (25_000.0, 0.8),
    (50_000.0, 0.6),
    (75_000.0, 0.4),
]
_FAR_FACTOR: float = 0.2


def distance_factor(distance_to_finish_m: float) -> float:
    """PCS distance-to-finish multiplier for a climb's score."""
    for cutoff, factor in _DISTANCE_FACTOR_BINS:
        if distance_to_finish_m <= cutoff:
            return factor
    return _FAR_FACTOR


def profile_score(
    segments: list[RouteSegment],
    total_distance_m: float | None = None,
    min_climb_length_m: float = 200.0,
) -> float:
    """PCS profile score for the whole route."""
    if not segments:
        return 0.0
    if total_distance_m is None:
        total_distance_m = segments[-1].end_distance_m
    climbs = extract_climbs(segments, min_climb_length_m)
    return sum(
        c.climb_score * distance_factor(total_distance_m - c.start_distance_m)
        for c in climbs
    )


def profile_score_final(
    segments: list[RouteSegment],
    total_distance_m: float | None = None,
    window_m: float = 25_000.0,
    min_climb_length_m: float = 200.0,
) -> float:
    """PCS profile score restricted to the last `window_m` metres.

    Climbs whose start lies before the window are excluded. Climbs
    starting inside the window keep their normal distance-to-finish
    factor.
    """
    if not segments:
        return 0.0
    if total_distance_m is None:
        total_distance_m = segments[-1].end_distance_m
    cutoff = total_distance_m - window_m
    climbs = [
        c
        for c in extract_climbs(segments, min_climb_length_m)
        if c.start_distance_m >= cutoff
    ]
    return sum(
        c.climb_score * distance_factor(total_distance_m - c.start_distance_m)
        for c in climbs
    )


def climb_centroid(
    segments: list[RouteSegment],
    min_climb_length_m: float = 200.0,
    min_climb_score: float = 1.0,
) -> float | None:
    """Weighted centre of mass of climbing difficulty, as a route fraction.

    Each climb's midpoint is weighted by its PCS climb score. The result
    is a number between 0 (all climbing at the start) and 1 (all climbing
    at the end). Returns `None` if no qualifying climbs exist.

    Parameters
    ----------
    segments : list of RouteSegment
        Simplified route segments.
    min_climb_length_m : float, optional
        Minimum climb length in metres. Default 200.
    min_climb_score : float, optional
        Ignore climbs below this score. Default 1.0.

    Returns
    -------
    float or None
        Climb centroid as a fraction of total route distance, or None
        if the route has no qualifying climbs.
    """
    if not segments:
        return None
    total_dist = segments[-1].end_distance_m
    if total_dist <= 0:
        return None
    climbs = [
        c
        for c in extract_climbs(segments, min_climb_length_m)
        if c.climb_score >= min_climb_score
    ]
    if not climbs:
        return None
    weighted_pos = sum(
        c.climb_score * (c.start_distance_m + c.end_distance_m) / 2 for c in climbs
    )
    total_score = sum(c.climb_score for c in climbs)
    return weighted_pos / (total_dist * total_score)


def max_climb_score(
    segments: list[RouteSegment],
    min_climb_length_m: float = 200.0,
) -> float:
    """Largest per-climb PCS score on the route (no distance factor).

    Useful when you care about the *presence* of a meaningful climb
    rather than its placement relative to the finish — e.g. when
    classifying rides by whether a gradient-aware estimator should
    beat a flat-speed baseline.
    """
    climbs = extract_climbs(segments, min_climb_length_m)
    return max((c.climb_score for c in climbs), default=0.0)


def cumulative_climb_score(
    segments: list[RouteSegment],
    min_climb_length_m: float = 200.0,
) -> float:
    """Sum of per-climb PCS scores on the route (no distance factor).

    Grows with both the size and the number of climbs, so it tracks
    how much gradient-aware routing should win over a flat baseline
    across the whole ride.
    """
    return sum(c.climb_score for c in extract_climbs(segments, min_climb_length_m))


@dataclass(frozen=True)
class MaxClimbClassification:
    """PCS-inspired classification by the hardest climb on the route.

    Uses the PCS per-climb formula `(steepness/2)^2 * length_km`
    *without* the distance-to-finish factor: here the goal is to flag
    routes where gradient-aware estimation should outperform a
    flat-speed baseline, and placement relative to the finish is
    irrelevant for that.
    """

    difficulty: str  # "pancake" | "rolling" | "minor_hills" | "hills" | "mountains"
    max_climb_score: float
    cumulative_climb_score: float
    n_climbs: int


def classify_by_max_climb(
    segments: list[RouteSegment],
    pancake_threshold: float = 0.5,
    rolling_threshold: float = 2.0,
    flat_threshold: float = 5.0,
    mountain_threshold: float = 50.0,
    min_climb_length_m: float = 200.0,
    min_climb_score: float = 1.0,
) -> MaxClimbClassification:
    """Classify a route by its hardest climb (PCS-inspired).

    Uses five difficulty levels that reflect the estimator's behavior
    at each tier:

    - *pancake*: max climb score < 0.5 — no meaningful gradient at all.
      Gradient-aware estimation adds nothing over a moving average.
    - *rolling*: max climb score in [0.5, 2) — minor undulations.
      The physics model helps slightly; corrections are noise.
    - *minor_hills*: max climb score in [2, 5) — small but real hills.
      Gradient awareness helps; short climbs are the signal.
    - *hills*: max climb score in [5, 50) — proper climbs but
      individually short (Belgian bergs, Flemish classics).
    - *mountains*: max climb score >= 50 — at least one sustained
      climb (≈ 10 km @ 4.5% or 5 km @ 6.3%).

    Parameters
    ----------
    segments : list of RouteSegment
        Simplified route segments.
    pancake_threshold : float, optional
        Score cutoff between pancake and rolling. Default 0.5.
    rolling_threshold : float, optional
        Score cutoff between rolling and minor_hills. Default 2.0.
    flat_threshold : float, optional
        Score cutoff between minor_hills and hills. Default 5.0.
    mountain_threshold : float, optional
        Score cutoff between hills and mountains. Default 50.0.
    min_climb_length_m : float, optional
        Minimum climb length in metres. Default 200.
    min_climb_score : float, optional
        Ignore climbs below this score. Default 1.0.

    Returns
    -------
    MaxClimbClassification
    """
    climbs = [
        c
        for c in extract_climbs(segments, min_climb_length_m)
        if c.climb_score >= min_climb_score
    ]
    max_score = max((c.climb_score for c in climbs), default=0.0)
    cum_score = sum(c.climb_score for c in climbs)

    if max_score < pancake_threshold:
        difficulty = "pancake"
    elif max_score < rolling_threshold:
        difficulty = "rolling"
    elif max_score < flat_threshold:
        difficulty = "minor_hills"
    elif max_score < mountain_threshold:
        difficulty = "hills"
    else:
        difficulty = "mountains"

    return MaxClimbClassification(
        difficulty=difficulty,
        max_climb_score=max_score,
        cumulative_climb_score=cum_score,
        n_climbs=len(climbs),
    )


@dataclass(frozen=True)
class PCSClassification:
    """Two-axis PCS stage classification."""

    difficulty: str  # "flat" | "hills" | "mountains"
    finish: str  # "flat" | "uphill"
    profile_score: float
    profile_score_final: float

    @property
    def icon(self) -> str:
        """PCS-style icon label."""
        if self.difficulty == "flat":
            return "flat"
        return f"{self.difficulty}, {self.finish} finish"


def classify(
    segments: list[RouteSegment],
    total_distance_m: float | None = None,
    flat_threshold: float = 50.0,
    mountain_threshold: float = 150.0,
    uphill_finish_threshold: float = 25.0,
) -> PCSClassification:
    """Classify a route on PCS's two independent axes.

    Parameters
    ----------
    segments : list of RouteSegment
        Simplified route segments.
    total_distance_m : float or None, optional
        Route length. Defaults to `segments[-1].end_distance_m`.
    flat_threshold : float, optional
        Profile-score cutoff between flat and hills. Default 50.
    mountain_threshold : float, optional
        Profile-score cutoff between hills and mountains. Default 150.
    uphill_finish_threshold : float, optional
        `profile_score_final` cutoff for an uphill finish. Default 25.

    Returns
    -------
    PCSClassification
    """
    ps = profile_score(segments, total_distance_m)
    ps_final = profile_score_final(segments, total_distance_m)

    if ps < flat_threshold:
        difficulty = "flat"
    elif ps < mountain_threshold:
        difficulty = "hills"
    else:
        difficulty = "mountains"

    finish = "uphill" if ps_final >= uphill_finish_threshold else "flat"
    return PCSClassification(difficulty, finish, ps, ps_final)
