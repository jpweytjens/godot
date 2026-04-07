"""ETA estimators for cycling ride prediction."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from godot.segmentation import RouteSegment, decimate_to_gradient_segments

if TYPE_CHECKING:
    from godot.ride import Ride

ROLLING_WINDOW_S = 300.0
EWMA_SPAN_S = 3600.0
MAX_MIN_PERIODS = 300


def _default_min_periods(window_s: float) -> int:
    """At most 300 (5 min), at least 10 % of the window."""
    return int(min(MAX_MIN_PERIODS, 0.10 * window_s))


class BaseEstimator:
    """Base for ETA estimators. Subclasses must implement `predict`."""

    @staticmethod
    def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Element-wise division, returning NaN where either operand is zero."""
        return (numerator / denominator).where((denominator != 0) & (numerator != 0))

    def predict(self, ride: Ride) -> pd.Series:
        """Return estimated speed in m/s at each row.

        Parameters
        ----------
        ride : Ride
            Prepared ride from `load_ride`.

        Returns
        -------
        pd.Series
            Speed in m/s at each row. NaN where insufficient data exists.
        """
        raise NotImplementedError


class AvgSpeedEstimator(BaseEstimator):
    """Expanding-window average speed estimator.

    Parameters
    ----------
    moving_only : bool, optional
        If True (default), denominator is total moving time.
        If False, denominator is total elapsed time.
    min_periods : int, optional
        Minimum cumulative seconds before emitting a value. Default 60.
        Prevents noisy estimates at ride start where the expanding
        average is essentially instantaneous speed.
    """

    def __init__(self, moving_only: bool = True, min_periods: int = 60) -> None:
        self._moving_only = moving_only
        self._min_periods = min_periods

    def __str__(self) -> str:
        mode = "moving" if self._moving_only else "elapsed"
        return f"avg speed ({mode} time)"

    def __repr__(self) -> str:
        return (
            f"AvgSpeedEstimator(moving_only={self._moving_only!r}, "
            f"min_periods={self._min_periods!r})"
        )

    def predict(self, ride: Ride) -> pd.Series:
        df = ride.df
        dd = df["delta_distance"]
        dt = df["delta_time"]
        if self._moving_only:
            moving = (~df["paused"]).astype(float)
            dd, dt = dd * moving, dt * moving
        cum_dd = dd.cumsum()
        cum_dt = dt.cumsum()
        raw = self.safe_divide(cum_dd, cum_dt)
        return raw.where(cum_dt >= self._min_periods)


class RollingAvgSpeedEstimator(BaseEstimator):
    """Rolling-window average speed estimator.

    Parameters
    ----------
    window_s : float, optional
        Rolling window in seconds. Defaults to ROLLING_WINDOW_S (300 s).
    moving_only : bool, optional
        If True (default), only accumulate distance and time while moving.
    min_periods : int, optional
        Minimum number of non-NaN observations required in the window.
        Defaults to `window_s` (i.e. a full window of data). Prevents
        unreliable estimates from thin post-pause windows.
    """

    def __init__(
        self,
        window_s: float | None = None,
        moving_only: bool = True,
        min_periods: int | None = None,
    ) -> None:
        self._window_s = ROLLING_WINDOW_S if window_s is None else window_s
        self._moving_only = moving_only
        self._min_periods = (
            _default_min_periods(self._window_s) if min_periods is None else min_periods
        )

    def __str__(self) -> str:
        mode = "moving" if self._moving_only else "elapsed"
        return f"rolling avg speed ({int(self._window_s)}s, {mode} time)"

    def __repr__(self) -> str:
        return (
            f"RollingAvgSpeedEstimator(window_s={self._window_s!r}, "
            f"moving_only={self._moving_only!r}, "
            f"min_periods={self._min_periods!r})"
        )

    def predict(self, ride: Ride) -> pd.Series:
        df = ride.df
        dd = df["delta_distance"]
        dt = df["delta_time"]
        if self._moving_only:
            dd, dt = dd.where(~df["paused"]), dt.where(~df["paused"])
        idx = pd.DatetimeIndex(df["time"])
        window = f"{int(self._window_s)}s"
        dd_roll = (
            pd.Series(dd.values, index=idx)
            .rolling(window, min_periods=self._min_periods)
            .sum()
        )
        dt_roll = (
            pd.Series(dt.values, index=idx)
            .rolling(window, min_periods=self._min_periods)
            .sum()
        )
        return pd.Series(self.safe_divide(dd_roll, dt_roll).values, index=df.index)


class RollingMedianSpeedEstimator(BaseEstimator):
    """Rolling-window median speed estimator.

    Computes per-row instantaneous speed and takes the rolling median,
    which is more robust to outliers than the mean-based rolling estimator.

    Parameters
    ----------
    window_s : float, optional
        Rolling window in seconds. Defaults to ROLLING_WINDOW_S (300 s).
    moving_only : bool, optional
        If True (default), NaN out speed during pauses before windowing.
    min_periods : int, optional
        Minimum non-NaN observations in window. Defaults to `window_s`.
    """

    def __init__(
        self,
        window_s: float | None = None,
        moving_only: bool = True,
        min_periods: int | None = None,
    ) -> None:
        self._window_s = ROLLING_WINDOW_S if window_s is None else window_s
        self._moving_only = moving_only
        self._min_periods = (
            _default_min_periods(self._window_s) if min_periods is None else min_periods
        )

    def __str__(self) -> str:
        mode = "moving" if self._moving_only else "elapsed"
        return f"rolling median speed ({int(self._window_s)}s, {mode} time)"

    def __repr__(self) -> str:
        return (
            f"RollingMedianSpeedEstimator(window_s={self._window_s!r}, "
            f"moving_only={self._moving_only!r}, "
            f"min_periods={self._min_periods!r})"
        )

    def predict(self, ride: Ride) -> pd.Series:
        df = ride.df
        speed = self.safe_divide(df["delta_distance"], df["delta_time"])
        if self._moving_only:
            speed = speed.where(~df["paused"])
        idx = pd.DatetimeIndex(df["time"])
        window = f"{int(self._window_s)}s"
        rolled = (
            pd.Series(speed.values, index=idx)
            .rolling(window, min_periods=self._min_periods)
            .median()
        )
        return pd.Series(rolled.values, index=df.index)


class EWMASpeedEstimator(BaseEstimator):
    """Exponentially weighted moving average speed estimator.

    Computes per-row instantaneous speed and applies EWMA smoothing.

    Parameters
    ----------
    span_s : float, optional
        EWMA span in seconds (number of observations at 1 Hz).
        Defaults to EWMA_SPAN_S (3600 s). Ignored when `alpha` is set.
    alpha : float, optional
        Smoothing factor override (0 < alpha <= 1). When provided,
        used directly instead of deriving from `span_s`.
    moving_only : bool, optional
        If True (default), NaN out speed during pauses before smoothing.
    min_periods : int, optional
        Minimum non-NaN observations. Defaults to `span_s`.
    """

    def __init__(
        self,
        span_s: float | None = None,
        alpha: float | None = None,
        moving_only: bool = True,
        min_periods: int | None = None,
    ) -> None:
        self._span_s = EWMA_SPAN_S if span_s is None else span_s
        self._alpha = alpha
        self._moving_only = moving_only
        self._min_periods = (
            _default_min_periods(self._span_s) if min_periods is None else min_periods
        )

    def __str__(self) -> str:
        mode = "moving" if self._moving_only else "elapsed"
        if self._alpha is not None:
            return f"EWMA speed (α={self._alpha}, {mode} time)"
        return f"EWMA speed ({int(self._span_s)}s span, {mode} time)"

    def __repr__(self) -> str:
        return (
            f"EWMASpeedEstimator(span_s={self._span_s!r}, "
            f"alpha={self._alpha!r}, "
            f"moving_only={self._moving_only!r}, "
            f"min_periods={self._min_periods!r})"
        )

    def predict(self, ride: Ride) -> pd.Series:
        df = ride.df
        speed = self.safe_divide(df["delta_distance"], df["delta_time"])
        if self._moving_only:
            speed = speed.where(~df["paused"])
        ewm_kwargs = {"min_periods": self._min_periods}
        if self._alpha is not None:
            ewm_kwargs["alpha"] = self._alpha
        else:
            ewm_kwargs["span"] = self._span_s
        smoothed = speed.ewm(**ewm_kwargs).mean()
        return smoothed


class DEWMASpeedEstimator(BaseEstimator):
    """Double (weighted) EWMA speed estimator.

    Blends a slow and a fast EWMA to balance responsiveness with stability.

    Parameters
    ----------
    slow_span_s : float, optional
        Span for the slow EWMA in seconds. Default 3600 (60 min).
    fast_span_s : float, optional
        Span for the fast EWMA in seconds. Default 600 (10 min).
    slow_alpha : float, optional
        Alpha override for the slow EWMA.
    fast_alpha : float, optional
        Alpha override for the fast EWMA.
    slow_weight : float, optional
        Weight for the slow component. Default 0.7.
    fast_weight : float, optional
        Weight for the fast component. Default 0.3.
    moving_only : bool, optional
        If True (default), NaN out speed during pauses.
    min_periods : int, optional
        Minimum non-NaN observations. Defaults to `slow_span_s`.
    """

    def __init__(
        self,
        slow_span_s: float = 3600.0,
        fast_span_s: float = 600.0,
        slow_alpha: float | None = None,
        fast_alpha: float | None = None,
        slow_weight: float = 0.7,
        fast_weight: float = 0.3,
        moving_only: bool = True,
        min_periods: int | None = None,
    ) -> None:
        mp = _default_min_periods(slow_span_s) if min_periods is None else min_periods
        self._slow = EWMASpeedEstimator(
            span_s=slow_span_s,
            alpha=slow_alpha,
            moving_only=moving_only,
            min_periods=mp,
        )
        self._fast = EWMASpeedEstimator(
            span_s=fast_span_s,
            alpha=fast_alpha,
            moving_only=moving_only,
            min_periods=mp,
        )
        self._slow_weight = slow_weight
        self._fast_weight = fast_weight

    def __str__(self) -> str:
        return (
            f"DEWMA speed ({self._fast} × {self._fast_weight} "
            f"+ {self._slow} × {self._slow_weight})"
        )

    def __repr__(self) -> str:
        return (
            f"DEWMASpeedEstimator(slow={self._slow!r}, fast={self._fast!r}, "
            f"slow_weight={self._slow_weight!r}, fast_weight={self._fast_weight!r})"
        )

    def predict(self, ride: Ride) -> pd.Series:
        slow = self._slow.predict(ride)
        fast = self._fast.predict(ride)
        return self._slow_weight * slow + self._fast_weight * fast


class LerpSpeedEstimator(BaseEstimator):
    """Blended prior-ramp + EWMA speed estimator.

    Builds a slow component that ramps linearly from a global prior toward
    the cumulative average speed over `ramp_s` moving seconds, then blends
    it with a fast EWMA for responsiveness.

    Parameterse
    ----------
    prior_ms : float
        Prior speed in m/s (e.g. from `compute_global_prior`).
    fast_span_s : float
        EWMA span for the fast component in seconds. Default 600 (10 min).
    ramp_s : float
        Moving seconds over which the slow component transitions from
        prior to cumulative average. Default 600.
    fast_weight : float
        Weight of the fast EWMA in the final blend. Default 0.15
        (i.e. 85 % slow, 15 % fast).
    moving_only : bool
        If True (default), ignore paused rows in all computations.
    """

    def __init__(
        self,
        prior_ms: float = 5.0,
        fast_span_s: float = 600.0,
        ramp_s: float = 600.0,
        fast_weight: float = 0.15,
        moving_only: bool = True,
    ) -> None:
        self._prior_ms = prior_ms
        self._fast_span_s = fast_span_s
        self._ramp_s = ramp_s
        self._fast_weight = fast_weight
        self._moving_only = moving_only

    def __str__(self) -> str:
        mode = "moving" if self._moving_only else "elapsed"
        prior_kmh = self._prior_ms * 3.6
        return (
            f"lerp speed (ramp={int(self._ramp_s)}s, fast={int(self._fast_span_s)}s, "
            f"w={self._fast_weight}, prior={prior_kmh:.1f}km/h, {mode})"
        )

    def __repr__(self) -> str:
        return (
            f"LerpSpeedEstimator(prior_ms={self._prior_ms!r}, "
            f"fast_span_s={self._fast_span_s!r}, ramp_s={self._ramp_s!r}, "
            f"fast_weight={self._fast_weight!r}, moving_only={self._moving_only!r})"
        )

    def predict(self, ride: Ride) -> pd.Series:
        df = ride.df
        dd = df["delta_distance"]
        dt = df["delta_time"]

        # Pause masking
        if self._moving_only:
            moving = (~df["paused"]).astype(float)
            dd_m, dt_m = dd * moving, dt * moving
        else:
            moving = pd.Series(1.0, index=df.index)
            dd_m, dt_m = dd, dt

        # Slow component: prior → cumulative average over ramp_s seconds
        cum_avg = self.safe_divide(dd_m.cumsum(), dt_m.cumsum()).fillna(self._prior_ms)
        ramp = (moving.cumsum() / self._ramp_s).clip(0.0, 1.0)
        slow = lerp(self._prior_ms, cum_avg, ramp)

        # Fast component: prior-seeded EWMA
        inst_speed = self.safe_divide(dd, dt)
        if self._moving_only:
            inst_speed = inst_speed.where(~df["paused"])
        fast = (
            _seed_prior(inst_speed, self._prior_ms)
            .ewm(span=self._fast_span_s, min_periods=1)
            .mean()
        )

        return lerp(slow, fast, self._fast_weight)


def _seed_prior(speed: pd.Series, prior_ms: float) -> pd.Series:
    """Replace leading NaNs with the prior so EWMA starts immediately."""
    leading_nan = speed.isna().cumprod().astype(bool)
    return speed.where(~leading_nan, prior_ms)


def lerp(start, end, weight):
    """Blend from `start` toward `end` as `weight` goes from 0 to 1."""
    return start * (1 - weight) + end * weight


class AdaptiveLerpSpeedEstimator(BaseEstimator):
    """Pause-aware blended speed estimator.

    Combines three signals:

    - **slow**: cumulative average speed (total time, so pauses have cost),
      with exponential decay toward a floor during pauses to prevent
      ETA skyrocketing.
    - **fast**: 60-minute EWMA of instantaneous speed (moving-only),
      seeded with a prior for immediate estimates.
    - **final**: ``lerp(slow, fast, fast_weight)``

    During a pause, ``confidence = ewma(is_moving, span=tau)`` decays
    exponentially, blending the slow component from ``avg_total`` toward
    a floor derived from the ride data:

        floor = cumsum(dd) / (cumsum(dt_elapsed) + k * tau)

    ``tau`` encodes typical pause length; ``k`` (default 2) accounts for
    longer-than-typical pauses.  The prior only seeds the fast EWMA at
    ride start.

    Parameters
    ----------
    prior_ms : float
        Prior speed in m/s for seeding the fast EWMA.
    tau : float
        Confidence decay span in seconds. Controls how quickly the slow
        component falls back to the floor during a pause.
    k : float
        Floor multiplier on tau. ``floor = distance / (elapsed + k*tau)``.
    fast_span_s : float
        EWMA span for the fast component.
    fast_weight : float
        Weight of the fast component in the final blend.
    """

    def __init__(
        self,
        prior_ms: float = 5.0,
        tau: float = 300.0,
        k: float = 2.0,
        fast_span_s: float = 3600.0,
        fast_weight: float = 0.15,
    ) -> None:
        self._prior_ms = prior_ms
        self._tau = tau
        self._k = k
        self._fast_span_s = fast_span_s
        self._fast_weight = fast_weight

    def __str__(self) -> str:
        prior_kmh = self._prior_ms * 3.6
        return (
            f"adaptive lerp (τ={int(self._tau)}s, k={self._k}, "
            f"fast={int(self._fast_span_s)}s, w={self._fast_weight}, "
            f"prior={prior_kmh:.1f}km/h)"
        )

    def __repr__(self) -> str:
        return (
            f"AdaptiveLerpSpeedEstimator(prior_ms={self._prior_ms!r}, "
            f"tau={self._tau!r}, k={self._k!r}, "
            f"fast_span_s={self._fast_span_s!r}, "
            f"fast_weight={self._fast_weight!r})"
        )

    def predict(self, ride: Ride) -> pd.Series:
        df = ride.df
        dd = df["delta_distance"]
        dt = df["delta_time"]
        is_moving = (~df["paused"]).astype(float)

        # Cumulative totals (elapsed time, including pauses)
        cum_dd = dd.cumsum()
        cum_dt = dt.cumsum()

        # Slow component: avg_total (includes pause cost)
        avg_total = self.safe_divide(cum_dd, cum_dt)

        # Confidence: exponential decay during pauses
        confidence = is_moving.ewm(span=self._tau, min_periods=1).mean()

        # Floor: avg if you added k*tau extra pause seconds
        floor = self.safe_divide(cum_dd, cum_dt + self._k * self._tau)

        # Blend: avg_total when riding, decays toward floor during pause
        slow = lerp(floor, avg_total, confidence)

        # Fast component: 60-min EWMA, moving-only, seeded with prior
        inst_speed = self.safe_divide(dd, dt)
        inst_speed = inst_speed.where(~df["paused"])
        fast = (
            _seed_prior(inst_speed, self._prior_ms)
            .ewm(span=self._fast_span_s, min_periods=1)
            .mean()
        )

        return lerp(slow, fast, self._fast_weight)


class OracleAdaptiveLerpEstimator(BaseEstimator):
    """Adaptive lerp using the ride's actual moving average as prior.

    Computes `prior_ms` from ride data inside `predict()`, then delegates
    to an `AdaptiveLerpSpeedEstimator`. This tests the estimator mechanics
    with a perfect prior.

    Parameters
    ----------
    tau : float
        Confidence decay span in seconds.
    k : float
        Floor multiplier on tau.
    fast_span_s : float
        EWMA span for the fast component.
    fast_weight : float
        Weight of the fast component in the final blend.
    """

    def __init__(
        self,
        tau: float = 300.0,
        k: float = 2.0,
        fast_span_s: float = 3600.0,
        fast_weight: float = 0.15,
    ) -> None:
        self._tau = tau
        self._k = k
        self._fast_span_s = fast_span_s
        self._fast_weight = fast_weight

    def __str__(self) -> str:
        return (
            f"oracle adaptive lerp (τ={int(self._tau)}s, k={self._k}, "
            f"fast={int(self._fast_span_s)}s, w={self._fast_weight})"
        )

    def __repr__(self) -> str:
        return (
            f"OracleAdaptiveLerpEstimator(tau={self._tau!r}, k={self._k!r}, "
            f"fast_span_s={self._fast_span_s!r}, fast_weight={self._fast_weight!r})"
        )

    def _oracle_ms(self, ride: Ride) -> float:
        df = ride.df
        moving = ~df["paused"]
        return float(
            df.loc[moving, "delta_distance"].sum() / df.loc[moving, "delta_time"].sum()
        )

    def predict(self, ride: Ride) -> pd.Series:
        inner = AdaptiveLerpSpeedEstimator(
            prior_ms=self._oracle_ms(ride),
            tau=self._tau,
            k=self._k,
            fast_span_s=self._fast_span_s,
            fast_weight=self._fast_weight,
        )
        return inner.predict(ride)


class NoisyOracleAdaptiveLerpEstimator(OracleAdaptiveLerpEstimator):
    """Adaptive lerp using a noisy version of the ride's actual moving average.

    Simulates a good-but-imperfect gradient-aware prior by adding
    Gaussian noise to the oracle speed.

    Parameters
    ----------
    tau : float
        Confidence decay span in seconds.
    k : float
        Floor multiplier on tau.
    fast_span_s : float
        EWMA span for the fast component.
    fast_weight : float
        Weight of the fast component in the final blend.
    cv : float
        Coefficient of variation for the noise. Default 0.10 (10%).
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        tau: float = 300.0,
        k: float = 2.0,
        fast_span_s: float = 3600.0,
        fast_weight: float = 0.15,
        cv: float = 0.10,
        seed: int = 42,
    ) -> None:
        super().__init__(tau=tau, k=k, fast_span_s=fast_span_s, fast_weight=fast_weight)
        self._cv = cv
        self._rng = np.random.default_rng(seed)

    def __str__(self) -> str:
        return (
            f"noisy oracle adaptive lerp (τ={int(self._tau)}s, k={self._k}, "
            f"fast={int(self._fast_span_s)}s, w={self._fast_weight}, cv={self._cv})"
        )

    def __repr__(self) -> str:
        return (
            f"NoisyOracleAdaptiveLerpEstimator(tau={self._tau!r}, k={self._k!r}, "
            f"fast_span_s={self._fast_span_s!r}, fast_weight={self._fast_weight!r}, "
            f"cv={self._cv!r})"
        )

    def predict(self, ride: Ride) -> pd.Series:
        oracle_ms = self._oracle_ms(ride)
        noisy_ms = float(self._rng.normal(oracle_ms, oracle_ms * self._cv))
        inner = AdaptiveLerpSpeedEstimator(
            prior_ms=noisy_ms,
            tau=self._tau,
            k=self._k,
            fast_span_s=self._fast_span_s,
            fast_weight=self._fast_weight,
        )
        return inner.predict(ride)


class PriorEWMASpeedEstimator(BaseEstimator):
    """EWMA speed estimator seeded with a prior.

    Uses the prior speed for initial observations so the EWMA produces
    estimates from the first row. The prior is gradually diluted as
    real data accumulates.

    Parameters
    ----------
    span_s : float, optional
        EWMA span in seconds. Defaults to EWMA_SPAN_S (3600 s).
    alpha : float, optional
        Smoothing factor override.
    prior_ms : float
        Prior speed in m/s (e.g. from `compute_global_prior`).
    moving_only : bool, optional
        If True (default), NaN out speed during pauses before smoothing.
    """

    def __init__(
        self,
        span_s: float | None = None,
        alpha: float | None = None,
        prior_ms: float = 5.0,
        moving_only: bool = True,
    ) -> None:
        self._span_s = EWMA_SPAN_S if span_s is None else span_s
        self._alpha = alpha
        self._prior_ms = prior_ms
        self._moving_only = moving_only

    def __str__(self) -> str:
        mode = "moving" if self._moving_only else "elapsed"
        prior_kmh = self._prior_ms * 3.6
        if self._alpha is not None:
            return (
                f"EWMA+prior speed (α={self._alpha}, prior={prior_kmh:.1f}km/h, {mode})"
            )
        return f"EWMA+prior speed ({int(self._span_s)}s, prior={prior_kmh:.1f}km/h, {mode})"

    def __repr__(self) -> str:
        return (
            f"PriorEWMASpeedEstimator(span_s={self._span_s!r}, "
            f"alpha={self._alpha!r}, prior_ms={self._prior_ms!r}, "
            f"moving_only={self._moving_only!r})"
        )

    def predict(self, ride: Ride) -> pd.Series:
        df = ride.df
        speed = self.safe_divide(df["delta_distance"], df["delta_time"])
        if self._moving_only:
            speed = speed.where(~df["paused"])
        speed = _seed_prior(speed, self._prior_ms)
        ewm_kwargs: dict = {"min_periods": 1}
        if self._alpha is not None:
            ewm_kwargs["alpha"] = self._alpha
        else:
            ewm_kwargs["span"] = self._span_s
        return speed.ewm(**ewm_kwargs).mean()


class GradientPriorEstimator(BaseEstimator):
    """Static gradient-aware ETA estimator.

    Uses decimated route segments and empirical speed ratios per gradient
    bin to estimate remaining time. For each row, sums
    ``segment_distance / (v_flat * ratio[bin])`` over all remaining segments.

    Parameters
    ----------
    v_flat_kmh : float
        Assumed flat-ground speed in km/h.
    ratios : dict[int, float]
        Mapping of gradient bin (left-edge %) to speed ratio relative to flat.
    """

    def __init__(self, v_flat_kmh: float, ratios: dict[int, float]) -> None:
        self._v_flat_ms = v_flat_kmh / 3.6
        self._ratios = ratios

    def __str__(self) -> str:
        return f"gradient prior ({self._v_flat_ms * 3.6:.1f} km/h flat)"

    def __repr__(self) -> str:
        return (
            f"GradientPriorEstimator(v_flat_kmh={self._v_flat_ms * 3.6!r}, "
            f"ratios=<{len(self._ratios)} bins>)"
        )

    def _ratio_for(self, gradient_frac: float) -> float:
        """Look up the speed ratio for a gradient, clamping to known bins."""
        bin_pct = math.floor(gradient_frac * 100)
        bin_pct = max(min(bin_pct, max(self._ratios)), min(self._ratios))
        return self._ratios.get(bin_pct, 1.0)

    def _ttg_from(self, distance_m: float, segments: list[RouteSegment]) -> float:
        """Time-to-go in seconds from `distance_m` to end of route."""
        ttg = 0.0
        for seg in segments:
            if seg.end_distance_m <= distance_m:
                continue
            start = max(seg.start_distance_m, distance_m)
            remaining = seg.end_distance_m - start
            ratio = self._ratio_for(seg.gradient)
            ttg += remaining / (self._v_flat_ms * ratio)
        return ttg

    def predict(self, ride: Ride) -> pd.Series:
        _, segments = decimate_to_gradient_segments(ride.df)
        df = ride.df
        total_dist = df["distance_m"].iloc[-1]

        # Precompute TTG at each segment start (+ end of route = 0)
        seg_starts = np.array([s.start_distance_m for s in segments] + [total_dist])
        seg_ttgs = np.empty(len(seg_starts))
        seg_ttgs[-1] = 0.0
        for i in range(len(segments) - 1, -1, -1):
            seg = segments[i]
            length = seg.end_distance_m - seg.start_distance_m
            ratio = self._ratio_for(seg.gradient)
            seg_ttgs[i] = seg_ttgs[i + 1] + length / (self._v_flat_ms * ratio)

        # For each row, interpolate TTG from precomputed segment boundaries
        distances = df["distance_m"].values
        idx = np.searchsorted(seg_starts, distances, side="right") - 1
        idx = idx.clip(0, len(segments) - 1)

        # TTG = ttg_at_next_boundary + partial_segment_time
        ttg = np.empty(len(distances))
        for i, (d, si) in enumerate(zip(distances, idx)):
            seg = segments[si]
            past_in_seg = d - seg.start_distance_m
            ratio = self._ratio_for(seg.gradient)
            partial_time = past_in_seg / (self._v_flat_ms * ratio)
            ttg[i] = seg_ttgs[si] - partial_time

        # Back-derive effective speed: remaining_distance / ttg
        remaining = total_dist - distances
        valid = (ttg > 0) & (remaining > 0)
        safe_ttg = np.where(valid, ttg, 1.0)
        speed = np.where(valid, remaining / safe_ttg, np.nan)
        return pd.Series(speed, index=df.index)


class PriorDEWMASpeedEstimator(BaseEstimator):
    """Double EWMA speed estimator seeded with a prior.

    Weighted blend of slow and fast `PriorEWMASpeedEstimator` components.

    Parameters
    ----------
    slow_span_s : float, optional
        Span for the slow EWMA. Default 3600 (60 min).
    fast_span_s : float, optional
        Span for the fast EWMA. Default 600 (10 min).
    slow_alpha : float, optional
        Alpha override for the slow EWMA.
    fast_alpha : float, optional
        Alpha override for the fast EWMA.
    slow_weight : float, optional
        Weight for the slow component. Default 0.7.
    fast_weight : float, optional
        Weight for the fast component. Default 0.3.
    prior_ms : float
        Prior speed in m/s.
    moving_only : bool, optional
        If True (default), NaN out speed during pauses.
    """

    def __init__(
        self,
        slow_span_s: float = 3600.0,
        fast_span_s: float = 600.0,
        slow_alpha: float | None = None,
        fast_alpha: float | None = None,
        slow_weight: float = 0.7,
        fast_weight: float = 0.3,
        prior_ms: float = 5.0,
        moving_only: bool = True,
    ) -> None:
        self._slow = PriorEWMASpeedEstimator(
            span_s=slow_span_s,
            alpha=slow_alpha,
            prior_ms=prior_ms,
            moving_only=moving_only,
        )
        self._fast = PriorEWMASpeedEstimator(
            span_s=fast_span_s,
            alpha=fast_alpha,
            prior_ms=prior_ms,
            moving_only=moving_only,
        )
        self._slow_weight = slow_weight
        self._fast_weight = fast_weight

    def __str__(self) -> str:
        return (
            f"DEWMA+prior speed ({self._fast} × {self._fast_weight} "
            f"+ {self._slow} × {self._slow_weight})"
        )

    def __repr__(self) -> str:
        return (
            f"PriorDEWMASpeedEstimator(slow={self._slow!r}, fast={self._fast!r}, "
            f"slow_weight={self._slow_weight!r}, fast_weight={self._fast_weight!r})"
        )

    def predict(self, ride: Ride) -> pd.Series:
        slow = self._slow.predict(ride)
        fast = self._fast.predict(ride)
        return self._slow_weight * slow + self._fast_weight * fast
