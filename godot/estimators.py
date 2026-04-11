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

        For ETA calculation: ``remaining_distance / predict()`` gives
        estimated remaining time. For most estimators this equals the
        instantaneous speed estimate. Gradient-aware estimators return
        an effective route-weighted average instead.

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

    def predict_current(self, ride: Ride) -> pd.Series:
        """Return estimated *instantaneous* speed in m/s at each row.

        What speed the rider is expected to be doing right now, given
        the current gradient / conditions. Defaults to `predict()`,
        which is correct for estimators where effective and instantaneous
        speed are the same (rolling averages, EWMA, etc.).

        Gradient-aware estimators override this to return
        ``v_flat * ratio[current_gradient]`` per row.
        """
        return self.predict(ride)


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
    - **final**: `lerp(slow, fast, fast_weight)`

    During a pause, `confidence = ewma(is_moving, span=tau)` decays
    exponentially, blending the slow component from `avg_total` toward
    a floor derived from the ride data:

        floor = cumsum(dd) / (cumsum(dt_elapsed) + k * tau)

    `tau` encodes typical pause length; `k` (default 2) accounts for
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
        Floor multiplier on tau. `floor = distance / (elapsed + k*tau)`.
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
    `segment_distance / (v_flat * ratio[bin])` over all remaining segments.

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

    def predict_current(self, ride: Ride) -> pd.Series:
        """Predicted instantaneous speed based on current segment gradient."""
        _, segments = decimate_to_gradient_segments(ride.df)
        df = ride.df
        distances = df["distance_m"].values
        seg_ends = np.array([s.end_distance_m for s in segments])
        idx = np.searchsorted(seg_ends, distances, side="left").clip(
            max=len(segments) - 1
        )
        speed = np.array(
            [self._v_flat_ms * self._ratio_for(segments[i].gradient) for i in idx]
        )
        return pd.Series(speed, index=df.index)


def physics_gradient_ratios(
    mass_kg: float,
    v_flat_ms: float,
    cda: float = 0.35,
    crr: float = 0.005,
    rho: float = 1.225,
    grad_min_pct: int = -20,
    grad_max_pct: int = 20,
) -> dict[int, float]:
    """Speed ratio per gradient from a constant-power cubic model.

    Solves `r**3 + (beta*cos(theta) + alpha_prime*sin(theta)) * r = 1 + beta`
    for each integer gradient bin, where
    `alpha_prime = m * g / (k_a * v_flat**2)`,
    `beta = Crr * alpha_prime`, `k_a = 0.5 * rho * CdA`
    and `r = v(theta) / v_flat`. Cardano's closed form applies with
    `-q/2 = (1 + beta) / 2`.

    CdA and Crr are held fixed — they shift the cubic coefficients but
    residual error is expected to be absorbed by downstream EWMA. Setting
    `crr=0` recovers the drag-only model.

    Parameters
    ----------
    mass_kg : float
        Rider + bike mass (kg).
    v_flat_ms : float
        Reference flat-ground speed (m/s).
    cda : float, optional
        Drag area CdA (m**2). Default 0.35.
    crr : float, optional
        Rolling resistance coefficient. Default 0.005 (good road tires
        on tarmac). Pass 0 to recover the drag-only model.
    rho : float, optional
        Air density (kg/m**3). Default 1.225.
    grad_min_pct, grad_max_pct : int, optional
        Inclusive integer gradient range in percent. Default -20..20.

    Returns
    -------
    dict[int, float]
        Integer gradient (%) -> speed ratio relative to `v_flat_ms`.
    """
    g = 9.81
    k_a = 0.5 * rho * cda
    alpha_prime = (mass_kg * g) / (k_a * v_flat_ms**2)
    beta = crr * alpha_prime
    half_rhs = (1 + beta) / 2  # = -q / 2

    ratios: dict[int, float] = {}
    for pct in range(grad_min_pct, grad_max_pct + 1):
        theta = math.atan(pct / 100)
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        p = beta * cos_t + alpha_prime * sin_t

        disc = half_rhs**2 + (p**3) / 27
        if disc >= 0:
            sqrt_d = math.sqrt(disc)
            r = math.cbrt(half_rhs + sqrt_d) + math.cbrt(half_rhs - sqrt_d)
        else:
            # Casus irreducibilis — trig form, k=0 gives the largest root
            amp = 2 * math.sqrt(-p / 3)
            phi = math.acos(-3 * half_rhs / p * math.sqrt(-3 / p))
            r = amp * math.cos(phi / 3)

        ratios[pct] = r
    return ratios


class PhysicsGradientPriorEstimator(GradientPriorEstimator):
    """Gradient-aware ETA estimator with ratios from a constant-power model.

    Builds speed ratios from rider mass and an assumed flat speed via the
    cubic power balance
    `P = 0.5 * rho * CdA * v**3 + Crr * m * g * cos(theta) * v
    + m * g * sin(theta) * v`,
    then delegates to `GradientPriorEstimator` for forward-looking time-to-go.

    The only rider input is mass; CdA and Crr are held at typical fixed
    values.

    Parameters
    ----------
    mass_kg : float
        Rider + bike mass (kg).
    v_flat_kmh : float
        Assumed flat-ground speed (km/h).
    cda : float, optional
        Drag area CdA (m**2). Default 0.35.
    crr : float, optional
        Rolling resistance coefficient. Default 0.005.
    rho : float, optional
        Air density (kg/m**3). Default 1.225.
    grad_min_pct, grad_max_pct : int, optional
        Inclusive range of gradient bins to precompute.
    """

    def __init__(
        self,
        mass_kg: float,
        v_flat_kmh: float,
        cda: float = 0.35,
        crr: float = 0.005,
        rho: float = 1.225,
        grad_min_pct: int = -20,
        grad_max_pct: int = 20,
    ) -> None:
        ratios = physics_gradient_ratios(
            mass_kg=mass_kg,
            v_flat_ms=v_flat_kmh / 3.6,
            cda=cda,
            crr=crr,
            rho=rho,
            grad_min_pct=grad_min_pct,
            grad_max_pct=grad_max_pct,
        )
        super().__init__(v_flat_kmh=v_flat_kmh, ratios=ratios)
        self._mass_kg = mass_kg
        self._cda = cda
        self._crr = crr
        self._rho = rho

    def __str__(self) -> str:
        return (
            f"physics gradient prior ({self._v_flat_ms * 3.6:.1f} km/h flat, "
            f"m={self._mass_kg:.0f}kg, CdA={self._cda}, Crr={self._crr})"
        )

    def __repr__(self) -> str:
        return (
            f"PhysicsGradientPriorEstimator(mass_kg={self._mass_kg!r}, "
            f"v_flat_kmh={self._v_flat_ms * 3.6!r}, cda={self._cda!r}, "
            f"crr={self._crr!r}, rho={self._rho!r})"
        )


class AdaptivePhysicsEstimator(BaseEstimator):
    """Physics gradient prior with dual-EWMA calibration.

    Combines the forward-looking gradient-aware TTG from
    `PhysicsGradientPriorEstimator` with two learned corrections:

    - **slow**: EWMA of ``actual_speed / physics_predicted_speed``,
      optionally restricted to near-flat rows (|gradient| < `cal_max_grad`).
      Learns the rider's true v_flat without gradient-correlated bias.
    - **fast**: EWMA of the residual after slow correction, using all
      gradients. Catches short-term deviations (wind, fatigue, drafting).

    The combined correction scales the physics prediction:
    ``calibrated = physics * slow * fast``. At ride start both corrections
    are 1.0, so the pure physics prior is used.

    Parameters
    ----------
    mass_kg : float
        Rider + bike mass (kg).
    v_flat_kmh : float
        Prior flat-ground speed (km/h).
    slow_span_s : float, optional
        Span for the slow calibration EWMA. Default 3600 (60 min).
    fast_span_s : float, optional
        Span for the fast correction EWMA. Default 300 (5 min).
    cal_max_grad : float, optional
        Maximum |gradient| (as fraction) for slow calibration data.
        Default 0.02 (2%). Set to 1.0 to use all gradients.
    cda : float, optional
        Drag area CdA (m**2). Default 0.35.
    crr : float, optional
        Rolling resistance coefficient. Default 0.005.
    rho : float, optional
        Air density (kg/m**3). Default 1.225.
    """

    def __init__(
        self,
        mass_kg: float,
        v_flat_kmh: float,
        slow_span_s: float = 3600.0,
        fast_span_s: float = 300.0,
        cal_max_grad: float = 0.02,
        cda: float = 0.35,
        crr: float = 0.005,
        rho: float = 1.225,
    ) -> None:
        self._physics = PhysicsGradientPriorEstimator(
            mass_kg=mass_kg, v_flat_kmh=v_flat_kmh, cda=cda, crr=crr, rho=rho
        )
        self._slow_span_s = slow_span_s
        self._fast_span_s = fast_span_s
        self._cal_max_grad = cal_max_grad

    def __str__(self) -> str:
        cal = (
            f"|g|<{self._cal_max_grad * 100:.0f}%"
            if self._cal_max_grad < 1.0
            else "all"
        )
        return (
            f"adaptive physics (slow={int(self._slow_span_s)}s [{cal}], "
            f"fast={int(self._fast_span_s)}s, {self._physics})"
        )

    def __repr__(self) -> str:
        return (
            f"AdaptivePhysicsEstimator(physics={self._physics!r}, "
            f"slow_span_s={self._slow_span_s!r}, fast_span_s={self._fast_span_s!r}, "
            f"cal_max_grad={self._cal_max_grad!r})"
        )

    def _corrections(self, ride: Ride) -> tuple[pd.Series, pd.Series]:
        """Compute slow and fast EWMA correction factors."""
        df = ride.df
        moving = ~df["paused"]

        # Actual instantaneous speed (moving rows only)
        actual = self.safe_divide(df["delta_distance"], df["delta_time"])
        actual = actual.where(moving)

        # Physics predicted speed at current gradient
        predicted = self._physics.predict_current(ride)

        # Residual ratio: actual / predicted
        ratio = self.safe_divide(actual, predicted)

        # Slow EWMA: optionally flat-only
        _, segments = decimate_to_gradient_segments(df)
        seg_ends = np.array([s.end_distance_m for s in segments])
        seg_idx = np.searchsorted(seg_ends, df["distance_m"].values, side="left").clip(
            max=len(segments) - 1
        )
        gradients = pd.Series([segments[i].gradient for i in seg_idx], index=df.index)
        ratio_slow = ratio.where(gradients.abs() <= self._cal_max_grad)
        slow = (
            ratio_slow.ewm(span=self._slow_span_s, min_periods=1)
            .mean()
            .ffill()
            .fillna(1.0)
        )

        # Fast EWMA: residual after slow correction, all gradients
        ratio_fast = self.safe_divide(actual, predicted * slow)
        fast = ratio_fast.ewm(span=self._fast_span_s, min_periods=1).mean().fillna(1.0)

        return slow, fast

    def predict(self, ride: Ride) -> pd.Series:
        slow, _ = self._corrections(ride)
        base = self._physics.predict(ride)
        return base * slow

    def predict_current(self, ride: Ride) -> pd.Series:
        slow, fast = self._corrections(ride)
        base = self._physics.predict_current(ride)
        return base * slow * fast


def _row_gradients(ride: Ride) -> tuple[pd.Series, list[RouteSegment]]:
    """Per-row gradient (fraction) and segment list from decimated route."""
    df = ride.df
    _, segments = decimate_to_gradient_segments(df)
    seg_ends = np.array([s.end_distance_m for s in segments])
    seg_idx = np.searchsorted(seg_ends, df["distance_m"].values, side="left").clip(
        max=len(segments) - 1
    )
    gradients = pd.Series([segments[i].gradient for i in seg_idx], index=df.index)
    return gradients, segments


def _gradient_bin_pct(gradient_frac: pd.Series, bin_size: int = 1) -> pd.Series:
    """Convert fractional gradient to binned percent (floor).

    With ``bin_size=1`` (default), returns integer percent: 0.054 → 5.
    With ``bin_size=3``, groups into 3%-wide bins: 0.054 → 3 (covers [3%, 6%)).
    """
    pct = np.floor(gradient_frac * 100)
    if bin_size > 1:
        pct = np.floor(pct / bin_size) * bin_size
    return pct.astype(int)


class BinnedAdaptivePhysicsEstimator(BaseEstimator):
    """Physics gradient prior with slow v_flat EWMA and per-bin fast corrections.

    Like `AdaptivePhysicsEstimator` but the fast EWMA runs per gradient
    bin. Each bin independently learns a correction for its gradient, so a
    climb-specific error doesn't contaminate descent predictions. Per-bin
    corrections are applied to both `predict()` (TTG) and `predict_current()`.

    Parameters
    ----------
    mass_kg : float
        Rider + bike mass (kg).
    v_flat_kmh : float
        Prior flat-ground speed (km/h).
    slow_span_s : float, optional
        Span for the slow calibration EWMA. Default 3600 (60 min).
    fast_span_s : float, optional
        Span for the per-bin fast EWMA. Default 300 (5 min).
    bin_size : int, optional
        Gradient bin width in percent. Default 1 (1% bins).
        Use 3 for coarser bins with more observations per bucket.
    cal_max_grad : float, optional
        Maximum |gradient| (fraction) for slow calibration. Default 0.02.
    cda, crr, rho : float, optional
        Physics model parameters.
    """

    def __init__(
        self,
        mass_kg: float,
        v_flat_kmh: float,
        slow_span_s: float = 3600.0,
        fast_span_s: float = 300.0,
        bin_size: int = 1,
        cal_max_grad: float = 0.02,
        cda: float = 0.35,
        crr: float = 0.005,
        rho: float = 1.225,
    ) -> None:
        self._physics = PhysicsGradientPriorEstimator(
            mass_kg=mass_kg, v_flat_kmh=v_flat_kmh, cda=cda, crr=crr, rho=rho
        )
        self._slow_span_s = slow_span_s
        self._fast_span_s = fast_span_s
        self._bin_size = bin_size
        self._cal_max_grad = cal_max_grad

    def __str__(self) -> str:
        cal = (
            f"|g|<{self._cal_max_grad * 100:.0f}%"
            if self._cal_max_grad < 1.0
            else "all"
        )
        return (
            f"binned adaptive physics (slow={int(self._slow_span_s)}s [{cal}], "
            f"fast={int(self._fast_span_s)}s/{self._bin_size}%bin, {self._physics})"
        )

    def __repr__(self) -> str:
        return (
            f"BinnedAdaptivePhysicsEstimator(physics={self._physics!r}, "
            f"slow_span_s={self._slow_span_s!r}, fast_span_s={self._fast_span_s!r}, "
            f"bin_size={self._bin_size!r}, cal_max_grad={self._cal_max_grad!r})"
        )

    def _corrections(self, ride: Ride) -> tuple[pd.Series, pd.Series]:
        """Slow scalar EWMA + per-bin fast EWMA corrections."""
        df = ride.df
        moving = ~df["paused"]

        actual = self.safe_divide(df["delta_distance"], df["delta_time"])
        actual = actual.where(moving)

        predicted = self._physics.predict_current(ride)
        ratio = self.safe_divide(actual, predicted)

        gradients, _ = _row_gradients(ride)
        grad_bins = _gradient_bin_pct(gradients, self._bin_size)

        # Slow EWMA: flat-only (or all), scalar correction
        ratio_slow = ratio.where(gradients.abs() <= self._cal_max_grad)
        slow = (
            ratio_slow.ewm(span=self._slow_span_s, min_periods=1)
            .mean()
            .ffill()
            .fillna(1.0)
        )

        # Per-bin fast EWMA: residual after slow correction
        ratio_fast = self.safe_divide(actual, predicted * slow)
        fast = pd.Series(1.0, index=df.index)
        for bin_pct, grp in ratio_fast.groupby(grad_bins):
            bin_ewma = grp.ewm(span=self._fast_span_s, min_periods=1).mean()
            fast.loc[bin_ewma.index] = bin_ewma
        fast = fast.fillna(1.0)

        return slow, fast

    def predict(self, ride: Ride) -> pd.Series:
        slow, fast = self._corrections(ride)
        gradients, segments = _row_gradients(ride)
        df = ride.df
        v_flat = self._physics._v_flat_ms
        grad_bins = _gradient_bin_pct(gradients, self._bin_size)
        bs = self._bin_size

        total_dist = df["distance_m"].iloc[-1]
        distances = df["distance_m"].values

        seg_base_speeds = np.array(
            [v_flat * self._physics._ratio_for(s.gradient) for s in segments]
        )
        seg_bins = np.array(
            [int(np.floor(s.gradient * 100 / bs) * bs) for s in segments]
        )
        seg_start_dists = np.array([s.start_distance_m for s in segments])
        seg_end_dists = np.array([s.end_distance_m for s in segments])

        # Scan forward: accumulate per-bin fast corrections and compute TTG
        ttg = np.empty(len(distances))
        running_bin_fast: dict[int, float] = {}

        for i in range(len(df)):
            b = int(grad_bins.iloc[i])
            f = fast.iloc[i]
            if not np.isnan(f):
                running_bin_fast[b] = f

            slow_i = slow.iloc[i]
            d = distances[i]

            si = int(np.searchsorted(seg_end_dists, d, side="right"))
            si = min(si, len(segments) - 1)

            t = 0.0
            for j in range(si, len(segments)):
                seg_start = (
                    max(seg_start_dists[j], d) if j == si else seg_start_dists[j]
                )
                seg_len = seg_end_dists[j] - seg_start
                if seg_len <= 0:
                    continue
                bin_corr = running_bin_fast.get(seg_bins[j], 1.0)
                t += seg_len / (seg_base_speeds[j] * slow_i * bin_corr)

            ttg[i] = t

        remaining = total_dist - distances
        valid = (ttg > 0) & (remaining > 0)
        safe_ttg = np.where(valid, ttg, 1.0)
        speed = np.where(valid, remaining / safe_ttg, np.nan)
        return pd.Series(speed, index=df.index)

    def predict_current(self, ride: Ride) -> pd.Series:
        slow, fast = self._corrections(ride)
        base = self._physics.predict_current(ride)
        return base * slow * fast


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
