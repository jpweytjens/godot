"""ETA estimators for cycling ride prediction."""

from __future__ import annotations

import bisect
import math
from collections import deque
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


# ---------------------------------------------------------------------------
# v_flat estimators — estimate flat-ground speed from observed ride data
# ---------------------------------------------------------------------------


class VFlatEstimator:
    """Base class for flat-ground speed estimators.

    Subclasses implement `estimate` which returns a per-row v_flat series
    (m/s) given a ride and a set of gradient speed ratios.
    """

    @staticmethod
    def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Element-wise division, returning NaN where either operand is zero."""
        return (numerator / denominator).where((denominator != 0) & (numerator != 0))

    def estimate(
        self,
        ride: Ride,
        ratios: dict[int, float],
        v_flat_init_ms: float,
    ) -> pd.Series:
        """Return estimated v_flat (m/s) at each row.

        Parameters
        ----------
        ride : Ride
            Prepared ride from `load_ride`.
        ratios : dict[int, float]
            Gradient bin (%) → speed ratio relative to flat.
        v_flat_init_ms : float
            Initial v_flat guess (m/s).

        Returns
        -------
        pd.Series
            v_flat estimate (m/s) at each row.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class StaticVFlat(VFlatEstimator):
    """Returns the initial v_flat unchanged for every row."""

    def estimate(
        self,
        ride: Ride,
        ratios: dict[int, float],
        v_flat_init_ms: float,
    ) -> pd.Series:
        return pd.Series(v_flat_init_ms, index=ride.df.index)

    def __str__(self) -> str:
        return "static"


class OracleVFlat(VFlatEstimator):
    """Returns the actual whole-ride flat moving average at every row.

    This is a cheating estimator — it uses future information to return
    the true flat-section average speed as a constant. Useful for
    isolating the EWMA correction layer from v_flat estimation errors.

    Parameters
    ----------
    max_grad : float
        Max |gradient| (fraction) to consider flat. Default 0.02.
    """

    def __init__(self, max_grad: float = 0.02) -> None:
        self._max_grad = max_grad

    def estimate(
        self,
        ride: Ride,
        ratios: dict[int, float],
        v_flat_init_ms: float,
    ) -> pd.Series:
        df = ride.df
        moving = ~df["paused"]
        gradients, _ = _row_gradients(ride)

        flat_mask = moving & (gradients.abs() < self._max_grad)
        flat_dist = df.loc[flat_mask, "delta_distance"].sum()
        flat_time = df.loc[flat_mask, "delta_time"].sum()
        v_flat_ms = (flat_dist / flat_time) if flat_time > 0 else v_flat_init_ms

        return pd.Series(v_flat_ms, index=df.index)

    def __str__(self) -> str:
        return "oracle"

    def __repr__(self) -> str:
        return f"OracleVFlat(max_grad={self._max_grad!r})"


class FlatSpeedVFlat(VFlatEstimator):
    """Cumulative average moving speed on flat sections.

    Uses only rows where ``|gradient| < max_grad`` and the rider is
    moving. Returns the expanding mean of instantaneous speed on those
    rows, forward-filled for non-flat rows. Falls back to
    ``v_flat_init_ms`` until the first flat observation.

    Parameters
    ----------
    max_grad : float, optional
        Maximum absolute gradient (fraction) to consider flat.
        Default 0.02 (2%).
    """

    def __init__(self, max_grad: float = 0.02) -> None:
        self._max_grad = max_grad

    def estimate(
        self,
        ride: Ride,
        ratios: dict[int, float],
        v_flat_init_ms: float,
    ) -> pd.Series:
        df = ride.df
        moving = ~df["paused"]
        gradients, _ = _row_gradients(ride)

        flat_moving = moving & (gradients.abs() < self._max_grad)
        speed = self.safe_divide(df["delta_distance"], df["delta_time"])
        flat_speed = speed.where(flat_moving)

        # Cumulative mean of flat-only speed, forward-filled
        cum_sum = flat_speed.cumsum()
        cum_count = flat_moving.cumsum()
        v_flat = self.safe_divide(cum_sum, cum_count).ffill().fillna(v_flat_init_ms)

        return v_flat

    def __str__(self) -> str:
        return f"flat-speed(|g|<{self._max_grad * 100:.0f}%)"

    def __repr__(self) -> str:
        return f"FlatSpeedVFlat(max_grad={self._max_grad!r})"


class WeightedGainVFlat(VFlatEstimator):
    """Adaptive v_flat using gradient-weighted gain schedule.

    Back-derives v_flat from each observation using the empirical speed
    ratios: ``v_flat_obs = v_actual / ratio[gradient]``. Updates via a
    weighted gain schedule that converges fast (Welford-like) then
    floors to a slow EWMA for drift tracking.

    Observations are weighted by ``cos²(θ) * min(dt, τ) / τ``, which
    downweights steep gradients (noisy ratio lookup) and very short
    time steps (unreliable speed).

    Parameters
    ----------
    tau_s : float, optional
        Segment duration saturation (seconds). Default 30.
    lambda_slow : float, optional
        Minimum gain floor for long-term drift tracking. Default 0.002.
    """

    def __init__(self, tau_s: float = 30.0, lambda_slow: float = 0.002) -> None:
        self._tau_s = tau_s
        self._lambda_slow = lambda_slow

    def estimate(
        self,
        ride: Ride,
        ratios: dict[int, float],
        v_flat_init_ms: float,
    ) -> pd.Series:
        df = ride.df
        moving = ~df["paused"]
        gradients, _ = _row_gradients(ride)

        # Precompute per-row ratio from the ratio table
        grad_pct = np.floor(gradients.values * 100).astype(int)
        min_bin = min(ratios)
        max_bin = max(ratios)
        row_ratios = np.array(
            [ratios.get(max(min_bin, min(max_bin, g)), 1.0) for g in grad_pct]
        )

        actual_speed = self.safe_divide(df["delta_distance"], df["delta_time"]).values
        dt = df["delta_time"].values

        v_flat = v_flat_init_ms
        W = 0.0
        out = np.empty(len(df))

        for i in range(len(df)):
            if not moving.iloc[i] or np.isnan(actual_speed[i]) or row_ratios[i] <= 0:
                out[i] = v_flat
                continue

            v_flat_obs = actual_speed[i] / row_ratios[i]
            if v_flat_obs <= 0 or not np.isfinite(v_flat_obs):
                out[i] = v_flat
                continue

            theta = math.atan(gradients.iloc[i])
            w = math.cos(theta) ** 2 * min(dt[i], self._tau_s) / self._tau_s

            W += w
            gain = max(w / W, self._lambda_slow)
            v_flat += gain * (v_flat_obs - v_flat)
            v_flat = max(v_flat, 2.0)  # floor ~7 km/h

            out[i] = v_flat

        return pd.Series(out, index=df.index)

    def __str__(self) -> str:
        return f"weighted-gain(τ={self._tau_s:.0f}s, λ={self._lambda_slow})"

    def __repr__(self) -> str:
        return (
            f"WeightedGainVFlat(tau_s={self._tau_s!r}, "
            f"lambda_slow={self._lambda_slow!r})"
        )


class EwmaLockVFlat(VFlatEstimator):
    """Flat-only EWMA with stability lock.

    Runs an EWMA on flat-section speeds and locks v_flat once the
    estimate stabilises. After locking, v_flat is frozen for the
    remainder of the ride — all subsequent drift is left to the
    slow/fast EWMA corrections.

    Parameters
    ----------
    max_grad : float
        Max |gradient| (fraction) to consider flat. Default 0.02.
    ewma_span_s : float
        EWMA span in seconds. Default 60.
    min_flat_s : float
        Min cumulative flat moving seconds before lock check. Default 60.
    stability_window_s : float
        Look-back window (flat seconds) for stability check. Default 60.
    lock_threshold_kmh : float
        Max EWMA drift (km/h) to declare stable and lock. Default 1.0.
    """

    def __init__(
        self,
        max_grad: float = 0.02,
        ewma_span_s: float = 60.0,
        min_flat_s: float = 60.0,
        stability_window_s: float = 60.0,
        lock_threshold_kmh: float = 1.0,
    ) -> None:
        self._max_grad = max_grad
        self._ewma_span_s = ewma_span_s
        self._min_flat_s = min_flat_s
        self._stability_window_s = stability_window_s
        self._lock_threshold_ms = lock_threshold_kmh / 3.6

    def estimate(
        self,
        ride: Ride,
        ratios: dict[int, float],
        v_flat_init_ms: float,
    ) -> pd.Series:
        df = ride.df
        moving = ~df["paused"]
        gradients, _ = _row_gradients(ride)
        speed = self.safe_divide(df["delta_distance"], df["delta_time"]).values
        grad_vals = gradients.abs().values

        alpha = 2.0 / (self._ewma_span_s + 1.0)
        ewma = np.nan
        flat_cum_s = 0.0
        locked_value = np.nan
        # Ring buffer: (flat_cumtime_s, ewma_value)
        history: deque[tuple[float, float]] = deque()

        out = np.empty(len(df))
        dt = df["delta_time"].values

        for i in range(len(df)):
            if not np.isnan(locked_value):
                out[i] = locked_value
                continue

            is_flat_moving = (
                moving.iloc[i]
                and grad_vals[i] < self._max_grad
                and not np.isnan(speed[i])
                and speed[i] > 0
            )

            if is_flat_moving:
                flat_cum_s += dt[i]
                if np.isnan(ewma):
                    ewma = speed[i]
                else:
                    ewma += alpha * (speed[i] - ewma)

                history.append((flat_cum_s, ewma))

                # Stability check
                if flat_cum_s >= self._min_flat_s:
                    cutoff = flat_cum_s - self._stability_window_s
                    # Trim old entries beyond the window
                    while len(history) > 1 and history[0][0] < cutoff:
                        history.popleft()
                    if history[0][0] <= cutoff + 1.0:
                        old_ewma = history[0][1]
                        if abs(ewma - old_ewma) < self._lock_threshold_ms:
                            locked_value = ewma

            out[i] = ewma if not np.isnan(ewma) else v_flat_init_ms

        # Forward-fill through gaps where ewma was NaN (non-flat before first flat)
        result = pd.Series(out, index=df.index)
        return result

    def __str__(self) -> str:
        th = self._lock_threshold_ms * 3.6
        return f"ewma-lock(span={self._ewma_span_s:.0f}s, lock={th:.1f}kmh)"

    def __repr__(self) -> str:
        th = self._lock_threshold_ms * 3.6
        return (
            f"EwmaLockVFlat(max_grad={self._max_grad!r}, "
            f"ewma_span_s={self._ewma_span_s!r}, "
            f"min_flat_s={self._min_flat_s!r}, "
            f"stability_window_s={self._stability_window_s!r}, "
            f"lock_threshold_kmh={th!r})"
        )


class MedianLockVFlat(VFlatEstimator):
    """Expanding median of back-derived v_flat with stability lock.

    Back-derives ``v_flat_obs = v_actual / ratio[gradient]`` from every
    moving observation and tracks the expanding median. Locks once the
    median stabilises.

    The median naturally filters outlier v_flat derivations from steep
    gradients where the physics ratio is imprecise.

    Parameters
    ----------
    min_obs : int
        Min observations before lock check. Default 60.
    stability_window_obs : int
        Compare current median to snapshot this many observations ago.
        Default 60.
    lock_threshold_kmh : float
        Max median drift (km/h) to declare stable and lock. Default 1.0.
    """

    def __init__(
        self,
        min_obs: int = 60,
        stability_window_obs: int = 60,
        lock_threshold_kmh: float = 1.0,
    ) -> None:
        self._min_obs = min_obs
        self._stability_window_obs = stability_window_obs
        self._lock_threshold_ms = lock_threshold_kmh / 3.6

    def estimate(
        self,
        ride: Ride,
        ratios: dict[int, float],
        v_flat_init_ms: float,
    ) -> pd.Series:
        df = ride.df
        moving = ~df["paused"]
        gradients, _ = _row_gradients(ride)

        grad_pct = np.floor(gradients.values * 100).astype(int)
        min_bin = min(ratios)
        max_bin = max(ratios)
        row_ratios = np.array(
            [ratios.get(max(min_bin, min(max_bin, g)), 1.0) for g in grad_pct]
        )

        actual_speed = self.safe_divide(df["delta_distance"], df["delta_time"]).values

        sorted_obs: list[float] = []
        obs_count = 0
        locked_value = np.nan
        last_snapshot_median = np.nan
        last_snapshot_count = 0
        current_median = np.nan

        out = np.empty(len(df))

        for i in range(len(df)):
            if not np.isnan(locked_value):
                out[i] = locked_value
                continue

            if (
                moving.iloc[i]
                and not np.isnan(actual_speed[i])
                and actual_speed[i] > 0
                and row_ratios[i] > 0
            ):
                v_flat_obs = actual_speed[i] / row_ratios[i]
                if np.isfinite(v_flat_obs) and v_flat_obs > 0:
                    bisect.insort(sorted_obs, v_flat_obs)
                    obs_count += 1
                    n = len(sorted_obs)
                    current_median = (
                        sorted_obs[n // 2]
                        if n % 2 == 1
                        else (sorted_obs[n // 2 - 1] + sorted_obs[n // 2]) / 2.0
                    )

                    # Take snapshot every stability_window_obs observations
                    if obs_count == self._min_obs:
                        last_snapshot_median = current_median
                        last_snapshot_count = obs_count
                    elif (
                        obs_count > self._min_obs
                        and (obs_count - last_snapshot_count)
                        >= self._stability_window_obs
                    ):
                        if (
                            abs(current_median - last_snapshot_median)
                            < self._lock_threshold_ms
                        ):
                            locked_value = current_median
                        last_snapshot_median = current_median
                        last_snapshot_count = obs_count

            out[i] = current_median if not np.isnan(current_median) else v_flat_init_ms

        return pd.Series(out, index=df.index)

    def __str__(self) -> str:
        th = self._lock_threshold_ms * 3.6
        return f"median-lock(min={self._min_obs}, lock={th:.1f}kmh)"

    def __repr__(self) -> str:
        th = self._lock_threshold_ms * 3.6
        return (
            f"MedianLockVFlat(min_obs={self._min_obs!r}, "
            f"stability_window_obs={self._stability_window_obs!r}, "
            f"lock_threshold_kmh={th!r})"
        )


class PriorFreeVFlat(VFlatEstimator):
    """Prior-free v_flat estimator with ride-time-scaled skip period.

    Uses ``v_flat_init_ms`` only to estimate total ride time (via the
    route segments and gradient ratios), which sets the skip period.
    During the skip the estimator returns ``v_flat_init_ms`` so the ETA
    estimator falls back to the bare realistic prior. After the skip,
    an expanding mean of ``v_actual / ratio[gradient]`` takes over —
    no EWMA, no prior contamination.

    Designed for use with `realistic_physics_ratios` where the gradient
    ratios are accurate enough that every observation yields a usable
    v_flat signal. The ETA estimator's own correction layers handle
    short-term deviations.

    Parameters
    ----------
    skip_fraction : float
        Fraction of estimated ride time to skip before accumulating.
        Default 0.01 (1%).
    min_skip_s : float
        Floor on skip period (moving seconds). Default 20.
    max_skip_s : float
        Ceiling on skip period (moving seconds). Default 300 (5 min).
    """

    def __init__(
        self,
        skip_fraction: float = 0.01,
        min_skip_s: float = 20.0,
        max_skip_s: float = 300.0,
    ) -> None:
        self._skip_fraction = skip_fraction
        self._min_skip_s = min_skip_s
        self._max_skip_s = max_skip_s

    def estimate(
        self,
        ride: Ride,
        ratios: dict[int, float],
        v_flat_init_ms: float,
    ) -> pd.Series:
        df = ride.df
        gradients, _ = _row_gradients(ride)
        grad_pct = np.floor(gradients.values * 100).astype(int)
        min_bin = min(ratios)
        max_bin = max(ratios)
        grad_pct = np.clip(grad_pct, min_bin, max_bin)
        row_ratios = np.array([ratios.get(g, 1.0) for g in grad_pct])

        speed = df["speed_ms"].values
        paused = df["paused"].values
        dt = df["delta_time"].values

        # Estimate ride time to scale skip period
        _, segments = decimate_to_gradient_segments(df)
        est_time_s = sum(
            (s.end_distance_m - s.start_distance_m)
            / max(
                0.5,
                v_flat_init_ms
                * ratios.get(
                    max(min_bin, min(max_bin, math.floor(s.gradient * 100))),
                    1.0,
                ),
            )
            for s in segments
        )
        skip_s = max(
            self._min_skip_s,
            min(self._max_skip_s, self._skip_fraction * est_time_s),
        )

        elapsed_moving = 0.0
        cum_sum = 0.0
        count = 0
        v_flat = v_flat_init_ms
        result = np.empty(len(df))

        for i in range(len(df)):
            if not paused[i] and speed[i] > 0.5 and row_ratios[i] > 0.05:
                elapsed_moving += dt[i]
                if elapsed_moving > skip_s:
                    cum_sum += speed[i] / row_ratios[i]
                    count += 1
                    v_flat = cum_sum / count

            result[i] = v_flat

        return pd.Series(result, index=df.index)

    def __str__(self) -> str:
        return (
            f"prior-free(skip={self._skip_fraction:.0%}, "
            f"[{self._min_skip_s:.0f}-{self._max_skip_s:.0f}]s)"
        )

    def __repr__(self) -> str:
        return (
            f"PriorFreeVFlat(skip_fraction={self._skip_fraction!r}, "
            f"min_skip_s={self._min_skip_s!r}, "
            f"max_skip_s={self._max_skip_s!r})"
        )


class AdaptiveGradientPriorEstimator(BaseEstimator):
    """Gradient-aware ETA estimator with adaptive v_flat.

    Like `GradientPriorEstimator` but v_flat is updated per row by a
    `VFlatEstimator`. The per-row TTG sums
    ``segment_distance / (v_flat[i] * ratio[bin])`` over remaining
    segments, using the current v_flat estimate.

    Parameters
    ----------
    v_flat_kmh : float
        Initial flat-ground speed guess (km/h).
    ratios : dict[int, float]
        Gradient bin (%) → speed ratio relative to flat.
    vflat_estimator : VFlatEstimator
        Strategy for updating v_flat during the ride.
    """

    def __init__(
        self,
        v_flat_kmh: float,
        ratios: dict[int, float],
        vflat_estimator: VFlatEstimator,
    ) -> None:
        self._v_flat_init_ms = v_flat_kmh / 3.6
        self._v_flat_ms = self._v_flat_init_ms  # compat with _ratio_for users
        self._ratios = ratios
        self._vflat_est = vflat_estimator

    def __str__(self) -> str:
        return (
            f"adaptive gradient prior "
            f"({self._v_flat_init_ms * 3.6:.1f} km/h init, "
            f"vflat={self._vflat_est})"
        )

    def __repr__(self) -> str:
        return (
            f"AdaptiveGradientPriorEstimator("
            f"v_flat_kmh={self._v_flat_init_ms * 3.6!r}, "
            f"ratios=<{len(self._ratios)} bins>, "
            f"vflat_estimator={self._vflat_est!r})"
        )

    def _ratio_for(self, gradient_frac: float) -> float:
        bin_pct = math.floor(gradient_frac * 100)
        bin_pct = max(min(bin_pct, max(self._ratios)), min(self._ratios))
        return self._ratios.get(bin_pct, 1.0)

    def predict(self, ride: Ride) -> pd.Series:
        _, segments = decimate_to_gradient_segments(ride.df)
        df = ride.df
        total_dist = df["distance_m"].iloc[-1]
        distances = df["distance_m"].values

        v_flat_series = self._vflat_est.estimate(
            ride, self._ratios, self._v_flat_init_ms
        )
        v_flat_arr = v_flat_series.values

        seg_starts = np.array([s.start_distance_m for s in segments])
        seg_ends = np.array([s.end_distance_m for s in segments])
        seg_ratios = np.array([self._ratio_for(s.gradient) for s in segments])

        speeds = np.empty(len(distances))
        for i in range(len(df)):
            d = distances[i]
            vf = v_flat_arr[i]

            si = int(np.searchsorted(seg_ends, d, side="right"))
            si = min(si, len(segments) - 1)

            ttg = 0.0
            for j in range(si, len(segments)):
                s_start = max(seg_starts[j], d) if j == si else seg_starts[j]
                s_len = seg_ends[j] - s_start
                if s_len <= 0:
                    continue
                ttg += s_len / (vf * seg_ratios[j])

            remaining = total_dist - d
            if ttg > 0 and remaining > 0:
                speeds[i] = remaining / ttg
            else:
                speeds[i] = np.nan

        return pd.Series(speeds, index=df.index)

    def predict_current(self, ride: Ride) -> pd.Series:
        _, segments = decimate_to_gradient_segments(ride.df)
        df = ride.df
        distances = df["distance_m"].values
        seg_ends = np.array([s.end_distance_m for s in segments])
        idx = np.searchsorted(seg_ends, distances, side="left").clip(
            max=len(segments) - 1
        )

        v_flat_series = self._vflat_est.estimate(
            ride, self._ratios, self._v_flat_init_ms
        )

        speed = np.array(
            [
                v_flat_series.iloc[i] * self._ratio_for(segments[idx[i]].gradient)
                for i in range(len(df))
            ]
        )
        return pd.Series(speed, index=df.index)


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


def realistic_physics_ratios(
    mass_kg: float,
    v_flat_ms: float,
    cda: float = 0.35,
    crr: float = 0.005,
    rho: float = 1.225,
    headwind_ms: float = 2.22,
    power_watts: float | None = None,
    climb_effort: float = 0.5,
    descent_confidence: float = 0.65,
    grad_min_pct: int = -20,
    grad_max_pct: int = 20,
) -> dict[int, float]:
    """Speed ratio per gradient from a realistic power model.

    Improves on `physics_gradient_ratios` with three corrections:

    1. **Headwind-corrected P_flat** — back-solves rider power assuming
       an effective headwind, giving a realistic wattage. If `power_watts`
       is provided directly, uses that instead.
    2. **Gradient-dependent power** — on climbs, rider pushes harder
       (interpolated by `climb_effort`); on descents, rider backs off
       proportional to gravity's contribution (scaled by
       `descent_confidence`).
    3. **Freewheel descent cap** — caps descent speed at
       `v_flat + descent_confidence * (v_freewheel - v_flat)` when the
       gravity-only terminal velocity exceeds v_flat (Nee & Herterich
       2022).

    The headwind also enters the per-gradient solver, so drag on descents
    is realistic rather than zero-wind.

    Parameters
    ----------
    mass_kg : float
        Rider + bike mass (kg).
    v_flat_ms : float
        Reference flat-ground speed (m/s).
    cda : float, optional
        Drag area CdA (m**2). Default 0.35.
    crr : float, optional
        Rolling resistance coefficient. Default 0.005.
    rho : float, optional
        Air density (kg/m**3). Default 1.225.
    headwind_ms : float, optional
        Effective headwind (m/s). Default 2.22 (~8 km/h).
    power_watts : float or None, optional
        If provided, used as P_flat directly (skips headwind back-solve).
    climb_effort : float, optional
        Fraction of extra power for constant-speed climbing (0-1).
        Default 0.5.
    descent_confidence : float, optional
        Rider descent confidence (0-1). Controls both power reduction on
        descents and freewheel speed cap. Default 0.65.
    grad_min_pct, grad_max_pct : int, optional
        Inclusive integer gradient range in percent. Default -20..20.

    Returns
    -------
    dict[int, float]
        Integer gradient (%) -> speed ratio relative to `v_flat_ms`.
    """
    g = 9.81
    k_a = 0.5 * rho * cda

    # --- P_flat: headwind-corrected or user-provided ---
    if power_watts is not None:
        p_flat = float(power_watts)
    else:
        p_flat = (
            k_a * (v_flat_ms + headwind_ms) ** 2 * v_flat_ms
            + crr * mass_kg * g * v_flat_ms
        )

    ratios: dict[int, float] = {}
    for pct in range(grad_min_pct, grad_max_pct + 1):
        theta = math.atan(pct / 100)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # --- Gradient-dependent power ---
        if pct > 0:
            # Climb: rider pushes harder
            p_constant_speed = (
                k_a * (v_flat_ms + headwind_ms) ** 2
                + crr * mass_kg * g * cos_t
                + mass_kg * g * sin_t
            ) * v_flat_ms
            p_g = p_flat + climb_effort * (p_constant_speed - p_flat)
        elif pct < 0:
            # Descent: rider backs off proportional to gravity
            p_grav = mass_kg * g * abs(sin_t) * v_flat_ms
            p_g = max(0.0, p_flat - (1 - descent_confidence) * p_grav)
        else:
            p_g = p_flat

        # --- Solve cubic with headwind ---
        # k_a*v³ + 2*k_a*v_wind*v² + (k_a*v_wind² + m*g*(Crr*cos+sin))*v - P = 0
        c3 = k_a
        c2 = 2 * k_a * headwind_ms
        c1 = k_a * headwind_ms**2 + mass_kg * g * (crr * cos_t + sin_t)
        c0 = -p_g

        if p_g <= 0 and c1 < 0:
            # Terminal velocity with zero power: solve quadratic
            # c3*v² + c2*v + c1 = 0  (after factoring out v)
            disc = c2**2 - 4 * c3 * c1
            v = (-c2 + math.sqrt(disc)) / (2 * c3) if disc >= 0 else 0.0
        elif p_g <= 0:
            v = 0.0
        else:
            roots = np.roots([c3, c2, c1, c0])
            real_pos = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
            v = max(real_pos) if real_pos else 0.0

        # --- Freewheel cap on descents ---
        if pct < 0:
            v_freewheel = math.sqrt(2 * mass_kg * g * abs(sin_t) / (rho * cda))
            if v_freewheel > v_flat_ms:
                v_cap = v_flat_ms + descent_confidence * (v_freewheel - v_flat_ms)
                v = min(v, v_cap)

        ratios[pct] = v / v_flat_ms
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


class RealisticPhysicsEstimator(GradientPriorEstimator):
    """Gradient-aware ETA estimator with a realistic power model.

    Builds speed ratios from `realistic_physics_ratios` which adds headwind
    correction, gradient-dependent power (climb effort + descent backing-off),
    and a freewheel descent speed cap. Delegates to `GradientPriorEstimator`
    for forward-looking time-to-go.

    Parameters
    ----------
    mass_kg : float
        Rider + bike mass (kg).
    v_flat_kmh : float
        Assumed flat-ground speed (km/h).
    headwind_kmh : float, optional
        Effective headwind (km/h). Default 8.0.
    power_watts : float or None, optional
        If provided, used as P_flat directly.
    climb_effort : float, optional
        Fraction of extra power for constant-speed climbing (0-1).
        Default 0.5.
    descent_confidence : float, optional
        Rider descent confidence (0-1). Default 0.65.
    cda, crr, rho : float, optional
        Aerodynamic and rolling resistance parameters.
    """

    def __init__(
        self,
        mass_kg: float,
        v_flat_kmh: float,
        cda: float = 0.35,
        crr: float = 0.005,
        rho: float = 1.225,
        headwind_kmh: float = 8.0,
        power_watts: float | None = None,
        climb_effort: float = 0.5,
        descent_confidence: float = 0.65,
        grad_min_pct: int = -20,
        grad_max_pct: int = 20,
    ) -> None:
        ratios = realistic_physics_ratios(
            mass_kg=mass_kg,
            v_flat_ms=v_flat_kmh / 3.6,
            cda=cda,
            crr=crr,
            rho=rho,
            headwind_ms=headwind_kmh / 3.6,
            power_watts=power_watts,
            climb_effort=climb_effort,
            descent_confidence=descent_confidence,
            grad_min_pct=grad_min_pct,
            grad_max_pct=grad_max_pct,
        )
        super().__init__(v_flat_kmh=v_flat_kmh, ratios=ratios)
        self._mass_kg = mass_kg
        self._cda = cda
        self._crr = crr
        self._rho = rho
        self._headwind_kmh = headwind_kmh
        self._climb_effort = climb_effort
        self._descent_confidence = descent_confidence

    def __str__(self) -> str:
        return (
            f"realistic physics prior ({self._v_flat_ms * 3.6:.1f} km/h flat, "
            f"m={self._mass_kg:.0f}kg, wind={self._headwind_kmh:.0f}km/h, "
            f"effort={self._climb_effort}, conf={self._descent_confidence})"
        )

    def __repr__(self) -> str:
        return (
            f"RealisticPhysicsEstimator(mass_kg={self._mass_kg!r}, "
            f"v_flat_kmh={self._v_flat_ms * 3.6!r}, "
            f"headwind_kmh={self._headwind_kmh!r}, "
            f"climb_effort={self._climb_effort!r}, "
            f"descent_confidence={self._descent_confidence!r})"
        )


def _slow_correction(
    ratio: pd.Series,
    gradients: pd.Series,
    moving: pd.Series,
    cal_max_grad: float,
    slow_span_s: float,
    ramp_s: float,
) -> pd.Series:
    """Slow v_flat correction with startup ramp.

    Blends a fast-converging expanding mean (startup) into a stable
    EWMA (long-term) over `ramp_s` moving seconds.

    Parameters
    ----------
    ratio : pd.Series
        actual_speed / physics_predicted_speed per row.
    gradients : pd.Series
        Per-row gradient as fraction.
    moving : pd.Series
        Boolean mask, True when riding.
    cal_max_grad : float
        |gradient| threshold for calibration data.
    slow_span_s : float
        EWMA span for long-term correction.
    ramp_s : float
        Moving seconds over which startup hands off to EWMA.
    """
    ratio_flat = ratio.where(gradients.abs() <= cal_max_grad)

    # Startup: expanding mean using ALL gradients (converges fast,
    # accepts gradient bias because a biased correction beats no correction)
    startup = ratio.expanding(min_periods=1).mean().ffill().fillna(1.0)

    # Long-term: stable EWMA on flat-only data
    ewma = ratio_flat.ewm(span=slow_span_s, min_periods=1).mean().ffill().fillna(1.0)

    # Ramp: startup → EWMA over ramp_s moving seconds
    moving_s = moving.astype(float).cumsum()
    ramp = (moving_s / ramp_s).clip(0.0, 1.0)
    return lerp(startup, ewma, ramp)


class AdaptivePhysicsEstimator(BaseEstimator):
    """Physics gradient prior with dual-EWMA calibration.

    Combines the forward-looking gradient-aware TTG from
    `PhysicsGradientPriorEstimator` with two learned corrections:

    - **slow**: v_flat correction with startup ramp. An expanding mean
      converges fast at ride start, then hands off to a stable EWMA
      over `ramp_s` moving seconds. Optionally restricted to near-flat
      rows (|gradient| < `cal_max_grad`).
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
    ramp_s : float, optional
        Moving seconds for startup→EWMA handoff. Default 600 (10 min).
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
        ramp_s: float = 600.0,
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
        self._ramp_s = ramp_s
        self._cal_max_grad = cal_max_grad

    def __str__(self) -> str:
        cal = (
            f"|g|<{self._cal_max_grad * 100:.0f}%"
            if self._cal_max_grad < 1.0
            else "all"
        )
        return (
            f"adaptive physics (slow={int(self._slow_span_s)}s [{cal}], "
            f"fast={int(self._fast_span_s)}s, ramp={int(self._ramp_s)}s, {self._physics})"
        )

    def __repr__(self) -> str:
        return (
            f"AdaptivePhysicsEstimator(physics={self._physics!r}, "
            f"slow_span_s={self._slow_span_s!r}, fast_span_s={self._fast_span_s!r}, "
            f"ramp_s={self._ramp_s!r}, cal_max_grad={self._cal_max_grad!r})"
        )

    def _corrections(self, ride: Ride) -> tuple[pd.Series, pd.Series]:
        """Compute slow and fast EWMA correction factors."""
        df = ride.df
        moving = ~df["paused"]

        actual = self.safe_divide(df["delta_distance"], df["delta_time"])
        actual = actual.where(moving)

        predicted = self._physics.predict_current(ride)
        ratio = self.safe_divide(actual, predicted)

        gradients, _ = _row_gradients(ride)

        slow = _slow_correction(
            ratio,
            gradients,
            moving,
            self._cal_max_grad,
            self._slow_span_s,
            self._ramp_s,
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


class BinnedAdaptiveEstimator(BaseEstimator):
    """Gradient prior with slow v_flat EWMA and per-bin fast corrections.

    Wraps any `GradientPriorEstimator` (empirical or physics-based) and
    learns two correction layers on top:

    - **slow**: v_flat correction with startup ramp (expanding mean →
      EWMA handoff), optionally restricted to near-flat rows.
    - **fast**: per-gradient-bin EWMA of the residual after slow
      correction. Each bin independently learns a correction so a
      climb-specific error doesn't contaminate descent predictions.

    Parameters
    ----------
    prior : GradientPriorEstimator
        Base gradient-aware estimator providing speed ratios and TTG.
    slow_span_s : float, optional
        Span for the slow calibration EWMA. Default 3600 (60 min).
    fast_span_s : float, optional
        Span for the per-bin fast EWMA. Default 300 (5 min).
    ramp_s : float, optional
        Moving seconds for startup→EWMA handoff. Default 600 (10 min).
    bin_size : int, optional
        Gradient bin width in percent. Default 1 (1% bins).
        Use 3 for coarser bins with more observations per bucket.
    cal_max_grad : float, optional
        Maximum |gradient| (fraction) for slow calibration. Default 0.02.
    """

    def __init__(
        self,
        prior: GradientPriorEstimator,
        slow_span_s: float = 3600.0,
        fast_span_s: float = 300.0,
        ramp_s: float = 600.0,
        bin_size: int = 1,
        cal_max_grad: float = 0.02,
    ) -> None:
        self._prior = prior
        self._slow_span_s = slow_span_s
        self._fast_span_s = fast_span_s
        self._ramp_s = ramp_s
        self._bin_size = bin_size
        self._cal_max_grad = cal_max_grad

    def __str__(self) -> str:
        cal = (
            f"|g|<{self._cal_max_grad * 100:.0f}%"
            if self._cal_max_grad < 1.0
            else "all"
        )
        return (
            f"binned adaptive (slow={int(self._slow_span_s)}s [{cal}], "
            f"fast={int(self._fast_span_s)}s/{self._bin_size}%bin, "
            f"ramp={int(self._ramp_s)}s, {self._prior})"
        )

    def __repr__(self) -> str:
        return (
            f"BinnedAdaptiveEstimator(prior={self._prior!r}, "
            f"slow_span_s={self._slow_span_s!r}, fast_span_s={self._fast_span_s!r}, "
            f"ramp_s={self._ramp_s!r}, bin_size={self._bin_size!r}, "
            f"cal_max_grad={self._cal_max_grad!r})"
        )

    def _corrections(self, ride: Ride) -> tuple[pd.Series, pd.Series]:
        """Slow correction with startup ramp + per-bin fast EWMA."""
        df = ride.df
        moving = ~df["paused"]

        actual = self.safe_divide(df["delta_distance"], df["delta_time"])
        actual = actual.where(moving)

        predicted = self._prior.predict_current(ride)
        ratio = self.safe_divide(actual, predicted)

        gradients, _ = _row_gradients(ride)
        grad_bins = _gradient_bin_pct(gradients, self._bin_size)

        slow = _slow_correction(
            ratio,
            gradients,
            moving,
            self._cal_max_grad,
            self._slow_span_s,
            self._ramp_s,
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
        v_flat = self._prior._v_flat_ms
        grad_bins = _gradient_bin_pct(gradients, self._bin_size)
        bs = self._bin_size

        total_dist = df["distance_m"].iloc[-1]
        distances = df["distance_m"].values

        seg_base_speeds = np.array(
            [v_flat * self._prior._ratio_for(s.gradient) for s in segments]
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
        base = self._prior.predict_current(ride)
        return base * slow * fast


class TrustedBinnedAdaptiveEstimator(BinnedAdaptiveEstimator):
    """Binned adaptive estimator with trust ramp and clamped corrections.

    Subclass of `BinnedAdaptiveEstimator` that guards the per-bin fast
    EWMA against noisy or starved bins:

    - **Trust ramp**: blends the raw EWMA toward 1.0 (no correction)
      when a bin has few observations. Full trust requires `trust_n`
      observations.
    - **Observation window**: optionally counts only observations
      within the last `trust_window_s` seconds so trust can decay
      when a bin hasn't been visited recently.
    - **Clamping**: hard bounds on the per-bin correction prevent
      catastrophic TTG blowups.

    Parameters
    ----------
    prior : GradientPriorEstimator
        Base gradient-aware estimator providing speed ratios and TTG.
    trust_n : int, optional
        Observations in a bin needed for full trust. Default 60.
    trust_window_s : float or None, optional
        Rolling window (seconds) for counting observations. ``None``
        means cumulative (trust grows monotonically). Default None.
    corr_min : float, optional
        Hard floor on per-bin correction. Default 0.5.
    corr_max : float, optional
        Hard ceiling on per-bin correction. Default 1.5.
    slow_span_s, fast_span_s, ramp_s, bin_size, cal_max_grad
        Passed through to `BinnedAdaptiveEstimator`.
    """

    def __init__(
        self,
        prior: GradientPriorEstimator,
        trust_n: int = 60,
        trust_window_s: float | None = None,
        corr_min: float = 0.5,
        corr_max: float = 1.5,
        slow_span_s: float = 3600.0,
        fast_span_s: float = 300.0,
        ramp_s: float = 600.0,
        bin_size: int = 1,
        cal_max_grad: float = 0.02,
    ) -> None:
        super().__init__(
            prior=prior,
            slow_span_s=slow_span_s,
            fast_span_s=fast_span_s,
            ramp_s=ramp_s,
            bin_size=bin_size,
            cal_max_grad=cal_max_grad,
        )
        self._trust_n = trust_n
        self._trust_window_s = trust_window_s
        self._corr_min = corr_min
        self._corr_max = corr_max

    def __str__(self) -> str:
        cal = (
            f"|g|<{self._cal_max_grad * 100:.0f}%"
            if self._cal_max_grad < 1.0
            else "all"
        )
        win = f"{int(self._trust_window_s)}s" if self._trust_window_s else "cumul"
        return (
            f"trusted binned adaptive (slow={int(self._slow_span_s)}s [{cal}], "
            f"fast={int(self._fast_span_s)}s/{self._bin_size}%bin, "
            f"trust={self._trust_n}obs/{win}, "
            f"clamp=[{self._corr_min},{self._corr_max}], "
            f"ramp={int(self._ramp_s)}s, {self._prior})"
        )

    def __repr__(self) -> str:
        return (
            f"TrustedBinnedAdaptiveEstimator(prior={self._prior!r}, "
            f"trust_n={self._trust_n!r}, trust_window_s={self._trust_window_s!r}, "
            f"corr_min={self._corr_min!r}, corr_max={self._corr_max!r}, "
            f"slow_span_s={self._slow_span_s!r}, fast_span_s={self._fast_span_s!r}, "
            f"ramp_s={self._ramp_s!r}, bin_size={self._bin_size!r}, "
            f"cal_max_grad={self._cal_max_grad!r})"
        )

    def _corrections(self, ride: Ride) -> tuple[pd.Series, pd.Series]:
        """Slow correction + trust-ramped, clamped per-bin fast EWMA."""
        df = ride.df
        moving = ~df["paused"]

        actual = self.safe_divide(df["delta_distance"], df["delta_time"])
        actual = actual.where(moving)

        predicted = self._prior.predict_current(ride)
        ratio = self.safe_divide(actual, predicted)

        gradients, _ = _row_gradients(ride)
        grad_bins = _gradient_bin_pct(gradients, self._bin_size)

        slow = _slow_correction(
            ratio,
            gradients,
            moving,
            self._cal_max_grad,
            self._slow_span_s,
            self._ramp_s,
        )

        # Per-bin fast EWMA with trust ramp and clamping
        ratio_fast = self.safe_divide(actual, predicted * slow)
        fast = pd.Series(1.0, index=df.index)
        elapsed = df["delta_time"].cumsum()

        for bin_pct, grp in ratio_fast.groupby(grad_bins):
            bin_ewma = grp.ewm(span=self._fast_span_s, min_periods=1).mean()

            # Observation count: cumulative or windowed
            if self._trust_window_s is None:
                count = np.arange(1, len(grp) + 1, dtype=float)
            else:
                bin_elapsed = elapsed.loc[grp.index].values
                count = np.empty(len(grp))
                for k in range(len(grp)):
                    count[k] = np.sum(
                        bin_elapsed[: k + 1] >= bin_elapsed[k] - self._trust_window_s
                    )

            trust = pd.Series(np.clip(count / self._trust_n, 0.0, 1.0), index=grp.index)
            blended = trust * bin_ewma + (1 - trust) * 1.0
            fast.loc[grp.index] = blended.clip(self._corr_min, self._corr_max)

        fast = fast.fillna(1.0)
        return slow, fast


class OracleTrustedBinnedEstimator(TrustedBinnedAdaptiveEstimator):
    """Trusted binned estimator with oracle (perfect) v_flat.

    Before each prediction, computes the true flat-section average speed
    from the full ride and patches the prior's v_flat. This isolates the
    slow/fast EWMA correction performance from v_flat estimation error.

    Parameters
    ----------
    max_grad : float
        Max |gradient| (fraction) to consider flat for the oracle.
        Default 0.02.
    **kwargs
        Forwarded to `TrustedBinnedAdaptiveEstimator`.
    """

    def __init__(self, max_grad: float = 0.02, **kwargs) -> None:
        super().__init__(**kwargs)
        self._oracle_max_grad = max_grad

    def _set_oracle_vflat(self, ride: Ride) -> None:
        df = ride.df
        moving = ~df["paused"]
        gradients, _ = _row_gradients(ride)
        flat_mask = moving & (gradients.abs() < self._oracle_max_grad)
        flat_dist = df.loc[flat_mask, "delta_distance"].sum()
        flat_time = df.loc[flat_mask, "delta_time"].sum()
        if flat_time > 0:
            self._prior._v_flat_ms = flat_dist / flat_time

    def predict(self, ride: Ride) -> pd.Series:
        self._set_oracle_vflat(ride)
        return super().predict(ride)

    def predict_current(self, ride: Ride) -> pd.Series:
        self._set_oracle_vflat(ride)
        return super().predict_current(ride)

    def __str__(self) -> str:
        return f"oracle {super().__str__()}"


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
