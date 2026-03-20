"""Online ETA estimators for cycling ride prediction."""

import numpy as np

ROLLING_WINDOW_S = 300.0


class AvgSpeedEstimator:
    """Estimates remaining time using cumulative average moving speed.

    Tracks total distance and time while moving (speed >= 1 km/h),
    and uses the resulting average speed to predict remaining time.
    """

    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._total_distance_m = 0.0
        self._total_time_s = 0.0
        self._prev_distance_m: float | None = None
        self._prev_timestamp_ms: int | None = None

    def update(
        self,
        timestamp_ms: int,
        distance_m: float,
        speed_kmh: float,
        elevation_m: float,
    ) -> None:
        """Ingest one data point and update the cumulative average.

        Parameters
        ----------
        timestamp_ms : int
            Unix timestamp in milliseconds.
        distance_m : float
            Cumulative distance from ride start in meters.
        speed_kmh : float
            Current speed in km/h.
        elevation_m : float
            Current elevation in meters (unused, kept for protocol compatibility).
        """
        if self._prev_timestamp_ms is not None and speed_kmh >= 1.0:
            dt_s = (timestamp_ms - self._prev_timestamp_ms) / 1000.0
            dd_m = distance_m - self._prev_distance_m
            if dt_s > 0 and dd_m >= 0:
                self._total_distance_m += dd_m
                self._total_time_s += dt_s
        self._prev_distance_m = distance_m
        self._prev_timestamp_ms = timestamp_ms

    def predict(
        self,
        current_distance_m: float,
        total_distance_m: float,
        now_ms: int,
    ) -> float:
        """Predict remaining ride time in seconds.

        Parameters
        ----------
        current_distance_m : float
            Current position in meters from ride start.
        total_distance_m : float
            Total route distance in meters.
        now_ms : int
            Current timestamp in milliseconds (unused).

        Returns
        -------
        float
            Estimated remaining seconds, or nan if no data yet.
        """
        if self._total_time_s == 0 or self._total_distance_m == 0:
            return np.nan
        avg_speed_ms = self._total_distance_m / self._total_time_s
        remaining_m = max(total_distance_m - current_distance_m, 0.0)
        return remaining_m / avg_speed_ms


class RollingAvgSpeedEstimator:
    """Estimates remaining time using rolling average speed over a time window.

    Only moving observations (speed >= 1 km/h) are kept. The buffer is
    trimmed to the most recent ``window_s`` seconds on each update.

    Parameters
    ----------
    window_s : float or None, optional
        Rolling window in seconds. Defaults to ROLLING_WINDOW_S (300 s).
    """

    def __init__(self, window_s: float | None = None) -> None:
        self._window_s = ROLLING_WINDOW_S if window_s is None else window_s
        self._observations: list[tuple[int, float]] = []

    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._observations = []

    def update(
        self,
        timestamp_ms: int,
        distance_m: float,
        speed_kmh: float,
        elevation_m: float,
    ) -> None:
        """Ingest one data point and maintain the rolling buffer.

        Parameters
        ----------
        timestamp_ms : int
            Unix timestamp in milliseconds.
        distance_m : float
            Cumulative distance in meters (unused, kept for protocol compatibility).
        speed_kmh : float
            Current speed in km/h.
        elevation_m : float
            Current elevation in meters (unused, kept for protocol compatibility).
        """
        if speed_kmh >= 1.0:
            self._observations.append((timestamp_ms, speed_kmh))
        cutoff_ms = timestamp_ms - int(self._window_s * 1000)
        self._observations = [(t, v) for t, v in self._observations if t >= cutoff_ms]

    def predict(
        self,
        current_distance_m: float,
        total_distance_m: float,
        now_ms: int,
    ) -> float:
        """Predict remaining ride time in seconds.

        Parameters
        ----------
        current_distance_m : float
            Current position in meters from ride start.
        total_distance_m : float
            Total route distance in meters.
        now_ms : int
            Current timestamp in milliseconds (unused).

        Returns
        -------
        float
            Estimated remaining seconds, or nan if no observations in window.
        """
        if not self._observations:
            return np.nan
        rolling_avg_ms = np.mean([v for _, v in self._observations]) / 3.6
        remaining_m = max(total_distance_m - current_distance_m, 0.0)
        return remaining_m / rolling_avg_ms
