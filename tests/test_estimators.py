import numpy as np
import pandas as pd
from eta.estimators import AvgSpeedEstimator, RollingAvgSpeedEstimator
from eta.ride import Ride


def make_ride_with_pause(
    moving_points: int = 50,
    pause_points: int = 120,
    speed_kmh: float = 20.0,
) -> Ride:
    """Synthetic ride: move → pause → move.

    The pause is longer than a typical rolling window (default 120 s > 60 s)
    so the window can fall entirely within the paused region.
    """
    speed_ms = speed_kmh / 3.6
    dt = 1.0  # 1-second intervals

    n = moving_points * 2 + pause_points
    paused = np.zeros(n, dtype=bool)
    paused[moving_points : moving_points + pause_points] = True

    delta_time = np.full(n, dt)
    delta_distance = np.where(paused, 0.0, speed_ms * dt)
    delta_distance[0] = 0.0
    distance_m = np.cumsum(delta_distance)
    time_s = np.cumsum(delta_time) - dt

    df = pd.DataFrame(
        {
            "time": pd.to_datetime((time_s * 1000).astype(int), unit="ms"),
            "distance_m": distance_m,
            "elevation_m": np.zeros(n),
            "speed_kmh": np.where(paused, 0.0, speed_kmh),
            "paused": paused,
            "delta_distance": delta_distance,
            "delta_time": delta_time,
        }
    )
    return Ride(
        name="pause_test",
        label="pause test",
        df=df,
        route_type="flat",
        contains_pauses=True,
        distance_method="haversine",
        speed_smoothed=False,
        distance=distance_m[-1],
        total_time=time_s[-1],
        ride_time=(moving_points * 2) * dt,
        paused_time=pause_points * dt,
    )


class TestSafeDivide:
    def test_no_inf_in_rolling_estimator_during_pause(self):
        """Rolling window smaller than pause must not produce inf."""
        ride = make_ride_with_pause(pause_points=120)
        est = RollingAvgSpeedEstimator(window_s=60, moving_only=True)
        result = est.predict(ride)

        assert not np.isinf(result).any(), (
            "inf values found in rolling estimator output"
        )

    def test_no_inf_in_avg_estimator_during_pause(self):
        """Cumulative estimator must not produce inf."""
        ride = make_ride_with_pause(pause_points=120)
        est = AvgSpeedEstimator(moving_only=True)
        result = est.predict(ride)

        assert not np.isinf(result).any(), "inf values found in avg estimator output"

    def test_deep_pause_rows_are_nan_when_moving_only(self):
        """Pause rows beyond the rolling window reach should be NaN."""
        ride = make_ride_with_pause(moving_points=50, pause_points=120)
        est = RollingAvgSpeedEstimator(window_s=60, moving_only=True)
        result = est.predict(ride)

        # Rows deep into the pause (beyond window reach of moving data)
        deep_pause = result.iloc[50 + 60 : 50 + 120]
        assert deep_pause.isna().all(), "deep pause rows should be NaN"

    def test_moving_rows_are_finite(self):
        """Non-paused rows (after first) should have finite positive speed."""
        ride = make_ride_with_pause(moving_points=50, pause_points=120)
        est = RollingAvgSpeedEstimator(window_s=60, moving_only=True)
        result = est.predict(ride)
        pause_mask = ride.df["paused"]
        moving = result[~pause_mask].iloc[1:]  # skip first row (no prior data)

        assert (moving > 0).all(), "moving rows should have positive speed"
        assert np.isfinite(moving).all(), "moving rows should be finite"
