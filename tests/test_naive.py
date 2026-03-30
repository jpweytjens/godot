import numpy as np
import pandas as pd
from eta.estimators import AvgSpeedEstimator, RollingAvgSpeedEstimator
from eta.ride import Ride


def make_ride(timestamps_ms, distances_m, speeds_kmh, paused=None):
    n = len(timestamps_ms)
    dist = np.array(distances_m, dtype=float)
    time_s = np.array(timestamps_ms, dtype=float) / 1000
    if paused is None:
        paused_arr = np.array([s < 1.0 for s in speeds_kmh])
    else:
        paused_arr = np.array(paused)
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(timestamps_ms, unit="ms"),
            "distance_m": dist,
            "speed_kmh": speeds_kmh,
            "elevation_m": np.zeros(n),
            "paused": paused_arr,
            "delta_distance": np.diff(dist, prepend=dist[0]).clip(min=0),
            "delta_time": np.diff(time_s, prepend=time_s[0]),
        }
    )
    # First row deltas should be zero
    df.loc[df.index[0], ["delta_distance", "delta_time"]] = 0.0
    total_dist = float(dist[-1])
    total_time = float(time_s[-1] - time_s[0])
    moving_time = float(df.loc[~df["paused"], "delta_time"].sum())
    return Ride(
        name="test",
        label="test",
        df=df,
        route_type="flat",
        contains_pauses=paused_arr.any(),
        distance_method="haversine",
        speed_smoothed=False,
        distance=total_dist,
        total_time=total_time,
        ride_time=moving_time,
        paused_time=total_time - moving_time,
    )


def constant_ride(n=20, speed_kmh=20.0, interval_s=1):
    speed_ms = speed_kmh / 3.6
    ts = [i * interval_s * 1000 for i in range(n)]
    dist = [i * speed_ms * interval_s for i in range(n)]
    return make_ride(ts, dist, [speed_kmh] * n)


class TestAvgSpeedEstimator:
    def test_nan_before_moving(self):
        ride = make_ride([0, 1000, 2000], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        speed = AvgSpeedEstimator().predict(ride)
        assert speed.isna().all()

    def test_constant_speed(self):
        speed_ms = 20 / 3.6
        ride = constant_ride(n=20, speed_kmh=20.0)
        speed = AvgSpeedEstimator().predict(ride)
        # After the first interval the expanding average should converge
        assert abs(speed.iloc[-1] - speed_ms) < 0.01

    def test_moving_only_excludes_stopped_time(self):
        # 10s moving at 20 km/h, then 10s stopped, then 10s moving again
        speed_ms = 20 / 3.6
        ts = list(range(0, 30_000, 1000))
        dist = (
            [i * speed_ms for i in range(10)]
            + [10 * speed_ms] * 10
            + [(10 + i) * speed_ms for i in range(10)]
        )
        speeds = [20.0] * 10 + [0.0] * 10 + [20.0] * 10
        ride = make_ride(ts, dist, speeds)

        moving = AvgSpeedEstimator(moving_only=True).predict(ride)
        total = AvgSpeedEstimator(moving_only=False).predict(ride)

        # moving-only speed >= total-time speed (same distance, less time in denom)
        assert moving.iloc[-1] > total.iloc[-1]
        # moving-only should recover ~20 km/h
        assert abs(moving.iloc[-1] - speed_ms) < 0.5

    def test_total_time_slower_when_stopped(self):
        # Stopped ride: speed never >= 1 km/h but distance recorded (e.g. GPS drift)
        ts = [0, 1000, 2000, 3000]
        dist = [0.0, 0.0, 0.0, 0.0]
        speeds = [0.0, 0.0, 0.0, 0.0]
        ride = make_ride(ts, dist, speeds)
        speed_moving = AvgSpeedEstimator(moving_only=True).predict(ride)
        speed_total = AvgSpeedEstimator(moving_only=False).predict(ride)
        assert speed_moving.isna().all()
        # total: cum_dd=0 always → 0/dt = 0
        # First row is NaN (no prior point); rest are 0/dt = 0
        assert speed_total.iloc[1:].eq(0).all()


class TestRollingAvgSpeedEstimator:
    def test_constant_speed(self):
        speed_ms = 20 / 3.6
        ride = constant_ride(n=60, speed_kmh=20.0)
        speed = RollingAvgSpeedEstimator(window_s=30).predict(ride)
        assert abs(speed.iloc[-1] - speed_ms) < 0.01

    def test_window_uses_only_recent_data(self):
        # First 100s at 10 km/h, then 10s at 30 km/h
        # With window_s=10 only the last 10s (30 km/h) should matter
        speed_slow_ms = 10 / 3.6
        speed_fast_ms = 30 / 3.6
        n_slow = 100
        n_fast = 10
        ts = list(range(0, (n_slow + n_fast) * 1000, 1000))
        dist = [i * speed_slow_ms for i in range(n_slow)] + [
            n_slow * speed_slow_ms + (i + 1) * speed_fast_ms for i in range(n_fast)
        ]
        speeds = [10.0] * n_slow + [30.0] * n_fast
        ride = make_ride(ts, dist, speeds)

        speed = RollingAvgSpeedEstimator(window_s=10).predict(ride)
        assert abs(speed.iloc[-1] - speed_fast_ms) < 0.5

    def test_moving_only_ignores_stopped_intervals(self):
        speed_ms = 20 / 3.6
        ts = list(range(0, 20_000, 1000))
        dist = [i * speed_ms for i in range(10)] + [10 * speed_ms] * 10
        speeds = [20.0] * 10 + [0.0] * 10
        ride = make_ride(ts, dist, speeds)

        moving = RollingAvgSpeedEstimator(window_s=300, moving_only=True).predict(ride)
        total = RollingAvgSpeedEstimator(window_s=300, moving_only=False).predict(ride)

        # After stop, moving-only should still reflect the moving speed
        assert moving.iloc[-1] > total.iloc[-1]
