import numpy as np
import pandas as pd
from godot.benchmark import backtest, compute_metrics
from godot.ride import Ride


class ConstantSpeedEstimator:
    """Test double: always predicts 20 km/h regardless of ride data."""

    def predict(self, ride: Ride) -> pd.Series:
        return pd.Series(20 / 3.6, index=ride.df.index)


def make_synthetic_ride(
    n: int = 100, speed_kmh: float = 20.0, total_km: float = 10.0
) -> Ride:
    """Synthetic ride at constant speed on flat terrain."""
    total_m = total_km * 1000
    dist = np.linspace(0, total_m, n)
    speed_ms = speed_kmh / 3.6
    time_s = dist / speed_ms
    df = pd.DataFrame(
        {
            "time": pd.to_datetime((time_s * 1000).astype(int), unit="ms"),
            "distance_m": dist,
            "elevation_m": np.zeros(n),
            "speed_kmh": np.full(n, speed_kmh),
            "paused": np.zeros(n, dtype=bool),
            "delta_distance": np.diff(dist, prepend=0).clip(min=0),
            "delta_time": np.diff(time_s, prepend=0),
        }
    )
    return Ride(
        name="synthetic",
        label="synthetic",
        df=df,
        route_type="flat",
        contains_pauses=False,
        distance_method="haversine",
        speed_smoothed=False,
        distance=total_m,
        total_time=time_s[-1],
        ride_time=time_s[-1],
        paused_time=0.0,
    )


def test_backtest_columns():
    ride = make_synthetic_ride()
    result = backtest(ride, ConstantSpeedEstimator())
    assert list(result.columns) == [
        "time",
        "distance_m",
        "speed_ms",
        "eta_remaining_s",
        "ata_remaining_s",
        "delta_s",
    ]


def test_backtest_length():
    ride = make_synthetic_ride(n=50)
    result = backtest(ride, ConstantSpeedEstimator())
    assert len(result) == 50


def test_backtest_delta_near_zero_for_perfect_estimator():
    """A perfect estimator at constant speed should produce near-zero delta."""
    ride = make_synthetic_ride(speed_kmh=20.0)
    result = backtest(ride, ConstantSpeedEstimator())
    # Ignore last point where remaining == 0
    assert result["delta_s"].iloc[:-1].abs().max() < 1.0


def test_backtest_ata_decreasing():
    ride = make_synthetic_ride()
    result = backtest(ride, ConstantSpeedEstimator())
    assert (result["ata_remaining_s"].diff().dropna() <= 0).all()


def test_backtest_delta_equals_eta_minus_ata():
    ride = make_synthetic_ride()
    result = backtest(ride, ConstantSpeedEstimator())
    expected = result["eta_remaining_s"] - result["ata_remaining_s"]
    assert (result["delta_s"] - expected).abs().max() < 1e-9


def test_compute_metrics():
    ride = make_synthetic_ride(speed_kmh=20.0)
    result = backtest(ride, ConstantSpeedEstimator())
    metrics = compute_metrics(result, warmup_distance_m=0.0)
    assert "mae_min" in metrics
    assert "rmse_min" in metrics
    # Perfect estimator → near-zero error
    assert metrics["mae_min"] < 0.1
    assert metrics["rmse_min"] < 0.1
