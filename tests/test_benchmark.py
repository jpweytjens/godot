import numpy as np
import pandas as pd
from benchmark import backtest


class ConstantSpeedEstimator:
    """Test double: always predicts 20 km/h regardless of ride data."""

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(20 / 3.6, index=df.index)


def make_synthetic_ride(
    n: int = 100, speed_kmh: float = 20.0, total_km: float = 10.0
) -> pd.DataFrame:
    """Synthetic ride at constant speed on flat terrain."""
    total_m = total_km * 1000
    dist = np.linspace(0, total_m, n)
    speed_ms = speed_kmh / 3.6
    time_s = dist / speed_ms
    return pd.DataFrame(
        {
            "timestamp_ms": (time_s * 1000).astype(int),
            "distance_m": dist,
            "elevation_m": np.zeros(n),
            "speed_kmh": np.full(n, speed_kmh),
        }
    )


def test_backtest_columns():
    df = make_synthetic_ride()
    result = backtest(df, ConstantSpeedEstimator())
    assert list(result.columns) == [
        "distance_m",
        "eta_remaining_s",
        "ata_remaining_s",
        "delta_s",
    ]


def test_backtest_length():
    df = make_synthetic_ride(n=50)
    result = backtest(df, ConstantSpeedEstimator())
    assert len(result) == 50


def test_backtest_delta_near_zero_for_perfect_estimator():
    """A perfect estimator at constant speed should produce near-zero delta."""
    df = make_synthetic_ride(speed_kmh=20.0)
    result = backtest(df, ConstantSpeedEstimator())
    # Ignore last point where remaining == 0
    assert result["delta_s"].iloc[:-1].abs().max() < 1.0


def test_backtest_ata_decreasing():
    df = make_synthetic_ride()
    result = backtest(df, ConstantSpeedEstimator())
    assert (result["ata_remaining_s"].diff().dropna() <= 0).all()


def test_backtest_delta_equals_eta_minus_ata():
    df = make_synthetic_ride()
    result = backtest(df, ConstantSpeedEstimator())
    expected = result["eta_remaining_s"] - result["ata_remaining_s"]
    assert (result["delta_s"] - expected).abs().max() < 1e-9
