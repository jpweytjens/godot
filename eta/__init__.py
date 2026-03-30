"""Cycling ETA estimator — prototype package."""

from eta.benchmark import backtest, compute_metrics
from eta.estimators import AvgSpeedEstimator, BaseEstimator, RollingAvgSpeedEstimator
from eta.pause import AddCurrentPause, NoPause
from eta.gpx import (
    read_gpx,
)
from eta.plot import (
    comparison_errors,
    eta_countdown,
    eta_error,
    speed_comparison,
)
from eta.ride import Ride, load_ride

__all__ = [
    # gpx
    "read_gpx",
    # ride
    "Ride",
    "load_ride",
    # estimators
    "BaseEstimator",
    "AvgSpeedEstimator",
    "RollingAvgSpeedEstimator",
    # pause
    "AddCurrentPause",
    "NoPause",
    # benchmark
    "backtest",
    "compute_metrics",
    # plot
    "eta_countdown",
    "eta_error",
    "speed_comparison",
    "comparison_errors",
]
