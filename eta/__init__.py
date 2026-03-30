"""Cycling ETA estimator — prototype package."""

from eta.benchmark import backtest, compute_metrics
from eta.estimators import AvgSpeedEstimator, BaseEstimator, RollingAvgSpeedEstimator
from eta.gpx import (
    read_gpx,
)
from eta.plot import comparison_errors, eta_error, speed_actual, speed_estimated
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
    # benchmark
    "backtest",
    "compute_metrics",
    # plot
    "speed_actual",
    "speed_estimated",
    "eta_error",
    "comparison_errors",
]
