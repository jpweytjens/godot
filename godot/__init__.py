"""Cycling ETA estimator — prototype package."""

from godot.benchmark import backtest, compute_metrics
from godot.estimators import (
    AvgSpeedEstimator,
    BaseEstimator,
    RollingAvgSpeedEstimator,
)
from godot.pause import AddElapsed, NanPauses, NoPause, WallClockPause
from godot.fit import read_fit
from godot.gpx import (
    read_gpx,
)
from godot.plot import (
    comparison_errors,
    elevation_profile,
    eta_countdown,
    eta_error,
    speed_comparison,
)
from godot.ride import Ride, load_ride, compute_global_prior
from godot.segmentation import decimate_to_gradient_segments, visvalingam_whyatt


__all__ = [
    # fit
    "read_fit",
    # gpx
    "read_gpx",
    # ride
    "Ride",
    "load_ride",
    "compute_global_prior",
    # segmentation
    "decimate_to_gradient_segments",
    "visvalingam_whyatt",
    # estimators
    "BaseEstimator",
    "AvgSpeedEstimator",
    "RollingAvgSpeedEstimator",
    # pause
    "AddElapsed",
    "NanPauses",
    "NoPause",
    "WallClockPause",
    # benchmark
    "backtest",
    "compute_metrics",
    # plot
    "elevation_profile",
    "eta_countdown",
    "eta_error",
    "speed_comparison",
    "comparison_errors",
]
