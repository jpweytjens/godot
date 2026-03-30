"""Pause strategies for speed-vector adjustment during stopped periods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import pandas as pd

if TYPE_CHECKING:
    from eta.ride import Ride


class PauseStrategy(Protocol):
    """Protocol for pause-aware speed adjustment.

    Implementations receive the raw estimated speed Series and return
    a (possibly modified) version.  The backtest applies this *before*
    computing ``remaining_distance / speed``, so NaN values propagate
    naturally into ETA.
    """

    def adjust(self, speed: pd.Series, ride: Ride) -> pd.Series:
        """Return an adjusted speed Series.

        Parameters
        ----------
        speed : pd.Series
            Estimated speed in m/s at each row.
        ride : Ride
            Prepared ride from `load_ride`.

        Returns
        -------
        pd.Series
            Adjusted speed. NaN where no estimate should be produced.
        """
        ...


class NoPause:
    """Leave speed as-is — naive baseline where pauses stay in the signal."""

    def adjust(self, speed: pd.Series, ride: Ride) -> pd.Series:
        return speed


class NanPauses:
    """NaN out speed during paused moments.

    During a pause the estimator has no new data, so reporting a speed
    is misleading.  Setting it to NaN makes ETA undefined while stopped
    and prevents the error metric from diverging.
    """

    def adjust(self, speed: pd.Series, ride: Ride) -> pd.Series:
        return speed.where(~ride.df["paused"])
