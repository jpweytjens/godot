"""Pause strategies for speed and ETA adjustment during stopped periods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import pandas as pd

from godot.gpx import pause_run_id

if TYPE_CHECKING:
    from godot.ride import Ride


class PauseStrategy(Protocol):
    """Protocol for pause-aware ETA adjustment.

    Two hooks:

    1. `adjust` — modify the estimated speed Series (e.g. NaN during pauses).
    2. `fill_pauses` — fill NaN ETA values produced by NaN speed
       (e.g. forward-fill, subtract elapsed pause, etc.).

    The backtest applies them as::

        speed_ms = pause_strategy.adjust(estimator.predict(ride), ride)
        eta_s    = pause_strategy.fill_pauses(remaining_m / speed_ms, ride)
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

    def fill_pauses(self, eta: pd.Series, ride: Ride) -> pd.Series:
        """Fill NaN ETA values caused by paused speed.

        Parameters
        ----------
        eta : pd.Series
            Raw ETA in seconds (NaN during pauses when speed was NaN).
        ride : Ride
            Prepared ride from `load_ride`.

        Returns
        -------
        pd.Series
            ETA with pause gaps filled (or unchanged if no NaN).
        """
        ...


class NoPause:
    """Leave speed and ETA unchanged — naive baseline."""

    def adjust(self, speed: pd.Series, ride: Ride) -> pd.Series:
        return speed

    def fill_pauses(self, eta: pd.Series, ride: Ride) -> pd.Series:
        return eta


class NanPauses:
    """NaN out speed during pauses, forward-fill ETA to hold constant.

    During a pause the estimator has no new data, so speed is set to NaN.
    ETA is forward-filled from the last riding value: error grows at 1 s/s
    (ATA ticks down, ETA holds).
    """

    def adjust(self, speed: pd.Series, ride: Ride) -> pd.Series:
        return speed.where(~ride.df["paused"])

    def fill_pauses(self, eta: pd.Series, ride: Ride) -> pd.Series:
        return eta.ffill()


def _elapsed_pause(ride: Ride) -> pd.Series:
    """Cumulative seconds within each pause run (0 during riding)."""
    df = ride.df
    run_id = pause_run_id(df["paused"])
    return df.groupby(run_id)["delta_time"].cumsum() * df["paused"]


class WallClockPause(NanPauses):
    """ETA follows the wall clock during pauses.

    Freezes the moving-time estimate and lets the clock tick: ETA
    decreases at 1 s/s during a pause (matching ATA), keeping the
    error flat. Once the pause ends, the prior resumes normally.
    """

    def fill_pauses(self, eta: pd.Series, ride: Ride) -> pd.Series:
        return eta.ffill() - _elapsed_pause(ride)


class AddElapsed(NanPauses):
    """NaN speed + add elapsed pause to forward-filled ETA.

    ETA increases during a pause while ATA decreases — error grows
    at 2 s/s.  Useful as a comparison baseline.
    """

    def fill_pauses(self, eta: pd.Series, ride: Ride) -> pd.Series:
        return eta.ffill() + _elapsed_pause(ride)
