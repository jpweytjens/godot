"""Pause strategies for ETA adjustment during stopped periods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import pandas as pd

if TYPE_CHECKING:
    from eta.ride import Ride


class PauseStrategy(Protocol):
    """Protocol for pause-aware ETA adjustment.

    Returns a Series of seconds to add to the speed-based ETA estimate
    at each row.
    """

    def adjust(self, ride: Ride) -> pd.Series:
        """Return seconds to add to ETA at each row.

        Parameters
        ----------
        ride : Ride
            Prepared ride from `load_ride`.

        Returns
        -------
        pd.Series
            Seconds to add. Zero when riding, positive when paused.
        """
        ...


class NoPause:
    """No pause adjustment — suitable for total-time speed estimators."""

    def adjust(self, ride: Ride) -> pd.Series:
        return pd.Series(0.0, index=ride.df.index)


class AddCurrentPause:
    """Add elapsed seconds of the current ongoing pause to ETA.

    Each pause segment accumulates independently. When riding resumes,
    the adjustment drops back to zero.
    """

    def adjust(self, ride: Ride) -> pd.Series:
        df = ride.df
        run_id = (df["paused"] != df["paused"].shift()).cumsum()
        return df.groupby(run_id)["delta_time"].cumsum() * df["paused"]
