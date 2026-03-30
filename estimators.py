"""ETA estimators for cycling ride prediction."""

import pandas as pd

ROLLING_WINDOW_S = 300.0


class BaseEstimator:
    """Base class providing data transformation helpers for speed estimators.

    Parameters
    ----------
    pause_kmh : float, optional
        Speed threshold below which a point is considered stopped.
        Default 1.0 km/h.
    """

    def __init__(self, pause_kmh: float = 1.0) -> None:
        self._pause_kmh = pause_kmh

    def _deltas(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Return per-row distance (m) and time (s) deltas.

        Parameters
        ----------
        df : pd.DataFrame
            Ride DataFrame with time and distance_m columns.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            dd : distance delta in metres (clipped to >= 0).
            dt : time delta in seconds.
        """
        dd = df["distance_m"].diff().clip(lower=0).fillna(0)
        dt = df["time"].diff().dt.total_seconds().fillna(0)
        return dd, dt

    def _moving_mask(self, df: pd.DataFrame) -> pd.Series:
        """Return a float mask that is 1.0 while moving (speed >= 1 km/h), else 0.0.

        Parameters
        ----------
        df : pd.DataFrame
            Ride DataFrame with speed_kmh column.

        Returns
        -------
        pd.Series
            Float mask aligned with df's index.
        """
        return (df["speed_kmh"] >= self._pause_kmh).astype(float)

    def _datetime_index(self, df: pd.DataFrame):
        """Return a DatetimeIndex from the time column for time-based rolling.

        Parameters
        ----------
        df : pd.DataFrame
            Ride DataFrame with time column.

        Returns
        -------
        pd.DatetimeIndex
        """
        return pd.DatetimeIndex(df["time"])

    def predict(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class AvgSpeedEstimator(BaseEstimator):
    """Expanding-window average speed estimator.

    Parameters
    ----------
    moving_only : bool, optional
        If True (default), denominator is total moving time (speed >= 1 km/h).
        If False, denominator is total elapsed time.
    """

    def __init__(self, moving_only: bool = True, pause_kmh: float = 1.0) -> None:
        super().__init__(pause_kmh=pause_kmh)
        self._moving_only = moving_only

    def predict(self, df: pd.DataFrame) -> pd.Series:
        dd, dt = self._deltas(df)
        if self._moving_only:
            mask = self._moving_mask(df)
            dd, dt = dd * mask, dt * mask
        cum_dd = dd.cumsum()
        cum_dt = dt.cumsum()
        return (cum_dd / cum_dt).where(cum_dt > 0)


class RollingAvgSpeedEstimator(BaseEstimator):
    """Rolling-window average speed estimator.

    Parameters
    ----------
    window_s : float, optional
        Rolling window in seconds. Defaults to ROLLING_WINDOW_S (300 s).
    moving_only : bool, optional
        If True (default), only accumulate distance and time while moving
        (speed >= 1 km/h).
    """

    def __init__(
        self,
        window_s: float | None = None,
        moving_only: bool = True,
        pause_kmh: float = 1.0,
    ) -> None:
        super().__init__(pause_kmh=pause_kmh)
        self._window_s = ROLLING_WINDOW_S if window_s is None else window_s
        self._moving_only = moving_only

    def predict(self, df: pd.DataFrame) -> pd.Series:
        dd, dt = self._deltas(df)
        if self._moving_only:
            mask = self._moving_mask(df)
            dd, dt = dd * mask, dt * mask
        idx = self._datetime_index(df)
        window = f"{int(self._window_s)}s"
        dd_roll = pd.Series(dd.values, index=idx).rolling(window, min_periods=1).sum()
        dt_roll = pd.Series(dt.values, index=idx).rolling(window, min_periods=1).sum()
        return pd.Series((dd_roll / dt_roll).values, index=df.index)
