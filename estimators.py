"""ETA estimators for cycling ride prediction."""

import pandas as pd

ROLLING_WINDOW_S = 300.0


class AvgSpeedEstimator:
    """Estimates speed using a backward-looking expanding window average.

    Parameters
    ----------
    moving_only : bool, optional
        If True (default), denominator is total moving time (speed >= 1 km/h),
        giving average moving speed. If False, denominator is total elapsed time.
    """

    def __init__(self, moving_only: bool = True) -> None:
        self._moving_only = moving_only

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return estimated speed (m/s) at each row using an expanding window.

        Parameters
        ----------
        df : pd.DataFrame
            Ride DataFrame with timestamp_ms, distance_m, speed_kmh columns.

        Returns
        -------
        pd.Series
            Speed in m/s at each row. NaN until sufficient data is accumulated.
        """
        dd = df["distance_m"].diff().clip(lower=0).fillna(0)
        dt = df["timestamp_ms"].diff().fillna(0) / 1000.0
        if self._moving_only:
            is_moving = (df["speed_kmh"] >= 1.0).astype(float)
            cum_dd = (dd * is_moving).cumsum()
            cum_dt = (dt * is_moving).cumsum()
        else:
            cum_dd = dd.cumsum()
            cum_dt = dt.cumsum()
        return (cum_dd / cum_dt).where(cum_dt > 0)


class RollingAvgSpeedEstimator:
    """Estimates speed using a backward-looking rolling time window.

    Parameters
    ----------
    window_s : float, optional
        Rolling window size in seconds. Defaults to ROLLING_WINDOW_S (300 s).
    moving_only : bool, optional
        If True (default), only accumulate time and distance while moving
        (speed >= 1 km/h).
    """

    def __init__(self, window_s: float | None = None, moving_only: bool = True) -> None:
        self._window_s = ROLLING_WINDOW_S if window_s is None else window_s
        self._moving_only = moving_only

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return estimated speed (m/s) at each row using a rolling window.

        Parameters
        ----------
        df : pd.DataFrame
            Ride DataFrame with timestamp_ms, distance_m, speed_kmh columns.

        Returns
        -------
        pd.Series
            Speed in m/s at each row.
        """
        idx = pd.to_datetime(df["timestamp_ms"], unit="ms")
        dd = df["distance_m"].diff().clip(lower=0).fillna(0)
        dt = df["timestamp_ms"].diff().fillna(0) / 1000.0
        if self._moving_only:
            is_moving = (df["speed_kmh"] >= 1.0).astype(float)
            dd = dd * is_moving
            dt = dt * is_moving
        window = f"{int(self._window_s)}s"
        dd_roll = pd.Series(dd.values, index=idx).rolling(window, min_periods=1).sum()
        dt_roll = pd.Series(dt.values, index=idx).rolling(window, min_periods=1).sum()
        speed = (dd_roll / dt_roll).values
        return pd.Series(speed, index=df.index)
