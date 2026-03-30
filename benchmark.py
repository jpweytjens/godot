from typing import Protocol

import pandas as pd


class Estimator(Protocol):
    """Protocol for ETA estimators used in backtesting.

    Estimators are stateless functions over the full ride DataFrame.
    Implement predict() to return estimated speed (m/s) at each row.
    """

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return estimated speed in m/s at each row.

        Parameters
        ----------
        df : pd.DataFrame
            Ride DataFrame with columns: time, distance_m, elevation_m, speed_kmh.

        Returns
        -------
        pd.Series
            Speed in m/s at each row. NaN where insufficient data exists.
        """
        ...


def backtest(
    df: pd.DataFrame, estimator: Estimator, min_speed_kmh: float | None = None
) -> pd.DataFrame:
    """Run an estimator over a ride DataFrame and record ETA vs ATA.

    Parameters
    ----------
    df : pd.DataFrame
        Ride DataFrame with columns: time, distance_m, elevation_m, speed_kmh.
    estimator : Estimator
        Estimator implementing predict().
    min_speed_kmh : float, optional
        Speed threshold below which ETA is set to NaN (stopped / near-stopped).
        Defaults to 5.0 km/h.

    Returns
    -------
    pd.DataFrame
        Columns: time, distance_m, speed_ms, eta_remaining_s, ata_remaining_s, delta_s.
        delta_s = eta_remaining_s - ata_remaining_s (positive = overestimate).
    """
    if min_speed_kmh is None:
        min_speed_kmh = 5.0
    speed_ms = estimator.predict(df)
    remaining_m = df["distance_m"].iloc[-1] - df["distance_m"]
    ata_s = (df["time"].iloc[-1] - df["time"]).dt.total_seconds()
    eta_s = remaining_m / speed_ms.where(speed_ms >= min_speed_kmh / 3.6)
    return pd.DataFrame(
        {
            "time": df["time"].values,
            "distance_m": df["distance_m"].values,
            "speed_ms": speed_ms.values,
            "eta_remaining_s": eta_s.values,
            "ata_remaining_s": ata_s.values,
            "delta_s": (eta_s - ata_s).values,
        }
    )
