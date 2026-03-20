from typing import Protocol

import pandas as pd


class Estimator(Protocol):
    """Protocol for online ETA estimators used in backtesting.

    Estimators are stateful and process ride data incrementally.
    Call reset() before each backtest run, update() for each data
    point in order, and predict() to get the current ETA.
    """

    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        ...

    def update(
        self,
        timestamp_ms: int,
        distance_m: float,
        speed_kmh: float,
        elevation_m: float,
    ) -> None:
        """Ingest one data point from the ride.

        Parameters
        ----------
        timestamp_ms : int
            Unix timestamp in milliseconds.
        distance_m : float
            Cumulative distance from ride start in meters.
        speed_kmh : float
            Current speed in km/h.
        elevation_m : float
            Current elevation in meters.
        """
        ...

    def predict(
        self,
        current_distance_m: float,
        total_distance_m: float,
        now_ms: int,
    ) -> float:
        """Predict remaining ride time in seconds.

        Parameters
        ----------
        current_distance_m : float
            Current position in meters from ride start.
        total_distance_m : float
            Total route distance in meters.
        now_ms : int
            Current timestamp in milliseconds.

        Returns
        -------
        float
            Estimated remaining time in seconds, or nan if insufficient data.
        """
        ...


def backtest(df: pd.DataFrame, estimator: Estimator) -> pd.DataFrame:
    """Feed ride data row-by-row to an estimator and record ETA vs ATA.

    Parameters
    ----------
    df : pd.DataFrame
        Ride DataFrame with columns: timestamp_ms, distance_m, elevation_m, speed_kmh.
    estimator : Estimator
        Estimator instance implementing the Estimator protocol.

    Returns
    -------
    pd.DataFrame
        Columns: distance_m, eta_remaining_s, ata_remaining_s, delta_s.
        delta_s = eta_remaining_s - ata_remaining_s (positive = overestimate).
    """
    total_distance_m = df["distance_m"].iloc[-1]
    total_time_ms = df["timestamp_ms"].iloc[-1]
    estimator.reset()

    records = []
    for _, row in df.iterrows():
        estimator.update(
            int(row["timestamp_ms"]),
            row["distance_m"],
            row["speed_kmh"],
            row["elevation_m"],
        )
        eta_s = estimator.predict(
            row["distance_m"], total_distance_m, int(row["timestamp_ms"])
        )
        ata_s = (total_time_ms - row["timestamp_ms"]) / 1000.0
        records.append(
            {
                "distance_m": row["distance_m"],
                "eta_remaining_s": eta_s,
                "ata_remaining_s": ata_s,
                "delta_s": eta_s - ata_s,
            }
        )

    return pd.DataFrame(records)
