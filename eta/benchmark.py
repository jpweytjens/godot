from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import pandas as pd

if TYPE_CHECKING:
    from eta.ride import Ride


class Estimator(Protocol):
    """Protocol for ETA estimators used in backtesting.

    Estimators are stateless functions over the full ride DataFrame.
    Implement predict() to return estimated speed (m/s) at each row.
    """

    def predict(self, ride: Ride) -> pd.Series:
        """Return estimated speed in m/s at each row.

        Parameters
        ----------
        ride : Ride
            Prepared ride from `load_ride`.

        Returns
        -------
        pd.Series
            Speed in m/s at each row. NaN where insufficient data exists.
        """
        ...


def backtest(
    ride: Ride, estimator: Estimator, min_speed_kmh: float | None = None
) -> pd.DataFrame:
    """Run an estimator over a ride and record ETA vs ATA.

    Parameters
    ----------
    ride : Ride
        Prepared ride from `load_ride`.
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
    df = ride.df
    speed_ms = estimator.predict(ride)
    remaining_m = ride.distance - df["distance_m"]
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


def compute_metrics(
    result_df: pd.DataFrame, warmup_distance_m: float
) -> dict[str, float]:
    """Compute MAE and RMSE for a single estimator backtest result.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of `backtest` with `distance_m` and `delta_s` columns.
    warmup_distance_m : float
        Distance threshold; rows below this are excluded.

    Returns
    -------
    dict[str, float]
        Keys: ``mae_min``, ``rmse_min`` (both in minutes).
    """
    trimmed = result_df[result_df["distance_m"] >= warmup_distance_m][
        "delta_s"
    ].dropna()
    return {
        "mae_min": trimmed.abs().mean() / 60,
        "rmse_min": (trimmed**2).mean() ** 0.5 / 60,
    }
