from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import pandas as pd

from eta.pause import NoPause, PauseStrategy

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
    ride: Ride,
    estimator: Estimator,
    pause_strategy: PauseStrategy | None = None,
) -> pd.DataFrame:
    """Run an estimator over a ride and record ETA vs ATA.

    Parameters
    ----------
    ride : Ride
        Prepared ride from `load_ride`.
    estimator : Estimator
        Estimator implementing predict().
    pause_strategy : PauseStrategy, optional
        Strategy for adjusting ETA during pauses. Defaults to `NoPause()`.

    Returns
    -------
    pd.DataFrame
        Columns: time, distance_m, speed_ms, eta_remaining_s, ata_remaining_s, delta_s.
        delta_s = eta_remaining_s - ata_remaining_s (positive = overestimate).
    """
    if pause_strategy is None:
        pause_strategy = NoPause()
    df = ride.df
    speed_ms = estimator.predict(ride)
    speed_ms = pause_strategy.adjust(speed_ms, ride)
    remaining_m = ride.distance - df["distance_m"]
    ata_s = (df["time"].iloc[-1] - df["time"]).dt.total_seconds()
    eta_s = pause_strategy.fill_pauses(remaining_m / speed_ms, ride)
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
    """Compute accuracy metrics for a single estimator backtest result.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of `backtest` with `distance_m`, `delta_s`, and
        `ata_remaining_s` columns.
    warmup_distance_m : float
        Distance threshold; rows below this are excluded.

    Returns
    -------
    dict[str, float]
        Keys: ``mae_min``, ``rmse_min`` (minutes),
        ``mpe_pct``, ``mape_pct`` (percentage).
    """
    trimmed = result_df[result_df["distance_m"] >= warmup_distance_m].dropna(
        subset=["delta_s"]
    )
    delta = trimmed["delta_s"]
    ata = trimmed["ata_remaining_s"]
    relative = (delta / ata).where(ata > 0)
    return {
        "mae_min": delta.abs().mean() / 60,
        "rmse_min": (delta**2).mean() ** 0.5 / 60,
        "mpe_pct": relative.mean() * 100,
        "mape_pct": relative.abs().mean() * 100,
    }
