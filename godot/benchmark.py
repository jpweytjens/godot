from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import pandas as pd

from godot.pause import NoPause, PauseStrategy

if TYPE_CHECKING:
    from godot.ride import Ride


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
    predict_current = getattr(estimator, "predict_current", estimator.predict)
    current_speed_ms = predict_current(ride)
    speed_ms = pause_strategy.adjust(speed_ms, ride)
    current_speed_ms = pause_strategy.adjust(current_speed_ms, ride)
    remaining_m = ride.distance - df["distance_m"]

    ata_s = (df["time"].iloc[-1] - df["time"]).dt.total_seconds()
    eta_s = pause_strategy.fill_pauses(remaining_m / speed_ms, ride)

    # Remaining moving time: total moving seconds minus moving seconds elapsed
    moving_dt = df["delta_time"].where(~df["paused"], 0.0)
    moving_elapsed = moving_dt.cumsum()
    ata_moving_s = moving_elapsed.iloc[-1] - moving_elapsed

    return pd.DataFrame(
        {
            "time": df["time"].values,
            "distance_m": df["distance_m"].values,
            "speed_ms": speed_ms.values,
            "current_speed_ms": current_speed_ms.values,
            "eta_remaining_s": eta_s.values,
            "ata_remaining_s": ata_s.values,
            "ata_moving_s": ata_moving_s.values,
            "delta_s": (eta_s - ata_s).values,
            "delta_moving_s": (eta_s - ata_moving_s).values,
        }
    )


def compute_metrics(
    result_df: pd.DataFrame,
    warmup_distance_m: float,
    moving_only: bool = False,
) -> dict[str, float]:
    """Compute accuracy metrics for a single estimator backtest result.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of `backtest`.
    warmup_distance_m : float
        Distance threshold; rows below this are excluded.
    moving_only : bool, optional
        If True, measure against remaining *moving* time instead of
        wall-clock time. Use for estimators that predict moving speed.

    Returns
    -------
    dict[str, float]
        Keys: `mae_min`, `rmse_min` (minutes),
        `mpe_pct`, `mape_pct` (percentage).
    """
    delta_col = "delta_moving_s" if moving_only else "delta_s"
    ata_col = "ata_moving_s" if moving_only else "ata_remaining_s"
    trimmed = result_df[result_df["distance_m"] >= warmup_distance_m].dropna(
        subset=[delta_col]
    )
    delta = trimmed[delta_col]
    ata = trimmed[ata_col]
    relative = (delta / ata).where(ata > 0)
    return {
        "mae_min": delta.abs().mean() / 60,
        "rmse_min": (delta**2).mean() ** 0.5 / 60,
        "mpe_pct": relative.mean() * 100,
        "mape_pct": relative.abs().mean() * 100,
    }
