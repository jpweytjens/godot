from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Protocol

import numpy as np
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
            "paused": df["paused"].values,
        }
    )


def _settling_time_s(
    elapsed_s: np.ndarray,
    abs_err_s: np.ndarray,
    tol_s: float,
    window_s: float,
) -> float:
    """Settling time: first moment after which `|err|` stays below `tol_s`
    for the next `window_s` seconds. `elapsed_s` must be non-decreasing.
    Returns NaN if the error never settles."""
    n = len(elapsed_s)
    if n == 0:
        return float("nan")
    # Forward-looking rolling max of abs_err over [t, t + window_s], linear
    # via a monotonic-decreasing deque keyed on index.
    fwd_max = np.empty(n)
    dq: deque[int] = deque()
    j = 0
    for i in range(n):
        while j < n and elapsed_s[j] <= elapsed_s[i] + window_s:
            while dq and abs_err_s[dq[-1]] <= abs_err_s[j]:
                dq.pop()
            dq.append(j)
            j += 1
        while dq and dq[0] < i:
            dq.popleft()
        fwd_max[i] = abs_err_s[dq[0]] if dq else abs_err_s[i]
    settled = fwd_max <= tol_s
    if not settled.any():
        return float("nan")
    return float(elapsed_s[int(settled.argmax())])


def compute_metrics(
    result_df: pd.DataFrame,
    warmup_distance_m: float,
    moving_only: bool = False,
    settling_rel_pct: float = 0.05,
    settling_floor_s: float = 60.0,
    settling_window_s: float = 120.0,
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
    settling_rel_pct : float, optional
        Relative tolerance for settling time, as a fraction of the initial
        remaining ETA. Default 0.05 (5%).
    settling_floor_s : float, optional
        Absolute floor on the settling tolerance in seconds. Default 60.
    settling_window_s : float, optional
        Sustained-below-tolerance window in seconds (measured in moving
        time when `moving_only=True`). Default 120.

    Returns
    -------
    dict[str, float]
        Keys: `mae_min`, `rmse_min` (minutes),
        `mpe_pct`, `mape_pct` (percentage),
        `settle_min` (minutes of moving time from warmup cutoff; NaN if
        the error never settled).
    """
    delta_col = "delta_moving_s" if moving_only else "delta_s"
    ata_col = "ata_moving_s" if moving_only else "ata_remaining_s"
    trimmed = result_df[result_df["distance_m"] >= warmup_distance_m].dropna(
        subset=[delta_col]
    )
    # Exclude paused rows — error is frozen during pauses and doesn't
    # reflect speed prediction quality.
    if "paused" in trimmed.columns:
        trimmed = trimmed[~trimmed["paused"]]
    delta = trimmed[delta_col]
    ata = trimmed[ata_col]
    relative = (delta / ata).where(ata > 0)

    # Settling time: elapsed (moving) seconds from the warmup cutoff until
    # |delta| stays below max(floor, rel_pct * initial_eta) for window_s.
    if len(trimmed) > 0 and float(ata.iloc[0]) > 0:
        initial_eta_s = float(ata.iloc[0])
        tol_s = max(settling_floor_s, settling_rel_pct * initial_eta_s)
        elapsed_s = (ata.iloc[0] - ata).values
        settle_s = _settling_time_s(
            elapsed_s, delta.abs().values, tol_s, settling_window_s
        )
    else:
        settle_s = float("nan")

    return {
        "mae_min": delta.abs().mean() / 60,
        "rmse_min": (delta**2).mean() ** 0.5 / 60,
        "mpe_pct": relative.mean() * 100,
        "mape_pct": relative.abs().mean() * 100,
        "settle_min": settle_s / 60,
    }
