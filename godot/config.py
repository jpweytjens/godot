"""Shared rider/ride configuration.

One frozen dataclass holds the ride-level constants that every backtest
script and estimator needs: rider + bike mass, flat-ground prior speed,
and the empirical/realistic gradient ratio tables. Pass a `RideConfig`
instance around instead of duplicating module-level constants.

The dataclass uses `cached_property` for the ratio tables so the
parquet read and physics cubic solve happen once per instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RideConfig:
    """Rider, bike, and environment constants shared by estimators.

    Parameters
    ----------
    rider_mass_kg : float
        Rider mass (kg). Default 85.
    bike_mass_kg : float
        Bike mass (kg). Default 10.
    v_flat_kmh : float
        Preset flat-ground speed (km/h). Default 28.8.
    gradient_ratios_path : Path
        Path to the empirical gradient-ratio parquet. Default
        `data/gradient_ratios.parquet`.
    """

    rider_mass_kg: float = 85.0
    bike_mass_kg: float = 10.0
    v_flat_kmh: float = 28.8
    gradient_ratios_path: Path = field(
        default_factory=lambda: Path("data/gradient_ratios.parquet")
    )

    @property
    def total_mass_kg(self) -> float:
        return self.rider_mass_kg + self.bike_mass_kg

    @property
    def v_flat_ms(self) -> float:
        return self.v_flat_kmh / 3.6

    @cached_property
    def empirical_ratios(self) -> dict[int, float]:
        """Empirical speed ratios by gradient bin (percent)."""
        return pd.read_parquet(self.gradient_ratios_path)["mean_ratio"].to_dict()

    @cached_property
    def realistic_ratios(self) -> dict[int, float]:
        """Speed ratios from the realistic physics model (default params)."""
        from godot.estimators import realistic_physics_ratios

        return realistic_physics_ratios(
            mass_kg=self.total_mass_kg,
            v_flat_ms=self.v_flat_ms,
        )


DEFAULT_CONFIG = RideConfig()
