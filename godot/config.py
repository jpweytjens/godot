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
from godot.convert import kmh_to_ms


@dataclass(frozen=True)
class RideConfig:
    """Rider, bike, environment and estimator hyperparameters.

    Single configuration object passed to every estimator. Grouped
    into: rider/bike mass, flat-ground prior, aerodynamics + rolling
    resistance, power model, correction-layer spans (BinnedAdaptive
    family), and trust / self-calibration settings. Defaults match
    the values historically hardcoded in each estimator's `__init__`.
    """

    # Rider + bike
    rider_mass_kg: float = 85.0
    bike_mass_kg: float = 10.0

    # Flat-ground prior speed (km/h)
    v_flat_kmh: float = 28.8

    # Aerodynamics / rolling resistance
    cda: float = 0.35
    crr: float = 0.005
    rho: float = 1.225

    # Power model
    headwind_kmh: float = 8.0
    power_watts: float | None = None
    ftp_watts: float | None = None
    climb_effort: float = 0.5
    descent_confidence: float = 0.65
    p_max_multiplier: float = 1.8
    descent_decay_k: float = 0.4
    coast_p_grav_ratio: float = 3.0

    # Gradient bin range (%)
    grad_min_pct: int = -20
    grad_max_pct: int = 20

    # BinnedAdaptive correction spans
    slow_span_s: float = 3600.0
    fast_span_s: float = 300.0
    ramp_s: float = 600.0
    bin_size: int = 1
    cal_max_grad: float = 0.02

    # Trusted BinnedAdaptive
    trust_n: int = 60
    trust_window_s: float | None = None
    corr_min: float = 0.5
    corr_max: float = 1.5

    # Calibrating / PI physics
    skip_fraction: float = 0.01
    ewma_fraction: float = 0.10
    min_skip_s: float = 20.0
    max_skip_s: float = 300.0
    min_ewma_span_s: float = 120.0
    max_ewma_span_s: float = 3600.0

    # Data
    gradient_ratios_path: Path = field(
        default_factory=lambda: Path("data/gradient_ratios.parquet")
    )

    @property
    def total_mass_kg(self) -> float:
        return self.rider_mass_kg + self.bike_mass_kg

    @property
    def v_flat_ms(self) -> float:
        return kmh_to_ms(self.v_flat_kmh)

    @property
    def headwind_ms(self) -> float:
        return kmh_to_ms(self.headwind_kmh)

    @cached_property
    def empirical_ratios(self) -> dict[int, float]:
        """Empirical speed ratios by gradient bin (percent)."""
        return pd.read_parquet(self.gradient_ratios_path)["mean_ratio"].to_dict()

    @cached_property
    def cubic_power_ratios(self) -> dict[int, float]:
        """Speed ratios from the constant-power cubic model."""
        from godot.estimators import physics_gradient_ratios

        return physics_gradient_ratios(
            mass_kg=self.total_mass_kg,
            v_flat_ms=self.v_flat_ms,
            cda=self.cda,
            crr=self.crr,
            rho=self.rho,
            grad_min_pct=self.grad_min_pct,
            grad_max_pct=self.grad_max_pct,
        )

    @cached_property
    def realistic_ratios(self) -> dict[int, float]:
        """Speed ratios from the realistic power model."""
        from godot.estimators import realistic_physics_ratios

        return realistic_physics_ratios(
            mass_kg=self.total_mass_kg,
            v_flat_ms=self.v_flat_ms,
            cda=self.cda,
            crr=self.crr,
            rho=self.rho,
            headwind_ms=self.headwind_ms,
            power_watts=self.power_watts,
            climb_effort=self.climb_effort,
            descent_confidence=self.descent_confidence,
            grad_min_pct=self.grad_min_pct,
            grad_max_pct=self.grad_max_pct,
        )

    @cached_property
    def ftp_ratios(self) -> dict[int, float]:
        """Speed ratios from the FTP-aware power model."""
        from godot.estimators import very_realistic_physics_ratios

        return very_realistic_physics_ratios(
            mass_kg=self.total_mass_kg,
            v_flat_ms=self.v_flat_ms,
            cda=self.cda,
            crr=self.crr,
            rho=self.rho,
            headwind_ms=self.headwind_ms,
            power_watts=self.power_watts,
            ftp_watts=self.ftp_watts,
            climb_effort=self.climb_effort,
            p_max_multiplier=self.p_max_multiplier,
            descent_decay_k=self.descent_decay_k,
            grad_min_pct=self.grad_min_pct,
            grad_max_pct=self.grad_max_pct,
        )


DEFAULT_CONFIG = RideConfig()
