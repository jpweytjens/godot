"""Unit conversion helpers.

Centralizes the few scalar conversions used across the codebase so
there are no bare `/ 3.6` or `* 3.6` literals floating around. Functions
accept scalars, arrays, or pandas Series — anything that supports `*`
and `/` with a float.
"""

from __future__ import annotations

from typing import TypeVar

KMH_TO_MS = 1.0 / 3.6
MS_TO_KMH = 3.6

T = TypeVar("T")


def kmh_to_ms(value: T) -> T:
    """Convert km/h → m/s. Input can be scalar, ndarray, or Series."""
    return value / 3.6  # noqa: TID252  — literal is the canonical form


def ms_to_kmh(value: T) -> T:
    """Convert m/s → km/h. Input can be scalar, ndarray, or Series."""
    return value * 3.6
