"""Plot P(g)/P(0) vs gradient for the two behavioural models.

Isolates the rider-behaviour layer of `realistic_physics_ratios` and
`very_realistic_physics_ratios`: how the rider's target power varies
with gradient, relative to flat-ground power. No cubic solver — this
is just the power model.

- freewheel model: linear climb effort (unbounded) + linear descent
  backoff
- FTP model: climb effort capped at `1.2 * ftp_watts`, descent power
  decays exponentially
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd

import godot.theme  # noqa: F401
from godot.config import RideConfig


def p_flat(cfg: RideConfig) -> float:
    """Back-solve flat-ground rider power from v_flat and headwind."""
    if cfg.power_watts is not None:
        return float(cfg.power_watts)
    g = 9.81
    k_a = 0.5 * cfg.rho * cfg.cda
    v = cfg.v_flat_ms
    w = cfg.headwind_ms
    return k_a * (v + w) ** 2 * v + cfg.crr * cfg.total_mass_kg * g * v


def freewheel_power(cfg: RideConfig, pct: int, p0: float) -> float:
    g = 9.81
    k_a = 0.5 * cfg.rho * cfg.cda
    v = cfg.v_flat_ms
    w = cfg.headwind_ms
    m = cfg.total_mass_kg
    theta = math.atan(pct / 100)
    sin_t, cos_t = math.sin(theta), math.cos(theta)

    if pct > 0:
        p_const = (k_a * (v + w) ** 2 + cfg.crr * m * g * cos_t + m * g * sin_t) * v
        return p0 + cfg.climb_effort * (p_const - p0)
    if pct < 0:
        p_grav = m * g * abs(sin_t) * v
        return max(0.0, p0 - (1 - cfg.descent_confidence) * p_grav)
    return p0


def ftp_power(cfg: RideConfig, pct: int, p0: float) -> float:
    g = 9.81
    k_a = 0.5 * cfg.rho * cfg.cda
    v = cfg.v_flat_ms
    w = cfg.headwind_ms
    m = cfg.total_mass_kg
    theta = math.atan(pct / 100)
    sin_t, cos_t = math.sin(theta), math.cos(theta)

    if cfg.ftp_watts is not None:
        p_max = 1.2 * float(cfg.ftp_watts)
    else:
        p_max = cfg.p_max_multiplier * p0

    if pct > 0:
        p_const = (k_a * (v + w) ** 2 + cfg.crr * m * g * cos_t + m * g * sin_t) * v
        return min(p_max, p0 + cfg.climb_effort * (p_const - p0))
    if pct < 0:
        return p0 * math.exp(-cfg.descent_decay_k * abs(pct))
    return p0


def build_dataframe(cfg: RideConfig) -> pd.DataFrame:
    p0 = p_flat(cfg)
    gradients = range(cfg.grad_min_pct, cfg.grad_max_pct + 1)
    models = {
        "constant power": lambda cfg, pct, p0: p0,
        "freewheel model": freewheel_power,
        "FTP model": ftp_power,
    }
    rows = [
        {
            "gradient_pct": pct,
            "ratio": fn(cfg, pct, p0) / p0,
            "assumptions": name,
        }
        for name, fn in models.items()
        for pct in gradients
    ]
    return pd.DataFrame(rows)


def make_chart(df: pd.DataFrame) -> alt.Chart:
    line = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("gradient_pct:Q", title="gradient (%)"),
            y=alt.Y("ratio:Q", title="P(g) / P(0)"),
            color=alt.Color(
                "assumptions:N",
                title="assumptions",
                scale=alt.Scale(
                    domain=["constant power", "freewheel model", "FTP model"],
                    range=["#4477AA", "#EE6677", "#228833"],
                ),
            ),
        )
    )
    vline = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color="grey", strokeDash=[2, 2])
        .encode(x="x:Q")
    )
    hline = (
        alt.Chart(pd.DataFrame({"y": [1]}))
        .mark_rule(color="grey", strokeDash=[2, 2])
        .encode(y="y:Q")
    )
    return (
        (vline + hline + line)
        .properties(width=640, height=400, title="Rider power vs gradient")
        .configure_legend(disable=False, orient="top-left")
    )


def main() -> None:
    cfg = RideConfig(ftp_watts=250.0)
    df = build_dataframe(cfg)
    chart = make_chart(df)

    out = Path("scripts/power_curve.svg")
    chart.save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
