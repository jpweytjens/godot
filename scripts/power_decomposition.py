"""Smoke test: decompose v(g)/v_flat with measured power.

For every FIT file with power data, compute two curves vs gradient:

1. Physics residual `v_obs / v_phys`, where `v_phys` is the steady-state
   speed predicted by the cubic at the *measured* power. With power
   subtracted out, the residual should be:
   - flat across climbs (CdA/Crr/wind-only signal)
   - well below 1 on descents (rider braking — the comfort signal)

2. Behavioural power curve `P_obs / P_flat` — what the rider actually
   does at each gradient, replacing the hand-tuned `climb_effort` and
   `descent_decay_k` of `realistic_physics_ratios`.

Samples are aggregated across all FITs and binned by 1% gradient.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import numpy as np
import pandas as pd

import godot.theme  # noqa: F401
from godot.config import RideConfig
from godot.fit import read_fit
from godot.gpx import add_haversine_distance
from godot.segmentation import RouteSegment, decimate_to_gradient_segments


def solve_v_from_power_vec(
    p_watts: np.ndarray,
    gradient: np.ndarray,
    cfg: RideConfig,
    v_init: np.ndarray,
    n_iter: int = 8,
) -> np.ndarray:
    """Vectorised steady-state v from the cubic, via Newton's method.

    f(v) = k_a (v + w)^2 v + m g (Crr cosθ + sinθ) v - P = 0
    f'(v) = k_a (3v^2 + 4 w v + w^2) + m g (Crr cosθ + sinθ)

    Starting from `v_init` (typically the observed speed), 8 iterations
    converge to machine precision for normal cycling regimes.
    """
    g = 9.81
    k_a = 0.5 * cfg.rho * cfg.cda
    m = cfg.total_mass_kg
    w = cfg.headwind_ms
    theta = np.arctan(gradient)
    slope_term = m * g * (cfg.crr * np.cos(theta) + np.sin(theta))

    v = np.maximum(v_init, 0.5)
    for _ in range(n_iter):
        f = k_a * (v + w) ** 2 * v + slope_term * v - p_watts
        fprime = k_a * (3 * v**2 + 4 * w * v + w**2) + slope_term
        with np.errstate(divide="ignore", invalid="ignore"):
            v = v - f / fprime
        v = np.maximum(v, 0.1)
    return v


def assign_segment_gradient(
    distances_m: np.ndarray, segments: list[RouteSegment]
) -> np.ndarray:
    starts = np.array([s.start_distance_m for s in segments])
    grads = np.array([s.gradient for s in segments])
    idx = np.searchsorted(starts, distances_m, side="right") - 1
    idx = np.clip(idx, 0, len(segments) - 1)
    return grads[idx]


def process_ride(path: Path, cfg: RideConfig) -> pd.DataFrame | None:
    df = read_fit(path)
    needed = {"watts", "speed_ms", "elevation_m", "lat", "lon"}
    if not needed.issubset(df.columns):
        return None
    if df["watts"].notna().sum() < 200:
        return None
    df = df.pipe(add_haversine_distance)

    clean = df.dropna(subset=["distance_m", "elevation_m"])
    if len(clean) < 200:
        return None
    _, segs = decimate_to_gradient_segments(clean)
    if not segs:
        return None

    gradient = assign_segment_gradient(df["distance_m"].to_numpy(), segs)
    df = df.assign(gradient=gradient)

    df = df[
        (df["watts"] > 0)
        & (df["speed_ms"] > 1.0)
        & df["watts"].notna()
        & df["gradient"].notna()
    ]
    if len(df) < 200:
        return None

    flat_mask = df["gradient"].abs() < 0.01
    if flat_mask.sum() < 100:
        return None
    p_flat = df.loc[flat_mask, "watts"].median()
    if not (50 < p_flat < 500):
        return None
    v_flat = df.loc[flat_mask, "speed_ms"].median()
    if not (3 < v_flat < 20):
        return None

    v_phys = solve_v_from_power_vec(
        df["watts"].to_numpy(dtype=float),
        df["gradient"].to_numpy(dtype=float),
        cfg,
        v_init=df["speed_ms"].to_numpy(dtype=float),
    )
    df = df.assign(v_phys=v_phys)
    df = df[df["v_phys"].notna() & (df["v_phys"] > 0.1)]
    df = df.assign(
        residual=df["speed_ms"] / df["v_phys"],
        p_ratio=df["watts"] / p_flat,
        v_ratio=df["speed_ms"] / v_flat,
        ride=path.stem,
    )
    return df[["ride", "gradient", "residual", "p_ratio", "v_ratio"]]


def bin_curve(df: pd.DataFrame, value: str, min_count: int = 100) -> pd.DataFrame:
    bins = (
        df.assign(grad_pct=(df["gradient"] * 100).round().astype(int))
        .groupby("grad_pct")[value]
        .agg(["median", "count"])
        .reset_index()
    )
    bins = bins[bins["count"] >= min_count]
    return bins[(bins["grad_pct"] >= -15) & (bins["grad_pct"] <= 15)]


_MODEL_ORDER = ["constant power", "freewheel model", "FTP model"]
_MODEL_COLORS = ["#4477AA", "#EE6677", "#228833"]


def build_v_model_df(cfg: RideConfig) -> pd.DataFrame:
    curves = {
        "constant power": cfg.cubic_power_ratios,
        "freewheel model": cfg.realistic_ratios,
        "FTP model": cfg.ftp_ratios,
    }
    return pd.DataFrame(
        [
            {"grad_pct": pct, "ratio": r, "assumptions": name}
            for name, curve in curves.items()
            for pct, r in curve.items()
        ]
    )


def build_p_model_df(cfg: RideConfig) -> pd.DataFrame:
    """Reuse the P(g)/P(0) models from scripts/power_curve.py."""
    from power_curve import freewheel_power, ftp_power, p_flat

    p0 = p_flat(cfg)
    gradients = range(cfg.grad_min_pct, cfg.grad_max_pct + 1)
    rows = []
    for pct in gradients:
        rows.append({"grad_pct": pct, "ratio": 1.0, "assumptions": "constant power"})
        rows.append(
            {
                "grad_pct": pct,
                "ratio": freewheel_power(cfg, pct, p0) / p0,
                "assumptions": "freewheel model",
            }
        )
        rows.append(
            {
                "grad_pct": pct,
                "ratio": ftp_power(cfg, pct, p0) / p0,
                "assumptions": "FTP model",
            }
        )
    return pd.DataFrame(rows)


def _reference_rules() -> alt.LayerChart:
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
    return vline + hline


def _model_layer(models: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(models)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("grad_pct:Q"),
            y=alt.Y("ratio:Q"),
            color=alt.Color(
                "assumptions:N",
                scale=alt.Scale(domain=_MODEL_ORDER, range=_MODEL_COLORS),
                title="assumptions",
            ),
        )
    )


def _empirical_layer(empirical: pd.DataFrame, y_title: str) -> alt.Chart:
    return (
        alt.Chart(empirical)
        .mark_line(point=True, color="black", strokeWidth=2)
        .encode(
            x=alt.X("grad_pct:Q", title="gradient (%)"),
            y=alt.Y("median:Q", title=y_title),
        )
    )


def make_chart(
    residual: pd.DataFrame,
    v_empirical: pd.DataFrame,
    p_empirical: pd.DataFrame,
    v_models: pd.DataFrame,
    p_models: pd.DataFrame,
) -> alt.Chart:
    rules = _reference_rules()

    res_chart = (
        rules
        + alt.Chart(residual)
        .mark_line(point=True, color="#4477AA")
        .encode(
            x=alt.X("grad_pct:Q", title="gradient (%)"),
            y=alt.Y("median:Q", title="v_obs / v_phys"),
        )
    ).properties(width=360, height=320, title="Physics residual (CdA/Crr/braking)")

    v_chart = (
        rules + _model_layer(v_models) + _empirical_layer(v_empirical, "v(g) / v_flat")
    ).properties(width=360, height=320, title="Speed ratio: empirical vs models")

    p_chart = (
        rules + _model_layer(p_models) + _empirical_layer(p_empirical, "P(g) / P_flat")
    ).properties(width=360, height=320, title="Power ratio: empirical vs models")

    return (res_chart | v_chart | p_chart).configure_legend(
        disable=False, orient="top-left"
    )


def main() -> None:
    cfg = RideConfig()
    fits = sorted(Path("data/fit").glob("*.fit"))

    parts: list[pd.DataFrame] = []
    for p in fits:
        try:
            r = process_ride(p, cfg)
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        if r is not None and len(r) > 0:
            parts.append(r)

    if not parts:
        print("no usable rides")
        return

    samples = pd.concat(parts, ignore_index=True)
    print(f"{len(parts)} rides, {len(samples):,} samples")
    print(
        f"gradient range: {samples['gradient'].min():.3f} .. {samples['gradient'].max():.3f}"
    )

    residual = bin_curve(samples, "residual")
    v_empirical = bin_curve(samples, "v_ratio")
    p_empirical = bin_curve(samples, "p_ratio")

    cfg_models = RideConfig(ftp_watts=250.0)
    v_models = build_v_model_df(cfg_models)
    p_models = build_p_model_df(cfg_models)

    chart = make_chart(residual, v_empirical, p_empirical, v_models, p_models)
    out = Path("scripts/power_decomposition.svg")
    chart.save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
