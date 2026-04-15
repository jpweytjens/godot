"""Plot v(g)/v(0) vs gradient for the three physics models.

Compares:
- Constant-power cubic (`physics_gradient_ratios`)
- Modified realistic (`realistic_physics_ratios`) — headwind, climb effort,
  descent backoff, freewheel cap
- FTP / behavioural (`very_realistic_physics_ratios`) — P_max climb ceiling
  and exponential descent power decay
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd

import godot.theme  # noqa: F401  (registers "eta" theme)
from godot.config import RideConfig


def build_dataframe(cfg: RideConfig) -> pd.DataFrame:
    curves = {
        "constant power": cfg.cubic_power_ratios,
        "freewheel model": cfg.realistic_ratios,
        "FTP model": cfg.ftp_ratios,
    }
    rows = [
        {"gradient_pct": pct, "ratio": ratio, "assumptions": name}
        for name, ratios in curves.items()
        for pct, ratio in ratios.items()
    ]
    return pd.DataFrame(rows)


def make_chart(df: pd.DataFrame) -> alt.Chart:
    line = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("gradient_pct:Q", title="gradient (%)"),
            y=alt.Y("ratio:Q", title="v(g) / v(0)"),
            color=alt.Color("assumptions:N", title="assumptions"),
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
        .properties(width=640, height=400, title="Speed ratio vs gradient")
        .configure_legend(disable=False, orient="top-right")
    )


def main() -> None:
    cfg = RideConfig(ftp_watts=250.0)
    df = build_dataframe(cfg)
    chart = make_chart(df)

    out = Path("scripts/gradient_curve.svg")
    chart.save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
