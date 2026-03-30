"""Tufte-inspired Altair theme and Paul Tol color palettes.

Color palettes from Paul Tol's color schemes:
https://personal.sron.nl/~pault/
"""

import altair as alt

# ---------------------------------------------------------------------------
# Paul Tol color palettes
# ---------------------------------------------------------------------------

TOL_BRIGHT = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
]

TOL_VIBRANT = [
    "#EE7733",
    "#0077BB",
    "#33BBEE",
    "#EE3377",
    "#CC3311",
    "#009988",
    "#BBBBBB",
]

TOL_MUTED = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#DDDDDD",
]

COLORS = TOL_BRIGHT


# ---------------------------------------------------------------------------
# Tufte-inspired theme
# ---------------------------------------------------------------------------


@alt.theme.register("eta", enable=True)
def eta_theme() -> alt.theme.ThemeConfig:
    """Minimal theme: no grid, no view border, reduced data-ink."""
    return {
        "config": {
            "view": {"stroke": None},
            "axis": {
                "grid": False,
                "domain": True,
                "domainColor": "#888",
                "ticks": True,
                "tickColor": "#888",
                "tickSize": 4,
                "labelFont": "system-ui",
                "labelFontSize": 11,
                "titleFont": "system-ui",
                "titleFontSize": 12,
                "titleFontWeight": "normal",
            },
            "title": {
                "font": "system-ui",
                "fontSize": 13,
                "fontWeight": "normal",
                "anchor": "start",
            },
            "legend": {"disable": True},
            "range": {"category": TOL_BRIGHT},
        }
    }
