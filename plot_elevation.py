"""Plot original vs VW-simplified elevation profiles with gradient-colored segments."""

import math
from pathlib import Path

import altair as alt
import pandas as pd

import eta.theme  # noqa: F401 — registers the "eta" theme
from eta.ride import load_ride
from eta.segmentation import (
    build_segments,
    merge_short_segments,
    visvalingam_whyatt,
)

# Diverging gradient palette: green (downhill) → grey (flat) → red (uphill)
_GRADIENT_COLORS = {
    -12: "#1a9850",
    -9: "#66bd63",
    -6: "#a6d96a",
    -3: "#d9ef8b",
    0: "#cccccc",
    3: "#fee08b",
    6: "#fdae61",
    9: "#f46d43",
    12: "#d73027",
}


def _gradient_bin(gradient_frac: float) -> int:
    """Assign a gradient (fraction) to a 3%-wide bin, clamped to [-12, 12]."""
    pct = gradient_frac * 100
    b = math.floor(pct / 3) * 3
    return max(-21, min(21, b))


def plot_ride(ride_name: str, df: pd.DataFrame, segments_df: pd.DataFrame) -> alt.Chart:
    """Build a two-row chart: original profile + gradient-colored segments."""
    dist_km = df["distance_m"] / 1000

    original = (
        alt.Chart(
            pd.DataFrame({"distance_km": dist_km, "elevation_m": df["elevation_m"]})
        )
        .mark_line(strokeWidth=0.8, color="#555")
        .encode(
            x=alt.X("distance_km:Q").title("Distance (km)"),
            y=alt.Y("elevation_m:Q").title("Elevation (m)"),
        )
        .properties(
            width=900,
            height=200,
            title=f"Original elevation profile ({len(df):,} points)",
        )
    )

    n_segments = segments_df["segment_id"].nunique()
    n_points = n_segments + 1
    bin_labels = sorted(_GRADIENT_COLORS.keys())
    domain = [f"{b:+d}%" for b in bin_labels]
    range_ = [_GRADIENT_COLORS[b] for b in bin_labels]

    segmented = (
        alt.Chart(segments_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("distance_km:Q").title("Distance (km)"),
            y=alt.Y("elevation_m:Q").title("Elevation (m)"),
            color=alt.Color(
                "gradient_bin:N", scale=alt.Scale(domain=domain, range=range_)
            )
            .title("Gradient")
            .legend(orient="right"),
            detail="segment_id:N",
        )
        .properties(
            width=900,
            height=200,
            title=f"VW-simplified — gradient bins (3%) ({n_points:,} points)",
        )
    )

    return (
        (original & segmented)
        .resolve_scale(y="independent")
        .properties(title=alt.Title(ride_name))
        .configure_legend(disable=False)
    )


def build_segments_df(
    df: pd.DataFrame,
    min_area: float = 2.0,
    min_length_m: float = 200.0,
) -> pd.DataFrame:
    """Run the VW pipeline and return a long-form DataFrame for plotting."""
    points = list(zip(df["distance_m"], df["elevation_m"]))
    points = visvalingam_whyatt(points, min_area)
    points = merge_short_segments(points, min_length_m)
    segments = build_segments(points)

    rows = []
    for i, seg in enumerate(segments):
        b = _gradient_bin(seg.gradient)
        label = f"{b:+d}%"
        rows.append(
            {
                "distance_km": points[i][0] / 1000,
                "elevation_m": points[i][1],
                "gradient_bin": label,
                "segment_id": i,
            }
        )
        rows.append(
            {
                "distance_km": points[i + 1][0] / 1000,
                "elevation_m": points[i + 1][1],
                "gradient_bin": label,
                "segment_id": i,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Elevation profile plots")
    parser.add_argument(
        "paths", nargs="*", type=Path, help="GPX/FIT files (default: data/*)"
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=5000.0,
        help="VW min triangle area (default: 5000)",
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=200.0,
        help="Minimum segment length in meters (default: 200)",
    )
    args = parser.parse_args()

    paths = args.paths or sorted(
        list(Path("data").glob("**/*.gpx")) + list(Path("data").glob("**/*.fit"))
    )
    if not paths:
        parser.error("No files found. Place .gpx/.fit files in data/ or pass paths.")

    out_dir = Path("output") / "elevation"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        ride = load_ride(Path(path))
        seg_df = build_segments_df(
            ride.df,
            min_area=args.min_area,
            min_length_m=args.min_length,
        )
        chart = plot_ride(str(ride), ride.df, seg_df)
        out_path = out_dir / f"{ride.name}.png"
        chart.save(str(out_path), scale_factor=2)
        print(f"Saved {out_path}")
