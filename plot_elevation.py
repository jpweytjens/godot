"""Plot original vs VW-simplified elevation profiles with gradient-colored segments."""

from pathlib import Path

import eta.theme  # noqa: F401 — registers the "eta" theme
from eta.plot import elevation_profile
from eta.ride import load_ride

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
        chart = elevation_profile(
            str(ride),
            ride.df,
            min_area=args.min_area,
            min_length_m=args.min_length,
        )
        out_path = out_dir / f"{ride.name}.png"
        chart.save(str(out_path), scale_factor=2)
        print(f"Saved {out_path}")
