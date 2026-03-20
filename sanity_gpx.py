"""Sanity-check GPX parsing: elevation profile, speed, and distance method comparison."""

from pathlib import Path

import matplotlib.pyplot as plt

from eta import (
    add_haversine_distance,
    add_integrated_distance,
    add_smooth_speed,
    read_gpx,
)

for gpx_path in Path("data").glob("*.gpx"):
    raw = read_gpx(gpx_path)
    df_h = raw.pipe(add_haversine_distance).pipe(add_smooth_speed)
    df_i = raw.pipe(add_integrated_distance).pipe(add_smooth_speed)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle(gpx_path.stem)

    axes[0].plot(df_h["distance_m"] / 1000, df_h["elevation_m"])
    axes[0].set_ylabel("Elevation (m)")

    axes[1].plot(df_h["distance_m"] / 1000, df_h["speed_kmh"], label="Haversine dist")
    axes[1].plot(
        df_i["distance_m"] / 1000, df_i["speed_kmh"], label="Integrated dist", alpha=0.7
    )
    axes[1].set_ylabel("Speed (km/h)")
    axes[1].legend()

    axes[2].plot(df_h["distance_m"] / 1000, df_h["distance_m"] - df_i["distance_m"])
    axes[2].set_ylabel("Haversine - Integrated (m)")
    axes[2].set_xlabel("Distance (km)")
    axes[2].axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    out = f"sanity_{gpx_path.stem}.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")
    print(f"  Haversine total:   {df_h['distance_m'].iloc[-1] / 1000:.1f} km")
    print(f"  Integrated total:  {df_i['distance_m'].iloc[-1] / 1000:.1f} km")
