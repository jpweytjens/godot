# Cycling ETA — Project Conventions

## File paths
- Always use `pathlib.Path`, never bare strings for file paths.
- Discover GPX files with `Path("data").glob("*.gpx")` — do not hardcode filenames.

## GPX parsing
- `read_gpx(path)` returns a raw DataFrame: lat, lon, elevation, timestamp, speed_ms, plus any available extensions (hr, watts, cad, atemp).
- Distance is added via `.pipe()` functions: `add_haversine_distance` or `add_integrated_distance`.
- `add_smooth_speed` derives `speed_kmh` via 5s rolling mean of `speed_ms`.
- Keeping distance computation separate lets us compare methods and mirrors the Kotlin target.

## GPX data format
- 1-second recording interval, `<gpxtpx:speed>` (m/s) present in all trackpoints.
- Also available: watts, hr, cad, atemp — ignore for now, useful for future versions.
- Test file: `data/criquielion.gpx` (~18,777 points, 2025-08-23 ride).

## Context
This is a Python prototype for a gradient-aware cycling ETA estimator.
The production target is a Kotlin extension — keep algorithm logic portable.
