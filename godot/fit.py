"""FIT file parsing — mirrors the output of `read_gpx`."""

import gzip
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from garmin_fit_sdk import Decoder, Stream

SEMICIRCLES_TO_DEG = 180 / 2**31

OPTIONAL_FIELDS = {
    "heart_rate": "hr",
    "power": "watts",
    "cadence": "cad",
    "temperature": "atemp",
}


@lru_cache(maxsize=None)
def read_fit(path: Path) -> pd.DataFrame:
    """Parse a FIT file into a raw DataFrame.

    Parameters
    ----------
    path : Path
        Path to the FIT file.

    Returns
    -------
    pd.DataFrame
        Columns: time, lat, lon, elevation_m, speed_ms.
        Additional fields (hr, cad, watts, atemp) included if present.
        Matches the schema of `read_gpx` — pipe functions work unchanged.
    """
    if path.suffixes[-1:] == [".gz"] or str(path).endswith(".fit.gz"):
        with gzip.open(path, "rb") as f:
            stream = Stream.from_byte_array(f.read())
    else:
        stream = Stream.from_file(str(path))
    decoder = Decoder(stream)
    messages, errors = decoder.read(
        apply_scale_and_offset=True,
        convert_datetimes_to_dates=True,
        expand_components=True,
        merge_heart_rates=True,
    )
    if errors:
        raise ValueError(f"FIT decode errors: {errors}")

    # Filter: only cycling activities
    sessions = messages.get("session_mesgs", [])
    if sessions:
        sport = sessions[0].get("sport", "")
        if sport != "cycling":
            raise ValueError(f"Not a cycling activity (sport={sport!r})")

    records = messages.get("record_mesgs", [])
    if not records:
        return pd.DataFrame(columns=["time", "lat", "lon", "elevation_m", "speed_ms"])

    rows = []
    for rec in records:
        lat_sc = rec.get("position_lat")
        lon_sc = rec.get("position_long")
        if lat_sc is None or lon_sc is None:
            continue

        row = {
            "time": pd.Timestamp(rec["timestamp"]),
            "lat": lat_sc * SEMICIRCLES_TO_DEG,
            "lon": lon_sc * SEMICIRCLES_TO_DEG,
            "elevation_m": rec.get("enhanced_altitude", rec.get("altitude", 0.0)),
            "speed_ms": rec.get("enhanced_speed", rec.get("speed", np.nan)),
        }
        for fit_key, col in OPTIONAL_FIELDS.items():
            if fit_key in rec:
                row[col] = rec[fit_key]

        rows.append(row)

    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
