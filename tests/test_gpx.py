from pathlib import Path
import pandas as pd
import pytest
from godot.gpx import (
    read_gpx,
    add_haversine_distance,
    add_integrated_distance,
    add_smooth_speed,
)

GPX_FILES = list(Path("data").glob("*.gpx"))
pytestmark = pytest.mark.skipif(not GPX_FILES, reason="No GPX files in data/")
GPX = GPX_FILES[0] if GPX_FILES else None


def test_read_gpx_columns():
    df = read_gpx(GPX)
    for col in ["time", "lat", "lon", "elevation_m", "speed_ms"]:
        assert col in df.columns


def test_read_gpx_timestamps_sorted():
    df = read_gpx(GPX)
    assert (df["time"].diff().dropna() > pd.Timedelta(0)).all()


def test_read_gpx_no_distance_column():
    df = read_gpx(GPX)
    assert "distance_m" not in df.columns


def test_haversine_starts_at_zero():
    df = read_gpx(GPX).pipe(add_haversine_distance)
    assert df["distance_m"].iloc[0] == 0.0


def test_haversine_monotonic():
    df = read_gpx(GPX).pipe(add_haversine_distance)
    assert (df["distance_m"].diff().dropna() >= 0).all()


def test_integrated_starts_at_zero():
    df = read_gpx(GPX).pipe(add_integrated_distance)
    assert df["distance_m"].iloc[0] == 0.0


def test_integrated_monotonic():
    df = read_gpx(GPX).pipe(add_integrated_distance)
    assert (df["distance_m"].diff().dropna() >= 0).all()


def test_smooth_speed_non_negative():
    df = read_gpx(GPX).pipe(add_haversine_distance).pipe(add_smooth_speed)
    assert (df["speed_kmh"] >= 0).all()


def test_reasonable_total_distance_haversine():
    df = read_gpx(GPX).pipe(add_haversine_distance)
    assert 1_000 < df["distance_m"].iloc[-1] < 500_000


def test_reasonable_total_distance_integrated():
    df = read_gpx(GPX).pipe(add_integrated_distance)
    assert 1_000 < df["distance_m"].iloc[-1] < 500_000
