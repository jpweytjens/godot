import numpy as np
import pandas as pd
from segmentation import (
    decimate,
    perpendicular_distance,
    ramer_douglas_peucker,
    merge_short_segments,
    build_segments,
    gradient_at_distance,
)


def flat_df(n: int = 500, total_m: float = 10_000.0) -> pd.DataFrame:
    dist = np.linspace(0, total_m, n)
    return pd.DataFrame({"distance_m": dist, "elevation_m": np.zeros(n)})


def test_decimate_spacing():
    df = flat_df()
    pts = decimate(df, spacing_m=100.0)
    # All interior gaps must meet spacing; last gap may be shorter (final point always kept)
    interior_gaps = [pts[i + 1][0] - pts[i][0] for i in range(len(pts) - 2)]
    assert all(g >= 99.0 for g in interior_gaps)


def test_decimate_always_includes_last_point():
    df = flat_df(n=10, total_m=95.0)
    pts = decimate(df, spacing_m=20.0)
    assert pts[-1][0] == df["distance_m"].iloc[-1]


def test_perpendicular_distance_collinear():
    assert perpendicular_distance((5.0, 0.0), (0.0, 0.0), (10.0, 0.0)) == 0.0


def test_perpendicular_distance_right_angle():
    d = perpendicular_distance((0.0, 5.0), (0.0, 0.0), (10.0, 0.0))
    assert abs(d - 5.0) < 1e-9


def test_rdp_flat_collapses_to_endpoints():
    pts = [(i * 10.0, 0.0) for i in range(100)]
    result = ramer_douglas_peucker(pts, epsilon=0.1)
    assert len(result) == 2


def test_rdp_preserves_peak():
    pts = [(0.0, 0.0), (50.0, 10.0), (100.0, 0.0)]
    result = ramer_douglas_peucker(pts, epsilon=0.5)
    assert len(result) == 3


def test_merge_removes_short():
    pts = [(0.0, 0.0), (5.0, 1.0), (100.0, 5.0), (200.0, 3.0)]
    merged = merge_short_segments(pts, min_length_m=50.0)
    assert all(merged[i + 1][0] - merged[i][0] >= 50.0 for i in range(len(merged) - 1))


def test_merge_always_keeps_endpoints():
    pts = [(0.0, 0.0), (1.0, 0.5), (2.0, 1.0)]
    merged = merge_short_segments(pts, min_length_m=100.0)
    assert merged[0] == pts[0]
    assert merged[-1] == pts[-1]


def test_gradient_calculation():
    pts = [(0.0, 0.0), (1000.0, 50.0)]
    segs = build_segments(pts)
    assert abs(segs[0].gradient - 0.05) < 1e-9


def test_gradient_at_distance():
    pts = [(0.0, 0.0), (500.0, 25.0), (1000.0, 0.0)]
    segs = build_segments(pts)
    assert gradient_at_distance(250.0, segs) == segs[0].gradient
    assert gradient_at_distance(750.0, segs) == segs[1].gradient
