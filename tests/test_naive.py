import numpy as np
from eta import AvgSpeedEstimator, RollingAvgSpeedEstimator


class TestAvgSpeedEstimator:
    def test_nan_before_first_update(self):
        est = AvgSpeedEstimator()
        est.reset()
        assert np.isnan(est.predict(0, 10_000, 0))

    def test_constant_speed(self):
        est = AvgSpeedEstimator()
        est.reset()
        speed_ms = 20 / 3.6
        for i in range(3):
            t_ms = i * 10_000
            d_m = i * speed_ms * 10
            est.update(t_ms, d_m, 20.0, 0.0)
        remaining_s = est.predict(d_m, d_m + 5000, int(t_ms))
        assert abs(remaining_s - 5000 / speed_ms) < 0.1

    def test_ignores_stopped_points(self):
        est = AvgSpeedEstimator()
        est.reset()
        est.update(0, 0.0, 0.0, 0.0)
        est.update(10_000, 100.0, 0.0, 0.0)
        assert np.isnan(est.predict(100.0, 10_000.0, 10_000))

    def test_reset_clears_state(self):
        est = AvgSpeedEstimator()
        est.reset()
        est.update(0, 0.0, 20.0, 0.0)
        est.update(10_000, 55.6, 20.0, 0.0)
        est.reset()
        assert np.isnan(est.predict(0, 10_000, 0))


class TestRollingAvgSpeedEstimator:
    def test_nan_before_first_update(self):
        est = RollingAvgSpeedEstimator()
        est.reset()
        assert np.isnan(est.predict(0, 10_000, 0))

    def test_constant_speed(self):
        est = RollingAvgSpeedEstimator()
        est.reset()
        speed_ms = 20 / 3.6
        for i in range(5):
            est.update(i * 10_000, i * speed_ms * 10, 20.0, 0.0)
        remaining_s = est.predict(5 * speed_ms * 10, 5 * speed_ms * 10 + 5000, 50_000)
        assert abs(remaining_s - 5000 / speed_ms) < 0.5

    def test_window_trims_old_observations(self):
        est = RollingAvgSpeedEstimator(window_s=10.0)
        est.reset()
        est.update(0, 0.0, 10.0, 0.0)  # 100s old — trimmed
        est.update(100_000, 500.0, 30.0, 0.0)  # only this remains
        remaining_s = est.predict(500.0, 10_500.0, 100_000)
        expected = 10_000.0 / (30 / 3.6)
        assert abs(remaining_s - expected) < 1.0

    def test_ignores_stopped_points(self):
        est = RollingAvgSpeedEstimator()
        est.reset()
        est.update(0, 0.0, 0.0, 0.0)
        assert np.isnan(est.predict(0, 10_000, 0))

    def test_reset_clears_state(self):
        est = RollingAvgSpeedEstimator()
        est.reset()
        est.update(0, 0.0, 20.0, 0.0)
        est.reset()
        assert np.isnan(est.predict(0, 10_000, 0))
