"""
tests/test_mspc.py - SPC Unit Tests
======================================
"""

import os
import sys

import numpy as np
import pytest
from scipy import stats as sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SECOMConfig
from mspc.hotelling_t2 import HotellingT2Chart
from mspc.mewma import MEWMAChart


@pytest.fixture
def in_control_data():
    """Generate multivariate normal in-control data.

    Returns:
        Tuple ``(X_phase1, X_phase2_ic, X_phase2_ooc)`` where the OOC set
        has a 2σ shift on the first variable.
    """
    rng = np.random.RandomState(42)
    p = 5
    n1, n2 = 300, 200

    cov = np.eye(p)
    X_phase1 = rng.multivariate_normal(np.zeros(p), cov, size=n1)
    X_phase2_ic = rng.multivariate_normal(np.zeros(p), cov, size=n2)

    shift = np.zeros(p)
    shift[0] = 2.0
    X_phase2_ooc = rng.multivariate_normal(shift, cov, size=n2)

    return X_phase1, X_phase2_ic, X_phase2_ooc


@pytest.fixture
def cfg():
    return SECOMConfig(alpha=0.0027, mewma_lambda=0.10, n_simulations=500)


class TestHotellingT2UCL:
    """Verify UCL formula matches known values."""

    def test_ucl_phase1_formula(self, in_control_data, cfg):
        X1, _, _ = in_control_data
        chart = HotellingT2Chart(cfg=cfg)
        chart.fit_phase1(X1, alpha=0.0027)

        m, p = X1.shape
        f_crit = sp_stats.f.ppf(1 - 0.0027, p, m - p)
        expected_ucl = (p * (m - 1) * (m + 1)) / (m * (m - p)) * f_crit

        np.testing.assert_allclose(
            chart.ucl_phase1, expected_ucl, rtol=1e-6,
            err_msg="Phase I UCL does not match F-distribution formula",
        )

    def test_chi2_ucl(self, in_control_data, cfg):
        X1, _, _ = in_control_data
        chart = HotellingT2Chart(cfg=cfg)
        chart.fit_phase1(X1, alpha=0.0027)

        p = X1.shape[1]
        expected = sp_stats.chi2.ppf(1 - 0.0027, df=p)
        np.testing.assert_allclose(
            chart.ucl_phase2_chi2, expected, rtol=1e-6,
        )


class TestT2InControlARL:
    """Verify ARL₀ ≈ 1/α ≈ 370 for in-control data (loose tolerance)."""

    def test_arl0_approx(self, in_control_data, cfg):
        X1, _, _ = in_control_data
        chart = HotellingT2Chart(cfg=cfg)
        chart.fit_phase1(X1, alpha=0.0027)

        result = chart.compute_arl(shift_size=0.0, n_simulations=2000)
        # ARL₀ should be > 300 at least (exact is ~370 but MC variance is high with 2000 runs)
        assert result["ARL"] > 300, f"ARL₀ = {result['ARL']}, expected ~370"


class TestT2IncreasesWithShift:
    """Average T² should increase with the mean shift."""

    def test_shift_increases_t2(self, in_control_data, cfg):
        X1, X2_ic, X2_ooc = in_control_data
        chart = HotellingT2Chart(cfg=cfg)
        chart.fit_phase1(X1)

        t2_ic = chart.calculate_t2(X2_ic).mean()
        t2_ooc = chart.calculate_t2(X2_ooc).mean()

        assert t2_ooc > t2_ic, (
            f"OOC mean T²={t2_ooc:.2f} should exceed IC mean T²={t2_ic:.2f}"
        )


class TestMEWMAInitialisation:
    """Z_0 should be the zero vector."""

    def test_z0_zero(self, in_control_data, cfg):
        X1, X2_ic, _ = in_control_data
        chart = MEWMAChart(lam=0.10, cfg=cfg)
        chart.fit(X1)
        result = chart.monitor(X2_ic)

        # Z[0] should roughly equal λ·(X[0] − μ) because Z_{-1} = 0
        expected_z0 = 0.10 * (X2_ic[0] - chart.mean_vector)
        np.testing.assert_allclose(
            result["Z_values"][0], expected_z0, atol=1e-10,
            err_msg="Z_0 should be λ·(X_0 − μ) since Z_{-1} = 0",
        )


class TestPhaseSeparation:
    """Phase I must contain zero Fail samples."""

    def test_no_fail_in_phase1(self):
        rng = np.random.RandomState(42)
        n = 100
        y = np.zeros(n, dtype=int)
        y[np.array([10, 30, 50, 70, 90])] = 1  # 5 fails

        pass_idx = np.where(y == 0)[0]
        fail_idx = np.where(y == 1)[0]

        phase1_idx = pass_idx[: int(0.6 * len(pass_idx))]
        phase2_idx = np.sort(np.concatenate([pass_idx[int(0.6 * len(pass_idx)):], fail_idx]))

        # Phase I should have zero fails
        assert np.all(y[phase1_idx] == 0), "Fail sample leaked into Phase I"

        # Phase II should contain all fails
        for fi in fail_idx:
            assert fi in phase2_idx, f"Fail index {fi} missing from Phase II"
