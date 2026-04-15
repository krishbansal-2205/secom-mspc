"""
tests/test_preprocessing.py - Preprocessing Unit Tests
=========================================================
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SECOMConfig
from preprocessing.cleaner import SECOMCleaner


@pytest.fixture
def sample_data():
    """Create a synthetic dataset mimicking SECOM characteristics.

    Returns:
        Tuple of ``(X, y)`` where ``X`` has:
        - 200 samples, 50 features
        - 5 features with >50 % missing
        - 3 features with zero variance
        - correlation injected between some features
    """
    rng = np.random.RandomState(42)
    n, p = 200, 50
    X = pd.DataFrame(rng.randn(n, p), columns=[f"F{i:03d}" for i in range(1, p + 1)])

    # Inject >50 % missing in 5 features
    for col in ["F001", "F002", "F003", "F004", "F005"]:
        mask = rng.rand(n) < 0.60
        X.loc[mask, col] = np.nan

    # Inject zero-variance features
    X["F048"] = 5.0
    X["F049"] = 5.0
    X["F050"] = 5.0

    # Inject perfect correlation
    X["F046"] = X["F010"] * 1.0001 + 0.0001

    y = pd.Series(rng.choice([0, 1], size=n, p=[0.93, 0.07]))
    return X, y


@pytest.fixture
def cfg():
    """Create a test-specific configuration with tiny thresholds."""
    return SECOMConfig(
        missing_threshold=0.50,
        variance_threshold=0.01,
        correlation_threshold=0.95,
        outlier_iqr_multiplier=3.0,
    )


class TestMissingValueRemoval:
    """Verify that features with >50 % missing are dropped."""

    def test_drops_high_missing(self, sample_data, cfg):
        X, y = sample_data
        cleaner = SECOMCleaner(cfg=cfg)
        X_clean = cleaner.remove_high_missing_features(X.copy())

        # The 5 injected features should be gone
        for col in ["F001", "F002", "F003", "F004", "F005"]:
            assert col not in X_clean.columns, f"{col} should have been dropped"

    def test_keeps_low_missing(self, sample_data, cfg):
        X, _ = sample_data
        cleaner = SECOMCleaner(cfg=cfg)
        X_clean = cleaner.remove_high_missing_features(X.copy())

        assert "F010" in X_clean.columns, "F010 should be retained"


class TestConstantFeatureRemoval:
    """Verify zero-variance features are removed."""

    def test_drops_zero_var(self, sample_data, cfg):
        X, _ = sample_data
        cleaner = SECOMCleaner(cfg=cfg)
        # Remove missing first to avoid NaN issues
        X = cleaner.remove_high_missing_features(X.copy())
        X = X.fillna(0)
        X_clean = cleaner.remove_constant_features(X)

        for col in ["F048", "F049", "F050"]:
            if col in X.columns:
                assert col not in X_clean.columns, f"{col} should be dropped (zero var)"


class TestImputation:
    """Verify no NaN survives imputation."""

    def test_no_nan_after_imputation(self, sample_data, cfg):
        X, _ = sample_data
        cleaner = SECOMCleaner(cfg=cfg)
        X = cleaner.remove_high_missing_features(X.copy())
        X = cleaner.remove_constant_features(X)
        X_imp = cleaner.impute_missing_values(X)

        assert X_imp.isna().sum().sum() == 0, "NaN survived imputation"


class TestScaling:
    """Verify median ≈ 0 after RobustScaler."""

    def test_median_near_zero(self, sample_data, cfg):
        X, _ = sample_data
        cleaner = SECOMCleaner(cfg=cfg)
        X_clean = cleaner.fit_transform(X.copy())

        medians = X_clean.median().abs()
        assert medians.max() < 0.01, f"Max median = {medians.max():.6f}, expected ≈ 0"


class TestTransformConsistency:
    """Verify that fit_transform ≈ fit + transform."""

    def test_consistency(self, sample_data, cfg):
        X, _ = sample_data
        cleaner = SECOMCleaner(cfg=cfg)
        X_ft = cleaner.fit_transform(X.copy())

        # transform the same data
        X_t = cleaner.transform(X.copy())

        # Columns should match
        assert list(X_ft.columns) == list(X_t.columns), "Column mismatch"
        # Values should be very close
        np.testing.assert_allclose(
            X_ft.values, X_t.values, atol=1e-6,
            err_msg="fit_transform vs transform differ",
        )
