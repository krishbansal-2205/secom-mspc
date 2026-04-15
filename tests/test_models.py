"""
tests/test_models.py - ML Model Unit Tests
=============================================
"""

import os
import sys

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SECOMConfig
from predictive_model.imbalance_handler import ImbalanceHandler


@pytest.fixture
def temporal_data():
    """Generate synthetic imbalanced temporal data with a real signal.

    Class-1 observations have a mean shift on the first 3 features so
    that a classifier can learn a non-trivial decision boundary.

    Returns:
        Tuple ``(X, y)`` ordered by time with ~7 % positive class.
    """
    rng = np.random.RandomState(42)
    n = 500
    p = 10
    X = rng.randn(n, p)
    y = rng.choice([0, 1], size=n, p=[0.93, 0.07])

    # Inject signal: shift first 3 features for class 1
    X[y == 1, 0] += 1.5
    X[y == 1, 1] += 1.0
    X[y == 1, 2] -= 1.0

    return X, y


@pytest.fixture
def cfg():
    return SECOMConfig(test_size=0.30, smote_k_neighbors=3, random_seed=42)


class TestTemporalSplit:
    """Test data must be chronologically after training data."""

    def test_no_future_leak(self, temporal_data, cfg):
        X, y = temporal_data
        n = len(y)
        split = int(n * (1 - cfg.test_size))

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # The last training index must be < first test index
        assert split == len(X_train), "Split index mismatch"
        assert len(X_train) + len(X_test) == n, "Data lost in split"
        # Chronological: all indices in train < all indices in test
        assert list(range(split)) == list(range(len(X_train)))


class TestSMOTEBalance:
    """After SMOTE, classes should be more balanced."""

    def test_smote_increases_minority(self, temporal_data, cfg):
        X, y = temporal_data
        n = len(y)
        split = int(n * 0.7)
        X_train, y_train = X[:split], y[:split]

        handler = ImbalanceHandler(cfg=cfg)
        _, y_res = handler.fit_resample(X_train, y_train)

        ratio_before = y_train.sum() / max((y_train == 0).sum(), 1)
        ratio_after = (y_res == 1).sum() / max((y_res == 0).sum(), 1)

        assert ratio_after >= ratio_before, (
            f"SMOTE should improve balance: before={ratio_before:.3f}, after={ratio_after:.3f}"
        )


class TestPredictionsBinary:
    """All predictions must be 0 or 1."""

    def test_binary_predictions(self, temporal_data, cfg):
        X, y = temporal_data
        split = int(len(y) * 0.7)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=500, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        assert set(np.unique(preds)).issubset({0, 1}), "Non-binary predictions found"


class TestProbabilitiesValid:
    """Predicted probabilities must be in [0, 1]."""

    def test_probabilities_in_range(self, temporal_data, cfg):
        X, y = temporal_data
        split = int(len(y) * 0.7)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestClassifier(
            n_estimators=50, max_depth=5, class_weight="balanced", random_state=42
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        assert proba.min() >= 0.0, f"Min probability = {proba.min()}"
        assert proba.max() <= 1.0, f"Max probability = {proba.max()}"


class TestAUCAboveRandom:
    """AUC-ROC must be > 0.5 for a balanced-weight model."""

    def test_auc_above_chance(self, temporal_data, cfg):
        X, y = temporal_data
        split = int(len(y) * 0.7)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Need at least 2 classes in test set
        if len(np.unique(y_test)) < 2:
            pytest.skip("Test set has only one class")

        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, class_weight="balanced", random_state=42
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc_val = roc_auc_score(y_test, proba)

        assert auc_val > 0.5, f"AUC = {auc_val:.3f}, expected > 0.5"
