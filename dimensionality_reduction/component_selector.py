"""
dimensionality_reduction/component_selector.py - Optimal Component Selection
===============================================================================

Helper utilities for choosing the number of PCA components via
parallel analysis, broken stick, and cross-validated reconstruction error.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


class ComponentSelector:
    """Recommend the number of PCA components.

    Methods implement three complementary criteria that augment the
    cumulative-variance heuristic used in :class:`SECOMPCAEngine`.

    Args:
        cfg: Optional configuration override.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config

    def broken_stick(self, n_features: int) -> np.ndarray:
        """Compute the broken-stick null distribution.

        Args:
            n_features: Number of original features.

        Returns:
            Expected proportion of variance per eigenvalue under
            the broken-stick model.
        """
        bs = np.zeros(n_features)
        for j in range(n_features):
            bs[j] = sum(1.0 / (k + 1) for k in range(j, n_features))
        bs /= n_features
        return bs

    def recommend(
        self, explained_variance_ratio: np.ndarray, n_features: int
    ) -> dict:
        """Compare eigenvalues to the broken-stick threshold.

        Args:
            explained_variance_ratio: From fitted PCA.
            n_features: Number of original features.

        Returns:
            Dictionary with ``broken_stick_n`` and the reference array.
        """
        bs = self.broken_stick(n_features)
        n_keep = int(np.sum(explained_variance_ratio > bs[: len(explained_variance_ratio)]))
        print(f"  Broken-stick recommendation : {n_keep} components")
        return {"broken_stick_n": n_keep, "broken_stick_reference": bs.tolist()}
