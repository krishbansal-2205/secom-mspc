"""
preprocessing/feature_selector.py - Feature Reduction Pipeline
================================================================

Thin wrapper that combines the quality checker and cleaner into a
single callable for convenience.
"""

import os
import sys
from typing import Dict, Tuple

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from preprocessing.quality_checker import DataQualityChecker
from preprocessing.cleaner import SECOMCleaner


class FeatureSelector:
    """Convenience class that runs quality assessment then cleaning.

    Attributes:
        checker: Fitted :class:`DataQualityChecker`.
        cleaner: Fitted :class:`SECOMCleaner`.
    """

    def __init__(self, cfg=None):
        """Initialise with optional configuration override.

        Args:
            cfg: Configuration object; defaults to global singleton.
        """
        self.cfg = cfg or config
        self.checker = DataQualityChecker(self.cfg)
        self.cleaner = SECOMCleaner(self.cfg)

    def run(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, Dict]:
        """Execute quality check then full cleaning pipeline.

        Args:
            X: Raw sensor DataFrame.
            y: Binary target Series.

        Returns:
            ``(X_clean, report)`` where *X_clean* is the fully processed
            DataFrame and *report* is a dictionary of quality + cleaning
            information.
        """
        quality_report = self.checker.run_full_assessment(X, y)
        X_clean = self.cleaner.fit_transform(X.copy(), y)
        preproc_report = self.cleaner.get_preprocessing_report()

        combined = {"quality": quality_report, "preprocessing": preproc_report}
        return X_clean, combined
