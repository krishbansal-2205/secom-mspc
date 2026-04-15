"""
predictive_model/imbalance_handler.py - Class Imbalance Handler
=================================================================

Applies SMOTE to the training set only, ensuring no data leakage.
"""

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


class ImbalanceHandler:
    """Handle class imbalance via SMOTE.

    Args:
        cfg: Optional configuration override.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.smote: SMOTE = SMOTE(
            k_neighbors=self.cfg.smote_k_neighbors,
            random_state=self.cfg.random_seed,
        )

    def fit_resample(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE to the training data.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.

        Returns:
            ``(X_resampled, y_resampled)`` tuple.
        """
        print("\n── SMOTE Resampling ──")
        print(f"  Before: class 0={int((y_train==0).sum())}  class 1={int((y_train==1).sum())}")
        X_res, y_res = self.smote.fit_resample(X_train, y_train)
        print(f"  After : class 0={int((y_res==0).sum())}  class 1={int((y_res==1).sum())}")
        return X_res, y_res
