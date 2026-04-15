"""
predictive_model/feature_importance.py - Feature Importance Analysis
======================================================================

Stand-alone module for computing permutation importance and SHAP-like
importance from tree-based models.
"""

import os
import sys
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


class FeatureImportanceAnalyser:
    """Analyse feature importance for the best model.

    Args:
        cfg: Optional configuration override.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.fig_dir = os.path.join(self.cfg.figures_dir, "models")
        os.makedirs(self.fig_dir, exist_ok=True)

    def compute_permutation_importance(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        n_repeats: int = 10,
    ) -> pd.DataFrame:
        """Compute and plot permutation importance.

        Args:
            model: Fitted classifier.
            X_test: Test features.
            y_test: True labels.
            feature_names: Feature names.
            n_repeats: Repetitions per feature.

        Returns:
            DataFrame of importance scores.
        """
        print("\n── Permutation Importance ──")
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=self.cfg.random_seed,
            scoring="roc_auc",
            n_jobs=-1,
        )

        imp_mean = result.importances_mean
        imp_std = result.importances_std
        order = np.argsort(imp_mean)[::-1][:20]

        names = [feature_names[i] if i < len(feature_names) else f"Var{i}" for i in order]
        vals = imp_mean[order]
        errs = imp_std[order]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(names)), vals, xerr=errs, color="#55A868",
                edgecolor="black", lw=0.5, capsize=3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Permutation Importance (AUC-ROC)")
        ax.set_title("Permutation Feature Importance (Top 20)", fontweight="bold")
        plt.tight_layout()

        path = os.path.join(self.fig_dir, "permutation_importance.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")

        df = pd.DataFrame({
            "feature": [feature_names[i] if i < len(feature_names) else f"Var{i}" for i in range(len(imp_mean))],
            "importance_mean": np.round(imp_mean, 4),
            "importance_std": np.round(imp_std, 4),
        }).sort_values("importance_mean", ascending=False)
        df.reset_index(drop=True, inplace=True)
        return df
