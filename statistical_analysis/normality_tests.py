"""
statistical_analysis/normality_tests.py - Distribution Testing
================================================================

Provides batch normality testing (Shapiro-Wilk, Anderson-Darling,
D'Agostino-Pearson) and QQ plot generation.
"""

import os
import sys
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


class NormalityTester:
    """Batch normality testing for multivariate sensor data.

    Args:
        cfg: Optional configuration override.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.fig_dir = os.path.join(self.cfg.figures_dir, "normality")
        os.makedirs(self.fig_dir, exist_ok=True)

    def test_all(self, X: pd.DataFrame, sample_size: int = 200) -> pd.DataFrame:
        """Run Shapiro-Wilk and D'Agostino-Pearson tests on every feature.

        Args:
            X: Processed sensor DataFrame.
            sample_size: Random sample size per feature.

        Returns:
            DataFrame with test statistics and p-values per feature.
        """
        print("\n── Normality Tests ──")
        rng = np.random.RandomState(42)
        records = []
        for col in X.columns:
            vals = X[col].dropna().values
            if len(vals) < 20:
                continue
            sub = rng.choice(vals, size=min(sample_size, len(vals)), replace=False)

            # Shapiro-Wilk
            try:
                sw_stat, sw_p = stats.shapiro(sub)
            except Exception:
                sw_stat, sw_p = np.nan, np.nan

            # D'Agostino-Pearson
            try:
                dp_stat, dp_p = stats.normaltest(sub)
            except Exception:
                dp_stat, dp_p = np.nan, np.nan

            records.append({
                "feature": col,
                "shapiro_stat": round(float(sw_stat), 4) if not np.isnan(sw_stat) else None,
                "shapiro_p": float(sw_p) if not np.isnan(sw_p) else None,
                "dagostino_stat": round(float(dp_stat), 4) if not np.isnan(dp_stat) else None,
                "dagostino_p": float(dp_p) if not np.isnan(dp_p) else None,
                "is_normal_005": bool(sw_p > 0.05) if not np.isnan(sw_p) else False,
            })

        df = pd.DataFrame(records)
        n_normal = int(df["is_normal_005"].sum())
        print(f"  Normal at α=0.05  : {n_normal}/{len(df)}")
        print(f"  Non-normal        : {len(df) - n_normal}/{len(df)}")

        # save
        path = os.path.join(self.cfg.tables_dir, "normality_tests.csv")
        os.makedirs(self.cfg.tables_dir, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"  ✓ Saved → {path}")

        return df

    def plot_qq_grid(self, X: pd.DataFrame, n_features: int = 12) -> None:
        """QQ plots for a selection of features.

        Args:
            X: Processed sensor DataFrame.
            n_features: Number of features to plot.
        """
        cols = X.columns[:n_features]
        rows = int(np.ceil(n_features / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
        axes = axes.ravel()

        for i, col in enumerate(cols):
            stats.probplot(X[col].dropna().values, dist="norm", plot=axes[i])
            axes[i].set_title(col, fontsize=9)
        for j in range(len(cols), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("QQ Plots", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(self.fig_dir, "qq_plots.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")
