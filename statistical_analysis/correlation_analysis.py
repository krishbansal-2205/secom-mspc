"""
statistical_analysis/correlation_analysis.py - Multicollinearity Analysis
===========================================================================

Analyses pairwise correlations, VIF, and eigenvalue condition indices
to assess multicollinearity in the sensor data.
"""

import os
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


class CorrelationAnalyser:
    """Analyse and visualise multicollinearity.

    Args:
        cfg: Optional configuration override.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.fig_dir = os.path.join(self.cfg.figures_dir, "correlation")
        os.makedirs(self.fig_dir, exist_ok=True)

    def analyse(self, X: pd.DataFrame) -> dict:
        """Run full multicollinearity analysis.

        Args:
            X: Processed sensor DataFrame.

        Returns:
            Dictionary with high-correlation pairs, condition number,
            and eigenvalue spectrum.
        """
        print("\n── Correlation / Multicollinearity Analysis ──")
        corr = X.corr()

        # High-correlation pairs
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        pairs: List[dict] = []
        for col in upper.columns:
            for idx in upper.index:
                val = upper.loc[idx, col]
                if pd.notna(val) and abs(val) > self.cfg.correlation_threshold:
                    pairs.append({"feature_a": idx, "feature_b": col, "correlation": round(float(val), 4)})

        print(f"  Pairs with |r| > {self.cfg.correlation_threshold} : {len(pairs)}")

        # Eigenvalue condition number
        eigvals = np.linalg.eigvalsh(corr.values)
        eigvals_sorted = np.sort(eigvals)[::-1]
        condition_number = float(eigvals_sorted[0] / max(eigvals_sorted[-1], 1e-12))
        print(f"  Condition number  : {condition_number:.1f}")

        # Plot eigenvalue spectrum
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(range(1, len(eigvals_sorted) + 1), eigvals_sorted, "o-", ms=3, color="steelblue")
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Eigenvalue (log scale)")
        ax.set_title("Correlation Matrix Eigenvalue Spectrum", fontweight="bold")
        ax.axhline(1.0, ls="--", color="gray", lw=1, label="λ = 1")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(self.fig_dir, "eigenvalue_spectrum.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")

        return {
            "n_high_pairs": len(pairs),
            "condition_number": condition_number,
            "top_pairs": pairs[:20],
        }
