"""
mspc/mewma.py - Multivariate EWMA Control Chart
===================================================

Implements the MEWMA chart with time-varying covariance structure
for the first 50 observations and asymptotic limits thereafter.
"""

import os
import sys
from typing import Dict, List, Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


class MEWMAChart:
    """Multivariate EWMA control chart.

    The MEWMA statistic is defined as:

    .. math::

        Z_i = \\lambda (X_i - \\mu_0) + (1 - \\lambda) Z_{i-1}, \\quad Z_0 = 0

    with time-varying covariance:

    .. math::

        \\Sigma_{Z,i} = \\frac{\\lambda}{2-\\lambda}
                        \\left[1 - (1-\\lambda)^{2i}\\right] \\Sigma

    Args:
        lam: Smoothing parameter λ ∈ (0, 1].
        cfg: Optional configuration override.
    """

    def __init__(self, lam: Optional[float] = None, cfg=None):
        self.cfg = cfg or config
        self.lam: float = lam if lam is not None else self.cfg.mewma_lambda
        self.mean_vector: Optional[np.ndarray] = None
        self.cov_matrix: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.ucl_asymptotic: float = 0.0
        self._p: int = 0
        self._m: int = 0
        self.fig_dir = os.path.join(self.cfg.figures_dir, "mspc")
        os.makedirs(self.fig_dir, exist_ok=True)

    # =================================================================
    # Fitting (Phase I)
    # =================================================================
    def fit(
        self, X_phase1: np.ndarray, alpha: Optional[float] = None
    ) -> "MEWMAChart":
        """Estimate process parameters and compute UCL.

        Args:
            X_phase1: In-control data (m × p).
            alpha: False-alarm rate; defaults to ``config.alpha``.

        Returns:
            ``self`` for chaining.
        """
        alpha = alpha or self.cfg.alpha
        X = np.asarray(X_phase1, dtype=np.float64)
        self._m, self._p = X.shape

        self.mean_vector = X.mean(axis=0)
        self.cov_matrix = np.cov(X, rowvar=False)
        cond = np.linalg.cond(self.cov_matrix)
        if cond > 1e10:
            self.cov_inv = np.linalg.pinv(self.cov_matrix)
        else:
            self.cov_inv = np.linalg.inv(self.cov_matrix)

        # Asymptotic UCL (Lowry et al. 1992)
        # We use the configured mewma_L parameter for the theoretical limit.
        self.ucl_asymptotic = float(self.cfg.mewma_L ** 2)

        print(f"\n  MEWMA Setup:")
        print(f"    λ (smoothing)        : {self.lam}")
        print(f"    Dimensions (p)       : {self._p}")
        print(f"    Phase I size (m)     : {self._m}")
        print(f"    Asymptotic UCL       : {self.ucl_asymptotic:.4f}")
        return self

    # =================================================================
    # Monitoring (Phase II)
    # =================================================================
    def monitor(self, X_phase2: np.ndarray) -> Dict:
        """Run MEWMA monitoring on Phase II data.

       "Uses a constant asymptotic limit based on the Lowry (1992) mewma_L parameter."

        Args:
            X_phase2: Phase II data (n × p).

        Returns:
            Dictionary with Z values, T² statistics, UCLs, and signals.
        """
        X = np.asarray(X_phase2, dtype=np.float64)
        n, p = X.shape

        Z = np.zeros((n, p))
        t2_mewma = np.zeros(n)
        ucl_array = np.zeros(n)
        Z_prev = np.zeros(p)

        lam = self.lam

        for i in range(n):
            deviation = X[i] - self.mean_vector
            Z[i] = lam * deviation + (1 - lam) * Z_prev

            # Time-varying factor
            factor = (lam / (2 - lam)) * (1 - (1 - lam) ** (2 * (i + 1)))

            # T² = Z' Σ_{Z,i}^{-1} Z  =  (1/factor) * Z' Σ^{-1} Z
            inv_factor = 1.0 / max(factor, 1e-15)
            t2_mewma[i] = float(inv_factor * Z[i] @ self.cov_inv @ Z[i])

            # UCL is constant chi²(p) since the statistic is already
            # normalised by the time-varying factor.
            ucl_array[i] = self.ucl_asymptotic

            Z_prev = Z[i].copy()

        # All UCLs are the same constant
        ucl_array[:] = self.ucl_asymptotic

        signals = t2_mewma > ucl_array
        signal_idx = np.where(signals)[0]

        result: Dict = {
            "Z_values": Z,
            "t2_mewma": t2_mewma,
            "ucl_array": ucl_array,
            "ucl_asymptotic": self.ucl_asymptotic,
            "signals": signals,
            "signal_indices": signal_idx,
            "signal_rate": float(100 * signals.sum() / n),
        }

        print(f"  MEWMA Monitoring: {signals.sum()} signals ({result['signal_rate']:.1f}%)")
        return result

    # =================================================================
    # Plotting
    # =================================================================
    def plot_mewma_chart(
        self, results: Dict, y_true: Optional[np.ndarray] = None
    ) -> None:
        """Render the MEWMA chart with three panels.

        Args:
            results: Output from :meth:`monitor`.
            y_true: Optional true labels.
        """
        print("\n── MEWMA Chart ──")
        t2 = results["t2_mewma"]
        ucl_arr = results["ucl_array"]
        signals = results["signals"]
        Z = results["Z_values"]
        n = len(t2)
        x = np.arange(n)

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(20, 14),
            gridspec_kw={"height_ratios": [3, 2, 1]},
            sharex=True,
        )

        # Panel 1 – T² MEWMA
        ax1.plot(x, t2, color="#4C72B0", lw=0.8, alpha=0.8)
        ax1.axhline(results["ucl_asymptotic"], color="red", ls="--", lw=1.5,
                     label=f"Asymptotic UCL = {results['ucl_asymptotic']:.2f}")
        if ucl_arr[:50].max() > 0:
            ax1.plot(x[:50], ucl_arr[:50], color="gray", ls=":", lw=1,
                     label="Time-varying UCL (first 50)")

        if y_true is not None:
            y_arr = np.asarray(y_true)
            for idx in np.where(y_arr == 1)[0]:
                if idx < n:
                    ax1.axvspan(idx - 0.5, idx + 0.5, alpha=0.08, color="red")

        ax1.fill_between(
            x, 0, t2,
            where=~signals, color="#55A868", alpha=0.15,
        )
        ax1.fill_between(
            x, 0, t2,
            where=signals, color="crimson", alpha=0.25,
        )

        ax1.set_ylabel("T² MEWMA")
        ax1.set_title(f"MEWMA Control Chart (λ = {self.lam})",
                       fontweight="bold", fontsize=14)
        ax1.legend(loc="upper right")

        # Panel 2 – Z statistics for top 3 PCs
        for j in range(min(3, Z.shape[1])):
            ax2.plot(x, Z[:, j], lw=0.7, label=f"Z(PC{j+1})")
        ax2.axhline(0, color="gray", ls="-", lw=0.5)
        ax2.set_ylabel("MEWMA Z-statistic")
        ax2.set_title("MEWMA Z Statistics (Top 3 PCs)", fontweight="bold")
        ax2.legend(fontsize=8)

        # Panel 3 – signal indicator
        colours = np.where(signals, "red", "#55A868")
        ax3.bar(x, signals.astype(int), color=colours, width=1.0)
        if y_true is not None:
            y_arr = np.asarray(y_true)
            for idx in np.where(y_arr == 1)[0]:
                if idx < n:
                    ax3.axvline(idx, color="purple", lw=0.5, alpha=0.5)
        ax3.set_ylabel("Signal")
        ax3.set_xlabel("Observation index")
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(["In-ctrl", "OOC"])

        plt.tight_layout()
        path = os.path.join(self.fig_dir, "mewma_chart.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")

    # =================================================================
    def compare_sensitivity(
        self,
        X_phase2: np.ndarray,
        lambdas: Optional[List[float]] = None,
    ) -> None:
        """Run MEWMA with several λ values and compare.

        Args:
            X_phase2: Phase II data.
            lambdas: List of λ values to compare.
        """
        lambdas = lambdas or [0.05, 0.10, 0.20, 0.30]
        print("\n── MEWMA λ Comparison ──")
        if X_phase2.shape[1] != self._p:
            raise ValueError(
                f"Expected {self._p} features, got {X_phase2.shape[1]}"
            )
        fig, ax = plt.subplots(figsize=(18, 6))

        for lam in lambdas:
            chart = MEWMAChart(lam=lam, cfg=self.cfg)
            chart.mean_vector = self.mean_vector
            chart.cov_matrix = self.cov_matrix
            chart.cov_inv = self.cov_inv
            chart._p = self._p
            chart._m = self._m

            chart.ucl_asymptotic = float(self.cfg.mewma_L ** 2)

            res = chart.monitor(X_phase2)
            ax.plot(res["t2_mewma"], lw=0.7, label=f"λ={lam} (signals={res['signals'].sum()})")

        ax.axhline(self.ucl_asymptotic, color="red", ls="--", lw=1, label=f"UCL (λ={self.lam})")
        ax.set_xlabel("Observation index")
        ax.set_ylabel("T² MEWMA")
        ax.set_title("MEWMA Sensitivity: λ Comparison", fontweight="bold", fontsize=14)
        ax.legend(fontsize=8)
        plt.tight_layout()
        path = os.path.join(self.fig_dir, "mewma_lambda_comparison.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")

    # =================================================================
    # Persistence
    # =================================================================
    def save(self, filepath: str) -> None:
        """Save MEWMA chart.

        Args:
            filepath: Destination .pkl.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"  ✓ MEWMA chart saved → {filepath}")

    @staticmethod
    def load(filepath: str) -> "MEWMAChart":
        """Load a saved MEWMA chart.

        Args:
            filepath: Source .pkl.
        """
        return joblib.load(filepath)
