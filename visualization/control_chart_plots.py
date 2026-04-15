"""
visualization/control_chart_plots.py - SPC Chart Visualisations
==================================================================

Reusable plotting functions for T², MEWMA, and combined control
charts with consistent styling.
"""

import os
import sys
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


class ControlChartPlotter:
    """High-level helpers for rendering SPC chart figures.

    Args:
        cfg: Optional configuration override.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.fig_dir = os.path.join(self.cfg.figures_dir, "mspc")
        os.makedirs(self.fig_dir, exist_ok=True)

    def plot_dual_chart(
        self,
        t2_values: np.ndarray,
        t2_ucl: float,
        mewma_values: np.ndarray,
        mewma_ucl: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        filename: str = "dual_control_chart.png",
    ) -> None:
        """Side-by-side T² and MEWMA charts.

        Args:
            t2_values: T² statistics.
            t2_ucl: T² upper control limit.
            mewma_values: MEWMA T² statistics.
            mewma_ucl: MEWMA UCL array (time-varying).
            y_true: Optional ground-truth labels.
            filename: Output filename.
        """
        n = len(t2_values)
        x = np.arange(n)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

        # T²
        ax1.plot(x, t2_values, color="#4C72B0", lw=0.7)
        ax1.axhline(t2_ucl, color="red", ls="--", lw=1.5, label=f"UCL={t2_ucl:.1f}")
        if y_true is not None:
            for idx in np.where(np.asarray(y_true) == 1)[0]:
                if idx < n:
                    ax1.axvspan(idx - 0.5, idx + 0.5, alpha=0.06, color="red")
        ax1.set_ylabel("Hotelling T²")
        ax1.set_title("T² Control Chart", fontweight="bold")
        ax1.legend(fontsize=9)

        # MEWMA
        ax2.plot(x, mewma_values, color="#DD8452", lw=0.7)
        if isinstance(mewma_ucl, np.ndarray) and len(mewma_ucl) == n:
            ax2.plot(x, mewma_ucl, color="red", ls="--", lw=1, label="UCL (time-varying)")
        else:
            ax2.axhline(float(mewma_ucl), color="red", ls="--", lw=1.5, label="UCL")
        if y_true is not None:
            for idx in np.where(np.asarray(y_true) == 1)[0]:
                if idx < n:
                    ax2.axvspan(idx - 0.5, idx + 0.5, alpha=0.06, color="red")
        ax2.set_ylabel("MEWMA T²")
        ax2.set_xlabel("Observation index")
        ax2.set_title("MEWMA Control Chart", fontweight="bold")
        ax2.legend(fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.fig_dir, filename)
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")

    def plot_signal_timeline(
        self,
        t2_signals: np.ndarray,
        mewma_signals: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        filename: str = "signal_timeline.png",
    ) -> None:
        """Visualise when each chart fires a signal vs actual defects.

        Args:
            t2_signals: Boolean array of T² signals.
            mewma_signals: Boolean array of MEWMA signals.
            y_true: Optional true labels.
            filename: Output filename.
        """
        n = len(t2_signals)
        x = np.arange(n)

        fig, ax = plt.subplots(figsize=(18, 4))

        ax.scatter(x[t2_signals], np.ones(t2_signals.sum()) * 2,
                   marker="|", s=100, color="steelblue", label="T² signal")
        ax.scatter(x[mewma_signals], np.ones(mewma_signals.sum()) * 1,
                   marker="|", s=100, color="darkorange", label="MEWMA signal")

        if y_true is not None:
            fail_idx = np.where(np.asarray(y_true) == 1)[0]
            ax.scatter(fail_idx, np.zeros(len(fail_idx)),
                       marker="*", s=80, color="crimson", label="Actual defect")

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Defect", "MEWMA", "T²"])
        ax.set_xlabel("Observation index")
        ax.set_title("Signal Timeline", fontweight="bold")
        ax.legend(loc="upper right")
        plt.tight_layout()

        path = os.path.join(self.fig_dir, filename)
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")
