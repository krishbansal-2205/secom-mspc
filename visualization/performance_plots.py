"""
visualization/performance_plots.py - Model Performance Plots
===============================================================

Stand-alone helpers for model-comparison bar charts, calibration
curves, and learning curves.
"""

import os
import sys
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


class PerformancePlotter:
    """Model performance comparison plots.

    Args:
        cfg: Optional configuration override.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.fig_dir = os.path.join(self.cfg.figures_dir, "models")
        os.makedirs(self.fig_dir, exist_ok=True)

    def plot_metric_comparison(
        self,
        eval_df: pd.DataFrame,
        filename: str = "metric_comparison.png",
    ) -> None:
        """Grouped bar chart comparing key metrics across models.

        Args:
            eval_df: DataFrame from :class:`SECOMModelEvaluator`.
            filename: Output filename.
        """
        metrics = ["auc_roc", "recall", "precision", "f1"]
        available = [m for m in metrics if m in eval_df.columns]
        if not available:
            print("  ⚠ No metrics available for comparison plot.")
            return

        fig, ax = plt.subplots(figsize=(12, 7))
        n_models = len(eval_df)
        n_metrics = len(available)
        width = 0.8 / n_metrics
        x = np.arange(n_models)

        colours = plt.cm.Set2(np.linspace(0, 1, n_metrics))
        for i, metric in enumerate(available):
            ax.bar(x + i * width, eval_df[metric].values, width=width,
                   label=metric.replace("_", " ").title(), color=colours[i],
                   edgecolor="black", lw=0.5)

        ax.set_xticks(x + width * (n_metrics - 1) / 2)
        ax.set_xticklabels(eval_df["model"].values, rotation=15, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=14)
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()

        path = os.path.join(self.fig_dir, filename)
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")
