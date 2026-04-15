"""
statistical_analysis/eda.py - Exploratory Data Analysis
=========================================================

Generates descriptive statistics, feature distributions, temporal
quality plots, correlation heatmaps, and pair plots.
"""

import os
import sys
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


class ExploratoryDataAnalysis:
    """Full EDA suite for the SECOM dataset.

    All plots are saved to ``outputs/figures/eda/``.

    Args:
        X: Processed sensor DataFrame.
        y: Binary target Series.
        timestamps: Datetime Series for temporal analysis.
        cfg: Optional configuration override.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        timestamps: pd.Series,
        cfg=None,
    ):
        self.X = X
        self.y = y
        self.timestamps = timestamps
        self.cfg = cfg or config
        self.fig_dir = os.path.join(self.cfg.figures_dir, "eda")
        os.makedirs(self.fig_dir, exist_ok=True)
        self._summary_text = ""

    # ─────────────────────────────────────────────────────────────────
    def run_full_eda(self) -> None:
        """Execute every EDA step in sequence."""
        print("\n══════════════════════════════════════════════════")
        print("        EXPLORATORY DATA ANALYSIS")
        print("══════════════════════════════════════════════════")

        self.descriptive_statistics_report(self.X, self.y)
        self.plot_feature_distributions(self.X, self.y, n_features=20)
        self.plot_time_series_quality(self.X, self.y, self.timestamps)
        self.plot_correlation_heatmap(self.X)
        self.plot_pairplot_top_features(self.X, self.y, n=6)
        print("\n  ✓ EDA complete – all figures saved.")

    # ─────────────────────────────────────────────────────────────────
    def descriptive_statistics_report(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Compute comprehensive descriptive statistics per feature.

        Args:
            X: Processed sensor DataFrame.
            y: Binary target.

        Returns:
            DataFrame with one row per feature.
        """
        print("\n── Descriptive Statistics ──")

        desc = X.describe().T
        desc.columns = ["count", "mean", "std", "min", "q25", "median", "q75", "max"]
        desc["skewness"] = X.skew()
        desc["kurtosis"] = X.kurtosis()
        desc["cv"] = (X.std() / (X.mean().abs() + 1e-12)).abs()
        desc["range"] = desc["max"] - desc["min"]
        desc["iqr"] = desc["q75"] - desc["q25"]

        # Class-conditional stats
        pass_X = X.loc[y == 0]
        fail_X = X.loc[y == 1]
        desc["mean_pass"] = pass_X.mean()
        desc["mean_fail"] = fail_X.mean()
        desc["std_pass"] = pass_X.std()
        desc["std_fail"] = fail_X.std()

        os.makedirs(self.cfg.tables_dir, exist_ok=True)
        path = os.path.join(self.cfg.tables_dir, "descriptive_stats.csv")
        desc.to_csv(path)
        print(f"  Saved descriptive stats ({desc.shape}) → {path}")
        print(desc.head(10).to_string())
        return desc

    # ─────────────────────────────────────────────────────────────────
    def plot_feature_distributions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 20,
    ) -> None:
        """KDE plots of top-N most variable features split by class.

        Args:
            X: Processed sensor DataFrame.
            y: Binary target.
            n_features: How many features to plot.
        """
        print("\n── Feature Distributions (top 20) ──")
        top_feats = X.var().nlargest(n_features).index.tolist()

        rows, cols = 4, 5
        fig, axes = plt.subplots(rows, cols, figsize=(24, 18))
        axes = axes.ravel()

        for i, feat in enumerate(top_feats):
            ax = axes[i]
            pass_vals = X.loc[y == 0, feat].dropna()
            fail_vals = X.loc[y == 1, feat].dropna()

            pass_vals.plot.kde(ax=ax, color="steelblue", alpha=0.6, label="Pass")
            fail_vals.plot.kde(ax=ax, color="crimson", alpha=0.7, label="Fail")

            ax.axvline(pass_vals.mean(), color="steelblue", ls="--", lw=0.8)
            ax.axvline(fail_vals.mean(), color="crimson", ls="--", lw=0.8)

            # KS test
            ks_stat, ks_p = stats.ks_2samp(pass_vals, fail_vals)
            ax.set_title(f"{feat}\nKS p={ks_p:.2e}", fontsize=9)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

        for j in range(len(top_feats), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Feature Distributions – Pass vs Fail", fontsize=16, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(self.fig_dir, "feature_distributions_top20.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")

    # ─────────────────────────────────────────────────────────────────
    def plot_time_series_quality(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        timestamps: pd.Series,
    ) -> None:
        """Four-panel temporal quality analysis.

        Args:
            X: Processed sensor DataFrame.
            y: Binary target.
            timestamps: Datetime Series.
        """
        print("\n── Time-Series Quality ──")
        ts = pd.to_datetime(timestamps)

        fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=False)

        # Panel 1 – cumulative defects
        cum = y.cumsum()
        axes[0].plot(ts, cum, color="crimson", lw=1.5)
        axes[0].set_ylabel("Cumulative defects")
        axes[0].set_title("Cumulative Defect Count", fontweight="bold")
        axes[0].fill_between(ts, cum, alpha=0.15, color="crimson")

        # Panel 2 – rolling defect rate (7-day window)
        df_tmp = pd.DataFrame({"ts": ts, "y": y.values}).set_index("ts").sort_index()
        roll = df_tmp["y"].rolling("7D").mean() * 100
        axes[1].plot(roll.index, roll.values, color="darkorange", lw=1.5)
        axes[1].set_ylabel("Defect rate (%)")
        axes[1].set_title("Rolling 7-Day Defect Rate", fontweight="bold")
        axes[1].axhline(y.mean() * 100, ls="--", color="gray", lw=1, label="Overall rate")
        axes[1].legend()

        # Panel 3 – production volume per day
        daily = df_tmp.resample("D").count()
        axes[2].bar(daily.index, daily["y"], color="steelblue", width=0.8)
        axes[2].set_ylabel("Wafers/day")
        axes[2].set_title("Daily Production Volume", fontweight="bold")

        # Panel 4 – defect rate by shift
        hours = ts.dt.hour
        shifts = pd.cut(hours, bins=[-1, 8, 16, 24], labels=["Night", "Morning", "Evening"])
        shift_rate = y.groupby(shifts).mean() * 100
        axes[3].bar(shift_rate.index.astype(str), shift_rate.values,
                     color=["#2c3e50", "#e67e22", "#8e44ad"])
        axes[3].set_ylabel("Defect rate (%)")
        axes[3].set_title("Defect Rate by Shift", fontweight="bold")

        plt.tight_layout()
        path = os.path.join(self.fig_dir, "time_series_quality.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")

        # Trend insight
        first_half = y.iloc[: len(y) // 2].mean()
        second_half = y.iloc[len(y) // 2 :].mean()
        trend = "INCREASING" if second_half > first_half else "DECREASING"
        print(f"  Insight: defect rate is {trend} over time "
              f"({100*first_half:.1f}% → {100*second_half:.1f}%)")

    # ─────────────────────────────────────────────────────────────────
    def plot_correlation_heatmap(self, X: pd.DataFrame) -> None:
        """Clustered correlation heatmap of processed features.

        Args:
            X: Processed sensor DataFrame.
        """
        print("\n── Correlation Heatmap ──")
        n = min(50, X.shape[1])
        cols = X.var().nlargest(n).index.tolist()
        corr = X[cols].corr()

        g = sns.clustermap(
            corr,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            figsize=(14, 14),
            dendrogram_ratio=0.12,
            annot=False,
            linewidths=0,
        )
        g.fig.suptitle("Clustered Correlation Heatmap (top 50)", fontsize=14, fontweight="bold", y=1.01)
        path = os.path.join(self.fig_dir, "correlation_heatmap_clustered.png")
        g.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close("all")
        print(f"  ✓ Saved → {path}")

    # ─────────────────────────────────────────────────────────────────
    def plot_pairplot_top_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n: int = 6,
    ) -> None:
        """Pair plot of top-N features by class separability.

        Args:
            X: Processed sensor DataFrame.
            y: Binary target.
            n: Number of features to include.
        """
        print("\n── Pair Plot (top 6) ──")
        # Use simple t-test for quick ranking
        scores = {}
        for col in X.columns:
            try:
                t, p = stats.ttest_ind(
                    X.loc[y == 0, col].dropna(),
                    X.loc[y == 1, col].dropna(),
                    equal_var=False,
                )
                scores[col] = abs(t)
            except Exception:
                scores[col] = 0
        top = sorted(scores, key=scores.get, reverse=True)[:n]

        tmp = X[top].copy()
        tmp["Label"] = y.map({0: "Pass", 1: "Fail"}).values

        g = sns.pairplot(
            tmp,
            hue="Label",
            palette={"Pass": "steelblue", "Fail": "crimson"},
            diag_kind="kde",
            plot_kws={"alpha": 0.4, "s": 15},
            corner=True,
        )
        g.fig.suptitle("Pair Plot – Top 6 Discriminative Features", fontsize=14, fontweight="bold", y=1.02)
        path = os.path.join(self.fig_dir, "pairplot_top_features.png")
        g.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close("all")
        print(f"  ✓ Saved → {path}")

    # ─────────────────────────────────────────────────────────────────
    def generate_eda_summary(self) -> str:
        """Return a formatted text summary of key EDA findings.

        Returns:
            Multi-line string suitable for the final report.
        """
        n_features = self.X.shape[1]
        defect_rate = self.y.mean() * 100
        text = (
            f"EDA Summary\n"
            f"-----------\n"
            f"Features analysed   : {n_features}\n"
            f"Overall defect rate : {defect_rate:.1f}%\n"
            f"Plots generated in  : {self.fig_dir}\n"
        )
        return text
