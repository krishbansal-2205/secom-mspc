"""
preprocessing/quality_checker.py - Data Quality Assessment
============================================================

Performs a comprehensive quality audit of the raw SECOM data,
covering missing values, constant features, outliers, distributions,
and class separability.  All results are saved to disk.
"""

import json
import os
import sys
import warnings
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config

warnings.filterwarnings("ignore", category=RuntimeWarning)


class DataQualityChecker:
    """Run a full quality assessment on the SECOM dataset.

    Attributes:
        cfg: Global configuration object.
        fig_dir: Directory to save quality-check plots.
    """

    def __init__(self, cfg=None):
        """Initialise with optional alternate configuration.

        Args:
            cfg: Configuration object; defaults to global singleton.
        """
        self.cfg = cfg or config
        self.fig_dir = os.path.join(self.cfg.figures_dir, "quality")
        os.makedirs(self.fig_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # Master method
    # ─────────────────────────────────────────────────────────────────
    def run_full_assessment(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict:
        """Execute every quality check and compile a full report.

        Args:
            X: Raw sensor DataFrame (with NaNs).
            y: Binary target Series (0/1).

        Returns:
            Dictionary with keys ``missing``, ``constant``, ``outliers``,
            ``distributions``, ``separability``.
        """
        print("\n══════════════════════════════════════════════════")
        print("        DATA QUALITY ASSESSMENT")
        print("══════════════════════════════════════════════════")

        missing_df = self.assess_missing_values(X, y)
        constant_info = self.assess_constant_features(X)
        outlier_df = self.assess_outliers(X)
        dist_df = self.assess_distributions(X)
        sep_df = self.assess_class_separability(X, y)

        report: Dict = {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "missing_summary": {
                "total_missing": int(X.isna().sum().sum()),
                "pct_missing": round(float(X.isna().sum().sum() / X.size * 100), 2),
            },
            "constant_features": constant_info,
            "top_outlier_features": outlier_df.head(10)["feature"].tolist()
            if "feature" in outlier_df.columns
            else [],
            "n_normal_features": int(dist_df["is_normal"].sum())
            if "is_normal" in dist_df.columns
            else 0,
            "top_discriminative_features": sep_df.head(20)["feature"].tolist()
            if "feature" in sep_df.columns
            else [],
        }

        # Save report JSON
        os.makedirs(self.cfg.reports_dir, exist_ok=True)
        report_path = os.path.join(self.cfg.reports_dir, "quality_report.json")
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\n  ✓ Quality report saved → {report_path}")

        return report

    # ─────────────────────────────────────────────────────────────────
    # 1 · Missing values
    # ─────────────────────────────────────────────────────────────────
    def assess_missing_values(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Analyse per-feature missing-value statistics.

        Args:
            X: Raw sensor DataFrame.
            y: Optional target vector for class-specific missing analysis.

        Returns:
            DataFrame with one row per feature and columns for counts,
            percentages, and category labels.
        """
        print("\n── 1. Missing-Value Analysis ──")

        miss_count = X.isna().sum()
        miss_pct = X.isna().mean() * 100

        def _cat(p: float) -> str:
            if p == 0:
                return "None"
            if p < 5:
                return "Low"
            if p < 20:
                return "Moderate"
            if p < 50:
                return "High"
            return "Critical"

        records = []
        for col in X.columns:
            rec = {
                "feature": col,
                "missing_count": int(miss_count[col]),
                "missing_pct": round(float(miss_pct[col]), 2),
                "missing_category": _cat(float(miss_pct[col])),
            }
            if y is not None:
                pass_miss = float(X.loc[y == 0, col].isna().mean() * 100)
                fail_miss = float(X.loc[y == 1, col].isna().mean() * 100)
                rec["missing_in_pass_pct"] = round(pass_miss, 2)
                rec["missing_in_fail_pct"] = round(fail_miss, 2)
                rec["missing_difference"] = round(fail_miss - pass_miss, 2)
            records.append(rec)

        df = pd.DataFrame(records)

        # Summary table
        cat_counts = df["missing_category"].value_counts()
        for cat in ["None", "Low", "Moderate", "High", "Critical"]:
            cnt = cat_counts.get(cat, 0)
            print(f"    {cat:10s}: {cnt:>4d} features")

        # --- plots -------------------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Missing-Value Analysis", fontsize=16, fontweight="bold")

        # A – histogram of missing %
        axes[0, 0].hist(df["missing_pct"], bins=50, edgecolor="black", color="#4C72B0")
        axes[0, 0].set_xlabel("Missing %")
        axes[0, 0].set_ylabel("Feature count")
        axes[0, 0].set_title("A – Distribution of Missing %")

        # B – top 30 most missing
        top30 = df.nlargest(30, "missing_pct")
        axes[0, 1].barh(top30["feature"], top30["missing_pct"], color="#DD8452")
        axes[0, 1].set_xlabel("Missing %")
        axes[0, 1].set_title("B – Top 30 Most-Missing Features")
        axes[0, 1].invert_yaxis()

        # C – scatter Pass vs Fail
        if y is not None and "missing_in_pass_pct" in df.columns:
            axes[1, 0].scatter(
                df["missing_in_pass_pct"],
                df["missing_in_fail_pct"],
                alpha=0.5,
                s=15,
                c="#55A868",
            )
            lim = max(
                df["missing_in_pass_pct"].max(), df["missing_in_fail_pct"].max(), 1
            )
            axes[1, 0].plot([0, lim], [0, lim], "r--", lw=1)
            axes[1, 0].set_xlabel("Missing % (Pass)")
            axes[1, 0].set_ylabel("Missing % (Fail)")
            axes[1, 0].set_title("C – Missing in Pass vs Fail")
        else:
            axes[1, 0].text(0.5, 0.5, "No labels provided", ha="center", va="center")

        # D – heatmap (sample)
        n_samp = min(100, X.shape[0])
        n_feat = min(50, X.shape[1])
        sample_idx = np.random.RandomState(42).choice(X.shape[0], n_samp, replace=False)
        sample_cols = X.columns[:n_feat]
        sns.heatmap(
            X.iloc[sample_idx][sample_cols].isna().astype(int),
            cbar=False,
            cmap="YlOrRd",
            ax=axes[1, 1],
        )
        axes[1, 1].set_title(f"D – Missing Heatmap ({n_samp}×{n_feat})")
        axes[1, 1].set_xlabel("Features")
        axes[1, 1].set_ylabel("Samples")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(self.fig_dir, "missing_value_analysis.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"    ✓ Saved → {path}")
        return df

    # ─────────────────────────────────────────────────────────────────
    # 2 · Constant / near-zero-variance
    # ─────────────────────────────────────────────────────────────────
    def assess_constant_features(self, X: pd.DataFrame) -> Dict:
        """Identify zero-variance and near-zero-variance features.

        Args:
            X: Raw sensor DataFrame.

        Returns:
            Dictionary with ``zero_var``, ``near_zero_var``, and
            ``single_unique`` feature lists.
        """
        print("\n── 2. Constant / Near-Zero Variance Features ──")

        stds = X.std(skipna=True)
        means = X.mean(skipna=True).replace(0, np.nan)
        cv = (stds / means).abs()

        zero_var = stds[stds == 0].index.tolist()
        near_zero = cv[cv < 0.01].index.tolist()
        near_zero = [f for f in near_zero if f not in zero_var]
        single_unique = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]

        info = {
            "zero_var": zero_var,
            "near_zero_var": near_zero,
            "single_unique": single_unique,
        }
        print(f"    Zero-variance       : {len(zero_var)}")
        print(f"    Near-zero-variance  : {len(near_zero)}")
        print(f"    Single-unique-value : {len(single_unique)}")

        return info

    # ─────────────────────────────────────────────────────────────────
    # 3 · Outliers
    # ─────────────────────────────────────────────────────────────────
    def assess_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute per-feature outlier statistics.

        Args:
            X: Raw sensor DataFrame.

        Returns:
            DataFrame sorted by ``n_outliers_iqr`` descending.
        """
        print("\n── 3. Outlier Assessment ──")

        records = []
        for col in X.columns:
            vals = X[col].dropna()
            if len(vals) == 0:
                continue
            q1 = vals.quantile(0.25)
            q3 = vals.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 3.0 * iqr, q3 + 3.0 * iqr
            n_iqr = int(((vals < lower) | (vals > upper)).sum())

            z = (vals - vals.mean()) / (vals.std() + 1e-12)
            n_z = int((z.abs() > 4).sum())
            max_z = float(z.abs().max())

            records.append(
                {
                    "feature": col,
                    "n_outliers_iqr": n_iqr,
                    "n_outliers_zscore": n_z,
                    "n_outliers_pct": round(100 * n_iqr / len(vals), 2),
                    "max_z_score": round(max_z, 2),
                }
            )

        df = pd.DataFrame(records).sort_values("n_outliers_iqr", ascending=False)
        df.reset_index(drop=True, inplace=True)

        print("    Top 10 most outlier-prone features:")
        for _, row in df.head(10).iterrows():
            print(
                f"      {row['feature']}  IQR-outliers={row['n_outliers_iqr']}"
                f"  max|z|={row['max_z_score']}"
            )

        return df

    # ─────────────────────────────────────────────────────────────────
    # 4 · Distributions
    # ─────────────────────────────────────────────────────────────────
    def assess_distributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Test normality and classify the shape of each feature.

        Uses the Shapiro-Wilk test on a random sample of 200 observations
        per feature for computational speed.

        Args:
            X: Raw sensor DataFrame.

        Returns:
            DataFrame of normality results per feature.
        """
        print("\n── 4. Distribution Analysis ──")

        rng = np.random.RandomState(42)
        records = []

        for col in X.columns:
            vals = X[col].dropna().values
            if len(vals) < 8:
                continue

            sample = rng.choice(vals, size=min(200, len(vals)), replace=False)

            try:
                stat_sw, p_sw = stats.shapiro(sample)
            except Exception:
                stat_sw, p_sw = np.nan, np.nan

            sk = float(stats.skew(sample, nan_policy="omit"))
            ku = float(stats.kurtosis(sample, nan_policy="omit"))

            is_normal = bool(p_sw > 0.05) if not np.isnan(p_sw) else False

            if abs(sk) < 0.5 and abs(ku) < 1.0 and is_normal:
                shape = "normal"
            elif sk > 1.0:
                shape = "right_skewed"
            elif sk < -1.0:
                shape = "left_skewed"
            elif ku > 3.0:
                shape = "heavy_tailed"
            else:
                shape = "bimodal_suspect"

            records.append(
                {
                    "feature": col,
                    "shapiro_stat": round(float(stat_sw), 4) if not np.isnan(stat_sw) else None,
                    "p_value": round(float(p_sw), 6) if not np.isnan(p_sw) else None,
                    "skewness": round(sk, 3),
                    "kurtosis": round(ku, 3),
                    "is_normal": is_normal,
                    "shape": shape,
                }
            )

        df = pd.DataFrame(records)
        n_normal = int(df["is_normal"].sum())
        n_total = len(df)
        print(
            f"    Normal     : {n_normal:>4d} ({100 * n_normal / max(n_total, 1):.1f}%)"
        )
        print(
            f"    Non-normal : {n_total - n_normal:>4d} ({100 * (n_total - n_normal) / max(n_total, 1):.1f}%)"
        )

        # distribution summary bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        shape_counts = df["shape"].value_counts()
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
        shape_counts.plot.bar(ax=ax, color=colors[: len(shape_counts)], edgecolor="black")
        ax.set_title("Distribution Shape Summary", fontsize=14, fontweight="bold")
        ax.set_ylabel("Feature count")
        ax.set_xlabel("Shape category")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = os.path.join(self.fig_dir, "distribution_analysis.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"    ✓ Saved → {path}")

        return df

    # ─────────────────────────────────────────────────────────────────
    # 5 · Class separability
    # ─────────────────────────────────────────────────────────────────
    def assess_class_separability(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Identify features that discriminate Pass from Fail.

        Computes independent-samples *t*-test, Mann-Whitney U,
        Cohen's *d*, and point-biserial correlation for every feature.

        Args:
            X: Raw sensor DataFrame.
            y: Binary label Series (0/1).

        Returns:
            DataFrame sorted by |Cohen's d| descending.
        """
        print("\n── 5. Class Separability Analysis ──")

        records = []
        for col in X.columns:
            pass_vals = X.loc[y == 0, col].dropna().values
            fail_vals = X.loc[y == 1, col].dropna().values

            if len(pass_vals) < 5 or len(fail_vals) < 5:
                continue

            # t-test
            t_stat, t_p = stats.ttest_ind(pass_vals, fail_vals, equal_var=False)

            # Mann-Whitney
            try:
                u_stat, u_p = stats.mannwhitneyu(
                    pass_vals, fail_vals, alternative="two-sided"
                )
            except ValueError:
                u_stat, u_p = np.nan, np.nan

            # Cohen's d
            pooled_std = np.sqrt(
                (pass_vals.std() ** 2 + fail_vals.std() ** 2) / 2
            )
            cohens_d = (
                (fail_vals.mean() - pass_vals.mean()) / (pooled_std + 1e-12)
            )

            # Point-biserial
            combined = np.concatenate([pass_vals, fail_vals])
            labels = np.concatenate(
                [np.zeros(len(pass_vals)), np.ones(len(fail_vals))]
            )
            try:
                pb_r, pb_p = stats.pointbiserialr(labels, combined)
            except Exception:
                pb_r, pb_p = np.nan, np.nan

            records.append(
                {
                    "feature": col,
                    "t_stat": round(float(t_stat), 4),
                    "t_pvalue": float(t_p),
                    "u_stat": round(float(u_stat), 1) if not np.isnan(u_stat) else None,
                    "u_pvalue": float(u_p) if not np.isnan(u_p) else None,
                    "cohens_d": round(float(cohens_d), 4),
                    "abs_cohens_d": round(abs(float(cohens_d)), 4),
                    "pb_correlation": round(float(pb_r), 4) if not np.isnan(pb_r) else None,
                }
            )

        df = pd.DataFrame(records).sort_values("abs_cohens_d", ascending=False)
        df.reset_index(drop=True, inplace=True)

        print("    Top 20 most discriminative features:")
        for _, row in df.head(20).iterrows():
            print(
                f"      {row['feature']}  |d|={row['abs_cohens_d']:.3f}  "
                f"t-p={row['t_pvalue']:.2e}"
            )

        # Volcano plot
        fig, ax = plt.subplots(figsize=(12, 7))
        neg_log_p = -np.log10(df["t_pvalue"].clip(lower=1e-300))
        colours = np.where(
            (df["abs_cohens_d"] > 0.3) & (neg_log_p > 2), "#C44E52", "#4C72B0"
        )
        ax.scatter(df["cohens_d"], neg_log_p, c=colours, alpha=0.6, s=20)
        ax.axhline(-np.log10(0.05), ls="--", color="gray", lw=1, label="p = 0.05")
        ax.axvline(-0.3, ls=":", color="orange", lw=1)
        ax.axvline(0.3, ls=":", color="orange", lw=1, label="|d| = 0.3")
        ax.set_xlabel("Cohen's d", fontsize=12)
        ax.set_ylabel("-log₁₀(p-value)", fontsize=12)
        ax.set_title("Class Separability Volcano Plot", fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(self.fig_dir, "class_separability.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"    ✓ Saved → {path}")

        return df
