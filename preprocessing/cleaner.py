"""
preprocessing/cleaner.py - SECOM Data Cleaning Pipeline
=========================================================

A reproducible, serialisable cleaning pipeline that tracks every
transformation applied and every feature removed.
"""

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config

warnings.filterwarnings("ignore", category=FutureWarning)


class SECOMCleaner:
    """End-to-end preprocessing pipeline for the SECOM dataset.

    The pipeline executes six sequential steps:

    1. Remove features with >50 % missing values.
    2. Remove constant / near-zero-variance features.
    3. Impute remaining missing values (median strategy).
    4. Clip extreme outliers using the IQR rule.
    5. Remove highly correlated features (|r| > 0.95).
    6. Scale features with :class:`RobustScaler`.

    Attributes:
        cfg: Configuration object.
        dropped_missing_features: Features removed in step 1.
        dropped_constant_features: Features removed in step 2.
        dropped_correlated_features: Features removed in step 5.
        imputer: Fitted :class:`SimpleImputer`.
        scaler: Fitted :class:`RobustScaler`.
        feature_log: Per-step feature counts.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.dropped_missing_features: List[str] = []
        self.dropped_constant_features: List[str] = []
        self.dropped_correlated_features: List[str] = []
        self.retained_features: List[str] = []
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[RobustScaler] = None
        self.clip_bounds: Dict[str, Tuple[float, float]] = {}
        self.feature_log: Dict[str, int] = {}
        self.fig_dir = os.path.join(self.cfg.figures_dir, "preprocessing")
        os.makedirs(self.fig_dir, exist_ok=True)

    # =================================================================
    # Public API
    # =================================================================
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit the entire cleaning pipeline and transform the data.

        Args:
            X: Raw sensor DataFrame (with NaNs).
            y: Optional target; not used for fitting but kept for API
               compatibility.

        Returns:
            Cleaned, imputed, and scaled DataFrame.
        """
        n_orig = X.shape[1]
        self.feature_log["original"] = n_orig
        print("\n══════════════════════════════════════════════════")
        print("        PREPROCESSING PIPELINE")
        print("══════════════════════════════════════════════════")

        X = self.remove_high_missing_features(X)
        self.feature_log["after_missing"] = X.shape[1]
        print(f"  Step 1  Missing threshold  : {X.shape[1]} features "
              f"(removed {n_orig - X.shape[1]})")

        prev = X.shape[1]
        X = self.remove_constant_features(X)
        self.feature_log["after_constant"] = X.shape[1]
        print(f"  Step 2  Constant removal   : {X.shape[1]} features "
              f"(removed {prev - X.shape[1]})")

        X = self.impute_missing_values(X)
        self.feature_log["after_imputation"] = X.shape[1]
        print(f"  Step 3  Imputation         : {X.shape[1]} features (removed 0)")

        X = self.clip_outliers(X)
        self.feature_log["after_clipping"] = X.shape[1]
        print(f"  Step 4  Outlier clipping   : {X.shape[1]} features (removed 0)")

        prev = X.shape[1]
        X = self.remove_correlated_features(X)
        self.feature_log["after_correlation"] = X.shape[1]
        print(f"  Step 5  Correlation filter : {X.shape[1]} features "
              f"(removed {prev - X.shape[1]})")

        X = self.scale_features(X)
        self.feature_log["after_scaling"] = X.shape[1]
        print(f"  Step 6  Scaling            : {X.shape[1]} features (removed 0)")

        self.retained_features = X.columns.tolist()

        pct = 100 * (1 - X.shape[1] / n_orig)
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("PREPROCESSING COMPLETE")
        print(f"  Original features : {n_orig}")
        print(f"  Final features    : {X.shape[1]}")
        print(f"  Total reduction   : {pct:.1f}%")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return X

    # =================================================================
    # Step 1 – high missing
    # =================================================================
    def remove_high_missing_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop features where the fraction of NaN exceeds the threshold.

        Args:
            X: DataFrame with potential NaN values.

        Returns:
            Filtered DataFrame with high-missing columns removed.
        """
        miss_pct = X.isna().mean()
        to_drop = miss_pct[miss_pct > self.cfg.missing_threshold].index.tolist()
        self.dropped_missing_features = to_drop
        return X.drop(columns=to_drop)

    # =================================================================
    # Step 2 – constant features
    # =================================================================
    def remove_constant_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove zero-variance and near-zero-variance columns.

        Uses :class:`VarianceThreshold` with a tiny threshold to catch
        columns that have no practical variability.

        Args:
            X: DataFrame (may still contain NaN at this stage).

        Returns:
            DataFrame with constant columns removed.
        """
        # Temporarily fill NaN for variance computation
        X_filled = X.fillna(X.median())
        selector = VarianceThreshold(threshold=self.cfg.variance_threshold)
        try:
            selector.fit(X_filled)
        except ValueError:
            return X

        mask = selector.get_support()
        dropped = X.columns[~mask].tolist()
        self.dropped_constant_features = dropped
        return X.loc[:, mask]

    # =================================================================
    # Step 3 – imputation
    # =================================================================
    def impute_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute remaining NaN values with the column median.

        Args:
            X: DataFrame that may still contain NaN values.

        Returns:
            DataFrame with zero NaN values.

        Raises:
            AssertionError: If any NaN remains after imputation.
        """
        self.imputer = SimpleImputer(strategy=self.cfg.imputation_strategy)
        self._imputer_feature_names = list(X.columns)
        arr = self.imputer.fit_transform(X)
        X_out = pd.DataFrame(arr, columns=X.columns, index=X.index)
        assert X_out.isna().sum().sum() == 0, "NaN survived imputation!"
        return X_out

    # =================================================================
    # Step 4 – outlier clipping
    # =================================================================
    def clip_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip extreme values using the IQR rule.

        Values below ``Q1 - k·IQR`` or above ``Q3 + k·IQR`` are winsorised
        to those boundaries.

        Args:
            X: Imputed DataFrame (no NaN).

        Returns:
            DataFrame with clipped values.
        """
        X = X.copy()
        k = self.cfg.outlier_iqr_multiplier
        total_clipped = 0
        for col in X.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - k * iqr, q3 + k * iqr
            self.clip_bounds[col] = (lo, hi)
            clipped = ((X[col] < lo) | (X[col] > hi)).sum()
            total_clipped += clipped
            X[col] = X[col].clip(lower=lo, upper=hi)

        pct = 100 * total_clipped / X.size
        print(f"    Clipped {total_clipped:,} values ({pct:.2f}% of all data points)")
        return X

    # =================================================================
    # Step 5 – correlated features
    # =================================================================
    def remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove one feature from each highly-correlated pair.

        For every pair with |r| > correlation_threshold, the feature
        with the **lower** variance is dropped.

        Args:
            X: Imputed and clipped DataFrame.

        Returns:
            DataFrame with redundant columns removed.
        """
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        var = X.var()

        to_drop = set()
        pairs_found = 0
        for col in upper.columns:
            high = upper.index[upper[col] > self.cfg.correlation_threshold].tolist()
            for h in high:
                pairs_found += 1
                # drop the lower-variance feature
                if var[col] < var[h]:
                    to_drop.add(col)
                else:
                    to_drop.add(h)

        self.dropped_correlated_features = sorted(to_drop)
        print(f"    Highly correlated pairs found : {pairs_found}")
        print(f"    Features to drop              : {len(to_drop)}")

        # Save before/after heatmaps
        try:
            self._plot_correlation_heatmaps(X, to_drop)
        except Exception as exc:
            print(f"    ⚠ Could not save correlation heatmaps: {exc}")

        return X.drop(columns=list(to_drop))

    def _plot_correlation_heatmaps(
        self, X: pd.DataFrame, to_drop: set
    ) -> None:
        """Save side-by-side correlation heatmaps (before/after removal).

        Args:
            X: DataFrame before correlated feature removal.
            to_drop: Set of feature names being removed.
        """
        n_show = min(40, X.shape[1])
        cols_before = X.columns[:n_show]
        cols_after = [c for c in cols_before if c not in to_drop][:n_show]

        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        sns.heatmap(
            X[cols_before].corr(),
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            ax=axes[0],
            square=True,
            cbar_kws={"shrink": 0.7},
        )
        axes[0].set_title("Before Correlation Filter", fontsize=13, fontweight="bold")

        sns.heatmap(
            X[cols_after].corr(),
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            ax=axes[1],
            square=True,
            cbar_kws={"shrink": 0.7},
        )
        axes[1].set_title("After Correlation Filter", fontsize=13, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(self.fig_dir, "correlation_before_after.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"    ✓ Saved → {path}")

    # =================================================================
    # Step 6 – scaling
    # =================================================================
    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using :class:`RobustScaler`.

        After scaling the median of every column should be ≈ 0.

        Args:
            X: Imputed, clipped, and correlation-filtered DataFrame.

        Returns:
            Scaled DataFrame.
        """
        self.scaler = RobustScaler()
        arr = self.scaler.fit_transform(X)
        X_out = pd.DataFrame(arr, columns=X.columns, index=X.index)

        median_check = X_out.median().abs().max()
        print(f"    Post-scaling max|median| : {median_check:.6f} (should be ≈ 0)")
        return X_out

    # =================================================================
    # Transform (for new / Phase II data)
    # =================================================================
    def transform(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted pipeline to new data.

        Args:
            X_new: Raw DataFrame aligned to the original schema.

        Returns:
            Cleaned and scaled DataFrame.
        """
        # 1. drop missing features
        keep = [c for c in X_new.columns if c not in self.dropped_missing_features]
        X_new = X_new[keep]

        # 2. drop constant features
        keep = [c for c in X_new.columns if c not in self.dropped_constant_features]
        X_new = X_new[keep]

        # 3. impute
        if self.imputer is not None:
            # Align columns to exactly what imputer expects, padding
            # any missing features with NaN so transform() gets the
            # correct number of columns.
            expected = self._imputer_feature_names if hasattr(self, "_imputer_feature_names") else list(X_new.columns)
            X_aligned = pd.DataFrame(
                index=X_new.index, columns=expected, dtype=np.float64
            )
            for c in expected:
                if c in X_new.columns:
                    X_aligned[c] = X_new[c].values
            arr = self.imputer.transform(X_aligned)
            X_new = pd.DataFrame(arr, columns=expected, index=X_new.index)

        # 4. clip
        for col in X_new.columns:
            if col in self.clip_bounds:
                lo, hi = self.clip_bounds[col]
                X_new[col] = X_new[col].clip(lower=lo, upper=hi)

        # 5. drop correlated features
        keep = [c for c in X_new.columns if c not in self.dropped_correlated_features]
        X_new = X_new[keep]

        # 6. scale
        if self.scaler is not None:
            arr = self.scaler.transform(X_new)
            X_new = pd.DataFrame(arr, columns=X_new.columns, index=X_new.index)

        return X_new

    # =================================================================
    # Reports & persistence
    # =================================================================
    def get_preprocessing_report(self) -> Dict:
        """Return a full dictionary of preprocessing decisions.

        Returns:
            Dictionary with dropped features, imputer params, scaler
            params, and per-step feature counts.
        """
        return {
            "dropped_missing": self.dropped_missing_features,
            "dropped_constant": self.dropped_constant_features,
            "dropped_correlated": self.dropped_correlated_features,
            "retained_features": self.retained_features,
            "imputer_strategy": self.cfg.imputation_strategy,
            "scaler_type": self.cfg.scaler_type,
            "feature_log": self.feature_log,
        }

    def save(self, filepath: str) -> None:
        """Persist the fitted cleaner with joblib.

        Args:
            filepath: Destination path (e.g., ``outputs/models/cleaner.pkl``).
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"  ✓ Cleaner saved → {filepath}")

    @staticmethod
    def load(filepath: str) -> "SECOMCleaner":
        """Load a previously saved cleaner.

        Args:
            filepath: Path to the ``.pkl`` file.

        Returns:
            Restored :class:`SECOMCleaner` instance.
        """
        return joblib.load(filepath)
