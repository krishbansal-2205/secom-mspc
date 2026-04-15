"""
dimensionality_reduction/pca_engine.py - PCA Implementation
==============================================================

Fits PCA on the scaled SECOM data, provides scree/score/loading
visualisations, and computes Hotelling T² and SPE statistics
directly from the PCA model.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


class SECOMPCAEngine:
    """PCA wrapper tailored for the SECOM MSPC workflow.

    Attributes:
        pca: Underlying scikit-learn :class:`PCA` object.
        n_components: Number of retained components.
        explained_variance_ratio: Array of individual ratios.
        cumulative_variance: Array of cumulative sums.
        loadings: Loading matrix (n_features × n_components).
        scores: Score matrix from the training set.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.pca: Optional[PCA] = None
        self.pca_full: Optional[PCA] = None
        self.n_components: int = 0
        self.explained_variance_ratio: Optional[np.ndarray] = None
        self.cumulative_variance: Optional[np.ndarray] = None
        self.loadings: Optional[np.ndarray] = None
        self.scores: Optional[np.ndarray] = None
        self._n_features_in: int = 0
        self._y_train: Optional[np.ndarray] = None
        self.fig_dir = os.path.join(self.cfg.figures_dir, "pca")
        os.makedirs(self.fig_dir, exist_ok=True)

    # =================================================================
    # Fitting
    # =================================================================
    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> "SECOMPCAEngine":
        """Fit PCA on already-scaled training data.

        Steps:
            1. Optional sanity check that data is scaled.
            2. Full PCA to get the explained-variance curve.
            3. Select the number of components by cumulative variance.
            4. Refit PCA with the chosen number.
            5. Store loadings and scores.

        Args:
            X_train: Scaled feature DataFrame.
            y_train: Optional labels (stored for visualisations only).

        Returns:
            ``self`` for chaining.
        """
        print("\n══════════════════════════════════════════════════")
        print("        PCA DIMENSIONALITY REDUCTION")
        print("══════════════════════════════════════════════════")

        X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        self._n_features_in = X.shape[1]
        if y_train is not None:
            self._y_train = np.asarray(y_train)

        # 1 – sanity
        col_means = np.nanmean(X, axis=0)
        if np.abs(col_means).max() > 1.0:
            print("  ⚠ Data may not be centred; max|mean| = "
                  f"{np.abs(col_means).max():.3f}")

        # 2 – full PCA
        # Prevent rank deficiency when n_samples == n_features
        n_max = min(X.shape[0], X.shape[1]) - 1
        self.pca_full = PCA(n_components=n_max, random_state=self.cfg.random_seed)
        self.pca_full.fit(X)
        self.explained_variance_ratio = self.pca_full.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.explained_variance_ratio)

        # 3 – variance targets
        for target in self.cfg.variance_targets:
            n = int(np.searchsorted(self.cumulative_variance, target) + 1)
            reduction = 100 * (1 - n / X.shape[1])
            print(f"  {int(target*100)}% variance → {n} components ({reduction:.0f}% reduction)")

        self.n_components = int(
            np.argmax(self.cumulative_variance >= self.cfg.selected_variance) + 1
        )

        # 4 – refit with optimal n
        self.pca = PCA(n_components=self.n_components, random_state=self.cfg.random_seed)
        self.scores = self.pca.fit_transform(X)
        self.loadings = self.pca.components_.T

        print(f"\n  ▸ Selected : {self.n_components} components "
              f"({self.cumulative_variance[self.n_components-1]*100:.1f}% "
              f"variance retained)")
        print(f"  ▸ Shape    : {X.shape[1]}D → {self.n_components}D")

        return self

    # =================================================================
    # Transform
    # =================================================================
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data into the PCA subspace.

        Args:
            X: Array or DataFrame with the same number of features as
               the training set.

        Returns:
            Score matrix of shape ``(n_samples, n_components)``.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if X.shape[1] != self._n_features_in:
            raise ValueError(
                f"Expected {self._n_features_in} features, got {X.shape[1]}"
            )
        return self.pca.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform in one call.

        Args:
            X: Scaled feature DataFrame.
            y: Optional labels.

        Returns:
            Score matrix.
        """
        self.fit(X, y)
        return self.scores

    # =================================================================
    # Visualisations
    # =================================================================
    def plot_scree_plot(self) -> None:
        """Generate 2×2 figure: scree, cumulative, score PC1-2, score PC1-3."""
        print("\n── PCA Scree & Score Plots ──")

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        n_show = min(self.cfg.max_components_show, len(self.explained_variance_ratio))
        x_range = np.arange(1, n_show + 1)
        evr = self.explained_variance_ratio[:n_show] * 100

        # --- top-left: scree ---
        ax = axes[0, 0]
        ax.bar(x_range, evr, color="#4C72B0", edgecolor="black", alpha=0.7, label="Individual")
        ax.plot(x_range, evr, "o-", color="crimson", ms=3, label="Line")
        ax.axvline(self.n_components, ls="--", color="red", lw=1.5, label=f"n={self.n_components}")
        ax.set_xlabel("Component")
        ax.set_ylabel("Explained Variance (%)")
        ax.set_title("Scree Plot", fontweight="bold")
        ax.legend()

        # --- top-right: cumulative ---
        ax = axes[0, 1]
        cum = self.cumulative_variance[:n_show] * 100
        ax.plot(x_range, cum, "o-", color="#55A868", ms=3)
        ax.fill_between(x_range, cum, alpha=0.15, color="#55A868")
        for tgt in [80, 90, 95, 99]:
            ax.axhline(tgt, ls=":", color="gray", lw=0.8)
            n_at = int(np.searchsorted(self.cumulative_variance, tgt / 100) + 1)
            ax.axvline(n_at, ls=":", color="gray", lw=0.6)
            ax.annotate(f"{n_at} PCs → {tgt}%", xy=(n_at, tgt),
                        fontsize=7, color="gray")
        ax.set_xlabel("Component")
        ax.set_ylabel("Cumulative Variance (%)")
        ax.set_title("Cumulative Variance", fontweight="bold")

        # --- bottom-left: PC1 vs PC2 ---
        self._score_plot(axes[1, 0], 0, 1)

        # --- bottom-right: PC1 vs PC3 ---
        self._score_plot(axes[1, 1], 0, 2)

        plt.tight_layout()
        path = os.path.join(self.fig_dir, "pca_scree_and_scores.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")

    def _score_plot(self, ax, pc_x: int, pc_y: int) -> None:
        """Helper: scatter plot of two PC scores coloured by class."""
        if self._y_train is None or self.scores is None:
            return
        if pc_y >= self.scores.shape[1]:
            ax.text(0.5, 0.5, "Not enough components", ha="center")
            return

        y = self._y_train
        for label, colour, name in [(0, "steelblue", "Pass"), (1, "crimson", "Fail")]:
            mask = y == label
            ax.scatter(
                self.scores[mask, pc_x], self.scores[mask, pc_y],
                c=colour, label=name, alpha=0.4, s=12 if label == 0 else 40,
                edgecolors="none",
            )
            # centroid
            cx = self.scores[mask, pc_x].mean()
            cy = self.scores[mask, pc_y].mean()
            ax.scatter(cx, cy, c=colour, marker="X", s=180, edgecolors="black", lw=1.5)

            # 95% confidence ellipse
            try:
                cov = np.cov(self.scores[mask, pc_x], self.scores[mask, pc_y])
                eigvals, eigvecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
                from scipy.stats import chi2
                scale = np.sqrt(chi2.ppf(0.95, 2))
                w, h = 2 * scale * np.sqrt(eigvals)
                ell = Ellipse((cx, cy), w, h, angle=angle, fill=False,
                              edgecolor=colour, lw=1.5, ls="--")
                ax.add_patch(ell)
            except Exception:
                pass

        vx = self.explained_variance_ratio[pc_x] * 100
        vy = self.explained_variance_ratio[pc_y] * 100
        ax.set_xlabel(f"PC{pc_x+1} ({vx:.1f}%)")
        ax.set_ylabel(f"PC{pc_y+1} ({vy:.1f}%)")
        ax.set_title(f"Score Plot PC{pc_x+1} vs PC{pc_y+1}", fontweight="bold")
        ax.legend(fontsize=9)

    # ─────────────────────────────────────────────────────────────────
    def plot_loading_heatmap(
        self,
        feature_names: List[str],
        n_components: int = 10,
    ) -> None:
        """Heatmap of top-30 loadings on the first *n* components.

        Args:
            feature_names: Original feature names.
            n_components: Number of PCs to display.
        """
        print("\n── PCA Loading Heatmap ──")
        n_comp = min(n_components, self.loadings.shape[1])
        L = self.loadings[:, :n_comp]

        # Top 30 features by max absolute loading
        max_load = np.max(np.abs(L), axis=1)
        top_idx = np.argsort(max_load)[-30:][::-1]

        data = L[top_idx, :]
        row_labels = [feature_names[i] for i in top_idx] if len(feature_names) > max(top_idx) else [f"F{i}" for i in top_idx]
        col_labels = [f"PC{j+1}" for j in range(n_comp)]

        fig, ax = plt.subplots(figsize=(max(8, n_comp), 12))
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n_comp))
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.6)
        ax.set_title("PCA Loading Heatmap (top 30 features)", fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.fig_dir, "pca_loading_heatmap.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")

    # ─────────────────────────────────────────────────────────────────
    def plot_biplot_3d(self, y: pd.Series) -> None:
        """Interactive 3-D score plot using Plotly.

        Args:
            y: Binary label vector.
        """
        print("\n── PCA 3-D Biplot (Plotly) ──")
        scores = self.scores
        if scores.shape[1] < 3:
            print("  ⚠ Fewer than 3 components – skipping 3-D plot.")
            return

        y_arr = np.asarray(y)
        t2 = self.compute_hotelling_t2_from_pca(scores)

        labels = np.where(y_arr == 0, "Pass", "Fail")
        colours = np.where(y_arr == 0, "steelblue", "crimson")
        sizes = np.where(y_arr == 0, 3, 8)

        hover = [
            f"Idx {i} | {labels[i]} | T²={t2[i]:.2f}"
            for i in range(len(y_arr))
        ]

        fig = go.Figure()
        for lbl, colour in [("Pass", "steelblue"), ("Fail", "crimson")]:
            mask = labels == lbl
            fig.add_trace(go.Scatter3d(
                x=scores[mask, 0], y=scores[mask, 1], z=scores[mask, 2],
                mode="markers",
                marker=dict(size=sizes[mask], color=colour, opacity=0.6),
                text=np.array(hover)[mask],
                hoverinfo="text",
                name=lbl,
            ))

        vx = self.explained_variance_ratio[0] * 100
        vy = self.explained_variance_ratio[1] * 100
        vz = self.explained_variance_ratio[2] * 100
        fig.update_layout(
            title="PCA 3-D Score Plot",
            scene=dict(
                xaxis_title=f"PC1 ({vx:.1f}%)",
                yaxis_title=f"PC2 ({vy:.1f}%)",
                zaxis_title=f"PC3 ({vz:.1f}%)",
            ),
            width=900,
            height=700,
        )
        path = os.path.join(self.fig_dir, "pca_3d_biplot.html")
        fig.write_html(path)
        print(f"  ✓ Saved → {path}")

    # ─────────────────────────────────────────────────────────────────
    def plot_loading_bar_per_component(
        self,
        feature_names: List[str],
        n_components: int = 5,
    ) -> None:
        """Horizontal bar charts of top loadings per component.

        Args:
            feature_names: Original feature names.
            n_components: Number of PCs to show.
        """
        print("\n── PCA Loading Bars ──")
        n_comp = min(n_components, self.loadings.shape[1])

        fig, axes = plt.subplots(1, n_comp, figsize=(6 * n_comp, 8))
        if n_comp == 1:
            axes = [axes]

        for j in range(n_comp):
            load_j = self.loadings[:, j]
            top_idx = np.argsort(np.abs(load_j))[-15:][::-1]
            vals = load_j[top_idx]
            names = [feature_names[i] if i < len(feature_names) else f"F{i}" for i in top_idx]
            colours = ["steelblue" if v >= 0 else "crimson" for v in vals]

            ax = axes[j]
            ax.barh(range(len(names)), vals, color=colours, edgecolor="black", lw=0.5)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.invert_yaxis()
            for k, v in enumerate(vals):
                ax.text(v, k, f" {v:.3f}", va="center", fontsize=7)
            ax.set_title(f"PC{j+1}", fontweight="bold")
            ax.set_xlabel("Loading")

        plt.tight_layout()
        path = os.path.join(self.fig_dir, "pca_loading_bars.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")

    # =================================================================
    # PC interpretation
    # =================================================================
    def get_component_interpretation(self, feature_names: List[str]) -> Dict:
        """Auto-generate textual interpretation for each PC.

        Args:
            feature_names: Original feature names.

        Returns:
            Dictionary keyed by ``PC1``, ``PC2``, … with top positive/
            negative features and an interpretation string.
        """
        interp: Dict = {}
        for j in range(self.n_components):
            load_j = self.loadings[:, j]
            order = np.argsort(load_j)

            top_pos_idx = order[-5:][::-1]
            top_neg_idx = order[:5]

            def _names(idx_arr):
                return [feature_names[i] if i < len(feature_names) else f"F{i}" for i in idx_arr]

            top_pos = _names(top_pos_idx)
            top_neg = _names(top_neg_idx)
            var_pct = self.explained_variance_ratio[j] * 100

            interp[f"PC{j+1}"] = {
                "top_positive_features": top_pos,
                "top_negative_features": top_neg,
                "variance_explained_pct": round(var_pct, 2),
                "interpretation_note": (
                    f"PC{j+1} contrasts {', '.join(top_pos[:3])} "
                    f"vs {', '.join(top_neg[:3])}"
                ),
            }
        return interp

    # =================================================================
    # T² and SPE from PCA scores
    # =================================================================
    def compute_hotelling_t2_from_pca(self, X_pca: np.ndarray) -> np.ndarray:
        """Compute Hotelling T² in the PCA subspace.

        .. math::

            T^2_i = \\sum_{j=1}^{p} \\frac{z_{ij}^2}{\\lambda_j}

        Args:
            X_pca: Score matrix (n × p).

        Returns:
            1-D array of T² values.
        """
        eigenvalues = self.pca.explained_variance_
        t2 = np.sum(X_pca ** 2 / eigenvalues[np.newaxis, :], axis=1)
        return t2

    def compute_spe(
        self, X_original: np.ndarray, X_pca: np.ndarray
    ) -> np.ndarray:
        """Compute Squared Prediction Error (Q statistic).

        SPE = ||X − X̂||²  where X̂ is the PCA reconstruction.

        Args:
            X_original: Original scaled data (n × d).
            X_pca: PCA scores (n × p).

        Returns:
            1-D array of SPE values.
        """
        if isinstance(X_original, pd.DataFrame):
            X_original = X_original.values
        X_reconstructed = X_pca @ self.pca.components_ + self.pca.mean_
        residual = X_original - X_reconstructed
        spe = np.sum(residual ** 2, axis=1)
        return spe

    # =================================================================
    # Persistence
    # =================================================================
    def save(self, filepath: str) -> None:
        """Save the PCA engine to disk.

        Args:
            filepath: Destination .pkl path.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"  ✓ PCA engine saved → {filepath}")

    @staticmethod
    def load(filepath: str) -> "SECOMPCAEngine":
        """Load a saved PCA engine.

        Args:
            filepath: Source .pkl path.

        Returns:
            Restored engine.
        """
        return joblib.load(filepath)
