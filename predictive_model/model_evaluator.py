"""
predictive_model/model_evaluator.py - Model Evaluation Suite
===============================================================

Evaluates trained classifiers with bootstrap CI, ROC/PR curves,
confusion matrices, and threshold analysis.
"""

import os
import sys
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


class SECOMModelEvaluator:
    """Evaluate and compare multiple classifiers.

    Args:
        cfg: Optional configuration override.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.fig_dir = os.path.join(self.cfg.figures_dir, "models")
        os.makedirs(self.fig_dir, exist_ok=True)

    # =================================================================
    def evaluate_all_models(
        self,
        models: Dict,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """Compute all metrics for every model.

        Args:
            models: ``{name: fitted_model}`` dictionary.
            X_test: Test features.
            y_test: True labels.

        Returns:
            Comparison DataFrame.
        """
        print("\n══════════════════════════════════════════════════")
        print("        MODEL EVALUATION")
        print("══════════════════════════════════════════════════")

        records = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                y_prob = model.decision_function(X_test)

            # Bootstrap AUC 95 % CI
            rng = np.random.RandomState(self.cfg.random_seed)
            aucs_boot = []
            for _ in range(1000):
                idx = rng.choice(len(y_test), len(y_test), replace=True)
                if len(np.unique(y_test[idx])) < 2:
                    continue
                aucs_boot.append(roc_auc_score(y_test[idx], y_prob[idx]))
            aucs_boot = np.array(aucs_boot)
            auc_lower = float(np.percentile(aucs_boot, 2.5))
            auc_upper = float(np.percentile(aucs_boot, 97.5))

            # Optimal threshold (maximise G-mean)
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            gmeans = np.sqrt(tpr * (1 - fpr))
            best_idx = np.argmax(gmeans)
            optimal_thresh = float(thresholds[best_idx])

            # PR curve AUC
            prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
            auc_pr = float(auc(rec_curve, prec_curve))

            try:
                brier = float(brier_score_loss(y_test, y_prob))
            except Exception:
                brier = np.nan
            try:
                ll = float(log_loss(y_test, y_prob))
            except Exception:
                ll = np.nan

            rec = {
                "model": name,
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                "auc_roc": round(float(roc_auc_score(y_test, y_prob)), 4),
                "auc_roc_ci_lower": round(auc_lower, 4),
                "auc_roc_ci_upper": round(auc_upper, 4),
                "auc_pr": round(auc_pr, 4),
                "brier_score": round(brier, 4) if not np.isnan(brier) else None,
                "log_loss": round(ll, 4) if not np.isnan(ll) else None,
                "optimal_threshold": round(optimal_thresh, 4),
            }
            records.append(rec)

            print(f"\n  {name}")
            print(f"    AUC-ROC  : {rec['auc_roc']:.4f}  "
                  f"[{rec['auc_roc_ci_lower']:.4f}, {rec['auc_roc_ci_upper']:.4f}]")
            print(f"    Recall   : {rec['recall']:.4f}   "
                  f"F1: {rec['f1']:.4f}   Precision: {rec['precision']:.4f}")

        df = pd.DataFrame(records)
        os.makedirs(self.cfg.tables_dir, exist_ok=True)
        path = os.path.join(self.cfg.tables_dir, "model_comparison.csv")
        df.to_csv(path, index=False)
        print(f"\n  ✓ Saved → {path}")
        return df

    # =================================================================
    def plot_roc_curves(
        self, models: Dict, X_test: np.ndarray, y_test: np.ndarray
    ) -> None:
        """ROC curves for all models on one axes.

        Args:
            models: Fitted model dictionary.
            X_test: Test features.
            y_test: True labels.
        """
        print("\n── ROC Curves ──")
        fig, ax = plt.subplots(figsize=(10, 8))
        colours = plt.cm.tab10(np.linspace(0, 1, len(models)))

        for (name, model), colour in zip(models.items(), colours):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                y_prob = model.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_val = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, color=colour, lw=2, label=f"{name} (AUC={auc_val:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves – All Models", fontweight="bold", fontsize=14)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        plt.tight_layout()

        path = os.path.join(self.fig_dir, "roc_curves.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")

    # =================================================================
    def plot_precision_recall_curves(
        self, models: Dict, X_test: np.ndarray, y_test: np.ndarray
    ) -> None:
        """Precision-Recall curves for all models.

        Args:
            models: Fitted model dictionary.
            X_test: Test features.
            y_test: True labels.
        """
        print("\n── Precision-Recall Curves ──")
        fig, ax = plt.subplots(figsize=(10, 8))
        colours = plt.cm.tab10(np.linspace(0, 1, len(models)))
        prevalence = y_test.mean()

        for (name, model), colour in zip(models.items(), colours):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                y_prob = model.decision_function(X_test)
            prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
            auc_val = auc(rec_c, prec_c)
            ax.plot(rec_c, prec_c, color=colour, lw=2, label=f"{name} (AUC-PR={auc_val:.3f})")

        ax.axhline(prevalence, ls="--", color="gray", lw=1, label=f"Baseline ({prevalence:.3f})")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curves", fontweight="bold", fontsize=14)
        ax.legend(loc="best", fontsize=9)
        plt.tight_layout()

        path = os.path.join(self.fig_dir, "precision_recall_curves.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")

    # =================================================================
    def plot_confusion_matrices(
        self, models: Dict, X_test: np.ndarray, y_test: np.ndarray
    ) -> None:
        """Side-by-side confusion matrices for all models.

        Args:
            models: Fitted model dictionary.
            X_test: Test features.
            y_test: True labels.
        """
        print("\n── Confusion Matrices ──")
        n = len(models)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (name, model) in zip(axes, models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            cm_pct = cm / cm.sum() * 100
            annot = np.array([
                [f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" for j in range(2)]
                for i in range(2)
            ])
            sns.heatmap(
                cm, annot=annot, fmt="", cmap="YlOrRd",
                xticklabels=["Pass", "Fail"],
                yticklabels=["Pass", "Fail"],
                ax=ax, cbar=False,
            )
            ax.set_title(name, fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        plt.tight_layout()
        path = os.path.join(self.fig_dir, "confusion_matrices.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")

    # =================================================================
    def plot_threshold_analysis(
        self,
        best_model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Plot metrics vs decision threshold.

        Args:
            best_model: The model to analyse.
            X_test: Test features.
            y_test: True labels.
        """
        print("\n── Threshold Analysis ──")
        try:
            y_prob = best_model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_prob = best_model.decision_function(X_test)

        thresholds = np.linspace(0.01, 0.99, 200)
        precs, recs, f1s, gmeans = [], [], [], []

        for t in thresholds:
            y_p = (y_prob >= t).astype(int)
            p = precision_score(y_test, y_p, zero_division=0)
            r = recall_score(y_test, y_p, zero_division=0)
            f = f1_score(y_test, y_p, zero_division=0)
            spec = ((y_p == 0) & (y_test == 0)).sum() / max((y_test == 0).sum(), 1)
            g = np.sqrt(r * spec)
            precs.append(p)
            recs.append(r)
            f1s.append(f)
            gmeans.append(g)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(thresholds, precs, label="Precision", lw=2)
        ax.plot(thresholds, recs, label="Recall", lw=2)
        ax.plot(thresholds, f1s, label="F1 Score", lw=2)
        ax.plot(thresholds, gmeans, label="G-Mean", lw=2, ls="--")

        opt_idx = np.argmax(f1s)
        ax.axvline(thresholds[opt_idx], ls=":", color="red", lw=1.5,
                    label=f"Optimal (F1) @ {thresholds[opt_idx]:.2f}")
        ax.scatter(thresholds[opt_idx], f1s[opt_idx], color="red", s=100, zorder=5)

        ax.set_xlabel("Decision Threshold", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Threshold Analysis", fontweight="bold", fontsize=14)
        ax.legend(fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()

        path = os.path.join(self.fig_dir, "threshold_analysis.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")

    # =================================================================
    def plot_feature_importance(
        self,
        best_model,
        feature_names,
        pca_engine=None,
        original_feature_names=None,
    ) -> None:
        """Feature importance from the best model, mapped to original space.

        Args:
            best_model: Model with ``feature_importances_`` attribute.
            feature_names: PC-level feature names.
            pca_engine: Optional PCA engine for mapping back.
            original_feature_names: Original sensor names.
        """
        print("\n── Feature Importance ──")
        try:
            importances = best_model.feature_importances_
        except AttributeError:
            try:
                importances = np.abs(best_model.coef_[0])
            except AttributeError:
                print("  ⚠ Model does not support feature importance.")
                return

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # PC-level importance
        order = np.argsort(importances)[::-1][:20]
        names = [feature_names[i] if i < len(feature_names) else f"PC{i+1}" for i in order]
        vals = importances[order]

        axes[0].barh(range(len(names)), vals, color="steelblue", edgecolor="black", lw=0.5)
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels(names, fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Importance")
        axes[0].set_title("PC-Level Feature Importance", fontweight="bold")

        # Original-feature mapping
        if pca_engine is not None and original_feature_names is not None:
            loadings = pca_engine.loadings  # d × p
            orig_imp = np.zeros(loadings.shape[0])
            for j in range(len(importances)):
                if j < loadings.shape[1]:
                    orig_imp += np.abs(loadings[:, j]) * importances[j]
            top_orig = np.argsort(orig_imp)[-20:][::-1]
            orig_names = [original_feature_names[i] if i < len(original_feature_names) else f"F{i}" for i in top_orig]
            orig_vals = orig_imp[top_orig]

            axes[1].barh(range(len(orig_names)), orig_vals, color="#DD8452", edgecolor="black", lw=0.5)
            axes[1].set_yticks(range(len(orig_names)))
            axes[1].set_yticklabels(orig_names, fontsize=9)
            axes[1].invert_yaxis()
            axes[1].set_xlabel("Mapped Importance")
            axes[1].set_title("Original Feature Importance (via PCA)", fontweight="bold")
        else:
            axes[1].text(0.5, 0.5, "PCA mapping not available", ha="center")

        plt.tight_layout()
        path = os.path.join(self.fig_dir, "feature_importance.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")
