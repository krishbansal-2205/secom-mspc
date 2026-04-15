"""
mspc/combined_mspc.py - Integrated MSPC System
=================================================

Master class that combines T² and MEWMA charts, generates a unified
performance report, and produces the combined dashboard plot.
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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from mspc.hotelling_t2 import HotellingT2Chart
from mspc.mewma import MEWMAChart


class CombinedMSPCSystem:
    """Integrates Hotelling T² and MEWMA into a single monitoring system.

    The ``combined_signal`` flag is the logical OR of both charts' signals.

    Args:
        cfg: Optional configuration override.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.t2_chart = HotellingT2Chart(cfg=self.cfg)
        self.mewma_chart = MEWMAChart(cfg=self.cfg)
        self.fig_dir = os.path.join(self.cfg.figures_dir, "mspc")
        os.makedirs(self.fig_dir, exist_ok=True)

    # =================================================================
    def attach(
        self, t2_chart: HotellingT2Chart, mewma_chart: MEWMAChart
    ) -> "CombinedMSPCSystem":
        """Attach pre-fitted T² and MEWMA charts.

        Use this instead of :meth:`fit` when the individual charts have
        already been fitted in earlier pipeline stages, so the combined
        system reuses the exact same parameters and UCLs.

        Args:
            t2_chart: Already-fitted :class:`HotellingT2Chart`.
            mewma_chart: Already-fitted :class:`MEWMAChart`.

        Returns:
            ``self``.
        """
        self.t2_chart = t2_chart
        self.mewma_chart = mewma_chart
        print("\n  ✓ Combined MSPC system attached pre-fitted T² and MEWMA charts.")
        return self

    # =================================================================
    def fit(self, X_phase1: np.ndarray) -> "CombinedMSPCSystem":
        """Fit both T² and MEWMA charts on Phase I data.

        This is a convenience method for standalone use.  When running
        the full pipeline, prefer :meth:`attach` to reuse charts that
        were already fitted in Phase 7 / Phase 8.

        Args:
            X_phase1: In-control score matrix (m × p).

        Returns:
            ``self``.
        """
        self.t2_chart.fit_phase1(X_phase1)
        self.mewma_chart.fit(X_phase1)
        print("\n  ✓ Combined MSPC system fitted.")
        return self

    # =================================================================
    def monitor(
        self,
        X_phase2: np.ndarray,
        y_true: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Run both charts on Phase II data and merge results.

        Args:
            X_phase2: Phase II score matrix.
            y_true: Optional ground-truth labels.

        Returns:
            DataFrame with columns for every chart and the combined signal.
        """
        t2_res = self.t2_chart.monitor_phase2(X_phase2)
        mewma_res = self.mewma_chart.monitor(X_phase2)

        n = len(t2_res["t2_values"])
        df = pd.DataFrame({
            "observation_id": np.arange(n),
            "t2_value": t2_res["t2_values"],
            "t2_ucl": t2_res["ucl"],
            "t2_signal": t2_res["signals"],
            "mewma_value": mewma_res["t2_mewma"],
            "mewma_ucl": mewma_res["ucl_array"],
            "mewma_signal": mewma_res["signals"],
        })
        df["combined_signal"] = df["t2_signal"] | df["mewma_signal"]

        # Normalised combined score for AUC computation
        df["combined_score"] = np.maximum(
            df["t2_value"] / df["t2_ucl"],
            df["mewma_value"] / np.maximum(df["mewma_ucl"], 1e-9)
        )

        if y_true is not None:
            y_arr = np.asarray(y_true)
            df["true_label"] = y_arr

            def _type(row):
                sig = row["combined_signal"]
                lbl = row["true_label"]
                if sig and lbl == 1:
                    return "TP"
                if sig and lbl == 0:
                    return "FP"
                if not sig and lbl == 1:
                    return "FN"
                return "TN"

            df["signal_type"] = df.apply(_type, axis=1)

        os.makedirs(self.cfg.tables_dir, exist_ok=True)
        path = os.path.join(self.cfg.tables_dir, "mspc_results.csv")
        df.to_csv(path, index=False)
        print(f"  ✓ MSPC results saved → {path}")
        return df

    # =================================================================
    def generate_performance_report(
        self,
        results_df: pd.DataFrame,
        y_true: np.ndarray,
    ) -> Dict:
        """Compute and print performance metrics for each chart.

        Args:
            results_df: DataFrame from :meth:`monitor`.
            y_true: Ground-truth labels.

        Returns:
            Dictionary with per-chart metrics.
        """
        y = np.asarray(y_true)
        report: Dict = {}

        for name, sig_col, val_col in [
            ("T2", "t2_signal", "t2_value"),
            ("MEWMA", "mewma_signal", "mewma_value"),
            ("Combined", "combined_signal", "combined_score"),
        ]:
            preds = results_df[sig_col].astype(int).values
            tp = int(((preds == 1) & (y == 1)).sum())
            tn = int(((preds == 0) & (y == 0)).sum())
            fp = int(((preds == 1) & (y == 0)).sum())
            fn = int(((preds == 0) & (y == 1)).sum())

            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            prec = tp / max(tp + fp, 1)
            f1 = 2 * prec * sens / max(prec + sens, 1e-12)
            far = fp / max(fp + tn, 1)

            try:
                auc = roc_auc_score(y, results_df[val_col].values)
            except ValueError:
                auc = 0.0

            report[name] = {
                "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                "sensitivity": round(sens, 4),
                "specificity": round(spec, 4),
                "precision": round(prec, 4),
                "f1": round(f1, 4),
                "false_alarm_rate": round(far, 4),
                "auc_roc": round(auc, 4),
            }

        # Pretty print
        print("\n╔══════════════════════════════════════════════════════╗")
        print("║        MSPC SYSTEM PERFORMANCE REPORT                ║")
        print("╠══════════════════════════════════════════════════════╣")
        header = f"║ {'Metric':<22}│ {'T²':>7} │ {'MEWMA':>7} │ {'Combined':>9} ║"
        print(header)
        print("╠══════════════════════════════════════════════════════╣")
        for metric_key, label in [
            ("TP", "True Positives"),
            ("TN", "True Negatives"),
            ("FP", "False Positives"),
            ("FN", "False Negatives"),
        ]:
            vals = [report[c][metric_key] for c in ["T2", "MEWMA", "Combined"]]
            print(f"║ {label:<22}│ {vals[0]:>7} │ {vals[1]:>7} │ {vals[2]:>9} ║")
        print("╠══════════════════════════════════════════════════════╣")
        for metric_key, label in [
            ("sensitivity", "Sensitivity"),
            ("specificity", "Specificity"),
            ("precision", "Precision"),
            ("f1", "F1 Score"),
            ("false_alarm_rate", "False Alarm Rate"),
            ("auc_roc", "AUC-ROC"),
        ]:
            vals = [report[c][metric_key] for c in ["T2", "MEWMA", "Combined"]]
            print(f"║ {label:<22}│ {vals[0]:>7.3f} │ {vals[1]:>7.3f} │ {vals[2]:>9.3f} ║")
        print("╚══════════════════════════════════════════════════════╝")

        # Save
        import json
        path = os.path.join(self.cfg.reports_dir, "mspc_performance.json")
        os.makedirs(self.cfg.reports_dir, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"  ✓ Saved → {path}")

        return report

    # =================================================================
    def plot_combined_dashboard(
        self,
        results_df: pd.DataFrame,
        y_true: np.ndarray,
    ) -> None:
        """Create a large 6-panel combined dashboard figure.

        Args:
            results_df: DataFrame from :meth:`monitor`.
            y_true: Ground-truth labels.
        """
        print("\n── Combined MSPC Dashboard ──")
        y = np.asarray(y_true)
        n = len(results_df)
        x = np.arange(n)

        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

        # Row 1 – T² chart (full width)
        ax1 = fig.add_subplot(gs[0, :])
        t2 = results_df["t2_value"].values
        ucl = results_df["t2_ucl"].values[0]
        ax1.plot(x, t2, color="#4C72B0", lw=0.6)
        ax1.axhline(ucl, color="red", ls="--", lw=1.5, label=f"UCL={ucl:.1f}")
        for idx in np.where(y == 1)[0]:
            if idx < n:
                ax1.axvspan(idx - 0.5, idx + 0.5, alpha=0.06, color="red")
        ax1.set_ylabel("T²")
        ax1.set_title("Hotelling T² Control Chart", fontweight="bold")
        ax1.legend(loc="upper right", fontsize=8)

        # Row 2 – MEWMA chart (full width)
        ax2 = fig.add_subplot(gs[1, :])
        mewma_v = results_df["mewma_value"].values
        mewma_ucl = results_df["mewma_ucl"].values
        ax2.plot(x, mewma_v, color="#DD8452", lw=0.6)
        ax2.plot(x, mewma_ucl, color="red", ls="--", lw=1, label="UCL")
        for idx in np.where(y == 1)[0]:
            if idx < n:
                ax2.axvspan(idx - 0.5, idx + 0.5, alpha=0.06, color="red")
        ax2.set_ylabel("T² MEWMA")
        ax2.set_title("MEWMA Control Chart", fontweight="bold")
        ax2.legend(loc="upper right", fontsize=8)

        # Row 3 left – T² distribution pass vs fail
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.hist(t2[y == 0], bins=50, alpha=0.6, color="steelblue", label="Pass", density=True)
        ax3.hist(t2[y == 1], bins=30, alpha=0.7, color="crimson", label="Fail", density=True)
        ax3.axvline(ucl, color="red", ls="--", lw=1.5)
        ax3.set_xlabel("T² value")
        ax3.set_ylabel("Density")
        ax3.set_title("T² Distribution: Pass vs Fail", fontweight="bold")
        ax3.legend()

        # Row 3 right – performance metrics bar chart
        ax4 = fig.add_subplot(gs[2, 1])
        metrics = ["sensitivity", "specificity", "precision", "f1"]
        for i, chart in enumerate(["T2", "MEWMA", "Combined"]):
            try:
                preds = results_df[
                    {"T2": "t2_signal", "MEWMA": "mewma_signal",
                     "Combined": "combined_signal"}[chart]
                ].astype(int).values
                tp = ((preds == 1) & (y == 1)).sum()
                tn = ((preds == 0) & (y == 0)).sum()
                fp = ((preds == 1) & (y == 0)).sum()
                fn = ((preds == 0) & (y == 1)).sum()
                sens = tp / max(tp + fn, 1)
                spec = tn / max(tn + fp, 1)
                prec = tp / max(tp + fp, 1)
                f1_v = 2 * prec * sens / max(prec + sens, 1e-12)
                vals = [sens, spec, prec, f1_v]
            except Exception:
                vals = [0, 0, 0, 0]
            x_pos = np.arange(len(metrics)) + i * 0.25
            ax4.bar(x_pos, vals, width=0.22, label=chart)
        ax4.set_xticks(np.arange(len(metrics)) + 0.25)
        ax4.set_xticklabels([m.capitalize() for m in metrics])
        ax4.set_ylabel("Score")
        ax4.set_title("Performance Metrics", fontweight="bold")
        ax4.legend(fontsize=8)
        ax4.set_ylim(0, 1.05)

        # Row 4 left – ROC curves
        ax5 = fig.add_subplot(gs[3, 0])
        for col, label, colour in [
            ("t2_value", "T²", "steelblue"),
            ("mewma_value", "MEWMA", "darkorange"),
        ]:
            try:
                fpr, tpr, _ = roc_curve(y, results_df[col].values)
                auc_val = roc_auc_score(y, results_df[col].values)
                ax5.plot(fpr, tpr, color=colour, lw=2,
                         label=f"{label} (AUC={auc_val:.3f})")
            except ValueError:
                pass
        ax5.plot([0, 1], [0, 1], "k--", lw=1)
        ax5.set_xlabel("False Positive Rate")
        ax5.set_ylabel("True Positive Rate")
        ax5.set_title("ROC Curves", fontweight="bold")
        ax5.legend(fontsize=9)

        # Row 4 right – confusion matrices
        ax6 = fig.add_subplot(gs[3, 1])
        for k, (chart, sig_col) in enumerate([
            ("T²", "t2_signal"),
            ("MEWMA", "mewma_signal"),
            ("Combined", "combined_signal"),
        ]):
            cm = confusion_matrix(y, results_df[sig_col].astype(int).values, labels=[0, 1])
            ax_sub = fig.add_axes([0.52 + k * 0.155, 0.05, 0.13, 0.18])
            sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                        xticklabels=["Pass", "Fail"],
                        yticklabels=["Pass", "Fail"],
                        ax=ax_sub, cbar=False)
            ax_sub.set_title(chart, fontsize=10)
            ax_sub.set_xlabel("Predicted")
            ax_sub.set_ylabel("Actual")
        ax6.set_visible(False)

        path = os.path.join(self.fig_dir, "combined_mspc_dashboard.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved → {path}")
