"""
mspc/fault_diagnosis.py - Fault Diagnosis Engine
===================================================

Performs MYT T² decomposition, generates contribution charts,
maps PC-space contributions back to original sensors, and produces
an OCAP (Out-of-Control Action Plan) report.
"""

from __future__ import annotations
from config import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from mspc.hotelling_t2 import HotellingT2Chart

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FaultDiagnosisEngine:
    """Diagnose root causes of out-of-control signals.

    Args:
        cfg: Optional configuration override.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.fig_dir = os.path.join(self.cfg.figures_dir, "mspc")
        os.makedirs(self.fig_dir, exist_ok=True)

    # =================================================================
    def diagnose_signal(
        self,
        x_signal: np.ndarray,
        chart: "HotellingT2Chart",
        t2_value: float,
        ucl: float,
        feature_names: List[str],
        original_feature_names: Optional[List[str]] = None,
        pca_loadings: Optional[np.ndarray] = None,
    ) -> Dict:
        """Perform complete fault diagnosis for a single signal.

        Steps:
            1. MYT T² decomposition.
            2. Standardised value analysis.
            3. Original-sensor mapping (if PCA loadings given).
            4. OCAP report generation.

        Args:
            x_signal: Observation vector (PC scores, 1-D).
            chart: Fitted HotellingT2Chart instance.
            t2_value: The T² value for this observation.
            ucl: Current UCL.
            feature_names: PC names (e.g. ``["PC1", ..., "PCk"]``).
            original_feature_names: Optional original sensor names.
            pca_loadings: Optional PCA loading matrix (d × p) for
                mapping back to sensors.

        Returns:
            Dictionary with contributions, alert level, and actions.
        """
        if not hasattr(chart, 'mean_vector'):
            raise TypeError("chart must be a fitted HotellingT2Chart")

        x = np.asarray(x_signal, dtype=np.float64).ravel()
        p = len(x)
        # Use Phase I statistics directly from the chart
        mu = chart.mean_vector
        cov = chart.cov_matrix
        std = np.sqrt(np.diag(cov) + 1e-12)

        # Step 1 – MYT decomposition using parameter slices (not re-estimation)
        try:
            cov_inv = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(p)

        diff_full = x - mu
        t2_full = float(diff_full @ cov_inv @ diff_full)

        contributions = np.zeros(p)
        for j in range(p):
            mask = np.ones(p, dtype=bool)
            mask[j] = False
            x_sub = x[mask]
            mu_sub = mu[mask]
            cov_sub = cov[np.ix_(mask, mask)]
            try:
                inv_sub = np.linalg.pinv(cov_sub)
            except np.linalg.LinAlgError:
                inv_sub = np.eye(p - 1)
            diff_sub = x_sub - mu_sub
            t2_sub = float(diff_sub @ inv_sub @ diff_sub)
            contributions[j] = max(t2_full - t2_sub, 0.0)

        total_contrib = contributions.sum() + 1e-12
        std_values = (x - mu) / std

        # Step 2 – alert level
        exceedance = (t2_value - ucl) / ucl * 100
        if exceedance > 200:
            alert = "CRITICAL"
        elif exceedance > 100:
            alert = "HIGH"
        elif exceedance > 50:
            alert = "MEDIUM"
        else:
            alert = "LOW"

        # Step 3 – original-sensor mapping
        top_original = []
        if pca_loadings is not None and original_feature_names is not None:
            # importance = |loading| × contribution for each PC
            orig_importance = np.zeros(pca_loadings.shape[0])
            for j in range(p):
                orig_importance += np.abs(pca_loadings[:, j]
                                          ) * contributions[j]
            top_idx = np.argsort(orig_importance)[-5:][::-1]
            top_original = [
                {"sensor": original_feature_names[i],
                 "importance": round(float(orig_importance[i]), 3)}
                for i in top_idx
            ]

        # Step 4 – OCAP report
        order = np.argsort(contributions)[::-1]
        top_pcs = [
            (feature_names[j] if j < len(feature_names) else f"Var{j}",
             round(100 * contributions[j] / total_contrib, 1))
            for j in order[:5]
        ]

        print("\n╔═══════════════════════════════════════════════════╗")
        print("║           FAULT DIAGNOSIS REPORT                  ║")
        print("╠═══════════════════════════════════════════════════╣")
        print(f"║ T² Value    : {t2_value:<10.4f} (UCL = {ucl:.4f})       ║")
        print(
            f"║ Exceedance  : {exceedance:.1f}% above UCL                  ║")
        print(f"║ Alert Level : {'⚠️  ' + alert:<40}║")
        print("╠═══════════════════════════════════════════════════╣")
        print("║ Top Contributing Components:                      ║")
        for i, (pc, pct) in enumerate(top_pcs[:3]):
            print(
                f"║   {i+1}. {pc:<5} - Contribution: {pct:>5.1f}%               ║")
        print("╠═══════════════════════════════════════════════════╣")
        print("║ Recommended Actions:                              ║")
        if len(top_pcs) > 0:
            print(
                f"║   1. Investigate sensors associated with {top_pcs[0][0]:<8}║")
        if len(top_pcs) > 1:
            print(f"║   2. Check process parameters for {top_pcs[1][0]:<13}║")
        print("║   3. Review last maintenance log                  ║")
        print("╚═══════════════════════════════════════════════════╝")

        # Generate contribution chart
        self.plot_contribution_chart(
            contributions, t2_value, ucl, feature_names, std_values
        )

        result: Dict = {
            "t2_value": t2_value,
            "ucl": ucl,
            "exceedance_pct": round(exceedance, 1),
            "alert_level": alert,
            "contributions": {
                feature_names[j] if j < len(feature_names) else f"Var{j}":
                round(float(contributions[j]), 4)
                for j in order
            },
            "top_pcs": top_pcs,
            "std_values": {
                feature_names[j] if j < len(feature_names) else f"Var{j}":
                round(float(std_values[j]), 3)
                for j in range(p)
            },
            "top_original_sensors": top_original,
        }

        return result

    # =================================================================
    def plot_contribution_chart(
        self,
        contributions: np.ndarray,
        t2_value: float,
        ucl: float,
        feature_names: List[str],
        std_values: Optional[np.ndarray] = None,
    ) -> None:
        """Generate 2-panel contribution chart.

        Args:
            contributions: Per-variable contribution array.
            t2_value: Total T² for the observation.
            ucl: Upper control limit.
            feature_names: Variable names.
            std_values: Optional standardised values.
        """
        p = len(contributions)
        total = contributions.sum() + 1e-12

        fig, axes = plt.subplots(1, 2, figsize=(18, max(6, p * 0.3)))

        # Panel 1 – contribution bars
        ax = axes[0]
        order = np.argsort(contributions)[::-1]
        names = [feature_names[j] if j < len(
            feature_names) else f"Var{j}" for j in order]
        vals = contributions[order]
        pcts = vals / total * 100

        colours = []
        for pct in pcts:
            if pct > 20:
                colours.append("crimson")
            elif pct > 10:
                colours.append("darkorange")
            else:
                colours.append("steelblue")

        ax.barh(range(len(names)), vals, color=colours,
                edgecolor="black", lw=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(t2_value, ls="--", color="black",
                   lw=1, label=f"T² = {t2_value:.2f}")
        ax.set_xlabel("T² Contribution")
        ax.set_title("T² Contribution Chart", fontweight="bold")
        ax.legend(fontsize=8)

        # Panel 2 – standardised values
        ax2 = axes[1]
        if std_values is not None:
            std_ord = std_values[order]
            col2 = ["crimson" if abs(s) > 3 else "steelblue" for s in std_ord]
            ax2.barh(range(len(names)), std_ord,
                     color=col2, edgecolor="black", lw=0.5)
            ax2.set_yticks(range(len(names)))
            ax2.set_yticklabels(names, fontsize=8)
            ax2.invert_yaxis()
            for lim in [-3, -2, -1, 1, 2, 3]:
                ax2.axvline(lim, ls=":", color="gray", lw=0.5)
            ax2.axvline(-3, ls="--", color="red", lw=1)
            ax2.axvline(3, ls="--", color="red", lw=1, label="±3σ")
            ax2.set_xlabel("Standardised Value")
            ax2.set_title("Standardised Values", fontweight="bold")
            ax2.legend(fontsize=8)
        else:
            ax2.text(0.5, 0.5, "No std values", ha="center", va="center")

        plt.tight_layout()
        path = os.path.join(self.fig_dir, "fault_diagnosis_contribution.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")
