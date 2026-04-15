"""
mspc/arl_simulator.py - Average Run Length Simulator
=======================================================

Monte-Carlo simulation of ARL for T² and MEWMA charts under
various mean-shift magnitudes.
"""

import os
import sys
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


class ARLSimulator:
    """Simulate ARL tables for T² and MEWMA charts.

    Args:
        cfg: Optional configuration override.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.fig_dir = os.path.join(self.cfg.figures_dir, "mspc")
        os.makedirs(self.fig_dir, exist_ok=True)

    # =================================================================
    def simulate_arl_table(
        self,
        t2_chart,
        mewma_chart,
        shift_sizes: Optional[List[float]] = None,
        n_sim: int = 5000,
    ) -> pd.DataFrame:
        """Run ARL simulations for both charts at each shift size.

        Args:
            t2_chart: Fitted :class:`HotellingT2Chart`.
            mewma_chart: Fitted :class:`MEWMAChart`.
            shift_sizes: List of shift magnitudes in σ units.
            n_sim: Number of Monte-Carlo runs per (chart, shift) pair.

        Returns:
            DataFrame with ARL/SDRL for both charts at each shift.
        """
        shift_sizes = shift_sizes or self.cfg.shift_sizes
        print("\n══════════════════════════════════════════════════")
        print("        ARL SIMULATION (Monte Carlo)")
        print("══════════════════════════════════════════════════")

        rng = np.random.RandomState(self.cfg.random_seed)
        p = t2_chart._p
        max_rl = 5000

        # Cholesky for data generation
        L = np.linalg.cholesky(t2_chart.cov_matrix + 1e-10 * np.eye(p))
        shift_dir = np.zeros(p)
        shift_dir[0] = 1.0

        records = []
        for delta in shift_sizes:
            mu_shift = delta * shift_dir
            print(f"\n  Shift = {delta}σ …", end=" ")

            # ── T² ARL ──────────────────────────────────────────────
            t2_rls = np.ones(n_sim, dtype=int) * max_rl
            active_t2 = np.ones(n_sim, dtype=bool)
            ucl_t2 = t2_chart.ucl_phase2_F
            
            for rl in range(1, max_rl + 1):
                if not np.any(active_t2):
                    break
                n_active = active_t2.sum()
                diff = rng.randn(n_active, p) @ L.T + mu_shift
                t2_val = np.einsum('ij,jk,ik->i', diff, t2_chart.cov_inv, diff)
                signal = t2_val > ucl_t2
                
                if np.any(signal):
                    signaled_idx = np.where(active_t2)[0][signal]
                    t2_rls[signaled_idx] = rl
                    active_t2[signaled_idx] = False

            # ── MEWMA ARL ───────────────────────────────────────────
            mewma_rls = np.ones(n_sim, dtype=int) * max_rl
            active_mewma = np.ones(n_sim, dtype=bool)
            Z = np.zeros((n_sim, p))
            ucl_mewma = mewma_chart.ucl_asymptotic
            lam = mewma_chart.lam
            
            for rl in range(1, max_rl + 1):
                if not np.any(active_mewma):
                    break
                n_active = active_mewma.sum()
                deviation = rng.randn(n_active, p) @ L.T + mu_shift
                
                Z[active_mewma] = lam * deviation + (1 - lam) * Z[active_mewma]
                
                factor = (lam / (2 - lam)) * (1 - (1 - lam) ** (2 * rl))
                inv_factor = 1.0 / max(factor, 1e-15)
                
                Z_active = Z[active_mewma]
                t2_m = inv_factor * np.einsum('ij,jk,ik->i', Z_active, mewma_chart.cov_inv, Z_active)
                
                signal = t2_m > ucl_mewma
                if np.any(signal):
                    signaled_idx = np.where(active_mewma)[0][signal]
                    mewma_rls[signaled_idx] = rl
                    active_mewma[signaled_idx] = False

            records.append({
                "shift_sigma": delta,
                "T2_ARL": round(float(t2_rls.mean()), 1),
                "T2_SDRL": round(float(t2_rls.std()), 1),
                "T2_MRL": round(float(np.median(t2_rls)), 1),
                "MEWMA_ARL": round(float(mewma_rls.mean()), 1),
                "MEWMA_SDRL": round(float(mewma_rls.std()), 1),
                "MEWMA_MRL": round(float(np.median(mewma_rls)), 1),
            })
            print(f"T²={records[-1]['T2_ARL']:.0f}  MEWMA={records[-1]['MEWMA_ARL']:.0f}")

        df = pd.DataFrame(records)

        print("\n  ─── ARL Comparison Table ───")
        print(df.to_string(index=False))

        os.makedirs(self.cfg.tables_dir, exist_ok=True)
        path = os.path.join(self.cfg.tables_dir, "arl_comparison_table.csv")
        df.to_csv(path, index=False)
        print(f"\n  ✓ Saved → {path}")

        return df

    # =================================================================
    def plot_arl_curves(self, arl_table: pd.DataFrame) -> None:
        """Plot ARL vs shift-size curves for both charts.

        Args:
            arl_table: DataFrame from :meth:`simulate_arl_table`.
        """
        print("\n── ARL Curves ──")
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.semilogy(
            arl_table["shift_sigma"], arl_table["T2_ARL"],
            "o-", color="steelblue", lw=2, ms=7, label="Hotelling T²",
        )
        ax.semilogy(
            arl_table["shift_sigma"], arl_table["MEWMA_ARL"],
            "s-", color="crimson", lw=2, ms=7, label="MEWMA",
        )

        # confidence bands (± 1 SDRL / √n_sim ≈ small)
        ax.fill_between(
            arl_table["shift_sigma"],
            arl_table["T2_ARL"] - arl_table["T2_SDRL"] * 0.1,
            arl_table["T2_ARL"] + arl_table["T2_SDRL"] * 0.1,
            alpha=0.15, color="steelblue",
        )
        ax.fill_between(
            arl_table["shift_sigma"],
            arl_table["MEWMA_ARL"] - arl_table["MEWMA_SDRL"] * 0.1,
            arl_table["MEWMA_ARL"] + arl_table["MEWMA_SDRL"] * 0.1,
            alpha=0.15, color="crimson",
        )

        ax.axhline(self.cfg.target_arl0, ls="--", color="gray", lw=1,
                    label=f"Target ARL₀ = {self.cfg.target_arl0}")

        # annotations
        mid = len(arl_table) // 2
        if arl_table.iloc[mid]["MEWMA_ARL"] < arl_table.iloc[mid]["T2_ARL"]:
            ax.annotate("MEWMA better for\nsmall shifts", xy=(0.5, 200),
                        fontsize=10, color="crimson", ha="center")

        ax.set_xlabel("Shift Size (σ)", fontsize=12)
        ax.set_ylabel("Average Run Length (log scale)", fontsize=12)
        ax.set_title("ARL Comparison: T² vs MEWMA", fontweight="bold", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()

        path = os.path.join(self.fig_dir, "arl_curves_comparison.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")
