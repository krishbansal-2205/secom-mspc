"""
mspc/hotelling_t2.py - Hotelling T² Control Chart
====================================================

Implements the Phase I / Phase II Hotelling T² chart with exact
F-distribution and chi-squared UCLs, vectorised computation,
MYT decomposition for fault diagnosis, and Monte-Carlo ARL simulation.
"""

import os
import sys
from typing import Dict, List, Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


class HotellingT2Chart:
    """Hotelling T² multivariate control chart.

    Attributes:
        mean_vector: Estimated process mean (Phase I).
        cov_matrix: Estimated covariance matrix (Phase I).
        cov_inv: (Pseudo)-inverse of covariance.
        ucl_phase1: F-based upper control limit for Phase I.
        ucl_phase2_F: F-based UCL for Phase II.
        ucl_phase2_chi2: Chi-squared asymptotic UCL for Phase II.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.mean_vector: Optional[np.ndarray] = None
        self.cov_matrix: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.ucl_phase1: float = 0.0
        self.ucl_phase2_F: float = 0.0
        self.ucl_phase2_chi2: float = 0.0
        self._m: int = 0  # Phase I sample size
        self._p: int = 0  # number of variables
        self.fig_dir = os.path.join(self.cfg.figures_dir, "mspc")
        os.makedirs(self.fig_dir, exist_ok=True)

    # =================================================================
    # Phase I fitting
    # =================================================================
    def fit_phase1(
        self, X_phase1: np.ndarray, alpha: Optional[float] = None
    ) -> np.ndarray:
        """Estimate process parameters from in-control Phase I data.

        Args:
            X_phase1: Score matrix from Phase I (m × p).
            alpha: False-alarm rate; defaults to ``config.alpha``.

        Returns:
            Array of Phase I T² values.

        Raises:
            ValueError: If input contains NaN or m ≤ p.
        """
        alpha = alpha or self.cfg.alpha
        X = np.asarray(X_phase1, dtype=np.float64)

        # Step 1 – validate
        if np.isnan(X).any():
            raise ValueError("Phase I data contains NaN values.")
        m, p = X.shape
        if m <= p:
            raise ValueError(
                f"Phase I needs m > p. Got m={m}, p={p}. "
                "Reduce PCA components or increase Phase I size."
            )
        if m < 30:
            print(f"  ⚠ Phase I has only {m} observations (recommend ≥ 30).")
        self._m, self._p = m, p

        # Step 2 – parameter estimation
        self.mean_vector = X.mean(axis=0)
        self.cov_matrix = np.cov(X, rowvar=False)

        cond = np.linalg.cond(self.cov_matrix)
        if cond > 1e10:
            print(f"  ⚠ Covariance is ill-conditioned (κ = {cond:.1e}). "
                  "Using pseudo-inverse.")
            self.cov_inv = np.linalg.pinv(self.cov_matrix)
        else:
            self.cov_inv = np.linalg.inv(self.cov_matrix)

        # Step 3 – control limits
        f_crit = sp_stats.f.ppf(1 - alpha, p, m - p)
        self.ucl_phase1 = (p * (m - 1) * (m + 1)) / (m * (m - p)) * f_crit
        self.ucl_phase2_F = (p * (m + 1) * (m - 1)) / (m * (m - p)) * f_crit
        self.ucl_phase2_chi2 = float(sp_stats.chi2.ppf(1 - alpha, df=p))

        # Step 4 – Phase I T² values
        t2_phase1 = self.calculate_t2(X)
        n_ooc = int((t2_phase1 > self.ucl_phase1).sum())
        pct_ooc = 100 * n_ooc / m

        # Step 5 – summary
        print("┌────────────────────────────────────────────┐")
        print("│ HOTELLING T² CHART – PHASE I RESULTS       │")
        print("├────────────────────────────────────────────┤")
        print(f"│ Phase I sample size (m) : {m:<18}│")
        print(f"│ Number of variables (p) : {p:<18}│")
        print(f"│ Alpha (false alarm rate) : {alpha:<17}│")
        print(f"│ Expected ARL₀           : {int(1/alpha):<18}│")
        print("│                                            │")
        print(f"│ Mean vector norm        : {np.linalg.norm(self.mean_vector):<17.4f}│")
        print(f"│ Covariance condition #  : {cond:<17.1f}│")
        print("│                                            │")
        print("│ Control Limits:                            │")
        print(f"│   Phase I  UCL (F-dist) : {self.ucl_phase1:<17.4f}│")
        print(f"│   Phase II UCL (F-dist) : {self.ucl_phase2_F:<17.4f}│")
        print(f"│   Phase II UCL (χ²)     : {self.ucl_phase2_chi2:<17.4f}│")
        print("│                                            │")
        print("│ Phase I Performance:                       │")
        print(f"│   Out-of-control points : {n_ooc} ({pct_ooc:.1f}%)        │")
        expected = alpha * m
        print(f"│   Expected false alarms : {expected:.1f}               │")
        print("└────────────────────────────────────────────┘")

        return t2_phase1

    # =================================================================
    # T² calculation (vectorised)
    # =================================================================
    def calculate_t2(self, X: np.ndarray) -> np.ndarray:
        """Compute Hotelling T² for every row in *X*.

        Uses ``np.einsum`` for efficient batch computation.

        Args:
            X: Data matrix (n × p).

        Returns:
            1-D array of T² values.
        """
        X_c = np.asarray(X, dtype=np.float64) - self.mean_vector
        return np.einsum("ij,jk,ik->i", X_c, self.cov_inv, X_c)

    # =================================================================
    # Phase II monitoring
    # =================================================================
    def monitor_phase2(
        self,
        X_phase2: np.ndarray,
        y_true: Optional[np.ndarray] = None,
    ) -> Dict:
        """Apply Phase II monitoring to new data.

        Args:
            X_phase2: Phase II score matrix.
            y_true: Optional true labels for performance evaluation.

        Returns:
            Dictionary with ``t2_values``, ``ucl``, ``signals``,
            ``signal_indices``, ``signal_rate``, and optional
            ``performance`` metrics.
        """
        t2 = self.calculate_t2(X_phase2)
        ucl = self.ucl_phase2_F
        signals = t2 > ucl
        signal_idx = np.where(signals)[0]

        result: Dict = {
            "t2_values": t2,
            "ucl": ucl,
            "signals": signals,
            "signal_indices": signal_idx,
            "signal_rate": float(100 * signals.sum() / len(t2)),
        }

        if y_true is not None:
            y_true = np.asarray(y_true)
            tp = int(((signals) & (y_true == 1)).sum())
            tn = int(((~signals) & (y_true == 0)).sum())
            fp = int(((signals) & (y_true == 0)).sum())
            fn = int(((~signals) & (y_true == 1)).sum())
            result["performance"] = {
                "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                "sensitivity": tp / max(tp + fn, 1),
                "specificity": tn / max(tn + fp, 1),
                "precision": tp / max(tp + fp, 1),
                "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "false_alarm_rate": fp / max(fp + tn, 1),
            }
            print(f"\n  T² Phase II: {signals.sum()} signals "
                  f"({result['signal_rate']:.1f}%) | "
                  f"TP={tp} FP={fp} FN={fn}")

        return result

    # =================================================================
    # Plotting
    # =================================================================
    def plot_t2_chart(
        self,
        t2_values: np.ndarray,
        ucl: float,
        y_true: Optional[np.ndarray] = None,
        phase: str = "II",
        title: Optional[str] = None,
    ) -> None:
        """Render the T² control chart with colour-coded points.

        Args:
            t2_values: Array of T² statistics.
            ucl: Upper control limit.
            y_true: Optional true labels for TP/FP/FN colouring.
            phase: "I" or "II" label.
            title: Custom chart title.
        """
        print(f"\n── Hotelling T² Chart (Phase {phase}) ──")
        n = len(t2_values)
        x = np.arange(n)
        signals = t2_values > ucl

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(20, 10),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

        # Main chart
        ax1.plot(x, t2_values, color="#4C72B0", lw=0.6, alpha=0.7)
        ax1.axhline(ucl, color="red", ls="--", lw=1.5, label=f"UCL = {ucl:.2f}")
        ax1.axhline(0.8 * ucl, color="orange", ls=":", lw=1, label=f"Warning = {0.8*ucl:.2f}")

        if y_true is not None:
            y_true = np.asarray(y_true)
            # colour each point
            tp = signals & (y_true == 1)
            fp = signals & (y_true == 0)
            fn = (~signals) & (y_true == 1)
            tn = (~signals) & (y_true == 0)

            ax1.scatter(x[tn], t2_values[tn], c="#4C72B0", s=10, alpha=0.3, label=f"TN ({tn.sum()})")
            ax1.scatter(x[tp], t2_values[tp], c="red", marker="*", s=150, zorder=5, label=f"TP ({tp.sum()})")
            ax1.scatter(x[fp], t2_values[fp], c="orange", marker="^", s=80, zorder=4, label=f"FP ({fp.sum()})")
            ax1.scatter(x[fn], t2_values[fn], c="purple", marker="v", s=80, zorder=4, label=f"FN ({fn.sum()})")

            # Shade actual defect regions
            for idx in np.where(y_true == 1)[0]:
                ax1.axvspan(idx - 0.5, idx + 0.5, alpha=0.05, color="red")
        else:
            in_ctrl = ~signals
            ax1.scatter(x[in_ctrl], t2_values[in_ctrl], c="#4C72B0", s=10, alpha=0.3)
            ax1.scatter(x[signals], t2_values[signals], c="red", marker="*", s=100, zorder=5, label="Signal")

        ax1.set_ylabel("T² Statistic")
        ax1.set_title(
            title or f"Hotelling T² Control Chart – Phase {phase}",
            fontweight="bold", fontsize=14,
        )
        ax1.legend(loc="upper right", fontsize=8)

        # Info box
        info = f"n={n}  UCL={ucl:.2f}  Signals={signals.sum()}"
        ax1.text(0.01, 0.97, info, transform=ax1.transAxes, fontsize=9,
                 va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        # Signal indicator
        colours = np.where(signals, "red", "#55A868")
        ax2.bar(x, signals.astype(int), color=colours, width=1.0)
        ax2.set_ylabel("Signal")
        ax2.set_xlabel("Observation index")
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["In-ctrl", "OOC"])

        plt.tight_layout()
        path = os.path.join(self.fig_dir, f"hotelling_t2_chart_phase{phase}.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")

    # =================================================================
    # MYT decomposition
    # =================================================================
    def decompose_signal(
        self,
        x_obs: np.ndarray,
        X_phase1: np.ndarray,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Mason-Young-Tracy decomposition of a T² signal.

        For each variable *j*, the contribution is:

        .. math:: d_j = T^2_{\\text{full}} - T^2_{-j}

        Args:
            x_obs: Single observation vector (1-D).
            X_phase1: Phase I data for re-estimating sub-models.
            feature_names: Variable names.

        Returns:
            DataFrame sorted by contribution descending.
        """
        x = np.asarray(x_obs, dtype=np.float64).ravel()
        p = len(x)
        t2_full = float(self.calculate_t2(x.reshape(1, -1))[0])

        contributions = np.zeros(p)
        for j in range(p):
            mask = np.ones(p, dtype=bool)
            mask[j] = False
            x_sub = x[mask].reshape(1, -1)
            # Use stored Phase I parameters (slices), not re-estimation
            mu_sub = self.mean_vector[mask]
            cov_sub = self.cov_matrix[np.ix_(mask, mask)]
            try:
                inv_sub = np.linalg.pinv(cov_sub)
            except np.linalg.LinAlgError:
                inv_sub = np.eye(p - 1)
            diff = x_sub - mu_sub
            t2_minus_j = float((diff @ inv_sub @ diff.T)[0, 0])
            contributions[j] = max(t2_full - t2_minus_j, 0.0)

        total = contributions.sum() + 1e-12
        std_values = (x - self.mean_vector) / (np.sqrt(np.diag(self.cov_matrix)) + 1e-12)

        records = []
        for j in range(p):
            name = feature_names[j] if j < len(feature_names) else f"Var{j}"
            records.append({
                "variable": name,
                "contribution": round(float(contributions[j]), 4),
                "contribution_pct": round(100 * contributions[j] / total, 2),
                "std_value": round(float(std_values[j]), 3),
                "status": "⚠" if abs(std_values[j]) > 3 else "OK",
            })

        df = pd.DataFrame(records).sort_values("contribution", ascending=False)
        df.reset_index(drop=True, inplace=True)

        print("\n  Variable             | Contribution | Contr%  | Std | Status")
        print("  " + "─" * 63)
        for _, row in df.head(10).iterrows():
            var_name = str(row['variable'])[:20]
            print(f"  {var_name:<20} {row['contribution']:>10.3f}  "
                  f"{row['contribution_pct']:>7.1f}%  "
                  f"{row['std_value']:>6.2f}  {row['status']}")

        return df

    # =================================================================
    # Monte-Carlo ARL
    # =================================================================
    def compute_arl(
        self,
        distribution: str = "normal",
        shift_size: float = 0.0,
        n_simulations: int = 10_000,
    ) -> Dict:
        """Estimate ARL by Monte-Carlo simulation.

        Args:
            distribution: "normal" (default).
            shift_size: Mean-shift magnitude in σ units.
            n_simulations: Number of simulation runs.

        Returns:
            Dictionary with ARL, SDRL, MRL, and percentiles.
        """
        rng = np.random.RandomState(self.cfg.random_seed)
        p = self._p
        ucl = self.ucl_phase2_F

        # Build shifted mean
        shift_dir = np.zeros(p)
        shift_dir[0] = 1.0  # shift along first PC
        mu_shifted = self.mean_vector + shift_size * shift_dir

        min_eigval = np.linalg.eigvalsh(self.cov_matrix).min()
        nugget = max(1e-8, -min_eigval + 1e-8) if min_eigval < 0 else 1e-10
        L = np.linalg.cholesky(self.cov_matrix + nugget * np.eye(p))

        run_lengths = np.zeros(n_simulations, dtype=int)
        max_rl = 5000

        for sim in range(n_simulations):
            rl = 0
            for _ in range(max_rl):
                rl += 1
                z = rng.randn(p)
                x = L @ z + mu_shifted
                diff = x - self.mean_vector  # use stored Phase I mean
                t2 = float(diff @ self.cov_inv @ diff)
                if t2 > ucl:
                    break
            run_lengths[sim] = rl

        arl = float(run_lengths.mean())
        sdrl = float(run_lengths.std())
        mrl = float(np.median(run_lengths))
        p5 = float(np.percentile(run_lengths, 5))
        p95 = float(np.percentile(run_lengths, 95))
        ci_margin = 1.96 * sdrl / np.sqrt(n_simulations)

        return {
            "shift_size": shift_size,
            "ARL": round(arl, 2),
            "SDRL": round(sdrl, 2),
            "MRL": round(mrl, 2),
            "P5": round(p5, 2),
            "P95": round(p95, 2),
            "CI_95_lower": round(arl - ci_margin, 2),
            "CI_95_upper": round(arl + ci_margin, 2),
        }

    # =================================================================
    # Persistence
    # =================================================================
    def save(self, filepath: str) -> None:
        """Save chart to disk.

        Args:
            filepath: Destination .pkl.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"  ✓ T² chart saved → {filepath}")

    @staticmethod
    def load(filepath: str) -> "HotellingT2Chart":
        """Load a saved chart.

        Args:
            filepath: Source .pkl.
        """
        return joblib.load(filepath)
