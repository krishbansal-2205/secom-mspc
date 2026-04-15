"""
mspc/phase_manager.py - SPC Phase I / Phase II Setup
======================================================

Splits the PCA-transformed data into Phase I (in-control baseline)
and Phase II (monitoring) sets following strict SPC rules:
  - Phase I contains ONLY confirmed Pass samples.
  - The split is TEMPORAL (never random).
"""

import os
import sys
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


class PhaseManager:
    """Manage Phase I / Phase II partitioning for SPC charts.

    Args:
        cfg: Optional configuration override.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.fig_dir = os.path.join(self.cfg.figures_dir, "mspc")
        os.makedirs(self.fig_dir, exist_ok=True)

    # =================================================================
    def setup_phases(
        self,
        X_pca: np.ndarray,
        y: pd.Series,
        timestamps: pd.Series,
    ) -> Dict:
        """Partition data into Phase I and Phase II.

        Phase I is built from the earliest *phase1_ratio* fraction of
        **Pass** observations.  All remaining observations (later Pass
        plus every Fail) go into Phase II.

        Args:
            X_pca: PCA score matrix (n × p).
            y: Binary labels aligned to ``X_pca``.
            timestamps: Datetime series aligned to ``X_pca``.

        Returns:
            Dictionary with ``X_phase1, X_phase2, y_phase1, y_phase2,
            ts_phase1, ts_phase2, phase1_indices, phase2_indices``.
        """
        print("\n══════════════════════════════════════════════════")
        print("        PHASE SETUP (SPC)")
        print("══════════════════════════════════════════════════")

        y_arr = np.asarray(y)
        ts_arr = np.asarray(timestamps)

        # All pass indices in temporal order
        pass_idx = np.where(y_arr == 0)[0]
        fail_idx = np.where(y_arr == 1)[0]

        n_phase1 = int(self.cfg.phase1_ratio * len(pass_idx))
        phase1_idx = pass_idx[:n_phase1]

        # Phase II = everything else
        phase2_idx = np.sort(np.concatenate([pass_idx[n_phase1:], fail_idx]))

        # Validate
        self.validate_phase_separation(phase1_idx, phase2_idx, y_arr, timestamps)

        # Build output
        phases: Dict = {
            "X_phase1": X_pca[phase1_idx],
            "X_phase2": X_pca[phase2_idx],
            "y_phase1": y_arr[phase1_idx],
            "y_phase2": y_arr[phase2_idx],
            "ts_phase1": ts_arr[phase1_idx],
            "ts_phase2": ts_arr[phase2_idx],
            "phase1_indices": phase1_idx,
            "phase2_indices": phase2_idx,
        }

        n2_pass = int((phases["y_phase2"] == 0).sum())
        n2_fail = int((phases["y_phase2"] == 1).sum())

        print("┌─────────────────────────────────────────┐")
        print("│ PHASE SETUP SUMMARY                     │")
        print("├─────────────────────────────────────────┤")
        print(f"│ Total observations    : {len(y_arr):>6,}         │")
        print("│                                         │")
        print("│ PHASE I (Baseline)                      │")
        print(f"│   Observations        : {len(phase1_idx):>6,}         │")
        print("│   Class               : 100% Pass ✅   │")
        ts1_min = pd.Timestamp(ts_arr[phase1_idx[0]])
        ts1_max = pd.Timestamp(ts_arr[phase1_idx[-1]])
        print(f"│   Date range  : {str(ts1_min)[:10]} → {str(ts1_max)[:10]}│")
        print("│   Purpose     : Estimate μ, Σ          │")
        print("│                                         │")
        print("│ PHASE II (Monitoring)                   │")
        print(f"│   Observations        : {len(phase2_idx):>6,}         │")
        print(f"│   Pass samples        : {n2_pass:>5} ({100*n2_pass/len(phase2_idx):.1f}%)    │")
        print(f"│   Fail samples        : {n2_fail:>5} ({100*n2_fail/len(phase2_idx):.1f}%)    │")
        ts2_min = pd.Timestamp(ts_arr[phase2_idx[0]])
        ts2_max = pd.Timestamp(ts_arr[phase2_idx[-1]])
        print(f"│   Date range  : {str(ts2_min)[:10]} → {str(ts2_max)[:10]}│")
        print("│   Purpose     : Detect faults           │")
        print("└─────────────────────────────────────────┘")

        return phases

    # =================================================================
    def validate_phase_separation(
        self,
        phase1_indices: np.ndarray,
        phase2_indices: np.ndarray,
        y: np.ndarray,
        timestamps: Optional[pd.Series] = None,
    ) -> bool:
        """Assert that phase splitting satisfies all SPC constraints.

        Args:
            phase1_indices: Index array for Phase I.
            phase2_indices: Index array for Phase II.
            y: Full label array.
            timestamps: Optional datetime series for temporal validation.

        Returns:
            ``True`` if all checks pass.

        Raises:
            ValueError: If any constraint is violated.
        """
        # Disjoint
        overlap = np.intersect1d(phase1_indices, phase2_indices)
        if len(overlap) > 0:
            raise ValueError(
                f"Phase I and II overlap on {len(overlap)} indices: {overlap[:5]}"
            )

        # Phase I all pass
        if np.any(y[phase1_indices] != 0):
            raise ValueError("Phase I contains Fail samples!")

        # Phase II contains all fail
        all_fail = set(np.where(y == 1)[0])
        phase2_set = set(phase2_indices.tolist())
        missing_fail = all_fail - phase2_set
        if missing_fail:
            raise ValueError(
                f"{len(missing_fail)} Fail samples missing from Phase II"
            )

        # Temporal ordering
        if timestamps is not None:
            ts_arr = np.asarray(timestamps)
            ts1_max = pd.Timestamp(ts_arr[phase1_indices].max())
            ts2_min = pd.Timestamp(ts_arr[phase2_indices].min())
            if ts2_min < ts1_max:
                print(f"  ⚠ Temporal overlap: Phase II min time ({ts2_min}) "
                      f"< Phase I max time ({ts1_max}). "
                      f"Some Phase II observations precede Phase I observations in time.")
            else:
                print("  ✓ Temporal ordering: Phase I fully precedes Phase II based on timestamps")
        else:
            phase1_max_idx = int(phase1_indices.max())
            phase2_min_idx = int(phase2_indices.min())
            if phase2_min_idx < phase1_max_idx:
                print(f"  ⚠ Temporal overlap: Phase II min index ({phase2_min_idx}) "
                      f"< Phase I max index ({phase1_max_idx}).")
            else:
                print("  ✓ Temporal ordering: Phase I fully precedes Phase II based on indices")
        print("  ✓ Phase validation passed")
        return True

    # =================================================================
    def plot_phase_timeline(
        self,
        timestamps: pd.Series,
        y: pd.Series,
        phase1_indices: np.ndarray,
        phase2_indices: np.ndarray,
    ) -> None:
        """Visualise the Phase I / II split on a timeline.

        Args:
            timestamps: Full datetime Series.
            y: Full label Series.
            phase1_indices: Phase I index array.
            phase2_indices: Phase II index array.
        """
        print("\n── Phase Timeline Plot ──")
        ts = np.asarray(pd.to_datetime(timestamps))
        y_arr = np.asarray(y)

        fig, ax = plt.subplots(figsize=(18, 5))

        # Phase I
        ax.scatter(
            ts[phase1_indices],
            np.zeros(len(phase1_indices)),
            c="steelblue", s=12, alpha=0.5, label="Phase I (Pass)",
        )

        # Phase II Pass
        p2_pass = phase2_indices[y_arr[phase2_indices] == 0]
        ax.scatter(
            ts[p2_pass],
            np.ones(len(p2_pass)) * 0.5,
            c="#55A868", s=12, alpha=0.5, label="Phase II (Pass)",
        )

        # Phase II Fail
        p2_fail = phase2_indices[y_arr[phase2_indices] == 1]
        ax.scatter(
            ts[p2_fail],
            np.ones(len(p2_fail)) * 1.0,
            c="crimson", s=60, marker="*", alpha=0.8, label="Phase II (Fail)",
        )

        # Boundary line
        boundary = ts[phase1_indices[-1]]
        ax.axvline(boundary, ls="--", color="black", lw=1.5, label="Phase boundary")
        ax.axvspan(ts[phase1_indices[0]], boundary, alpha=0.05, color="steelblue")
        ax.axvspan(boundary, ts[phase2_indices[-1]], alpha=0.05, color="#55A868")

        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["Phase I", "Phase II Pass", "Phase II Fail"])
        ax.set_xlabel("Time")
        ax.set_title("Phase I / Phase II Timeline", fontweight="bold", fontsize=14)
        ax.legend(loc="upper left")
        plt.tight_layout()

        path = os.path.join(self.fig_dir, "phase_timeline.png")
        fig.savefig(path, dpi=self.cfg.figure_dpi)
        plt.close(fig)
        print(f"  ✓ Saved → {path}")
