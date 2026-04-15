"""
config.py - Central Configuration for the SECOM MSPC Project
=============================================================

All parameters used across the project are defined here in a single
dataclass for reproducibility and easy tuning.
"""

import os
from dataclasses import dataclass, field
from typing import List

# ──────────────────────────────────────────────────────────────────────
# Project root directory (resolved relative to this file)
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class SECOMConfig:
    """Complete configuration for the SECOM MSPC pipeline.

    Every tunable parameter is documented and grouped by pipeline stage.
    Changing values here propagates automatically to every module.
    """

    # ── Data parameters ──────────────────────────────────────────────
    secom_data_url: str = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "secom/secom.data"
    )
    secom_labels_url: str = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "secom/secom_labels.data"
    )
    raw_data_dir: str = os.path.join(PROJECT_ROOT, "data", "raw")
    processed_data_dir: str = os.path.join(PROJECT_ROOT, "data", "processed")
    random_seed: int = 42

    # ── Preprocessing parameters ─────────────────────────────────────
    missing_threshold: float = 0.50
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    outlier_iqr_multiplier: float = 3.0
    imputation_strategy: str = "median"
    scaler_type: str = "robust"

    # ── PCA parameters ───────────────────────────────────────────────
    variance_targets: List[float] = field(
        default_factory=lambda: [0.80, 0.90, 0.95, 0.99]
    )
    selected_variance: float = 0.95
    max_components_show: int = 50

    # ── MSPC parameters ─────────────────────────────────────────────
    alpha: float = 0.0027          # ≈ 3-sigma false-alarm rate
    phase1_ratio: float = 0.60    # fraction of PASS samples for Phase I
    phase1_pass_only: bool = True
    mewma_lambda: float = 0.10
    mewma_L: float = 3.50
    cusum_k: float = 0.50
    cusum_h: float = 4.00
    ucl_method_phase2: str = "F"   # "F" or "chi2"

    # ── ARL parameters ──────────────────────────────────────────────
    n_simulations: int = 10_000
    shift_sizes: List[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    )
    target_arl0: int = 370

    # ── ML model parameters ─────────────────────────────────────────
    test_size: float = 0.30
    cv_folds: int = 5
    smote_k_neighbors: int = 5
    rf_n_estimators: int = 300
    rf_max_depth: int = 10
    gb_n_estimators: int = 200
    gb_learning_rate: float = 0.05
    scoring_metric: str = "roc_auc"

    # ── Output parameters ────────────────────────────────────────────
    figures_dir: str = os.path.join(PROJECT_ROOT, "outputs", "figures")
    models_dir: str = os.path.join(PROJECT_ROOT, "outputs", "models")
    reports_dir: str = os.path.join(PROJECT_ROOT, "outputs", "reports")
    tables_dir: str = os.path.join(PROJECT_ROOT, "outputs", "tables")
    logs_dir: str = os.path.join(PROJECT_ROOT, "outputs", "logs")
    figure_dpi: int = 300
    figure_format: str = "png"

    def to_dict(self) -> dict:
        """Serialize all configuration to a plain dictionary."""
        import dataclasses
        return dataclasses.asdict(self)


# ──────────────────────────────────────────────────────────────────────
# Singleton instance used throughout the project
# ──────────────────────────────────────────────────────────────────────
config = SECOMConfig()
