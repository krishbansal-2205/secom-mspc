"""
main.py - Master Pipeline Runner
===================================

Executes the complete SECOM MSPC pipeline end-to-end:

    python main.py [--skip-download] [--skip-processed]
                   [--phases-only] [--models-only]

Every stage prints progress, saves outputs, and logs to
``outputs/logs/run.log``.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# ── project root on path ────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import config


# ═════════════════════════════════════════════════════════════════════
# Setup helpers
# ═════════════════════════════════════════════════════════════════════
def setup_logging() -> logging.Logger:
    """Create a logger that writes to both console and file.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    os.makedirs(config.logs_dir, exist_ok=True)
    logger = logging.getLogger("secom_mspc")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(os.path.join(config.logs_dir, "run.log"), mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def create_output_directories() -> None:
    """Create every output directory listed in the config."""
    dirs = [
        os.path.join(config.figures_dir, "eda"),
        os.path.join(config.figures_dir, "pca"),
        os.path.join(config.figures_dir, "mspc"),
        os.path.join(config.figures_dir, "models"),
        os.path.join(config.figures_dir, "quality"),
        os.path.join(config.figures_dir, "preprocessing"),
        os.path.join(config.figures_dir, "normality"),
        os.path.join(config.figures_dir, "correlation"),
        config.models_dir,
        config.reports_dir,
        config.tables_dir,
        config.logs_dir,
        config.raw_data_dir,
        config.processed_data_dir,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_run_parameters() -> None:
    """Persist all configuration parameters to JSON."""
    info = config.to_dict()
    info["run_timestamp"] = datetime.now().isoformat()
    info["python_version"] = sys.version

    try:
        import sklearn
        info["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import numpy
        info["numpy_version"] = numpy.__version__
    except ImportError:
        pass

    path = os.path.join(config.reports_dir, "run_parameters.json")
    with open(path, "w") as fh:
        json.dump(info, fh, indent=2, default=str)
    print(f"  ✓ Run parameters saved → {path}")


# ═════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════
def run_pipeline(
    skip_download: bool = False,
    skip_if_processed: bool = False,
    phases_only: bool = False,
    models_only: bool = False,
) -> dict:
    """Execute the full SECOM MSPC pipeline.

    Args:
        skip_download: If ``True``, skip the dataset download step.
        skip_if_processed: If ``True``, load cached processed data.
        phases_only: Run only MSPC phases (skip ML).
        models_only: Run only ML models (skip MSPC).

    Returns:
        Dictionary with all results from every stage.
    """
    t_start = time.time()

    print("┌─────────────────────────────────────────────────────┐")
    print("│ SECOM MSPC PROJECT – MASTER PIPELINE                │")
    print(f"│ Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<40}│")
    print("└─────────────────────────────────────────────────────┘")

    # ── Phase 0: Setup ───────────────────────────────────────────────
    logger = setup_logging()
    create_output_directories()
    save_run_parameters()
    logger.info("Pipeline started")

    np.random.seed(config.random_seed)

    all_results: dict = {"config": config.to_dict()}

    # ── Phase 1: Data Loading ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 1 – DATA LOADING")
    print("=" * 60)
    from data.loader import SECOMDataLoader

    loader = SECOMDataLoader()
    if not skip_download:
        loader.download_data()

    X, y, df_master = loader.load_data()
    timestamps = df_master["timestamp"]
    df_master = loader.create_time_features(df_master)
    logger.info(f"Data loaded: {X.shape}")

    # ── Phase 2: Data Quality ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2 – DATA QUALITY ASSESSMENT")
    print("=" * 60)
    from preprocessing.quality_checker import DataQualityChecker

    checker = DataQualityChecker()
    quality_report = checker.run_full_assessment(X, y)
    all_results["quality_report"] = quality_report
    logger.info("Quality assessment complete")

    # ── Phase 3: Preprocessing ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 3 – PREPROCESSING")
    print("=" * 60)
    from preprocessing.cleaner import SECOMCleaner

    cleaner = SECOMCleaner()
    X_processed = cleaner.fit_transform(X.copy(), y)
    cleaner.save(os.path.join(config.models_dir, "cleaner.pkl"))
    all_results["preprocessing_report"] = cleaner.get_preprocessing_report()
    logger.info(f"Preprocessing complete: {X.shape} → {X_processed.shape}")

    # ── Phase 4: EDA ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 4 – EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    from statistical_analysis.eda import ExploratoryDataAnalysis

    eda = ExploratoryDataAnalysis(X_processed, y, timestamps)
    eda.run_full_eda()
    logger.info("EDA complete")

    # ── Phase 5: PCA ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 5 – PCA DIMENSIONALITY REDUCTION")
    print("=" * 60)
    from dimensionality_reduction.pca_engine import SECOMPCAEngine

    pca_engine = SECOMPCAEngine()
    X_pca = pca_engine.fit_transform(X_processed, y)
    pca_engine.plot_scree_plot()
    feature_names_processed = list(X_processed.columns)
    pca_engine.plot_loading_heatmap(feature_names_processed)
    pca_engine.plot_biplot_3d(y)
    pca_engine.plot_loading_bar_per_component(feature_names_processed, n_components=5)
    pca_engine.save(os.path.join(config.models_dir, "pca_engine.pkl"))
    all_results["pca_summary"] = {
        "n_components": pca_engine.n_components,
        "variance_retained": float(pca_engine.cumulative_variance[pca_engine.n_components - 1]),
    }
    logger.info(f"PCA complete: {X_processed.shape[1]}D → {X_pca.shape[1]}D")

    pc_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]

    if not models_only:
        # ── Phase 6: Phase Setup ─────────────────────────────────────
        print("\n" + "=" * 60)
        print("  PHASE 6 – SPC PHASE SETUP")
        print("=" * 60)
        from mspc.phase_manager import PhaseManager

        phase_mgr = PhaseManager()
        phases = phase_mgr.setup_phases(X_pca, y, timestamps)
        phase_mgr.validate_phase_separation(
            phases["phase1_indices"], phases["phase2_indices"], np.asarray(y)
        )
        phase_mgr.plot_phase_timeline(
            timestamps, y, phases["phase1_indices"], phases["phase2_indices"]
        )
        logger.info(f"Phases set: Phase I={len(phases['phase1_indices'])}, "
                     f"Phase II={len(phases['phase2_indices'])}")

        # ── Phase 7: Hotelling T² ────────────────────────────────────
        print("\n" + "=" * 60)
        print("  PHASE 7 – HOTELLING T² CHART")
        print("=" * 60)
        from mspc.hotelling_t2 import HotellingT2Chart

        t2_chart = HotellingT2Chart()
        t2_phase1 = t2_chart.fit_phase1(phases["X_phase1"])
        t2_results = t2_chart.monitor_phase2(
            phases["X_phase2"], phases["y_phase2"]
        )
        t2_chart.plot_t2_chart(
            t2_results["t2_values"], t2_results["ucl"],
            y_true=phases["y_phase2"], phase="II",
        )
        t2_chart.save(os.path.join(config.models_dir, "t2_chart.pkl"))
        logger.info(f"T² chart: {t2_results['signals'].sum()} signals")

        # ── Phase 8: MEWMA ───────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  PHASE 8 – MEWMA CHART")
        print("=" * 60)
        from mspc.mewma import MEWMAChart

        mewma = MEWMAChart(lam=config.mewma_lambda)
        mewma.fit(phases["X_phase1"])
        mewma_results = mewma.monitor(phases["X_phase2"])
        mewma.plot_mewma_chart(mewma_results, phases["y_phase2"])
        mewma.compare_sensitivity(phases["X_phase2"])
        mewma.save(os.path.join(config.models_dir, "mewma_chart.pkl"))
        logger.info(f"MEWMA chart: {mewma_results['signals'].sum()} signals")

        # ── Phase 9: Combined MSPC ───────────────────────────────────
        print("\n" + "=" * 60)
        print("  PHASE 9 – COMBINED MSPC SYSTEM")
        print("=" * 60)
        from mspc.combined_mspc import CombinedMSPCSystem

        combined = CombinedMSPCSystem()
        combined.fit(phases["X_phase1"])
        results_df = combined.monitor(phases["X_phase2"], phases["y_phase2"])
        mspc_perf = combined.generate_performance_report(
            results_df, phases["y_phase2"]
        )
        combined.plot_combined_dashboard(results_df, phases["y_phase2"])
        all_results["mspc_performance"] = mspc_perf
        logger.info(
            f"Combined MSPC: Sens={mspc_perf.get('Combined', {}).get('sensitivity', 'N/A')}, "
            f"Spec={mspc_perf.get('Combined', {}).get('specificity', 'N/A')}"
        )

        # ── Phase 10: ARL Simulation ─────────────────────────────────
        print("\n" + "=" * 60)
        print("  PHASE 10 – ARL SIMULATION")
        print("=" * 60)
        from mspc.arl_simulator import ARLSimulator

        arl_sim = ARLSimulator()
        # Use fewer simulations and fewer shift points for speed
        arl_shifts = [0.0, 0.5, 1.0, 2.0, 3.0]
        arl_table = arl_sim.simulate_arl_table(
            t2_chart, mewma, shift_sizes=arl_shifts, n_sim=1000
        )
        arl_sim.plot_arl_curves(arl_table)
        all_results["arl_table"] = arl_table
        logger.info("ARL simulation complete")

        # ── Phase 11: Fault Diagnosis Example ────────────────────────
        print("\n" + "=" * 60)
        print("  PHASE 11 – FAULT DIAGNOSIS EXAMPLE")
        print("=" * 60)
        from mspc.fault_diagnosis import FaultDiagnosisEngine

        diag = FaultDiagnosisEngine()
        # Find first TP signal
        tp_mask = results_df["signal_type"] == "TP" if "signal_type" in results_df.columns else results_df["combined_signal"]
        tp_indices = results_df[tp_mask].index.tolist()
        if tp_indices:
            first_tp = tp_indices[0]
            obs_id = int(results_df.loc[first_tp, "observation_id"])
            x_signal = phases["X_phase2"][obs_id]
            t2_val = float(results_df.loc[first_tp, "t2_value"])
            fd_result = diag.diagnose_signal(
                x_signal, phases["X_phase1"],
                t2_val, t2_chart.ucl_phase2_F,
                pc_names,
                original_feature_names=feature_names_processed,
                pca_loadings=pca_engine.loadings,
            )
            all_results["fault_diagnosis"] = fd_result
            logger.info("Fault diagnosis example generated")
        else:
            print("  ⚠ No TP signal found for fault diagnosis example.")
            logger.info("No TP signal for fault diagnosis")

    if not phases_only:
        # ── Phase 12: ML Models ──────────────────────────────────────
        print("\n" + "=" * 60)
        print("  PHASE 12 – ML MODELS")
        print("=" * 60)
        from predictive_model.model_trainer import SECOMModelTrainer
        from predictive_model.model_evaluator import SECOMModelEvaluator

        trainer = SECOMModelTrainer()
        data = trainer.prepare_data(X_pca, y)
        models = trainer.train_all_models(
            data["X_train_smote"], data["y_train_smote"]
        )
        cv_results = trainer.cross_validate_all(
            models, data["X_train"], data["y_train"]
        )

        evaluator = SECOMModelEvaluator()
        eval_results = evaluator.evaluate_all_models(
            models, data["X_test"], data["y_test"]
        )
        evaluator.plot_roc_curves(models, data["X_test"], data["y_test"])
        evaluator.plot_precision_recall_curves(models, data["X_test"], data["y_test"])
        evaluator.plot_confusion_matrices(models, data["X_test"], data["y_test"])

        # Find best model
        best_model_name = eval_results.sort_values("auc_roc", ascending=False).iloc[0]["model"]
        best_model = models[best_model_name]
        evaluator.plot_threshold_analysis(best_model, data["X_test"], data["y_test"])
        evaluator.plot_feature_importance(
            best_model, pc_names, pca_engine, feature_names_processed
        )

        all_results["ml_comparison"] = eval_results
        all_results["best_model_name"] = best_model_name
        logger.info(f"ML models: best = {best_model_name}, "
                     f"AUC = {eval_results.sort_values('auc_roc', ascending=False).iloc[0]['auc_roc']}")

    # ── Phase 13: Report Generation ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 13 – REPORT GENERATION")
    print("=" * 60)
    from visualization.report_generator import ReportGenerator

    reporter = ReportGenerator()
    reporter.generate_html_report(all_results)
    logger.info("HTML report generated")

    # ── Phase 14: Final Summary ──────────────────────────────────────
    elapsed = time.time() - t_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    n_figs = sum(
        len([f for f in files if f.endswith(".png")])
        for _, _, files in os.walk(config.figures_dir)
    )
    n_models = len([f for f in os.listdir(config.models_dir) if f.endswith(".pkl")]) if os.path.isdir(config.models_dir) else 0
    n_tables = len([f for f in os.listdir(config.tables_dir) if f.endswith(".csv")]) if os.path.isdir(config.tables_dir) else 0

    pca_info = all_results.get("pca_summary", {})
    mspc_perf_final = all_results.get("mspc_performance", {})
    comb = mspc_perf_final.get("Combined", {})
    best_ml = all_results.get("best_model_name", "N/A")
    ml_df = all_results.get("ml_comparison")

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║             PIPELINE COMPLETE – FINAL SUMMARY            ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║ Total Runtime         : {minutes:>2d} minutes {seconds:>2d} seconds            ║")
    print("║                                                          ║")
    print("║ DATA:                                                    ║")
    print(f"║   Original features   : 591                              ║")
    print(f"║   Final features      : {X_processed.shape[1]:<4d} (after preprocessing)       ║")
    print(f"║   PCA components      : {pca_info.get('n_components', '?'):<4} "
          f"({100*pca_info.get('variance_retained', 0):.0f}% variance retained)    ║")
    print("║                                                          ║")
    if comb:
        print("║ MSPC PERFORMANCE:                                        ║")
        print(f"║   T² Sensitivity      : {100*mspc_perf_final.get('T2', {}).get('sensitivity', 0):.1f}%                           ║")
        print(f"║   T² Specificity      : {100*mspc_perf_final.get('T2', {}).get('specificity', 0):.1f}%                           ║")
        print(f"║   MEWMA Sensitivity   : {100*mspc_perf_final.get('MEWMA', {}).get('sensitivity', 0):.1f}%                           ║")
        print(f"║   Combined AUC-ROC    : {comb.get('auc_roc', 0):.3f}                           ║")
    print("║                                                          ║")
    if ml_df is not None and len(ml_df) > 0:
        best_row = ml_df.sort_values("auc_roc", ascending=False).iloc[0]
        print("║ ML MODEL PERFORMANCE:                                    ║")
        print(f"║   Best Model          : {best_ml:<33}║")
        print(f"║   AUC-ROC             : {best_row['auc_roc']:.3f}                           ║")
        print(f"║   Recall (defects)    : {100*best_row.get('recall', 0):.1f}%                           ║")
    print("║                                                          ║")
    print("║ OUTPUTS SAVED:                                           ║")
    print(f"║   Figures             : {n_figs} PNG files                     ║")
    print(f"║   HTML Report         : secom_mspc_complete_report.html  ║")
    print(f"║   Models              : {n_models} .pkl files                     ║")
    print(f"║   Tables              : {n_tables} .csv files                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("\n  To launch dashboard: streamlit run dashboard/app.py\n")

    logger.info("Pipeline completed successfully")
    return all_results


# ═════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════
def main() -> None:
    """Parse CLI arguments and launch the pipeline."""
    parser = argparse.ArgumentParser(
        description="SECOM MSPC Pipeline – Master Runner"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip raw data download (files must already exist).",
    )
    parser.add_argument(
        "--skip-processed", action="store_true",
        help="Use cached processed data if available.",
    )
    parser.add_argument(
        "--phases-only", action="store_true",
        help="Run only the MSPC (Phase I / II) stages.",
    )
    parser.add_argument(
        "--models-only", action="store_true",
        help="Run only the ML model training stages.",
    )
    args = parser.parse_args()

    run_pipeline(
        skip_download=args.skip_download,
        skip_if_processed=args.skip_processed,
        phases_only=args.phases_only,
        models_only=args.models_only,
    )


if __name__ == "__main__":
    main()
