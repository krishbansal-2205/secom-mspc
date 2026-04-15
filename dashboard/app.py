"""
dashboard/app.py - Streamlit MSPC Quality Monitor
====================================================

Five-page interactive dashboard for real-time MSPC monitoring,
PCA exploration, ML model comparison, and alert management.

Launch with::

    streamlit run dashboard/app.py
"""

import os
import sys

import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import streamlit as st
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("Install streamlit and plotly: pip install streamlit plotly")

from config import config
from dashboard.chart_components import build_t2_chart, build_mewma_chart, build_class_pie
from dashboard.alert_system import AlertSystem

# ─────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SECOM MSPC Quality Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_mspc_results() -> pd.DataFrame:
    """Load saved MSPC monitoring results.

    Returns:
        DataFrame from ``outputs/tables/mspc_results.csv``.
    """
    path = os.path.join(config.tables_dir, "mspc_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    """Load saved ML model comparison table.

    Returns:
        DataFrame from ``outputs/tables/model_comparison.csv``.
    """
    path = os.path.join(config.tables_dir, "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_arl_table() -> pd.DataFrame:
    """Load saved ARL comparison table.

    Returns:
        DataFrame from ``outputs/tables/arl_comparison_table.csv``.
    """
    path = os.path.join(config.tables_dir, "arl_comparison_table.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def image_or_placeholder(path: str, caption: str = "") -> None:
    """Display an image if it exists, otherwise show a placeholder.

    Args:
        path: Absolute or relative path to a PNG image.
        caption: Optional caption.
    """
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"Image not found: {os.path.basename(path)}. Run the pipeline first.")


# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
st.sidebar.title("🏭 SECOM MSPC")
st.sidebar.markdown("**Quality Monitor Dashboard**")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Control Charts", "🔬 PCA Analysis",
     "🤖 ML Models", "🚨 Alert Center"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
show_labels = st.sidebar.checkbox("Show actual labels", value=True)
st.sidebar.markdown("### Pipeline Parameters")
st.sidebar.info(
    f"**MEWMA λ:** {config.mewma_lambda}\n\n"
    f"**Alpha (α):** {config.alpha}"
)

# System status
results_df = load_mspc_results()
if len(results_df) > 0:
    st.sidebar.success("✅ System Online")
else:
    st.sidebar.error("⚠ No data – run the pipeline first")


# ═════════════════════════════════════════════════════════════════════
# PAGE 1 – HOME
# ═════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🏠 SECOM MSPC – Overview")

    if len(results_df) == 0:
        st.warning("No MSPC results found. Please run `python main.py` first.")
    else:
        # KPI cards
        n_total = len(results_df)
        defect_rate = results_df["true_label"].mean() * 100 if "true_label" in results_df else 0
        t2_signals = int(results_df["t2_signal"].sum())
        mewma_signals = int(results_df["mewma_signal"].sum())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Wafers Monitored", f"{n_total:,}")
        col2.metric("Defect Rate", f"{defect_rate:.1f}%")
        col3.metric("T² Signals", str(t2_signals))
        col4.metric("MEWMA Signals", str(mewma_signals))

        st.markdown("---")

        # Mini T² chart (last 100)
        last_n = min(100, n_total)
        tail = results_df.tail(last_n)
        t2_vals = tail["t2_value"].values
        ucl = tail["t2_ucl"].values[0] if "t2_ucl" in tail else 0
        y_true = tail["true_label"].values if "true_label" in tail and show_labels else None
        fig = build_t2_chart(t2_vals, ucl, y_true, title=f"T² (last {last_n} observations)")
        st.plotly_chart(fig, use_container_width=True)

        # Class distribution
        if "true_label" in results_df:
            fig_pie = build_class_pie(results_df["true_label"].values)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Recent alerts
        alert_sys = AlertSystem()
        alert_sys.generate_alerts(results_df)
        alert_df = alert_sys.get_alert_df()
        if len(alert_df) > 0:
            st.subheader("Recent Alerts (last 10)")
            st.dataframe(alert_df.tail(10), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE 2 – CONTROL CHARTS
# ═════════════════════════════════════════════════════════════════════
elif page == "📊 Control Charts":
    st.title("📊 Control Charts")

    if len(results_df) == 0:
        st.warning("No data. Run the pipeline first.")
    else:
        y_true = results_df["true_label"].values if "true_label" in results_df and show_labels else None

        # T² chart
        st.subheader("Hotelling T² Chart")
        t2_vals = results_df["t2_value"].values
        ucl = results_df["t2_ucl"].values[0]
        fig_t2 = build_t2_chart(t2_vals, ucl, y_true)
        st.plotly_chart(fig_t2, use_container_width=True)

        # MEWMA chart
        st.subheader("MEWMA Chart")
        mewma_vals = results_df["mewma_value"].values
        mewma_ucl = results_df["mewma_ucl"].values
        fig_mewma = build_mewma_chart(mewma_vals, mewma_ucl, y_true)
        st.plotly_chart(fig_mewma, use_container_width=True)

        # Download
        csv = results_df.to_csv(index=False)
        st.download_button("📥 Download MSPC Data (CSV)", csv,
                           "mspc_results.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════════
# PAGE 3 – PCA ANALYSIS
# ═════════════════════════════════════════════════════════════════════
elif page == "🔬 PCA Analysis":
    st.title("🔬 PCA Analysis")

    pca_dir = os.path.join(config.figures_dir, "pca")

    image_or_placeholder(os.path.join(pca_dir, "pca_scree_and_scores.png"),
                         "Scree & Score Plots")

    image_or_placeholder(os.path.join(pca_dir, "pca_loading_heatmap.png"),
                         "Loading Heatmap")

    biplot_path = os.path.join(pca_dir, "pca_3d_biplot.html")
    if os.path.exists(biplot_path):
        st.subheader("Interactive 3-D Score Plot")
        with open(biplot_path, "r", encoding="utf-8") as fh:
            html_content = fh.read()
        st.components.v1.html(html_content, height=700, scrolling=True)
    else:
        st.info("3-D biplot not found. Run the pipeline first.")

    image_or_placeholder(os.path.join(pca_dir, "pca_loading_bars.png"),
                         "Loading Bar Charts")


# ═════════════════════════════════════════════════════════════════════
# PAGE 4 – ML MODELS
# ═════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Models":
    st.title("🤖 ML Model Results")

    model_df = load_model_comparison()
    if len(model_df) == 0:
        st.warning("No model results. Run the pipeline first.")
    else:
        st.subheader("Model Comparison")
        st.dataframe(model_df, use_container_width=True)

        model_dir = os.path.join(config.figures_dir, "models")
        image_or_placeholder(os.path.join(model_dir, "roc_curves.png"), "ROC Curves")
        image_or_placeholder(os.path.join(model_dir, "confusion_matrices.png"), "Confusion Matrices")
        image_or_placeholder(os.path.join(model_dir, "threshold_analysis.png"), "Threshold Analysis")
        image_or_placeholder(os.path.join(model_dir, "feature_importance.png"), "Feature Importance")


# ═════════════════════════════════════════════════════════════════════
# PAGE 5 – ALERT CENTER
# ═════════════════════════════════════════════════════════════════════
elif page == "🚨 Alert Center":
    st.title("🚨 Alert Center")

    if len(results_df) == 0:
        st.warning("No data. Run the pipeline first.")
    else:
        alert_sys = AlertSystem()
        alert_sys.generate_alerts(results_df)
        summary = alert_sys.get_summary()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Alerts", summary["total"])
        col2.metric("True Positives", summary["TP"])
        col3.metric("False Positives", summary["FP"])
        col4.metric("Critical", summary["levels"].get("CRITICAL", 0))

        alert_df = alert_sys.get_alert_df()
        if len(alert_df) > 0:
            st.subheader("Alert Log")

            # Filter by level
            level_filter = st.multiselect(
                "Filter by level",
                ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            )
            filtered = alert_df[alert_df["alert_level"].isin(level_filter)]
            st.dataframe(filtered, use_container_width=True, height=400)

            # Expandable details for each alert
            st.subheader("Alert Details")
            for i, row in filtered.head(20).iterrows():
                with st.expander(f"Obs #{row['observation_id']} – {row['alert_level']}"):
                    st.write(f"**T² Value:** {row['t2_value']:.2f}  (UCL = {row['t2_ucl']:.2f})")
                    st.write(f"**MEWMA Value:** {row['mewma_value']:.2f}")
                    if row["true_label"] is not None:
                        label = "FAIL ❌" if row["true_label"] == 1 else "PASS ✅"
                        st.write(f"**True Label:** {label}")
                    st.markdown("**Recommended Actions:**")
                    st.markdown("1. Inspect process parameters\n"
                                "2. Check sensor calibration\n"
                                "3. Review maintenance log")
        else:
            st.success("No alerts – process is in control! ✅")

        # ARL table
        st.subheader("ARL Comparison")
        arl_df = load_arl_table()
        if len(arl_df) > 0:
            st.dataframe(arl_df, use_container_width=True)
        else:
            st.info("ARL table not generated yet.")


# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("SECOM MSPC v1.0 · Industry 4.0")

if __name__ == "__main__":
    print("=" * 60)
    print("  To launch the dashboard, run:")
    print("    streamlit run dashboard/app.py")
    print("=" * 60)
