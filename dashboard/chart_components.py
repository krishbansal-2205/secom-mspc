"""
dashboard/chart_components.py - Reusable Chart Widgets
========================================================

Plotly-based chart builders used by the Streamlit dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional


def build_t2_chart(
    t2_values: np.ndarray,
    ucl: float,
    y_true: Optional[np.ndarray] = None,
    title: str = "Hotelling T² Control Chart",
) -> go.Figure:
    """Build an interactive Plotly T² control chart.

    Args:
        t2_values: Array of T² statistics.
        ucl: Upper control limit.
        y_true: Optional true labels for colour coding.
        title: Chart title.

    Returns:
        Plotly :class:`Figure`.
    """
    n = len(t2_values)
    x = list(range(n))
    signals = t2_values > ucl

    fig = go.Figure()

    # Main line
    fig.add_trace(go.Scatter(
        x=x, y=t2_values, mode="lines",
        line=dict(color="steelblue", width=1),
        name="T²",
    ))

    # UCL
    fig.add_hline(y=ucl, line_dash="dash", line_color="red",
                  annotation_text=f"UCL = {ucl:.2f}")

    # Signal markers
    if y_true is not None:
        y_arr = np.asarray(y_true)
        tp = signals & (y_arr == 1)
        fp = signals & (y_arr == 0)
        fn = (~signals) & (y_arr == 1)

        if tp.any():
            fig.add_trace(go.Scatter(
                x=np.array(x)[tp].tolist(), y=t2_values[tp].tolist(),
                mode="markers", marker=dict(color="red", size=10, symbol="star"),
                name=f"TP ({tp.sum()})",
            ))
        if fp.any():
            fig.add_trace(go.Scatter(
                x=np.array(x)[fp].tolist(), y=t2_values[fp].tolist(),
                mode="markers", marker=dict(color="orange", size=8, symbol="triangle-up"),
                name=f"FP ({fp.sum()})",
            ))
        if fn.any():
            fig.add_trace(go.Scatter(
                x=np.array(x)[fn].tolist(), y=t2_values[fn].tolist(),
                mode="markers", marker=dict(color="purple", size=8, symbol="triangle-down"),
                name=f"FN ({fn.sum()})",
            ))

    fig.update_layout(
        title=title, xaxis_title="Observation", yaxis_title="T²",
        template="plotly_dark", height=400,
    )
    return fig


def build_mewma_chart(
    mewma_values: np.ndarray,
    ucl_array: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    title: str = "MEWMA Control Chart",
) -> go.Figure:
    """Build an interactive Plotly MEWMA chart.

    Args:
        mewma_values: MEWMA T² statistics.
        ucl_array: Time-varying UCL array.
        y_true: Optional true labels.
        title: Chart title.

    Returns:
        Plotly :class:`Figure`.
    """
    n = len(mewma_values)
    x = list(range(n))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=mewma_values.tolist(), mode="lines",
        line=dict(color="darkorange", width=1), name="MEWMA T²",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=ucl_array.tolist(), mode="lines",
        line=dict(color="red", width=1, dash="dash"), name="UCL",
    ))

    if y_true is not None:
        fail_idx = np.where(np.asarray(y_true) == 1)[0]
        for idx in fail_idx:
            if idx < n:
                fig.add_vrect(
                    x0=idx - 0.5, x1=idx + 0.5,
                    fillcolor="red", opacity=0.08, line_width=0,
                )

    fig.update_layout(
        title=title, xaxis_title="Observation", yaxis_title="T² MEWMA",
        template="plotly_dark", height=400,
    )
    return fig


def build_class_pie(y: np.ndarray) -> go.Figure:
    """Pie chart of class distribution.

    Args:
        y: Binary label array.

    Returns:
        Plotly :class:`Figure`.
    """
    n_pass = int((y == 0).sum())
    n_fail = int((y == 1).sum())
    fig = go.Figure(go.Pie(
        labels=["Pass", "Fail"], values=[n_pass, n_fail],
        marker_colors=["steelblue", "crimson"],
        hole=0.4,
    ))
    fig.update_layout(
        title="Class Distribution", template="plotly_dark", height=300,
    )
    return fig
