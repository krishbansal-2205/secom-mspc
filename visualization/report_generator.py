"""
visualization/report_generator.py - Automated HTML Report
============================================================

Generates a complete, self-contained HTML report with embedded
base-64 images and Bootstrap styling.
"""

import base64
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


class ReportGenerator:
    """Generate a production-quality HTML report.

    Args:
        cfg: Optional configuration override.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config

    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _embed_image(path: str) -> str:
        """Read an image and return a base-64 data URI.

        Args:
            path: Absolute or relative path to a PNG image.

        Returns:
            Base-64 data URI string, or a placeholder on error.
        """
        try:
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            return f"data:image/png;base64,{b64}"
        except FileNotFoundError:
            return ""

    # ─────────────────────────────────────────────────────────────────
    def _img_tag(self, relpath: str, caption: str = "") -> str:
        """Build an ``<img>`` tag with embedded base-64 data.

        Args:
            relpath: Path relative to the project root.
            caption: Optional caption shown below the image.

        Returns:
            HTML string.
        """
        uri = self._embed_image(relpath)
        if not uri:
            return f'<p class="text-muted">Image not found: {relpath}</p>'
        html = f'<img src="{uri}" class="img-fluid mb-2" alt="{caption}">'
        if caption:
            html += f'<p class="text-muted small">{caption}</p>'
        return html

    # ─────────────────────────────────────────────────────────────────
    def generate_html_report(self, all_results: Dict[str, Any]) -> None:
        """Build and save the complete HTML report.

        Args:
            all_results: Dictionary collected from every pipeline stage.
                Expected keys (all optional – missing items are skipped):
                ``quality_report``, ``preprocessing_report``,
                ``pca_summary``, ``mspc_performance``, ``ml_comparison``,
                ``arl_table``, ``fault_diagnosis``, ``config``.
        """
        print("\n══════════════════════════════════════════════════")
        print("        HTML REPORT GENERATION")
        print("══════════════════════════════════════════════════")

        figs = self.cfg.figures_dir
        quality_dir = os.path.join(figs, "quality")
        eda_dir = os.path.join(figs, "eda")
        pca_dir = os.path.join(figs, "pca")
        mspc_dir = os.path.join(figs, "mspc")
        model_dir = os.path.join(figs, "models")
        preproc_dir = os.path.join(figs, "preprocessing")

        # ── Build HTML ───────────────────────────────────────────────
        html_parts: list = []
        html_parts.append(self._header())

        # Section 1 – Executive Summary
        html_parts.append('<section id="executive-summary">')
        html_parts.append("<h2>1. Executive Summary</h2>")
        html_parts.append("<ul>")
        html_parts.append("<li><strong>Objective:</strong> Implement a Multivariate Statistical Process "
                          "Control system for semiconductor wafer quality prediction.</li>")
        html_parts.append("<li><strong>Dataset:</strong> SECOM – 1,567 wafer runs, 591 sensors, "
                          "~6.7 % defect rate.</li>")

        mspc_perf = all_results.get("mspc_performance", {})
        combined = mspc_perf.get("Combined", {})
        if combined:
            html_parts.append(f"<li><strong>MSPC Sensitivity:</strong> {combined.get('sensitivity', 'N/A')}</li>")
            html_parts.append(f"<li><strong>MSPC Specificity:</strong> {combined.get('specificity', 'N/A')}</li>")

        ml_comp = all_results.get("ml_comparison")
        if ml_comp is not None and len(ml_comp) > 0:
            best = ml_comp.iloc[0] if hasattr(ml_comp, "iloc") else {}
            html_parts.append(f"<li><strong>Best ML Model:</strong> "
                              f"{best.get('model', 'N/A')} (AUC = {best.get('auc_roc', 'N/A')})</li>")

        html_parts.append("<li><strong>Business Impact:</strong> Early detection of process faults "
                          "can prevent $50 K–$500 K per failed wafer batch.</li>")
        html_parts.append("</ul></section><hr>")

        # Section 2 – Dataset Overview
        html_parts.append('<section id="dataset-overview">')
        html_parts.append("<h2>2. Dataset Overview</h2>")
        html_parts.append(self._img_tag(os.path.join(quality_dir, "missing_value_analysis.png"),
                                        "Missing-value analysis"))
        html_parts.append(self._img_tag(os.path.join(eda_dir, "time_series_quality.png"),
                                        "Temporal quality overview"))
        html_parts.append("</section><hr>")

        # Section 3 – Preprocessing
        html_parts.append('<section id="preprocessing">')
        html_parts.append("<h2>3. Data Preprocessing</h2>")
        prep = all_results.get("preprocessing_report", {})
        log = prep.get("feature_log", {})
        if log:
            html_parts.append("<table class='table table-sm table-bordered'><tr>"
                              "<th>Step</th><th>Features</th></tr>")
            for step, cnt in log.items():
                html_parts.append(f"<tr><td>{step}</td><td>{cnt}</td></tr>")
            html_parts.append("</table>")
        html_parts.append(self._img_tag(os.path.join(preproc_dir, "correlation_before_after.png"),
                                        "Correlation before/after filtering"))
        html_parts.append("</section><hr>")

        # Section 4 – PCA
        html_parts.append('<section id="pca">')
        html_parts.append("<h2>4. PCA Results</h2>")
        html_parts.append(self._img_tag(os.path.join(pca_dir, "pca_scree_and_scores.png"),
                                        "Scree and score plots"))
        html_parts.append(self._img_tag(os.path.join(pca_dir, "pca_loading_heatmap.png"),
                                        "Loading heatmap"))
        biplot_path = os.path.join(pca_dir, "pca_3d_biplot.html")
        if os.path.exists(biplot_path):
            html_parts.append(f'<p><a href="{biplot_path}" target="_blank">'
                              "Open interactive 3-D biplot →</a></p>")
        html_parts.append("</section><hr>")

        # Section 5 – MSPC
        html_parts.append('<section id="mspc">')
        html_parts.append("<h2>5. MSPC Results</h2>")
        html_parts.append(self._img_tag(os.path.join(mspc_dir, "phase_timeline.png"),
                                        "Phase I / Phase II timeline"))
        html_parts.append(self._img_tag(os.path.join(mspc_dir, "hotelling_t2_chart_phaseII.png"),
                                        "Hotelling T² control chart"))
        html_parts.append(self._img_tag(os.path.join(mspc_dir, "mewma_chart.png"),
                                        "MEWMA control chart"))
        html_parts.append(self._img_tag(os.path.join(mspc_dir, "combined_mspc_dashboard.png"),
                                        "Combined MSPC dashboard"))
        if mspc_perf:
            html_parts.append("<h4>Performance comparison</h4>")
            html_parts.append("<pre>" + json.dumps(mspc_perf, indent=2) + "</pre>")
        html_parts.append(self._img_tag(os.path.join(mspc_dir, "arl_curves_comparison.png"),
                                        "ARL comparison"))
        html_parts.append("</section><hr>")

        # Section 6 – ML Models
        html_parts.append('<section id="ml-models">')
        html_parts.append("<h2>6. ML Model Results</h2>")
        if ml_comp is not None and isinstance(ml_comp, pd.DataFrame):
            html_parts.append(ml_comp.to_html(classes="table table-sm table-bordered", index=False))
        html_parts.append(self._img_tag(os.path.join(model_dir, "roc_curves.png"),
                                        "ROC curves"))
        html_parts.append(self._img_tag(os.path.join(model_dir, "confusion_matrices.png"),
                                        "Confusion matrices"))
        html_parts.append(self._img_tag(os.path.join(model_dir, "feature_importance.png"),
                                        "Feature importance"))
        html_parts.append("</section><hr>")

        # Section 7 – Fault Diagnosis
        html_parts.append('<section id="fault-diagnosis">')
        html_parts.append("<h2>7. Fault Diagnosis Example</h2>")
        html_parts.append(self._img_tag(os.path.join(mspc_dir, "fault_diagnosis_contribution.png"),
                                        "T² contribution chart"))
        fd = all_results.get("fault_diagnosis", {})
        if fd:
            html_parts.append("<pre>" + json.dumps(fd, indent=2, default=str) + "</pre>")
        html_parts.append("</section><hr>")

        # Section 8 – Conclusions
        html_parts.append('<section id="conclusions">')
        html_parts.append("<h2>8. Conclusions &amp; Recommendations</h2>")
        html_parts.append("<ol>")
        html_parts.append("<li>The MEWMA chart complements T² by detecting smaller, "
                          "sustained process shifts earlier.</li>")
        html_parts.append("<li>Combining MSPC with ML models provides both interpretable "
                          "SPC charts and predictive defect classification.</li>")
        html_parts.append("<li>Fault diagnosis pinpoints which sensors contribute most to "
                          "each out-of-control signal, guiding maintenance teams.</li>")
        html_parts.append("<li>Deploy the Streamlit dashboard for real-time monitoring of "
                          "incoming wafer data.</li>")
        html_parts.append("</ol></section><hr>")

        # Section 9 – Technical Appendix
        html_parts.append('<section id="appendix">')
        html_parts.append("<h2>9. Technical Appendix</h2>")
        cfg_dict = all_results.get("config", self.cfg.to_dict())
        html_parts.append("<h4>Parameters</h4>")
        html_parts.append("<pre>" + json.dumps(cfg_dict, indent=2, default=str) + "</pre>")
        html_parts.append(f"<p>Report generated: {datetime.now().isoformat()}</p>")
        html_parts.append("</section>")

        html_parts.append(self._footer())

        # ── Write file ───────────────────────────────────────────────
        os.makedirs(self.cfg.reports_dir, exist_ok=True)
        path = os.path.join(self.cfg.reports_dir, "secom_mspc_complete_report.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(html_parts))

        print(f"  ✓ HTML report saved → {path}")

    # ─── HTML template helpers ───────────────────────────────────────
    @staticmethod
    def _header() -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SECOM MSPC – Complete Report</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
      rel="stylesheet">
<style>
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
       padding: 30px 60px; max-width: 1200px; margin: auto; }
h2 { margin-top: 40px; color: #2c3e50; }
img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }
pre { background: #f8f9fa; padding: 15px; border-radius: 6px; font-size: 13px;
      max-height: 400px; overflow-y: auto; }
table { font-size: 13px; }
hr { border-top: 2px solid #e0e0e0; margin: 30px 0; }
</style>
</head>
<body>
<h1 class="mb-4">🏭 SECOM MSPC – Complete Analysis Report</h1>
<p class="lead">Industry 4.0 Multivariate Statistical Process Control System<br>
Semiconductor Manufacturing Quality Prediction</p>
<hr>
"""

    @staticmethod
    def _footer() -> str:
        return """
<footer class="mt-5 pt-3 border-top text-muted small">
<p>Generated by the SECOM MSPC Pipeline &bull;
   <a href="https://archive.ics.uci.edu/ml/datasets/SECOM">UCI SECOM Dataset</a></p>
</footer>
</body>
</html>"""
