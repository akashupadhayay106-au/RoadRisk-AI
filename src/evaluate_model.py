"""
RoadRisk AI — Model Evaluation & Metrics Module
================================================
Comprehensive evaluation metrics, per-class diagnostics,
confusion matrix computations, and Plotly visualization builders.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import plotly.figure_factory as ff
import plotly.graph_objects as go


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics with emphasis on Macro-F1
    due to severe class imbalance (Slight 86%, Serious 13%, Fatal 1%).
    """
    if labels is None:
        labels = sorted(list(set(y_true)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class metrics
    p_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    r_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class_metrics = {}
    for i, label in enumerate(labels):
        per_class_metrics[str(label)] = {
            "precision": round(float(p_per_class[i]), 4),
            "recall": round(float(r_per_class[i]), 4),
            "f1_score": round(float(f1_per_class[i]), 4),
        }

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "classes": labels,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_normalized, 4).tolist(),
    }


def plot_confusion_matrix_plotly(
    cm: List[List[int | float]],
    class_names: List[str],
    normalize: bool = False,
    title: str = "Confusion Matrix"
) -> go.Figure:
    """
    Render a polished Plotly interactive confusion matrix heatmap.
    """
    z = np.array(cm)
    if normalize:
        text = [[f"{val:.2%}" for val in row] for row in z]
        colorscale = "Blues"
    else:
        text = [[f"{int(val):,}" for val in row] for row in z]
        colorscale = "Reds"

    fig = ff.create_annotated_heatmap(
        z=z,
        x=class_names,
        y=class_names,
        annotation_text=text,
        colorscale=colorscale,
        showscale=True,
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Predicted Severity Class"),
        yaxis=dict(title="True Severity Class", autorange="reversed"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig
