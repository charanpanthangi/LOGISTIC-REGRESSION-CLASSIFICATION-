"""Visualization utilities for classification results."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


def plot_confusion_matrix(cm: np.ndarray, labels: tuple[str, str] = ("negative", "positive")) -> Path:
    """Plot and save a confusion matrix heatmap."""

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    output_path = OUTPUTS_DIR / "confusion_matrix.svg"
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Path:
    """Plot and save the ROC curve."""

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_scores, ax=ax)
    ax.set_title("Receiver Operating Characteristic")
    output_path = OUTPUTS_DIR / "roc_curve.svg"
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def plot_precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Path:
    """Plot and save the Precision-Recall curve."""

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_scores, ax=ax)
    ax.set_title("Precision-Recall Curve")
    output_path = OUTPUTS_DIR / "precision_recall_curve.svg"
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path
