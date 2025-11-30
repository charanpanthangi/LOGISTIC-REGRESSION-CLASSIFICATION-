"""Run a full logistic regression workflow on the breast cancer dataset."""
from __future__ import annotations

from pathlib import Path

from app.data import load_data
from app.evaluate import evaluate_classification
from app.model import predict_labels, predict_proba, train_model
from app.preprocess import split_and_scale
from app.visualize import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)


def run_pipeline(scale_features: bool = True) -> None:
    """Execute the end-to-end training, evaluation, and visualization pipeline."""

    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, scale=scale_features)

    model = train_model(X_train, y_train)
    y_test_proba = predict_proba(model, X_test)
    y_test_pred = predict_labels(model, X_test)

    metrics = evaluate_classification(y_test, y_test_pred)

    print("\n=== Evaluation Metrics ===")
    for key, value in metrics.items():
        if key == "confusion_matrix":
            print(f"{key}:\n{value}")
        else:
            print(f"{key}: {value:.3f}")

    cm_path = plot_confusion_matrix(metrics["confusion_matrix"])
    roc_path = plot_roc_curve(y_test, y_test_proba)
    pr_path = plot_precision_recall_curve(y_test, y_test_proba)

    print("\nSaved visualizations:")
    for path in (cm_path, roc_path, pr_path):
        print(f"- {path}")


if __name__ == "__main__":
    run_pipeline(scale_features=True)
