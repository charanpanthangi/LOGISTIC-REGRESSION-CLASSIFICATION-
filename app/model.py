"""Model training utilities using scikit-learn's LogisticRegression."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    solver: str = "lbfgs",
    max_iter: int = 200,
    random_state: int | None = 42,
) -> LogisticRegression:
    """Train a logistic regression classifier.

    Logistic regression maps inputs through the sigmoid function to produce
    probabilities between 0 and 1. A decision boundary of 0.5 translates these
    probabilities into class labels. Regularization (L2 by default) prevents
    overfitting by penalizing large weights.

    Args:
        X_train: Training features.
        y_train: Training labels.
        solver: Optimization algorithm. ``lbfgs`` handles L2 regularization well.
        max_iter: Maximum iterations for solver convergence.
        random_state: Seed for reproducibility.

    Returns:
        Trained ``LogisticRegression`` model.
    """

    model = LogisticRegression(
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def predict_proba(model: LogisticRegression, X: pd.DataFrame) -> np.ndarray:
    """Predict class probabilities using the trained model."""

    return model.predict_proba(X)[:, 1]


def predict_labels(model: LogisticRegression, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """Convert probabilities to class labels based on a decision threshold."""

    probabilities = predict_proba(model, X)
    return (probabilities >= threshold).astype(int)
