"""Preprocessing utilities: split data and apply optional scaling."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler | None]:
    """Split the dataset and optionally scale features.

    Standardizing features helps solvers like ``lbfgs`` converge faster because
    gradient updates operate on features with similar ranges.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction of data reserved for testing.
        random_state: Seed for reproducibility.
        scale: Whether to apply ``StandardScaler`` to the features.

    Returns:
        ``(X_train, X_test, y_train, y_test, scaler)`` tuple. The scaler is
        ``None`` when scaling is disabled.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler: StandardScaler | None = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), index=X_test.index, columns=X_test.columns
        )

    return X_train, X_test, y_train, y_test, scaler
