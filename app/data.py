"""Data loading utilities for the breast cancer binary classification dataset."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data(as_frame: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the scikit-learn breast cancer dataset.

    Args:
        as_frame: If ``True``, returns pandas objects for readability.

    Returns:
        Tuple containing feature matrix ``X`` and target vector ``y``.
    """

    dataset = load_breast_cancer(as_frame=as_frame)
    X = dataset.data
    y = dataset.target
    return X, y
