import numpy as np

from app.data import load_data
from app.model import predict_labels, train_model
from app.preprocess import split_and_scale


def test_model_training_and_prediction():
    X, y = load_data()
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)
    model = train_model(X_train, y_train)
    preds = predict_labels(model, X_test)
    assert preds.shape[0] == y_test.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})
