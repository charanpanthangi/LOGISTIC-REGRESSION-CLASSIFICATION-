from app.data import load_data


def test_load_data_shapes():
    X, y = load_data()
    assert len(X) == len(y)
    assert X.shape[0] > 0
    assert X.shape[1] == 30
