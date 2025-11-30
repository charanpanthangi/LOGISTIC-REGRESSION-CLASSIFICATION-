# Logistic Regression for Binary Classification

A beginner-friendly tutorial and production template that demonstrates logistic regression for binary classification using scikit-learn. The example uses the built-in **breast cancer** dataset to predict whether a tumor is malignant or benign.

## What is Logistic Regression?

Logistic regression is a linear model that predicts the probability of a binary outcome. It applies the **sigmoid function** to a linear combination of input features so the output stays between 0 and 1. A default decision boundary of 0.5 converts those probabilities into class labels (0 or 1). Because the model is linear in the weights, it is fast to train and interpret.

Key concepts:
- **Sigmoid function**: \(\sigma(z) = 1 / (1 + e^{-z})\). Maps any real number to (0, 1).
- **Decision boundary**: Threshold (commonly 0.5) used to turn probabilities into class predictions.
- **Probabilities vs. labels**: The model outputs probabilities; you pick a threshold to obtain labels.
- **Regularization**: Penalizes large weights to reduce overfitting. The default L2 regularization in scikit-learn works well for most cases.
- **Feature scaling**: Standardizing features helps solvers such as `lbfgs` converge faster and keeps coefficients comparable.

## When to Use It
- Binary classification problems (spam vs. not spam, churn vs. retain, pass vs. fail).
- When you want probabilities, not just labels.
- When interpretability and speed matter.

## Dataset
- **Source**: `sklearn.datasets.load_breast_cancer`
- **Task**: Predict if a breast cancer tumor is malignant (1) or benign (0).
- **Features**: 30 numeric measurements extracted from digitized images of fine needle aspirates.

## Project Structure
```
<repo-root>/
├── app/
│   ├── __init__.py
│   ├── data.py
│   ├── preprocess.py
│   ├── model.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── main.py
├── notebooks/
│   └── demo_logistic_regression.ipynb
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_evaluate.py
├── examples/
│   └── README_examples.md
├── requirements.txt
├── Dockerfile
├── .gitignore
├── LICENSE
└── README.md
```

## Training Pipeline
1. **Load data** (`app/data.py`).
2. **Split & scale** (`app/preprocess.py`) using `StandardScaler` (helps optimization).
3. **Train model** (`app/model.py`) with scikit-learn `LogisticRegression` (L2 regularization, `lbfgs` solver).
4. **Evaluate** (`app/evaluate.py`) with accuracy, precision, recall, F1-score, and confusion matrix.
5. **Visualize** (`app/visualize.py`) confusion matrix, ROC curve, and Precision-Recall curve (saved as SVG).
6. **Run the pipeline** (`app/main.py`).

## How to Run
### Local (Python 3.10+)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app/main.py
```

### Notebook Demo
```bash
jupyter notebook notebooks/demo_logistic_regression.ipynb
```

### Run Tests
```bash
pytest
```

### Docker
Build and run:
```bash
docker build -t logistic-regression-demo .
docker run --rm logistic-regression-demo
```

## Evaluation Metrics
- **Accuracy**: Overall correctness.
- **Precision**: Of predicted positives, how many are truly positive.
- **Recall**: Of true positives, how many are captured.
- **F1-score**: Harmonic mean of precision and recall.
- **Confusion matrix**: Counts of true/false positives/negatives.
- **ROC & PR curves**: Show trade-offs at different thresholds.

## Future Extensions
- Multiclass classification via one-vs-rest.
- Regularization tuning (adjust `C` and penalty types).
- Handling class imbalance with class weights or resampling.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
