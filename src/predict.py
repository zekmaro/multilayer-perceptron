import numpy as np
import pickle
from src.models.Model import Model

def evaluate_model(model_path, X_test, y_test):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(network, X_test)
    one_hot = np.zeros_like(y_pred)
    one_hot[np.arange(len(y_test)), y_test] = 1

    acc = model.get_accuracy(y_pred, y_test)
    precision = model.get_precision(y_pred, one_hot)
    recall = model.get_recall(y_pred, one_hot)
    f1 = model.get_f1_score(y_pred, one_hot)

    print(f"Model: {model_path}")
    print(f"  Accuracy:  {acc:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall:    {recall:.2f}")
    print(f"  F1 Score:  {f1:.2f}")
    print()

def main():
    X_test = np.load("saved/X_test.npy")
    y_test = np.load("saved/y_test.npy")

    model_paths = [
        "trained_models/momentum.pkl",
        "trained_models/rmsprop.pkl",
        "trained_models/adam.pkl",
        "trained_models/gradient_descent.pkl"
    ]

    for path in model_paths:
        evaluate_model(path, X_test, y_test)

if __name__ == "__main__":
    main()
