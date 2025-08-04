from src.header import LOSS_VALUES_MAP, ACCURACY_VALUES_MAP
from src.models.Visualizer import Visualizer
import numpy as np
import pickle
import os


def evaluate_model(
        model_path: str,
        network_path: str,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> None:
    """
    Evaluate the model using the test data and print the results.
    
    Args:
        model_path (str): Path to the saved model.
        network_path (str): Path to the saved network.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(network_path, "rb") as f:
        network = pickle.load(f)

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

    LOSS_VALUES_MAP[model.name] = model.loss_history
    ACCURACY_VALUES_MAP[model.name] = model.accuracy_history


def main() -> None:
    """Main function to evaluate all trained models and visualize results."""
    try:
        os.makedirs("trained_models", exist_ok=True)
        X_test = np.load("saved/X_test.npy")
        y_test = np.load("saved/y_test.npy")

        model_paths = [
            "trained_models/gradient_descent/model.pkl",
            "trained_models/rmsprop/model.pkl",
            "trained_models/adam/model.pkl",
            "trained_models/momentum/model.pkl"
        ]
        network_paths = [
            "trained_models/gradient_descent/network.pkl",
            "trained_models/rmsprop/network.pkl",
            "trained_models/adam/network.pkl",
            "trained_models/momentum/network.pkl"
        ]

        for model_path, network_path in zip(model_paths, network_paths):
            evaluate_model(model_path, network_path, X_test, y_test)
        
        visualizer = Visualizer()
        visualizer.compare_loss_histories(LOSS_VALUES_MAP)
        visualizer.compare_accuracy_histories(ACCURACY_VALUES_MAP)

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
