import numpy as np
import os
import pickle
from src.models.DenseLayer import DenseLayer
from src.models.Model import Model
from src.models.Optimizers import OPTIMIZER_CLASSES
from src.config import MODEL_CONFIGS

def main():
    X_train = np.load("saved/X_train.npy")
    y_train = np.load("saved/y_train.npy")

    for config in MODEL_CONFIGS:
        optimizer_class = OPTIMIZER_CLASSES[config["optimizer"]]
        optimizer = optimizer_class(**config.get("optimizer_params", {}))

        model = Model(name=config["name"], optimizer=optimizer)
        layers = [
            DenseLayer(16, activation_name='relu', input_dim=X_train.shape[1]),
            DenseLayer(8, activation_name='relu'),
            DenseLayer(2, activation_name='softmax')
        ]
        network = model.create_network(layers)

        model.fit(
            network,
            X_train,
            y_train,
            **config.get("fit_params", {})
        )

        model_filename = f"trained_models/{config['name']}.pkl"
        os.makedirs("trained_models", exist_ok=True)
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)

        print(f"Saved model: {model_filename}")

if __name__ == "__main__":
    main()
