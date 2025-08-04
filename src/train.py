from src.models.Optimizers import OPTIMIZER_CLASSES
from src.models.DenseLayer import DenseLayer
from src.config import MODEL_CONFIGS
from src.models.Model import Model
import numpy as np


def main() -> None:
    """
    Main function to train the model with different optimizers
    and save the trained models.
    """
    try: 
        X_train = np.load("saved/X_train.npy")
        y_train = np.load("saved/y_train.npy")

        for config in MODEL_CONFIGS:
            optimizer_class = OPTIMIZER_CLASSES[config["optimizer"]]
            optimizer = optimizer_class(**config.get("optimizer_params", {}))

            model = Model(name=config["name"], optimizer=optimizer)
            layers = [
                DenseLayer(**layer) for layer in config["layers"]
            ]
            network = model.create_network(layers, input_dim=X_train.shape[1])

            model.fit(
                network,
                X_train,
                y_train,
                **config.get("fit_params", {})
            )

            model_path = f"trained_models/{config['name']}"
            model.save(network, model_path, config=config)
    
    except Exception as e:
        print(f"An error occurred during training: {e}")


if __name__ == "__main__":
    main()
