from src.models.Preprocessing import Preprocessing
from src.models.DenseLayer import DenseLayer
from src.models.Visualizer import Visualizer
from src.models.Model import Model
from src.config import MODEL_CONFIGS
from src.header import (
    ACCURACY_VALUES_MAP,
    LOSS_VALUES_MAP,
    LABEL_MAPPING,
    MEAN_FEATURES,
    DROP_COLUMNS,
	DATA_PATH,
    COLUMNS,
)
import pandas as pd
import numpy as np


def load_and_prepare_data(processor: Preprocessing, target_column: str) -> None:
    """
    Load and prepare the dataset for analysis.

    Args:
        processor (Preprocessing): Instance of the Preprocessing class.
        target_column (str): Name of the target column to extract.
    """
    processor.load_data(header=True)
    processor.name_columns(COLUMNS)
    processor.extract_X_y(target_column=target_column, drop_columns=DROP_COLUMNS)
    processor.encode_target(LABEL_MAPPING)
    processor.check_nulls(processor.df)
    processor.check_uniqueness(processor.df)
    print(processor.df.describe().T)


def explore_dataset(df: pd.DataFrame, visualizer: Visualizer) -> None:
    """
    Explore the dataset using various visualizations.
    
    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        visualizer (Visualizer): Instance of the Visualizer class.
    """
    visualizer.plot_correlation_matrix(df.corr())
    visualizer.plot_histograms(df)
    df_mean = pd.DataFrame(df, columns=MEAN_FEATURES + ["diagnosis"])
    visualizer.plot_pairplot(df_mean, "diagnosis", "Pairplot of Mean Features")


def train_models(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        visualizer: Visualizer
    ) -> None:
    """
    Train models based on the configurations defined in MODEL_CONFIGS.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (np.dnarray): Training labels.
        X_test (pd.DataFrame): Testing features.
        y_test (np.dnarray): Testing labels.
    """
    for config in MODEL_CONFIGS:
        model = Model(name=config["name"])
        layers = [DenseLayer(**layer) for layer in config["layers"]]
        network = model.create_network(layers)

        model.fit(
            network,
            X_train,
            y_train,
            epochs=config["params"]["epochs"],
            batch_size=config["params"]["batch_size"],
            learning_rate=config["params"]["learning_rate"]
        )

        LOSS_VALUES_MAP[config["params"]["algorithm"]].append(model.loss_history)
        ACCURACY_VALUES_MAP[config["params"]["algorithm"]].append(model.accuracy_history)

        y_pred = model.predict(network, X_test)

        accuracy = model.get_accuracy(y_pred, y_test)
        precision = model.get_precision(y_pred, y_test)
        recall = model.get_recall(y_pred, y_test)
        F1_score = model.get_f1_score(y_pred, y_test)

        print(f"Model {config['name']} accuracy: {accuracy:.2f}")
        print(f"Model {config['name']} precision: {precision:.2f}")
        print(f"Model {config['name']} recall: {recall:.2f}")
        print(f"Model {config['name']} F1 score: {F1_score:.2f}")

    visualizer.compare_loss_histories(LOSS_VALUES_MAP)
    visualizer.compare_accuracy_histories(ACCURACY_VALUES_MAP)


def main():
    try: 
        processor = Preprocessing(DATA_PATH)
        load_and_prepare_data(processor, target_column="diagnosis")

        df = processor.df.copy()
        df['diagnosis'] = processor.y

        visualizer = Visualizer()
        explore_dataset(df, visualizer)

        processor.X = df.drop(columns=['diagnosis', 'id'])
        processor.y = df['diagnosis']

        processor.normalize_features()
        X_train, y_train, X_test, y_test = processor.split_data()
        print(X_train.shape)
        print(y_train.shape)

        train_models(X_train, y_train, X_test, y_test, visualizer)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
