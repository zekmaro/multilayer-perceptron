from src.models.Preprocessing import Preprocessing
from src.models.Visualizer import Visualizer
from src.header import (
    MEAN_FEATURES,
    LABEL_MAPPING,
    DROP_COLUMNS,
    DATA_PATH,
    COLUMNS
)
import pandas as pd
import numpy as np
import os


def explore_dataset(df: pd.DataFrame, visualizer: Visualizer) -> None:
    """
    Explore the dataset using various visualizations.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        visualizer (Visualizer): Instance of the Visualizer class.
    """
    os.makedirs("plots", exist_ok=True)
    visualizer.plot_correlation_matrix(df.corr())
    visualizer.plot_histograms(df)
    df_mean = pd.DataFrame(df, columns=MEAN_FEATURES + ["diagnosis"])
    visualizer.plot_pairplot(df_mean, "diagnosis", "Pairplot of Mean Features")


def main() -> None:
    """
    Main function to process the dataset,
    split it into training and testing sets, and save the splits.
    """
    try:
        os.makedirs("saved", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        processor = Preprocessing(DATA_PATH)
        processor.load_data(header=True)
        processor.name_columns(COLUMNS)
        processor.extract_X_y(target_column="diagnosis", drop_columns=DROP_COLUMNS)
        processor.encode_target(LABEL_MAPPING)
        processor.normalize_features()

        df = processor.X.copy()
        df["diagnosis"] = processor.y

        visualizer = Visualizer()
        explore_dataset(df, visualizer)

        # x = pd.read_csv("data_training.csv")
        # x.columns = COLUMNS
        # processor.X = x.drop(columns=["diagnosis", "id"])
        # print(processor.X.columns)
        # processor.y = x["diagnosis"]
        # processor.y = processor.y.map(LABEL_MAPPING)
        # processor.normalize_features()

        X_train, y_train, X_test, y_test = processor.split_data()

        # Save splits
        np.save("saved/X_train.npy", X_train)
        np.save("saved/y_train.npy", y_train)
        np.save("saved/X_test.npy", X_test)
        np.save("saved/y_test.npy", y_test)

        print("Data successfully split and saved.")
    
    except Exception as e:
        print(f"An error occurred during data processing: {e}")


if __name__ == "__main__":
    main()
