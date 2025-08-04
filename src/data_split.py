from src.models.Preprocessing import Preprocessing
from src.models.Visualizer import Visualizer
from src.header import COLUMNS, DROP_COLUMNS, LABEL_MAPPING, DATA_PATH, MEAN_FEATURES
import numpy as np
import pandas as pd


def explore_dataset(df: pd.DataFrame, visualizer: Visualizer) -> None:
    """
    Explore the dataset using various visualizations.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        visualizer (Visualizer): Instance of the Visualizer class.
    """
    # Correlation heatmap
    visualizer.plot_correlation_matrix(df.corr())

    # Histogram for each feature
    visualizer.plot_histograms(df)

    # Pairplot for selected mean features
    df_mean = pd.DataFrame(df, columns=MEAN_FEATURES + ["diagnosis"])
    visualizer.plot_pairplot(df_mean, "diagnosis", "Pairplot of Mean Features")


def main():
    try:
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
